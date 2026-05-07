"""
Analysis Orchestrator.

Coordinates the full analysis pipeline:
1. Data ingestion
2. Feature engineering
3. Metrics computation
4. Graph analysis
5. Archetype clustering
6. Risk scoring

Handles timeouts, errors, and partial results.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from ..core.config import settings
from ..core.types import TokenMint, HolderSnapshot
from ..data.client import SolanaDataClient
from ..graph import FundingGraph, detect_communities, find_shared_funders
from ..clustering import cluster_wallets, Archetype
from ..risk.scoring import RiskReport, generate_risk_report
from ..metrics.normalization import BaselineStatistics
from .features import FeatureEngineer, TemporalContext
from .metrics_pipeline import MetricsPipeline, MetricsResult

logger = structlog.get_logger()


@dataclass
class AnalysisResult:
    """Complete analysis result for a token."""

    # Token info
    mint: str
    holder_count: int
    total_supply: int

    # Computed data
    metrics: MetricsResult
    archetypes: dict[str, float]  # archetype -> proportion
    risk_report: RiskReport | None

    # Sell probabilities for top holders
    top_holder_sell_probs: list[float]

    # Graph stats
    graph_stats: dict[str, Any]

    # Metadata
    analysis_version: str
    computed_at: datetime
    latency_ms: int
    is_partial: bool
    warnings: list[str]

    def to_dict(self) -> dict:
        """Export as dictionary for API/Telegram output."""
        return {
            "mint": self.mint,
            "holder_count": self.holder_count,
            "total_supply": self.total_supply,
            "metrics": self.metrics.to_dict(),
            "archetypes": self.archetypes,
            "risk": self.risk_report.to_dict() if self.risk_report else None,
            "graph": self.graph_stats,
            "metadata": {
                "version": self.analysis_version,
                "computed_at": self.computed_at.isoformat(),
                "latency_ms": self.latency_ms,
                "is_partial": self.is_partial,
                "warnings": self.warnings,
            },
        }


class AnalysisOrchestrator:
    """
    Main orchestrator for token analysis.

    Coordinates all analysis steps with:
    - Timeout enforcement (SLA: 30s)
    - Error handling
    - Partial results fallback
    - Progress logging
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        data_client: SolanaDataClient | None = None,
        baseline: BaselineStatistics | None = None,
    ):
        self.data_client = data_client
        self.baseline = baseline or self._get_default_baseline()
        self.feature_engineer = FeatureEngineer()
        self.metrics_pipeline = MetricsPipeline()
        self._timeout = settings.sla_timeout_seconds

    async def analyze(
        self,
        mint: TokenMint,
        timeout: int | None = None,
    ) -> AnalysisResult:
        """
        Run full analysis for a token.

        Args:
            mint: Token mint address
            timeout: Optional timeout override (default: 30s)

        Returns:
            AnalysisResult with all computed data
        """
        start_time = datetime.now(timezone.utc)
        timeout = timeout or self._timeout
        warnings: list[str] = []
        is_partial = False

        logger.info("analysis_started", mint=mint, timeout=timeout)

        try:
            # Run with timeout
            result = await asyncio.wait_for(
                self._run_analysis(mint, warnings),
                timeout=timeout,
            )
            return result

        except asyncio.TimeoutError:
            logger.warning("analysis_timeout", mint=mint, timeout=timeout)
            warnings.append(f"Analysis timed out after {timeout}s")

            # Return partial results if available
            return await self._create_partial_result(mint, start_time, warnings)

        except Exception as e:
            logger.error("analysis_failed", mint=mint, error=str(e))
            raise

    async def _run_analysis(
        self,
        mint: TokenMint,
        warnings: list[str],
    ) -> AnalysisResult:
        """Run the full analysis pipeline."""
        start_time = datetime.now(timezone.utc)

        # Initialize client if needed
        if self.data_client is None:
            self.data_client = SolanaDataClient()

        # Step 1: Fetch holders
        logger.info("step_1_fetching_holders", mint=mint)
        snapshot = await self.data_client.get_token_holders(
            mint,
            limit=settings.max_holders_per_token,
        )

        if snapshot.holder_count > settings.max_holders_per_token:
            warnings.append(
                f"Sampled {settings.max_holders_per_token} of {snapshot.holder_count} holders"
            )

        # Step 2: Build funding graph
        logger.info("step_2_building_graph", holders=snapshot.holder_count)
        wallet_addresses = [b.wallet for b in snapshot.balances]
        funding_edges = await self.data_client.get_funding_edges(wallet_addresses[:500])

        funding_graph = FundingGraph()
        for wallet in wallet_addresses:
            funding_graph.add_wallet(wallet)
        funding_graph.add_edges_from_list(funding_edges)

        # Step 3: Compute features
        logger.info("step_3_computing_features")
        features = self.feature_engineer.compute_features(
            snapshot=snapshot,
            funding_graph=funding_graph,
            temporal_ctx=None,  # Would need historical data
        )

        # Step 4: Compute metrics
        logger.info("step_4_computing_metrics")

        # Find shared funders for coordination score
        shared_funders = find_shared_funders(funding_graph, wallet_addresses[:100])
        shared_funder_wallets: set[str] = set()
        for funder, funded in shared_funders.items():
            shared_funder_wallets.update(funded)

        metrics = self.metrics_pipeline.compute_all(
            snapshot=snapshot,
            funding_graph=funding_graph,
            cluster_wallets=wallet_addresses[:100],
            shared_funder_wallets=shared_funder_wallets,
        )

        # Step 5: Cluster into archetypes
        logger.info("step_5_clustering_archetypes")
        archetype_assignments = cluster_wallets(features)

        # Compute archetype distribution
        archetype_counts: dict[str, int] = {}
        for assignment in archetype_assignments.values():
            name = assignment.archetype.value
            archetype_counts[name] = archetype_counts.get(name, 0) + 1

        total = len(archetype_assignments)
        archetypes = {
            name: count / total
            for name, count in archetype_counts.items()
        } if total > 0 else {}

        # Step 6: Compute risk scores
        logger.info("step_6_computing_risk")

        # Placeholder sell probabilities (would come from hazard model)
        top_holder_sell_probs = [0.1] * min(10, len(features))

        # Build metrics dict for risk report
        metrics_dict = {
            "hhi": metrics.hhi,
            "entropy": metrics.shannon_entropy,
            "gini": metrics.gini_coefficient,
            "wdr": metrics.whale_dominance_ratio,
            "churn": metrics.churn_rate or metrics.hhi,  # Fallback
            "coordination": metrics.coordination_score or metrics.hhi,  # Fallback
            "funding_density": metrics.funding_density,
        }

        risk_report = generate_risk_report(
            mint=mint,
            metrics=metrics_dict,
            sell_probabilities=top_holder_sell_probs,
            baseline=self.baseline,
            liquidity_depth=None,  # Would need DEX data
            model_version=self.VERSION,
        )

        # Compute graph stats
        graph_stats = {
            "vertices": funding_graph.num_vertices,
            "edges": funding_graph.num_edges,
            "communities": len(detect_communities(funding_graph)),
            "shared_funder_count": len(shared_funders),
        }

        # Calculate latency
        end_time = datetime.now(timezone.utc)
        latency_ms = int((end_time - start_time).total_seconds() * 1000)

        logger.info(
            "analysis_completed",
            mint=mint,
            latency_ms=latency_ms,
            holders=snapshot.holder_count,
        )

        return AnalysisResult(
            mint=mint,
            holder_count=snapshot.holder_count,
            total_supply=snapshot.total_supply,
            metrics=metrics,
            archetypes=archetypes,
            risk_report=risk_report,
            top_holder_sell_probs=top_holder_sell_probs,
            graph_stats=graph_stats,
            analysis_version=self.VERSION,
            computed_at=end_time,
            latency_ms=latency_ms,
            is_partial=False,
            warnings=warnings,
        )

    async def _create_partial_result(
        self,
        mint: str,
        start_time: datetime,
        warnings: list[str],
    ) -> AnalysisResult:
        """Create partial result on timeout."""
        end_time = datetime.now(timezone.utc)
        latency_ms = int((end_time - start_time).total_seconds() * 1000)

        # Try to get at least basic holder info
        try:
            snapshot = await asyncio.wait_for(
                self.data_client.get_token_holders(mint, limit=100),
                timeout=5.0,
            )
            holder_count = snapshot.holder_count
            total_supply = snapshot.total_supply
        except Exception:
            holder_count = 0
            total_supply = 0

        warnings.append("Partial results only - full analysis timed out")

        # Return minimal result
        return AnalysisResult(
            mint=mint,
            holder_count=holder_count,
            total_supply=total_supply,
            metrics=None,  # type: ignore
            archetypes={},
            risk_report=None,
            top_holder_sell_probs=[],
            graph_stats={},
            analysis_version=self.VERSION,
            computed_at=end_time,
            latency_ms=latency_ms,
            is_partial=True,
            warnings=warnings,
        )

    def _get_default_baseline(self) -> BaselineStatistics:
        """Get default baseline statistics."""
        # These would come from database in production
        return BaselineStatistics(
            version="v1.0.0-default",
            hhi_mean=0.05,
            hhi_std=0.03,
            entropy_mean=4.5,
            entropy_std=1.2,
            gini_mean=0.75,
            gini_std=0.15,
            wdr_mean=0.30,
            wdr_std=0.15,
            churn_mean=0.10,
            churn_std=0.08,
            coordination_mean=0.15,
            coordination_std=0.10,
        )
