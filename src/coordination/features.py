"""
Multi-Evidence Coordination Features.

Computes pairwise similarity features across multiple evidence dimensions:
- Funding similarity (shared funder, amount, timing)
- Trading similarity (buy time, sequences, cadence, DEX routes)
- Behavioral similarity (holding, position size, profit-taking, exit)
- Cross-token similarity (co-participation, history, entity reuse)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence
from collections import defaultdict
import math
import statistics

import structlog

logger = structlog.get_logger()


@dataclass
class FundingSimilarityFeatures:
    """Funding-based coordination features for a wallet pair."""

    # Shared funder evidence
    shared_funder_binary: bool = False
    shared_funder_depth: int = 0  # 0 = no shared funder, 1 = direct, 2+ = indirect
    funder_overlap_jaccard: float = 0.0  # Jaccard similarity of funder sets

    # Amount similarity
    funding_amount_similarity: float = 0.0  # 0-1, 1 = identical amounts
    funding_amount_ratio: float = 0.0  # ratio of smaller to larger amount

    # Timing similarity
    funding_time_similarity: float = 0.0  # 0-1, 1 = simultaneous
    funding_time_gap_seconds: float = float("inf")


@dataclass
class TradingSimilarityFeatures:
    """Trading-based coordination features for a wallet pair."""

    # First buy timing
    first_buy_time_similarity: float = 0.0  # 0-1, 1 = simultaneous
    first_buy_time_gap_seconds: float = float("inf")

    # Sequence similarity
    buy_sequence_similarity: float = 0.0  # 0-1, correlation of buy sequences
    sell_sequence_similarity: float = 0.0  # 0-1, correlation of sell sequences
    trade_cadence_similarity: float = 0.0  # 0-1, similar trade frequency

    # Route similarity
    dex_route_similarity: float = 0.0  # 0-1, same DEX paths


@dataclass
class BehavioralSimilarityFeatures:
    """Behavioral coordination features for a wallet pair."""

    # Holding patterns
    holding_duration_similarity: float = 0.0  # 0-1
    position_size_similarity: float = 0.0  # 0-1

    # Profit-taking behavior
    profit_taking_similarity: float = 0.0  # 0-1, similar exit %-ages
    exit_timing_similarity: float = 0.0  # 0-1, exit at similar times


@dataclass
class CrossTokenSimilarityFeatures:
    """Cross-token coordination features for a wallet pair."""

    # Co-participation across tokens
    repeated_co_participation_count: int = 0  # Number of tokens both wallets traded
    shared_previous_tokens: int = 0  # Tokens traded before current one

    # Historical correlation
    historical_exit_correlation: float = 0.0  # Correlation of exits across tokens
    entity_reuse_score: float = 0.0  # Evidence of entity reuse patterns


@dataclass
class CoordinationFeatures:
    """Complete coordination feature set for a wallet pair."""

    wallet1: str
    wallet2: str

    funding: FundingSimilarityFeatures = field(default_factory=FundingSimilarityFeatures)
    trading: TradingSimilarityFeatures = field(default_factory=TradingSimilarityFeatures)
    behavioral: BehavioralSimilarityFeatures = field(default_factory=BehavioralSimilarityFeatures)
    cross_token: CrossTokenSimilarityFeatures = field(default_factory=CrossTokenSimilarityFeatures)

    # Computed aggregate
    evidence_types_present: int = 0
    evidence_vector: list[float] = field(default_factory=list)

    def count_evidence_types(self) -> int:
        """Count number of evidence types with non-trivial values."""
        count = 0

        # Funding evidence
        if self.funding.shared_funder_binary:
            count += 1
        if self.funding.funding_amount_similarity > 0.3:
            count += 1
        if self.funding.funding_time_similarity > 0.3:
            count += 1

        # Trading evidence
        if self.trading.first_buy_time_similarity > 0.3:
            count += 1
        if self.trading.buy_sequence_similarity > 0.3:
            count += 1

        # Behavioral evidence
        if self.behavioral.exit_timing_similarity > 0.3:
            count += 1

        # Cross-token evidence
        if self.cross_token.repeated_co_participation_count >= 2:
            count += 1

        return count

    def to_vector(self) -> list[float]:
        """Convert all features to a numeric vector for scoring."""
        return [
            # Funding (4 features)
            1.0 if self.funding.shared_funder_binary else 0.0,
            self.funding.funder_overlap_jaccard,
            self.funding.funding_amount_similarity,
            self.funding.funding_time_similarity,
            # Trading (4 features)
            self.trading.first_buy_time_similarity,
            self.trading.buy_sequence_similarity,
            self.trading.trade_cadence_similarity,
            self.trading.dex_route_similarity,
            # Behavioral (4 features)
            self.behavioral.holding_duration_similarity,
            self.behavioral.position_size_similarity,
            self.behavioral.profit_taking_similarity,
            self.behavioral.exit_timing_similarity,
            # Cross-token (2 features)
            min(self.cross_token.repeated_co_participation_count / 10.0, 1.0),
            self.cross_token.entity_reuse_score,
        ]


def _time_similarity(t1: Optional[datetime], t2: Optional[datetime], max_gap_hours: float = 24.0) -> tuple[float, float]:
    """
    Compute time similarity and gap.

    Returns (similarity, gap_seconds) where similarity is 1.0 for identical times
    and decays exponentially with time gap.
    """
    if t1 is None or t2 is None:
        return 0.0, float("inf")

    gap_seconds = abs((t1 - t2).total_seconds())
    max_gap_seconds = max_gap_hours * 3600

    if gap_seconds > max_gap_seconds:
        return 0.0, gap_seconds

    # Exponential decay: similarity = exp(-gap / scale)
    # Scale chosen so similarity ~0.1 at max_gap
    scale = max_gap_seconds / 2.3  # ln(10) ≈ 2.3
    similarity = math.exp(-gap_seconds / scale)

    return similarity, gap_seconds


def _amount_similarity(a1: float, a2: float) -> tuple[float, float]:
    """
    Compute amount similarity.

    Returns (similarity, ratio) where similarity is 1.0 for identical amounts.
    Uses log-scale comparison to handle different magnitudes.
    """
    if a1 <= 0 or a2 <= 0:
        return 0.0, 0.0

    # Ratio of smaller to larger
    ratio = min(a1, a2) / max(a1, a2)

    # Log-scale similarity: identical = 1.0, 10x difference ≈ 0.5
    log_ratio = math.log10(max(a1, a2) / min(a1, a2)) if min(a1, a2) > 0 else float("inf")
    similarity = 1.0 / (1.0 + log_ratio)

    return similarity, ratio


def _sequence_similarity(seq1: list[float], seq2: list[float]) -> float:
    """
    Compute similarity between two sequences using Pearson correlation.

    Returns 0.0 if sequences are too short or incomparable.
    """
    if len(seq1) < 2 or len(seq2) < 2:
        return 0.0

    # Align sequences to same length (truncate longer)
    min_len = min(len(seq1), len(seq2))
    s1 = seq1[:min_len]
    s2 = seq2[:min_len]

    if min_len < 2:
        return 0.0

    # Pearson correlation
    try:
        mean1 = statistics.mean(s1)
        mean2 = statistics.mean(s2)
        std1 = statistics.stdev(s1) if len(s1) > 1 else 0
        std2 = statistics.stdev(s2) if len(s2) > 1 else 0

        if std1 == 0 or std2 == 0:
            return 0.0

        covariance = sum((a - mean1) * (b - mean2) for a, b in zip(s1, s2)) / len(s1)
        correlation = covariance / (std1 * std2)

        # Convert correlation [-1, 1] to similarity [0, 1]
        # Both positive and negative correlation indicate coordination
        return abs(correlation)
    except (statistics.StatisticsError, ZeroDivisionError):
        return 0.0


def _jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


@dataclass
class WalletContext:
    """Context data for a single wallet used in feature computation."""

    address: str

    # Funding data
    funders: set[str] = field(default_factory=set)
    funding_amounts: list[float] = field(default_factory=list)
    funding_times: list[datetime] = field(default_factory=list)
    earliest_funding_time: Optional[datetime] = None
    total_funding: float = 0.0

    # Trading data
    first_buy_time: Optional[datetime] = None
    buy_timestamps: list[datetime] = field(default_factory=list)
    sell_timestamps: list[datetime] = field(default_factory=list)
    buy_amounts: list[float] = field(default_factory=list)
    sell_amounts: list[float] = field(default_factory=list)
    dex_routes: set[str] = field(default_factory=set)

    # Behavioral data
    holding_duration_days: float = 0.0
    position_size: float = 0.0
    profit_pct: float = 0.0
    exit_time: Optional[datetime] = None

    # Cross-token data
    tokens_traded: set[str] = field(default_factory=set)
    exit_times_by_token: dict[str, datetime] = field(default_factory=dict)


def compute_funding_features(
    ctx1: WalletContext,
    ctx2: WalletContext,
) -> FundingSimilarityFeatures:
    """Compute funding similarity features between two wallets."""
    features = FundingSimilarityFeatures()

    # Shared funder
    shared_funders = ctx1.funders & ctx2.funders
    features.shared_funder_binary = len(shared_funders) > 0
    features.funder_overlap_jaccard = _jaccard_similarity(ctx1.funders, ctx2.funders)

    # For shared funder depth, we'd need the full graph - set to 1 if binary is True
    if features.shared_funder_binary:
        features.shared_funder_depth = 1

    # Amount similarity
    if ctx1.total_funding > 0 and ctx2.total_funding > 0:
        features.funding_amount_similarity, features.funding_amount_ratio = _amount_similarity(
            ctx1.total_funding, ctx2.total_funding
        )

    # Timing similarity
    features.funding_time_similarity, features.funding_time_gap_seconds = _time_similarity(
        ctx1.earliest_funding_time, ctx2.earliest_funding_time, max_gap_hours=24.0
    )

    return features


def compute_trading_features(
    ctx1: WalletContext,
    ctx2: WalletContext,
) -> TradingSimilarityFeatures:
    """Compute trading similarity features between two wallets."""
    features = TradingSimilarityFeatures()

    # First buy timing
    features.first_buy_time_similarity, features.first_buy_time_gap_seconds = _time_similarity(
        ctx1.first_buy_time, ctx2.first_buy_time, max_gap_hours=1.0
    )

    # Buy sequence similarity (compare buy amount sequences)
    features.buy_sequence_similarity = _sequence_similarity(
        ctx1.buy_amounts, ctx2.buy_amounts
    )

    # Sell sequence similarity
    features.sell_sequence_similarity = _sequence_similarity(
        ctx1.sell_amounts, ctx2.sell_amounts
    )

    # Trade cadence similarity
    if ctx1.buy_timestamps and ctx2.buy_timestamps:
        # Compare number of trades (simple cadence measure)
        n1 = len(ctx1.buy_timestamps) + len(ctx1.sell_timestamps)
        n2 = len(ctx2.buy_timestamps) + len(ctx2.sell_timestamps)
        if n1 > 0 and n2 > 0:
            features.trade_cadence_similarity = min(n1, n2) / max(n1, n2)

    # DEX route similarity
    features.dex_route_similarity = _jaccard_similarity(ctx1.dex_routes, ctx2.dex_routes)

    return features


def compute_behavioral_features(
    ctx1: WalletContext,
    ctx2: WalletContext,
) -> BehavioralSimilarityFeatures:
    """Compute behavioral similarity features between two wallets."""
    features = BehavioralSimilarityFeatures()

    # Holding duration similarity
    if ctx1.holding_duration_days > 0 and ctx2.holding_duration_days > 0:
        ratio = min(ctx1.holding_duration_days, ctx2.holding_duration_days) / max(
            ctx1.holding_duration_days, ctx2.holding_duration_days
        )
        features.holding_duration_similarity = ratio

    # Position size similarity
    if ctx1.position_size > 0 and ctx2.position_size > 0:
        features.position_size_similarity, _ = _amount_similarity(
            ctx1.position_size, ctx2.position_size
        )

    # Profit-taking similarity (compare exit %ages)
    if ctx1.profit_pct != 0 or ctx2.profit_pct != 0:
        # Both made similar profit/loss
        if (ctx1.profit_pct > 0) == (ctx2.profit_pct > 0):  # Same direction
            if abs(ctx1.profit_pct) > 0 and abs(ctx2.profit_pct) > 0:
                ratio = min(abs(ctx1.profit_pct), abs(ctx2.profit_pct)) / max(
                    abs(ctx1.profit_pct), abs(ctx2.profit_pct)
                )
                features.profit_taking_similarity = ratio

    # Exit timing similarity
    features.exit_timing_similarity, _ = _time_similarity(
        ctx1.exit_time, ctx2.exit_time, max_gap_hours=24.0
    )

    return features


def compute_cross_token_features(
    ctx1: WalletContext,
    ctx2: WalletContext,
) -> CrossTokenSimilarityFeatures:
    """Compute cross-token similarity features between two wallets."""
    features = CrossTokenSimilarityFeatures()

    # Co-participation count
    shared_tokens = ctx1.tokens_traded & ctx2.tokens_traded
    features.repeated_co_participation_count = len(shared_tokens)
    features.shared_previous_tokens = len(shared_tokens)

    # Historical exit correlation
    if shared_tokens:
        exit_diffs = []
        for token in shared_tokens:
            t1 = ctx1.exit_times_by_token.get(token)
            t2 = ctx2.exit_times_by_token.get(token)
            if t1 and t2:
                gap_hours = abs((t1 - t2).total_seconds()) / 3600
                exit_diffs.append(gap_hours)

        if exit_diffs:
            # Average time gap between exits
            avg_gap = statistics.mean(exit_diffs)
            # Convert to similarity: 0 gap = 1.0, 24h gap = ~0.1
            features.historical_exit_correlation = math.exp(-avg_gap / 10.0)

    # Entity reuse score (based on co-participation strength)
    if features.repeated_co_participation_count >= 3:
        features.entity_reuse_score = min(features.repeated_co_participation_count / 10.0, 1.0)

    return features


def compute_pairwise_coordination_features(
    ctx1: WalletContext,
    ctx2: WalletContext,
) -> CoordinationFeatures:
    """
    Compute complete coordination features between two wallets.

    This is the main entry point for pairwise feature computation.
    """
    features = CoordinationFeatures(
        wallet1=ctx1.address,
        wallet2=ctx2.address,
    )

    features.funding = compute_funding_features(ctx1, ctx2)
    features.trading = compute_trading_features(ctx1, ctx2)
    features.behavioral = compute_behavioral_features(ctx1, ctx2)
    features.cross_token = compute_cross_token_features(ctx1, ctx2)

    features.evidence_types_present = features.count_evidence_types()
    features.evidence_vector = features.to_vector()

    return features


def build_wallet_context(
    address: str,
    funding_graph=None,
    trade_events: list = None,
    holder_data: dict = None,
) -> WalletContext:
    """
    Build WalletContext from available data sources.

    Args:
        address: Wallet address
        funding_graph: FundingGraph instance (optional)
        trade_events: List of TradeEvent objects (optional)
        holder_data: Dict with balance/position data (optional)

    Returns:
        WalletContext populated with available data
    """
    ctx = WalletContext(address=address)

    # Extract funding data from graph
    if funding_graph is not None:
        try:
            funders = funding_graph.get_funders(address)
            ctx.funders = set(funders) if funders else set()

            # Get funding edge data
            for funder in ctx.funders:
                edge_data = funding_graph._graph.edges.get((funder, address), {})
                if edge_data:
                    amount = edge_data.get("amount_lamports", edge_data.get("amount", 0))
                    if amount:
                        ctx.funding_amounts.append(float(amount))
                        ctx.total_funding += float(amount)
                    timestamp = edge_data.get("timestamp")
                    if timestamp:
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp)
                        ctx.funding_times.append(timestamp)
                        if ctx.earliest_funding_time is None or timestamp < ctx.earliest_funding_time:
                            ctx.earliest_funding_time = timestamp
        except Exception as e:
            logger.debug("funding_graph_extraction_error", wallet=address[:8], error=str(e))

    # Extract trading data
    if trade_events:
        wallet_events = [e for e in trade_events if e.wallet_address == address]
        buys = [e for e in wallet_events if e.trade_type == "buy"]
        sells = [e for e in wallet_events if e.trade_type == "sell"]

        ctx.buy_timestamps = [e.timestamp for e in buys]
        ctx.sell_timestamps = [e.timestamp for e in sells]

        if buys:
            ctx.first_buy_time = min(e.timestamp for e in buys)

        # Extract amounts if available
        for e in buys:
            if hasattr(e, "amount"):
                ctx.buy_amounts.append(float(e.amount))
        for e in sells:
            if hasattr(e, "amount"):
                ctx.sell_amounts.append(float(e.amount))

        # Extract DEX routes if available
        for e in wallet_events:
            if hasattr(e, "dex") and e.dex:
                ctx.dex_routes.add(e.dex)

        # Extract tokens traded
        ctx.tokens_traded = set(e.token_mint for e in wallet_events)

        # Extract exit times by token
        for e in sells:
            token = e.token_mint
            if token not in ctx.exit_times_by_token or e.timestamp > ctx.exit_times_by_token[token]:
                ctx.exit_times_by_token[token] = e.timestamp

        if sells:
            ctx.exit_time = max(e.timestamp for e in sells)

    # Extract holder data
    if holder_data:
        ctx.position_size = float(holder_data.get("balance", 0))
        ctx.holding_duration_days = float(holder_data.get("holding_duration_days", 0))
        ctx.profit_pct = float(holder_data.get("profit_pct", 0))

    return ctx
