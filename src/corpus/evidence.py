"""
Evidence Package System (Sprint 10 - Deliverable 2).

Provides domain-specific evidence packages for each label type:
- Exit Event Evidence
- Coordination Evidence
- Wallet Behaviour Evidence
- Token Outcome Evidence
- Launch Trajectory Evidence
- Entity Link Evidence

Each evidence package supports:
- Structured data for the domain
- JSON serialization
- Confidence computation
- Human-readable summaries
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import structlog

from .schema import LabelDomain

logger = structlog.get_logger()


# ============================================================================
# Base Evidence Package
# ============================================================================


@dataclass
class EvidencePackage(ABC):
    """Base class for evidence packages."""

    domain: LabelDomain
    collected_at: datetime
    data_version: str

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        pass

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), default=str)

    @abstractmethod
    def summary(self) -> str:
        """Human-readable summary of evidence."""
        pass

    @abstractmethod
    def confidence_factors(self) -> list[str]:
        """List of factors contributing to confidence."""
        pass


# ============================================================================
# Exit Event Evidence
# ============================================================================


@dataclass
class ExitEventEvidence(EvidencePackage):
    """Evidence for exit event classification."""

    domain: LabelDomain = field(default=LabelDomain.EXIT_EVENT, init=False)

    # Transaction identity
    signature: str
    slot: int
    block_time: datetime | None

    # Token movement
    token_mint: str
    token_amount: int
    token_decimals: int

    # Balance changes
    token_balance_before: int
    token_balance_after: int
    sol_balance_change: int  # lamports

    # Quote asset
    quote_asset_received: bool
    quote_asset_mint: str | None
    quote_asset_amount: int | None

    # Program detection
    program_ids: list[str]
    dex_program: str | None
    lp_program: str | None
    bridge_program: str | None

    # Destination
    destination_address: str | None
    destination_type: str | None  # cex, wallet, pool, burn
    destination_is_known_cex: bool
    destination_cex_name: str | None

    # LP tokens
    lp_token_minted: bool
    lp_token_burned: bool
    lp_token_amount: int | None

    # Classifier output
    classifier_exit_type: str
    classifier_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "collected_at": self.collected_at.isoformat(),
            "data_version": self.data_version,
            "signature": self.signature,
            "slot": self.slot,
            "block_time": self.block_time.isoformat() if self.block_time else None,
            "token_mint": self.token_mint,
            "token_amount": self.token_amount,
            "token_decimals": self.token_decimals,
            "token_balance_before": self.token_balance_before,
            "token_balance_after": self.token_balance_after,
            "sol_balance_change": self.sol_balance_change,
            "quote_asset_received": self.quote_asset_received,
            "quote_asset_mint": self.quote_asset_mint,
            "quote_asset_amount": self.quote_asset_amount,
            "program_ids": self.program_ids,
            "dex_program": self.dex_program,
            "lp_program": self.lp_program,
            "bridge_program": self.bridge_program,
            "destination_address": self.destination_address,
            "destination_type": self.destination_type,
            "destination_is_known_cex": self.destination_is_known_cex,
            "destination_cex_name": self.destination_cex_name,
            "lp_token_minted": self.lp_token_minted,
            "lp_token_burned": self.lp_token_burned,
            "lp_token_amount": self.lp_token_amount,
            "classifier_exit_type": self.classifier_exit_type,
            "classifier_confidence": self.classifier_confidence,
        }

    def summary(self) -> str:
        amount_formatted = self.token_amount / (10 ** self.token_decimals)
        sol_change_formatted = self.sol_balance_change / 1e9

        parts = [
            f"Exit: {self.classifier_exit_type} (conf: {self.classifier_confidence:.2f})",
            f"Amount: {amount_formatted:.4f} tokens",
        ]

        if self.quote_asset_received:
            parts.append(f"SOL received: {sol_change_formatted:.4f}")
        if self.dex_program:
            parts.append(f"DEX: {self.dex_program}")
        if self.destination_type:
            parts.append(f"Destination: {self.destination_type}")

        return " | ".join(parts)

    def confidence_factors(self) -> list[str]:
        factors = []
        if self.dex_program:
            factors.append(f"dex_detected:{self.dex_program}")
        if self.quote_asset_received:
            factors.append("quote_received")
        if self.lp_token_minted:
            factors.append("lp_token_minted")
        if self.lp_token_burned:
            factors.append("lp_token_burned")
        if self.destination_is_known_cex:
            factors.append(f"known_cex:{self.destination_cex_name}")
        if self.bridge_program:
            factors.append(f"bridge:{self.bridge_program}")
        return factors


# ============================================================================
# Coordination Evidence
# ============================================================================


@dataclass
class CoordinationEvidence(EvidencePackage):
    """Evidence for coordination cluster classification."""

    domain: LabelDomain = field(default=LabelDomain.COORDINATION, init=False)

    # Cluster identity
    cluster_id: str
    token_mint: str
    wallet_addresses: list[str]
    cluster_size: int

    # Shared funders
    shared_funder_count: int
    shared_funder_addresses: list[str]
    shared_funder_ratio: float  # % of wallets with shared funder

    # Timing analysis
    timing_similarity_score: float  # 0-1
    mean_entry_time_diff_seconds: float
    median_entry_time_diff_seconds: float
    entry_timing_std_seconds: float

    # Amount analysis
    amount_similarity_score: float  # 0-1
    amount_coefficient_of_variation: float
    amount_range_ratio: float  # max/min

    # Trade sequence
    sequence_similarity_score: float  # 0-1
    sequence_alignment_length: int

    # Cross-token
    cross_token_co_participation: float  # Jaccard similarity
    tokens_in_common: list[str]

    # Statistical validation
    null_model_p_value: float | None
    z_score: float | None
    bootstrap_stability: float | None

    # Classifier output
    classifier_coordination_label: str
    classifier_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "collected_at": self.collected_at.isoformat(),
            "data_version": self.data_version,
            "cluster_id": self.cluster_id,
            "token_mint": self.token_mint,
            "wallet_addresses": self.wallet_addresses,
            "cluster_size": self.cluster_size,
            "shared_funder_count": self.shared_funder_count,
            "shared_funder_addresses": self.shared_funder_addresses,
            "shared_funder_ratio": self.shared_funder_ratio,
            "timing_similarity_score": self.timing_similarity_score,
            "mean_entry_time_diff_seconds": self.mean_entry_time_diff_seconds,
            "median_entry_time_diff_seconds": self.median_entry_time_diff_seconds,
            "entry_timing_std_seconds": self.entry_timing_std_seconds,
            "amount_similarity_score": self.amount_similarity_score,
            "amount_coefficient_of_variation": self.amount_coefficient_of_variation,
            "amount_range_ratio": self.amount_range_ratio,
            "sequence_similarity_score": self.sequence_similarity_score,
            "sequence_alignment_length": self.sequence_alignment_length,
            "cross_token_co_participation": self.cross_token_co_participation,
            "tokens_in_common": self.tokens_in_common,
            "null_model_p_value": self.null_model_p_value,
            "z_score": self.z_score,
            "bootstrap_stability": self.bootstrap_stability,
            "classifier_coordination_label": self.classifier_coordination_label,
            "classifier_confidence": self.classifier_confidence,
        }

    def summary(self) -> str:
        parts = [
            f"Coordination: {self.classifier_coordination_label} (conf: {self.classifier_confidence:.2f})",
            f"Cluster size: {self.cluster_size}",
            f"Shared funders: {self.shared_funder_ratio:.1%}",
            f"Timing sim: {self.timing_similarity_score:.2f}",
        ]
        if self.z_score is not None:
            parts.append(f"z-score: {self.z_score:.2f}")
        if self.null_model_p_value is not None:
            parts.append(f"p-value: {self.null_model_p_value:.4f}")
        return " | ".join(parts)

    def confidence_factors(self) -> list[str]:
        factors = []
        if self.shared_funder_ratio > 0.5:
            factors.append(f"high_shared_funder:{self.shared_funder_ratio:.2f}")
        if self.timing_similarity_score > 0.7:
            factors.append(f"high_timing_sim:{self.timing_similarity_score:.2f}")
        if self.z_score is not None and self.z_score > 2:
            factors.append(f"significant_z:{self.z_score:.2f}")
        if self.cross_token_co_participation > 0.3:
            factors.append(f"cross_token:{self.cross_token_co_participation:.2f}")
        return factors


# ============================================================================
# Wallet Behaviour Evidence
# ============================================================================


@dataclass
class WalletBehaviourEvidence(EvidencePackage):
    """Evidence for wallet behaviour profile classification."""

    domain: LabelDomain = field(default=LabelDomain.WALLET_BEHAVIOUR, init=False)

    # Wallet identity
    wallet_address: str
    observation_period_days: int

    # Entry timing
    mean_entry_timing_percentile: float  # 0-1 (0=earliest, 1=latest)
    entry_timing_std: float
    sniper_score: float  # How often in first 10%

    # Holding duration
    mean_holding_duration_hours: float
    median_holding_duration_hours: float
    holding_duration_std_hours: float

    # Trade behaviour
    total_trades: int
    trades_per_week: float
    win_rate: float  # % of profitable exits
    mean_pnl_percent: float

    # Position sizing
    mean_position_size_sol: float
    max_position_size_sol: float
    position_size_consistency: float  # 1 - CV

    # Archetype scores
    sniper_archetype_score: float
    accumulator_archetype_score: float
    liquidity_actor_score: float
    dormant_whale_score: float

    # Cross-token history
    tokens_participated: int
    behaviour_history_score: float
    behaviour_confidence: float

    # Coordination signals
    coordination_participation_count: int
    known_entity_links: int

    # Classifier output
    classifier_behaviour_label: str
    classifier_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "collected_at": self.collected_at.isoformat(),
            "data_version": self.data_version,
            "wallet_address": self.wallet_address,
            "observation_period_days": self.observation_period_days,
            "mean_entry_timing_percentile": self.mean_entry_timing_percentile,
            "entry_timing_std": self.entry_timing_std,
            "sniper_score": self.sniper_score,
            "mean_holding_duration_hours": self.mean_holding_duration_hours,
            "median_holding_duration_hours": self.median_holding_duration_hours,
            "holding_duration_std_hours": self.holding_duration_std_hours,
            "total_trades": self.total_trades,
            "trades_per_week": self.trades_per_week,
            "win_rate": self.win_rate,
            "mean_pnl_percent": self.mean_pnl_percent,
            "mean_position_size_sol": self.mean_position_size_sol,
            "max_position_size_sol": self.max_position_size_sol,
            "position_size_consistency": self.position_size_consistency,
            "sniper_archetype_score": self.sniper_archetype_score,
            "accumulator_archetype_score": self.accumulator_archetype_score,
            "liquidity_actor_score": self.liquidity_actor_score,
            "dormant_whale_score": self.dormant_whale_score,
            "tokens_participated": self.tokens_participated,
            "behaviour_history_score": self.behaviour_history_score,
            "behaviour_confidence": self.behaviour_confidence,
            "coordination_participation_count": self.coordination_participation_count,
            "known_entity_links": self.known_entity_links,
            "classifier_behaviour_label": self.classifier_behaviour_label,
            "classifier_confidence": self.classifier_confidence,
        }

    def summary(self) -> str:
        parts = [
            f"Behaviour: {self.classifier_behaviour_label} (conf: {self.classifier_confidence:.2f})",
            f"Trades: {self.total_trades}",
            f"Win rate: {self.win_rate:.1%}",
            f"Avg hold: {self.mean_holding_duration_hours:.1f}h",
        ]
        return " | ".join(parts)

    def confidence_factors(self) -> list[str]:
        factors = []
        if self.sniper_archetype_score > 0.7:
            factors.append(f"high_sniper:{self.sniper_archetype_score:.2f}")
        if self.accumulator_archetype_score > 0.7:
            factors.append(f"high_accumulator:{self.accumulator_archetype_score:.2f}")
        if self.total_trades >= 50:
            factors.append(f"sufficient_trades:{self.total_trades}")
        if self.behaviour_confidence > 0.8:
            factors.append(f"high_history_conf:{self.behaviour_confidence:.2f}")
        return factors


# ============================================================================
# Token Outcome Evidence
# ============================================================================


@dataclass
class TokenOutcomeEvidence(EvidencePackage):
    """Evidence for token outcome classification."""

    domain: LabelDomain = field(default=LabelDomain.TOKEN_OUTCOME, init=False)

    # Token identity
    token_mint: str
    token_name: str | None
    launch_timestamp: datetime

    # Outcome horizon
    outcome_horizon_hours: int
    observation_end: datetime

    # Liquidity trajectory
    initial_liquidity_sol: float
    peak_liquidity_sol: float
    final_liquidity_sol: float
    liquidity_decline_percent: float
    time_to_peak_hours: float

    # Price trajectory
    initial_price: float
    peak_price: float
    final_price: float
    price_decline_from_peak_percent: float
    price_volatility: float

    # Holder metrics
    initial_holder_count: int
    peak_holder_count: int
    final_holder_count: int
    holder_churn_rate: float

    # Concentration
    top_10_holder_concentration_initial: float
    top_10_holder_concentration_final: float
    concentration_change: float

    # Exit cascade indicators
    coordinated_exit_detected: bool
    large_holder_exit_count: int
    exit_cascade_timestamp: datetime | None

    # Volume analysis
    total_volume_sol: float
    wash_volume_estimate_percent: float

    # Classifier output
    classifier_outcome_label: str
    classifier_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "collected_at": self.collected_at.isoformat(),
            "data_version": self.data_version,
            "token_mint": self.token_mint,
            "token_name": self.token_name,
            "launch_timestamp": self.launch_timestamp.isoformat(),
            "outcome_horizon_hours": self.outcome_horizon_hours,
            "observation_end": self.observation_end.isoformat(),
            "initial_liquidity_sol": self.initial_liquidity_sol,
            "peak_liquidity_sol": self.peak_liquidity_sol,
            "final_liquidity_sol": self.final_liquidity_sol,
            "liquidity_decline_percent": self.liquidity_decline_percent,
            "time_to_peak_hours": self.time_to_peak_hours,
            "initial_price": self.initial_price,
            "peak_price": self.peak_price,
            "final_price": self.final_price,
            "price_decline_from_peak_percent": self.price_decline_from_peak_percent,
            "price_volatility": self.price_volatility,
            "initial_holder_count": self.initial_holder_count,
            "peak_holder_count": self.peak_holder_count,
            "final_holder_count": self.final_holder_count,
            "holder_churn_rate": self.holder_churn_rate,
            "top_10_holder_concentration_initial": self.top_10_holder_concentration_initial,
            "top_10_holder_concentration_final": self.top_10_holder_concentration_final,
            "concentration_change": self.concentration_change,
            "coordinated_exit_detected": self.coordinated_exit_detected,
            "large_holder_exit_count": self.large_holder_exit_count,
            "exit_cascade_timestamp": self.exit_cascade_timestamp.isoformat() if self.exit_cascade_timestamp else None,
            "total_volume_sol": self.total_volume_sol,
            "wash_volume_estimate_percent": self.wash_volume_estimate_percent,
            "classifier_outcome_label": self.classifier_outcome_label,
            "classifier_confidence": self.classifier_confidence,
        }

    def summary(self) -> str:
        parts = [
            f"Outcome: {self.classifier_outcome_label} (conf: {self.classifier_confidence:.2f})",
            f"Liquidity: {self.initial_liquidity_sol:.1f} → {self.final_liquidity_sol:.1f} SOL",
            f"Price decline: {self.price_decline_from_peak_percent:.1f}%",
        ]
        if self.coordinated_exit_detected:
            parts.append("Coordinated exit detected")
        return " | ".join(parts)

    def confidence_factors(self) -> list[str]:
        factors = []
        if self.liquidity_decline_percent > 90:
            factors.append(f"severe_liq_decline:{self.liquidity_decline_percent:.0f}%")
        if self.coordinated_exit_detected:
            factors.append("coordinated_exit")
        if self.wash_volume_estimate_percent > 50:
            factors.append(f"high_wash:{self.wash_volume_estimate_percent:.0f}%")
        if self.holder_churn_rate > 0.8:
            factors.append(f"high_churn:{self.holder_churn_rate:.2f}")
        return factors


# ============================================================================
# Launch Trajectory Evidence
# ============================================================================


@dataclass
class LaunchTrajectoryEvidence(EvidencePackage):
    """Evidence for launch trajectory classification."""

    domain: LabelDomain = field(default=LabelDomain.LAUNCH_TRAJECTORY, init=False)

    # Token identity
    token_mint: str
    launch_timestamp: datetime

    # First hour metrics
    first_hour_buyers: int
    first_hour_volume_sol: float
    first_minute_buyers: int
    first_block_buyers: int

    # Bot detection
    bot_buyer_count: int
    bot_buyer_ratio: float
    programmatic_buyer_signatures: list[str]

    # Insider signals
    insider_wallet_count: int
    insider_allocation_percent: float
    pre_launch_holder_count: int

    # Distribution analysis
    initial_distribution_gini: float
    early_holder_concentration: float
    distribution_entropy: float

    # Growth pattern
    organic_growth_score: float  # 0-1
    buyer_diversity_score: float  # 0-1
    temporal_spread_score: float  # 0-1

    # Coordination at launch
    coordinated_entry_detected: bool
    coordination_cluster_count: int

    # Classifier output
    classifier_trajectory_label: str
    classifier_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "collected_at": self.collected_at.isoformat(),
            "data_version": self.data_version,
            "token_mint": self.token_mint,
            "launch_timestamp": self.launch_timestamp.isoformat(),
            "first_hour_buyers": self.first_hour_buyers,
            "first_hour_volume_sol": self.first_hour_volume_sol,
            "first_minute_buyers": self.first_minute_buyers,
            "first_block_buyers": self.first_block_buyers,
            "bot_buyer_count": self.bot_buyer_count,
            "bot_buyer_ratio": self.bot_buyer_ratio,
            "programmatic_buyer_signatures": self.programmatic_buyer_signatures,
            "insider_wallet_count": self.insider_wallet_count,
            "insider_allocation_percent": self.insider_allocation_percent,
            "pre_launch_holder_count": self.pre_launch_holder_count,
            "initial_distribution_gini": self.initial_distribution_gini,
            "early_holder_concentration": self.early_holder_concentration,
            "distribution_entropy": self.distribution_entropy,
            "organic_growth_score": self.organic_growth_score,
            "buyer_diversity_score": self.buyer_diversity_score,
            "temporal_spread_score": self.temporal_spread_score,
            "coordinated_entry_detected": self.coordinated_entry_detected,
            "coordination_cluster_count": self.coordination_cluster_count,
            "classifier_trajectory_label": self.classifier_trajectory_label,
            "classifier_confidence": self.classifier_confidence,
        }

    def summary(self) -> str:
        parts = [
            f"Trajectory: {self.classifier_trajectory_label} (conf: {self.classifier_confidence:.2f})",
            f"First hour buyers: {self.first_hour_buyers}",
            f"Bot ratio: {self.bot_buyer_ratio:.1%}",
            f"Organic score: {self.organic_growth_score:.2f}",
        ]
        return " | ".join(parts)

    def confidence_factors(self) -> list[str]:
        factors = []
        if self.bot_buyer_ratio > 0.5:
            factors.append(f"high_bot_ratio:{self.bot_buyer_ratio:.2f}")
        if self.insider_allocation_percent > 30:
            factors.append(f"high_insider:{self.insider_allocation_percent:.0f}%")
        if self.organic_growth_score > 0.8:
            factors.append(f"organic:{self.organic_growth_score:.2f}")
        if self.coordinated_entry_detected:
            factors.append("coordinated_entry")
        return factors


# ============================================================================
# Entity Link Evidence
# ============================================================================


@dataclass
class EntityLinkEvidence(EvidencePackage):
    """Evidence for entity resolution link classification."""

    domain: LabelDomain = field(default=LabelDomain.ENTITY_RESOLUTION, init=False)

    # Wallet pair
    wallet_a: str
    wallet_b: str

    # Shared funders
    shared_funder_addresses: list[str]
    shared_funder_count: int
    funding_chain_length: int

    # Transfer chains
    direct_transfer_count: int
    indirect_transfer_count: int
    transfer_chain_paths: list[list[str]]

    # Co-participation
    tokens_co_participated: list[str]
    co_participation_jaccard: float
    temporal_overlap_score: float

    # Timing correlation
    entry_timing_correlation: float
    exit_timing_correlation: float
    trade_timing_correlation: float

    # Behavioural similarity
    archetype_similarity: float
    holding_pattern_similarity: float
    position_size_similarity: float

    # Migration signals
    migration_detected: bool
    migration_timestamp: datetime | None
    migration_completeness: float  # 0-1

    # Classifier output
    classifier_entity_label: str
    classifier_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "collected_at": self.collected_at.isoformat(),
            "data_version": self.data_version,
            "wallet_a": self.wallet_a,
            "wallet_b": self.wallet_b,
            "shared_funder_addresses": self.shared_funder_addresses,
            "shared_funder_count": self.shared_funder_count,
            "funding_chain_length": self.funding_chain_length,
            "direct_transfer_count": self.direct_transfer_count,
            "indirect_transfer_count": self.indirect_transfer_count,
            "transfer_chain_paths": self.transfer_chain_paths,
            "tokens_co_participated": self.tokens_co_participated,
            "co_participation_jaccard": self.co_participation_jaccard,
            "temporal_overlap_score": self.temporal_overlap_score,
            "entry_timing_correlation": self.entry_timing_correlation,
            "exit_timing_correlation": self.exit_timing_correlation,
            "trade_timing_correlation": self.trade_timing_correlation,
            "archetype_similarity": self.archetype_similarity,
            "holding_pattern_similarity": self.holding_pattern_similarity,
            "position_size_similarity": self.position_size_similarity,
            "migration_detected": self.migration_detected,
            "migration_timestamp": self.migration_timestamp.isoformat() if self.migration_timestamp else None,
            "migration_completeness": self.migration_completeness,
            "classifier_entity_label": self.classifier_entity_label,
            "classifier_confidence": self.classifier_confidence,
        }

    def summary(self) -> str:
        parts = [
            f"Entity link: {self.classifier_entity_label} (conf: {self.classifier_confidence:.2f})",
            f"Shared funders: {self.shared_funder_count}",
            f"Co-participation: {self.co_participation_jaccard:.2f}",
        ]
        if self.migration_detected:
            parts.append("Migration detected")
        return " | ".join(parts)

    def confidence_factors(self) -> list[str]:
        factors = []
        if self.shared_funder_count > 0:
            factors.append(f"shared_funders:{self.shared_funder_count}")
        if self.direct_transfer_count > 0:
            factors.append(f"direct_transfers:{self.direct_transfer_count}")
        if self.co_participation_jaccard > 0.5:
            factors.append(f"high_co_part:{self.co_participation_jaccard:.2f}")
        if self.migration_detected:
            factors.append(f"migration:{self.migration_completeness:.2f}")
        if self.archetype_similarity > 0.8:
            factors.append(f"archetype_sim:{self.archetype_similarity:.2f}")
        return factors


# ============================================================================
# Evidence Package Builder
# ============================================================================


class EvidencePackageBuilder:
    """Builder for creating evidence packages from raw data."""

    @staticmethod
    def from_json(json_str: str, domain: LabelDomain) -> EvidencePackage | None:
        """Parse evidence package from JSON."""
        try:
            data = json.loads(json_str)
            return EvidencePackageBuilder.from_dict(data, domain)
        except Exception as e:
            logger.error("evidence_parse_error", error=str(e))
            return None

    @staticmethod
    def from_dict(data: dict[str, Any], domain: LabelDomain) -> EvidencePackage | None:
        """Build evidence package from dictionary."""
        try:
            # Parse collected_at
            collected_at = datetime.fromisoformat(data["collected_at"])
            data_version = data["data_version"]

            if domain == LabelDomain.EXIT_EVENT:
                return ExitEventEvidence(
                    collected_at=collected_at,
                    data_version=data_version,
                    signature=data["signature"],
                    slot=data["slot"],
                    block_time=datetime.fromisoformat(data["block_time"]) if data.get("block_time") else None,
                    token_mint=data["token_mint"],
                    token_amount=data["token_amount"],
                    token_decimals=data["token_decimals"],
                    token_balance_before=data["token_balance_before"],
                    token_balance_after=data["token_balance_after"],
                    sol_balance_change=data["sol_balance_change"],
                    quote_asset_received=data["quote_asset_received"],
                    quote_asset_mint=data.get("quote_asset_mint"),
                    quote_asset_amount=data.get("quote_asset_amount"),
                    program_ids=data["program_ids"],
                    dex_program=data.get("dex_program"),
                    lp_program=data.get("lp_program"),
                    bridge_program=data.get("bridge_program"),
                    destination_address=data.get("destination_address"),
                    destination_type=data.get("destination_type"),
                    destination_is_known_cex=data.get("destination_is_known_cex", False),
                    destination_cex_name=data.get("destination_cex_name"),
                    lp_token_minted=data.get("lp_token_minted", False),
                    lp_token_burned=data.get("lp_token_burned", False),
                    lp_token_amount=data.get("lp_token_amount"),
                    classifier_exit_type=data["classifier_exit_type"],
                    classifier_confidence=data["classifier_confidence"],
                )

            # Add other domain builders as needed
            logger.warning("unsupported_evidence_domain", domain=domain.value)
            return None

        except Exception as e:
            logger.error("evidence_build_error", error=str(e), domain=domain.value)
            return None
