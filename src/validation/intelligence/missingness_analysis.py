"""
Missingness Analysis for Feature Impact Assessment.

Analyzes patterns and predictive power of missing data to determine
whether missingness is informative rather than merely inconvenient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import structlog

logger = structlog.get_logger()


@dataclass
class MissingnessPattern:
    """Pattern analysis for a feature's missingness."""

    feature_name: str
    missing_count: int
    missing_percentage: float
    missing_predicts_event: bool  # Does missingness predict sell event?
    missing_predicts_unknown: bool  # Does missingness predict UNKNOWN archetype?
    missing_predicts_anomaly: bool  # Does missingness predict high anomaly score?
    missing_predicts_coordination: bool  # Does missingness predict coordinated cluster?

    event_rate_when_missing: float
    event_rate_when_present: float
    rate_ratio: float  # missing_rate / present_rate

    chi_square_stat: Optional[float]
    chi_square_pvalue: Optional[float]

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "feature_name": self.feature_name,
            "missing_count": self.missing_count,
            "missing_percentage": self.missing_percentage,
            "missing_predicts_event": self.missing_predicts_event,
            "missing_predicts_unknown": self.missing_predicts_unknown,
            "missing_predicts_anomaly": self.missing_predicts_anomaly,
            "missing_predicts_coordination": self.missing_predicts_coordination,
            "event_rate_when_missing": self.event_rate_when_missing,
            "event_rate_when_present": self.event_rate_when_present,
            "rate_ratio": self.rate_ratio,
            "chi_square_pvalue": self.chi_square_pvalue,
        }


@dataclass
class MissingnessByCategory:
    """Missingness statistics by data category."""

    category: str
    features: list[str]
    total_missing: int
    total_possible: int
    missing_percentage: float
    any_missing_count: int  # Wallets with any feature missing in category
    any_missing_percentage: float


@dataclass
class MissingnessReport:
    """Complete missingness analysis report."""

    # Per-feature analysis
    feature_patterns: dict[str, MissingnessPattern]

    # By category
    category_stats: dict[str, MissingnessByCategory]

    # Overall
    total_wallets: int
    wallets_with_any_missing: int
    wallets_with_any_missing_pct: float

    # Predictive power summary
    informative_features: list[str]  # Features where missingness is predictive
    recommendation: str

    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "feature_patterns": {k: v.to_dict() for k, v in self.feature_patterns.items()},
            "category_stats": {k: {
                "category": v.category,
                "features": v.features,
                "missing_percentage": v.missing_percentage,
                "any_missing_percentage": v.any_missing_percentage,
            } for k, v in self.category_stats.items()},
            "total_wallets": self.total_wallets,
            "wallets_with_any_missing": self.wallets_with_any_missing,
            "wallets_with_any_missing_pct": self.wallets_with_any_missing_pct,
            "informative_features": self.informative_features,
            "recommendation": self.recommendation,
            "computed_at": self.computed_at.isoformat(),
        }


# Feature categories for grouping
FEATURE_CATEGORIES = {
    "temporal_history": [
        "entry_time_relative",
        "holding_duration",
        "position_volatility",
    ],
    "trade_history": [
        "trade_count",
        "burstiness",
        "swap_frequency",
        "delta_balance_7d",
        "delta_balance_30d",
    ],
    "price_data": [
        "entry_price_usd",
        "current_price_usd",
        "unrealized_pnl_ratio",
        "unrealized_pnl_usd",
        "price_change_1h_pct",
        "price_change_24h_pct",
        "price_change_7d_pct",
    ],
    "graph_data": [
        "in_degree",
        "out_degree",
        "eigenvector_centrality",
        "shared_funder_count",
        "total_funding_received",
        "largest_funder_share",
        "funding_hhi",
    ],
    "liquidity_data": [
        "liquidity_usd_current",
        "liquidity_usd_1h_avg",
        "liquidity_usd_24h_avg",
        "sell_pressure_vs_liquidity",
    ],
}


class MissingnessAnalyzer:
    """
    Analyzes missingness patterns and their predictive power.

    Determines whether missing data is:
    - Random (MCAR - Missing Completely At Random)
    - Informative (MAR/MNAR - predictive of outcomes)
    """

    def __init__(self, significance_threshold: float = 0.05):
        """
        Initialize analyzer.

        Args:
            significance_threshold: P-value threshold for significance
        """
        self.significance_threshold = significance_threshold

    def analyze(
        self,
        data: pd.DataFrame,
        event_col: Optional[str] = "event",
        archetype_col: Optional[str] = "archetype",
        anomaly_col: Optional[str] = "outlier_score",
        coordination_col: Optional[str] = "is_coordinated",
    ) -> MissingnessReport:
        """
        Perform comprehensive missingness analysis.

        Args:
            data: DataFrame with features and outcome columns
            event_col: Column indicating sell event (if available)
            archetype_col: Column with archetype assignment (if available)
            anomaly_col: Column with anomaly/outlier score (if available)
            coordination_col: Column indicating coordination cluster (if available)

        Returns:
            MissingnessReport with complete analysis
        """
        logger.info(
            "starting_missingness_analysis",
            n_wallets=len(data),
            n_columns=len(data.columns),
        )

        total_wallets = len(data)

        # Analyze each feature
        feature_patterns: dict[str, MissingnessPattern] = {}

        for col in data.columns:
            # Skip outcome columns
            if col in [event_col, archetype_col, anomaly_col, coordination_col]:
                continue

            pattern = self._analyze_feature(
                data, col, event_col, archetype_col, anomaly_col, coordination_col
            )
            if pattern is not None:
                feature_patterns[col] = pattern

        # Analyze by category
        category_stats = self._analyze_by_category(data)

        # Count wallets with any missing
        feature_cols = [c for c in data.columns if c not in [event_col, archetype_col, anomaly_col, coordination_col]]
        wallets_with_missing = data[feature_cols].isna().any(axis=1).sum()
        wallets_with_missing_pct = wallets_with_missing / total_wallets * 100

        # Identify informative features
        informative = [
            name for name, pattern in feature_patterns.items()
            if (pattern.missing_predicts_event or
                pattern.missing_predicts_unknown or
                pattern.missing_predicts_anomaly or
                pattern.missing_predicts_coordination)
        ]

        # Generate recommendation
        recommendation = self._generate_recommendation(
            feature_patterns, category_stats, informative
        )

        logger.info(
            "missingness_analysis_complete",
            informative_features=len(informative),
            wallets_with_missing_pct=wallets_with_missing_pct,
        )

        return MissingnessReport(
            feature_patterns=feature_patterns,
            category_stats=category_stats,
            total_wallets=total_wallets,
            wallets_with_any_missing=wallets_with_missing,
            wallets_with_any_missing_pct=wallets_with_missing_pct,
            informative_features=informative,
            recommendation=recommendation,
        )

    def _analyze_feature(
        self,
        data: pd.DataFrame,
        feature_col: str,
        event_col: Optional[str],
        archetype_col: Optional[str],
        anomaly_col: Optional[str],
        coordination_col: Optional[str],
    ) -> Optional[MissingnessPattern]:
        """Analyze missingness pattern for a single feature."""
        if feature_col not in data.columns:
            return None

        is_missing = data[feature_col].isna()
        missing_count = int(is_missing.sum())

        if missing_count == 0:
            return None

        missing_pct = missing_count / len(data) * 100

        # Initialize predictive flags
        predicts_event = False
        predicts_unknown = False
        predicts_anomaly = False
        predicts_coordination = False

        event_rate_missing = 0.0
        event_rate_present = 0.0
        chi_stat = None
        chi_pvalue = None

        # Test against sell event
        if event_col and event_col in data.columns:
            event_rate_missing = data.loc[is_missing, event_col].mean()
            event_rate_present = data.loc[~is_missing, event_col].mean()

            # Chi-square test
            chi_stat, chi_pvalue = self._chi_square_test(
                is_missing.values, data[event_col].values
            )

            if chi_pvalue is not None and chi_pvalue < self.significance_threshold:
                predicts_event = True

        # Test against UNKNOWN archetype
        if archetype_col and archetype_col in data.columns:
            is_unknown = data[archetype_col] == "unknown"
            _, pvalue = self._chi_square_test(is_missing.values, is_unknown.values)
            if pvalue is not None and pvalue < self.significance_threshold:
                predicts_unknown = True

        # Test against anomaly score (high = anomalous)
        if anomaly_col and anomaly_col in data.columns:
            # Use median split for anomaly
            anomaly_median = data[anomaly_col].median()
            is_anomalous = data[anomaly_col] > anomaly_median
            _, pvalue = self._chi_square_test(is_missing.values, is_anomalous.values)
            if pvalue is not None and pvalue < self.significance_threshold:
                predicts_anomaly = True

        # Test against coordination
        if coordination_col and coordination_col in data.columns:
            _, pvalue = self._chi_square_test(
                is_missing.values, data[coordination_col].values
            )
            if pvalue is not None and pvalue < self.significance_threshold:
                predicts_coordination = True

        # Rate ratio
        rate_ratio = event_rate_missing / event_rate_present if event_rate_present > 0 else 0.0

        return MissingnessPattern(
            feature_name=feature_col,
            missing_count=missing_count,
            missing_percentage=missing_pct,
            missing_predicts_event=predicts_event,
            missing_predicts_unknown=predicts_unknown,
            missing_predicts_anomaly=predicts_anomaly,
            missing_predicts_coordination=predicts_coordination,
            event_rate_when_missing=event_rate_missing,
            event_rate_when_present=event_rate_present,
            rate_ratio=rate_ratio,
            chi_square_stat=chi_stat,
            chi_square_pvalue=chi_pvalue,
        )

    def _chi_square_test(
        self,
        x: NDArray,
        y: NDArray,
    ) -> tuple[Optional[float], Optional[float]]:
        """Perform chi-square test of independence."""
        try:
            from scipy.stats import chi2_contingency

            # Create contingency table
            x_bool = x.astype(bool)
            y_bool = y.astype(bool)

            table = np.array([
                [(~x_bool & ~y_bool).sum(), (~x_bool & y_bool).sum()],
                [(x_bool & ~y_bool).sum(), (x_bool & y_bool).sum()],
            ])

            # Skip if any cell is 0
            if (table == 0).any():
                return None, None

            chi2, pvalue, _, _ = chi2_contingency(table)
            return float(chi2), float(pvalue)

        except Exception:
            return None, None

    def _analyze_by_category(
        self,
        data: pd.DataFrame,
    ) -> dict[str, MissingnessByCategory]:
        """Analyze missingness by feature category."""
        stats = {}

        for category, features in FEATURE_CATEGORIES.items():
            available_features = [f for f in features if f in data.columns]

            if not available_features:
                continue

            # Count missing per feature
            missing_counts = data[available_features].isna().sum()
            total_missing = int(missing_counts.sum())
            total_possible = len(data) * len(available_features)
            missing_pct = total_missing / total_possible * 100 if total_possible > 0 else 0.0

            # Count wallets with any missing in category
            any_missing = data[available_features].isna().any(axis=1)
            any_missing_count = int(any_missing.sum())
            any_missing_pct = any_missing_count / len(data) * 100

            stats[category] = MissingnessByCategory(
                category=category,
                features=available_features,
                total_missing=total_missing,
                total_possible=total_possible,
                missing_percentage=missing_pct,
                any_missing_count=any_missing_count,
                any_missing_percentage=any_missing_pct,
            )

        return stats

    def _generate_recommendation(
        self,
        patterns: dict[str, MissingnessPattern],
        category_stats: dict[str, MissingnessByCategory],
        informative: list[str],
    ) -> str:
        """Generate recommendation based on analysis."""
        parts = []

        # High missingness warning
        high_missing = [
            name for name, p in patterns.items()
            if p.missing_percentage > 30
        ]
        if high_missing:
            parts.append(f"HIGH MISSINGNESS (>30%): {', '.join(high_missing[:5])}")

        # Informative missingness
        if informative:
            parts.append(
                f"INFORMATIVE MISSINGNESS: {', '.join(informative[:5])} "
                "(consider keeping indicator columns)"
            )
        else:
            parts.append("No significant informative missingness detected")

        # Category-level issues
        for cat, stat in category_stats.items():
            if stat.missing_percentage > 50:
                parts.append(f"WARNING: {cat} has {stat.missing_percentage:.1f}% missing")

        if not parts:
            return "Missingness levels acceptable across all features"

        return "; ".join(parts)
