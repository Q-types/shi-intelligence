"""
Natural Language Risk Narratives for SHI.

Generates human-readable explanations for risk scores, regime changes,
and anomalies. Provides clear, actionable summaries with uncertainty bounds.

Key Features:
- Risk score narratives with feature breakdowns
- Regime change explanations
- Anomaly detection narratives
- Uncertainty-aware language
- Actionable insights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

import structlog

from .shap_explainer import SHAPExplanation, FeatureContribution
from ..temporal.regimes import HolderRegimeType

logger = structlog.get_logger()


class RiskLevel(Enum):
    """Risk level classification."""

    VERY_LOW = "very_low"  # < 0.2
    LOW = "low"  # 0.2 - 0.4
    MODERATE = "moderate"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # > 0.8


class TrendDirection(Enum):
    """Trend direction for metrics."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


@dataclass
class RiskNarrative:
    """Complete risk narrative with explanations."""

    summary: str  # One-sentence overview
    risk_level: RiskLevel
    confidence: str  # Textual confidence description
    key_drivers: List[str]  # Bullet points for main contributors
    actionable_insights: List[str]  # What to do about it
    uncertainty_note: Optional[str] = None  # Uncertainty caveat
    technical_details: Optional[str] = None  # For advanced users


@dataclass
class RegimeNarrative:
    """Narrative for regime changes."""

    transition_summary: str  # "Shifted from X to Y"
    reason: str  # Why it happened
    implications: List[str]  # What this means
    confidence: str
    historical_context: Optional[str] = None


@dataclass
class AnomalyNarrative:
    """Narrative for anomaly detection."""

    anomaly_type: str  # "Sybil cluster", "Wash trading", etc.
    evidence: List[str]  # Supporting evidence
    severity: str  # How unusual this is
    recommended_action: str
    confidence: str


class NarrativeGenerator:
    """
    Generates natural language explanations for SHI outputs.

    Converts SHAP explanations and model outputs into clear,
    actionable narratives with appropriate uncertainty language.
    """

    # Feature name to human-readable mapping
    FEATURE_DISPLAY_NAMES = {
        "hhi": "holder concentration (HHI)",
        "gini": "wealth inequality (Gini)",
        "top10_pct": "top 10 holders' share",
        "top50_pct": "top 50 holders' share",
        "churn_rate": "holder turnover rate",
        "mean_balance": "average holder balance",
        "median_balance": "median holder balance",
        "total_holders": "total number of holders",
        "betweenness_centrality": "network influence",
        "clustering_coefficient": "clustering tendency",
        "pagerank": "network importance",
        "anomaly_score": "anomaly score",
        "degree": "number of connections",
        "sell_event_rate": "selling frequency",
        "time_since_first_tx": "wallet age",
        "tx_frequency": "transaction frequency",
        "avg_tx_size": "average transaction size",
    }

    def __init__(self, verbose: bool = False):
        """
        Initialize narrative generator.

        Parameters
        ----------
        verbose : bool, optional
            Whether to include technical details, by default False
        """
        self.verbose = verbose

    def generate_risk_narrative(
        self,
        explanation: SHAPExplanation,
        token_symbol: Optional[str] = None,
    ) -> RiskNarrative:
        """
        Generate risk score narrative from SHAP explanation.

        Parameters
        ----------
        explanation : SHAPExplanation
            SHAP explanation for risk score
        token_symbol : Optional[str], optional
            Token symbol for context, by default None

        Returns
        -------
        RiskNarrative
            Complete narrative with actionable insights
        """
        risk_score = explanation.predicted_value
        risk_level = self._classify_risk_level(risk_score)

        # Build summary
        token_name = f"{token_symbol} " if token_symbol else ""
        summary = self._create_risk_summary(token_name, risk_score, risk_level)

        # Confidence description
        confidence = self._describe_confidence(explanation)

        # Key drivers (top positive contributors)
        key_drivers = self._extract_key_drivers(explanation)

        # Actionable insights based on risk level and drivers
        actionable_insights = self._generate_risk_insights(
            risk_level,
            explanation.top_contributors
        )

        # Uncertainty note
        uncertainty_note = self._create_uncertainty_note(explanation)

        # Technical details (if verbose)
        technical_details = None
        if self.verbose:
            technical_details = self._create_technical_details(explanation)

        return RiskNarrative(
            summary=summary,
            risk_level=risk_level,
            confidence=confidence,
            key_drivers=key_drivers,
            actionable_insights=actionable_insights,
            uncertainty_note=uncertainty_note,
            technical_details=technical_details,
        )

    def generate_regime_narrative(
        self,
        from_regime: HolderRegimeType,
        to_regime: HolderRegimeType,
        confidence: float,
        drivers: Optional[List[FeatureContribution]] = None,
    ) -> RegimeNarrative:
        """
        Generate narrative for regime transition.

        Parameters
        ----------
        from_regime : HolderRegimeType
            Previous regime
        to_regime : HolderRegimeType
            New regime
        confidence : float
            Transition confidence (0-1)
        drivers : Optional[List[FeatureContribution]], optional
            Features driving transition, by default None

        Returns
        -------
        RegimeNarrative
            Regime change narrative
        """
        # Transition summary
        transition_summary = (
            f"Holder regime shifted from {self._regime_display_name(from_regime)} "
            f"to {self._regime_display_name(to_regime)}"
        )

        # Reason (based on regime types)
        reason = self._explain_regime_transition(from_regime, to_regime, drivers)

        # Implications
        implications = self._regime_implications(to_regime)

        # Confidence
        confidence_desc = self._describe_regime_confidence(confidence)

        return RegimeNarrative(
            transition_summary=transition_summary,
            reason=reason,
            implications=implications,
            confidence=confidence_desc,
        )

    def generate_anomaly_narrative(
        self,
        anomaly_score: float,
        features: List[FeatureContribution],
        wallet_address: Optional[str] = None,
    ) -> AnomalyNarrative:
        """
        Generate narrative for anomaly detection.

        Parameters
        ----------
        anomaly_score : float
            Anomaly score (more negative = more anomalous)
        features : List[FeatureContribution]
            Feature contributions to anomaly
        wallet_address : Optional[str], optional
            Wallet being analyzed, by default None

        Returns
        -------
        AnomalyNarrative
            Anomaly explanation
        """
        # Classify anomaly type based on features
        anomaly_type = self._classify_anomaly_type(features)

        # Evidence from features
        evidence = self._extract_anomaly_evidence(features)

        # Severity
        severity = self._describe_anomaly_severity(anomaly_score)

        # Recommended action
        recommended_action = self._recommend_anomaly_action(anomaly_score, anomaly_type)

        # Confidence
        confidence = self._describe_anomaly_confidence(anomaly_score)

        return AnomalyNarrative(
            anomaly_type=anomaly_type,
            evidence=evidence,
            severity=severity,
            recommended_action=recommended_action,
            confidence=confidence,
        )

    # ---- Helper Methods ----

    def _classify_risk_level(self, score: float) -> RiskLevel:
        """Classify risk score into level."""
        if score < 0.2:
            return RiskLevel.VERY_LOW
        elif score < 0.4:
            return RiskLevel.LOW
        elif score < 0.6:
            return RiskLevel.MODERATE
        elif score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _create_risk_summary(
        self, token_name: str, score: float, level: RiskLevel
    ) -> str:
        """Create one-sentence risk summary."""
        level_descriptions = {
            RiskLevel.VERY_LOW: "exhibits very low sell pressure risk",
            RiskLevel.LOW: "shows low sell pressure risk",
            RiskLevel.MODERATE: "has moderate sell pressure risk",
            RiskLevel.HIGH: "faces high sell pressure risk",
            RiskLevel.VERY_HIGH: "is at very high risk of sell pressure",
        }

        return f"{token_name}{level_descriptions[level]} (score: {score:.2f})"

    def _describe_confidence(self, explanation: SHAPExplanation) -> str:
        """Describe confidence in prediction."""
        if explanation.prediction_std is None:
            return "High confidence"

        std = explanation.prediction_std
        if std < 0.05:
            return "Very high confidence"
        elif std < 0.10:
            return "High confidence"
        elif std < 0.15:
            return "Moderate confidence"
        else:
            return "Low confidence (high uncertainty)"

    def _extract_key_drivers(self, explanation: SHAPExplanation) -> List[str]:
        """Extract key drivers as bullet points."""
        drivers = []

        for contrib in explanation.positive_contributors[:5]:  # Top 5
            feature_name = self.FEATURE_DISPLAY_NAMES.get(
                contrib.feature_name, contrib.feature_name
            )

            # Direction and magnitude
            if contrib.contribution_pct > 20:
                impact = "significantly increases"
            elif contrib.contribution_pct > 10:
                impact = "moderately increases"
            else:
                impact = "slightly increases"

            drivers.append(
                f"{feature_name.capitalize()} ({contrib.feature_value:.3f}) "
                f"{impact} risk (+{contrib.contribution_pct:.1f}%)"
            )

        return drivers

    def _generate_risk_insights(
        self, level: RiskLevel, contributors: List[FeatureContribution]
    ) -> List[str]:
        """Generate actionable insights based on risk."""
        insights = []

        if level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            insights.append("Exercise caution - consider reducing position size")
            insights.append("Monitor holder concentration and whale movements closely")

            # Check for specific risk factors
            for contrib in contributors:
                if "hhi" in contrib.feature_name.lower() and contrib.shap_value > 0:
                    insights.append(
                        "High holder concentration detected - vulnerable to whale dumps"
                    )
                if "churn" in contrib.feature_name.lower() and contrib.shap_value > 0:
                    insights.append("High turnover rate - holders are exiting frequently")

        elif level == RiskLevel.MODERATE:
            insights.append("Moderate risk - maintain standard position sizing")
            insights.append("Set stop-loss orders as precaution")

        else:
            insights.append("Low risk environment - suitable for accumulation")
            insights.append("Monitor for regime changes that could shift risk")

        return insights

    def _create_uncertainty_note(self, explanation: SHAPExplanation) -> Optional[str]:
        """Create uncertainty caveat if applicable."""
        if explanation.confidence_interval is None:
            return None

        ci_lower, ci_upper = explanation.confidence_interval
        ci_width = ci_upper - ci_lower

        if ci_width > 0.3:
            return (
                f"Note: Prediction has wide confidence interval "
                f"({ci_lower:.2f} to {ci_upper:.2f}). "
                f"Exercise additional caution."
            )
        elif ci_width > 0.2:
            return (
                f"Moderate uncertainty in prediction "
                f"(95% CI: {ci_lower:.2f} - {ci_upper:.2f})"
            )

        return None

    def _create_technical_details(self, explanation: SHAPExplanation) -> str:
        """Create technical details for advanced users."""
        details = f"Baseline: {explanation.baseline_value:.3f}\n"
        details += f"Prediction: {explanation.predicted_value:.3f}\n"
        details += f"Total SHAP magnitude: {explanation.total_shap_magnitude:.3f}\n\n"

        details += "All feature contributions:\n"
        for name, value in sorted(
            explanation.all_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            sign = "+" if value >= 0 else ""
            details += f"  {name}: {sign}{value:.4f}\n"

        return details

    def _regime_display_name(self, regime: HolderRegimeType) -> str:
        """Human-readable regime name."""
        names = {
            HolderRegimeType.ACCUMULATION: "accumulation (decentralizing)",
            HolderRegimeType.DISTRIBUTION: "distribution (centralizing)",
            HolderRegimeType.COORDINATED_ACCUMULATION: "coordinated accumulation",
            HolderRegimeType.DECAY: "decay (holders exiting)",
            HolderRegimeType.STABLE: "stable",
        }
        return names.get(regime, regime.value)

    def _explain_regime_transition(
        self,
        from_regime: HolderRegimeType,
        to_regime: HolderRegimeType,
        drivers: Optional[List[FeatureContribution]],
    ) -> str:
        """Explain why regime changed."""
        if to_regime == HolderRegimeType.ACCUMULATION:
            return "New holders are entering, reducing concentration"
        elif to_regime == HolderRegimeType.DISTRIBUTION:
            return "Tokens consolidating into fewer wallets"
        elif to_regime == HolderRegimeType.DECAY:
            return "Holders are rapidly exiting positions"
        elif to_regime == HolderRegimeType.COORDINATED_ACCUMULATION:
            return "Coordinated buying detected among connected wallets"
        else:
            return "Holder structure has stabilized"

    def _regime_implications(self, regime: HolderRegimeType) -> List[str]:
        """Implications of current regime."""
        implications = {
            HolderRegimeType.ACCUMULATION: [
                "Increasing holder base reduces concentration risk",
                "May indicate growing organic interest",
                "Positive for long-term stability",
            ],
            HolderRegimeType.DISTRIBUTION: [
                "Concentration increasing - higher whale risk",
                "Could signal accumulation by large players",
                "Monitor for potential manipulation",
            ],
            HolderRegimeType.DECAY: [
                "High selling pressure likely",
                "Holder confidence declining",
                "High risk of further downside",
            ],
            HolderRegimeType.COORDINATED_ACCUMULATION: [
                "Possible coordinated manipulation",
                "Exercise extreme caution",
                "May precede pump-and-dump",
            ],
            HolderRegimeType.STABLE: [
                "Low volatility in holder structure",
                "Reduced risk of sudden changes",
                "Suitable for stable positions",
            ],
        }
        return implications.get(regime, ["Monitoring recommended"])

    def _describe_regime_confidence(self, confidence: float) -> str:
        """Describe confidence in regime detection."""
        if confidence > 0.9:
            return "Very high confidence"
        elif confidence > 0.75:
            return "High confidence"
        elif confidence > 0.6:
            return "Moderate confidence"
        else:
            return "Low confidence - regime uncertain"

    def _classify_anomaly_type(self, features: List[FeatureContribution]) -> str:
        """Classify type of anomaly based on features."""
        # Simple heuristic - could be more sophisticated
        for feature in features:
            if "clustering" in feature.feature_name.lower():
                return "Sybil cluster detected"
            elif "betweenness" in feature.feature_name.lower():
                return "Unusual network position"
            elif "tx_frequency" in feature.feature_name.lower():
                return "Abnormal transaction pattern"

        return "Anomalous wallet behavior"

    def _extract_anomaly_evidence(self, features: List[FeatureContribution]) -> List[str]:
        """Extract evidence for anomaly."""
        evidence = []
        for feature in features[:5]:
            feature_name = self.FEATURE_DISPLAY_NAMES.get(
                feature.feature_name, feature.feature_name
            )
            evidence.append(
                f"{feature_name.capitalize()}: {feature.feature_value:.3f} "
                f"(unusual by {abs(feature.contribution_pct):.1f}%)"
            )
        return evidence

    def _describe_anomaly_severity(self, score: float) -> str:
        """Describe how severe the anomaly is."""
        if score < -0.8:
            return "Extremely anomalous (top 1% most unusual)"
        elif score < -0.6:
            return "Highly anomalous (top 5% most unusual)"
        elif score < -0.4:
            return "Moderately anomalous"
        else:
            return "Slightly unusual"

    def _recommend_anomaly_action(self, score: float, anomaly_type: str) -> str:
        """Recommend action based on anomaly."""
        if score < -0.7:
            return "Investigate immediately - potential manipulation or bot activity"
        elif score < -0.5:
            return "Monitor closely - unusual behavior detected"
        else:
            return "Be aware - wallet shows some anomalous characteristics"

    def _describe_anomaly_confidence(self, score: float) -> str:
        """Describe confidence in anomaly detection."""
        if abs(score) > 0.8:
            return "Very high confidence in anomaly detection"
        elif abs(score) > 0.6:
            return "High confidence"
        elif abs(score) > 0.4:
            return "Moderate confidence"
        else:
            return "Low confidence - borderline anomaly"
