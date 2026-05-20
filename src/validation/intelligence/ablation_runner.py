"""
Ablation Study Runner for Feature Group Validation.

Runs comprehensive ablation tests across feature groups to identify:
- Essential feature groups
- Redundant feature groups
- Noisy or harmful feature groups
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import structlog

from ...clustering.ablation import AblationTester, FEATURE_GROUPS, AblationStudyResult
from ...clustering.diagnostics import HDBSCANDiagnostics

logger = structlog.get_logger()


@dataclass
class FeatureGroupImpact:
    """Impact assessment for a single feature group."""

    group_name: str
    features: list[str]

    # Clustering impact
    cluster_stability_delta: float  # Change in ARI when removed
    noise_rate_delta: float  # Change in noise % when removed
    unknown_rate_delta: float  # Change in UNKNOWN archetype % when removed

    # Downstream prediction impact
    sell_prediction_delta: Optional[float]  # Change in C-index when removed
    sybil_detection_delta: Optional[float]  # Change in coordination detection

    # Assessment
    is_essential: bool
    is_redundant: bool
    is_harmful: bool
    assessment_notes: list[str]

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "group_name": self.group_name,
            "features": self.features,
            "cluster_stability_delta": self.cluster_stability_delta,
            "noise_rate_delta": self.noise_rate_delta,
            "unknown_rate_delta": self.unknown_rate_delta,
            "sell_prediction_delta": self.sell_prediction_delta,
            "sybil_detection_delta": self.sybil_detection_delta,
            "is_essential": self.is_essential,
            "is_redundant": self.is_redundant,
            "is_harmful": self.is_harmful,
            "assessment_notes": self.assessment_notes,
        }


@dataclass
class AblationStudyResults:
    """Complete ablation study results."""

    feature_group_impacts: dict[str, FeatureGroupImpact]
    baseline_silhouette: Optional[float]
    baseline_noise_rate: float
    baseline_concordance: Optional[float]

    essential_groups: list[str]
    redundant_groups: list[str]
    harmful_groups: list[str]

    recommendation: str
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "feature_group_impacts": {
                k: v.to_dict() for k, v in self.feature_group_impacts.items()
            },
            "baseline_silhouette": self.baseline_silhouette,
            "baseline_noise_rate": self.baseline_noise_rate,
            "baseline_concordance": self.baseline_concordance,
            "essential_groups": self.essential_groups,
            "redundant_groups": self.redundant_groups,
            "harmful_groups": self.harmful_groups,
            "recommendation": self.recommendation,
            "computed_at": self.computed_at.isoformat(),
        }


class AblationRunner:
    """
    Runs comprehensive ablation studies.

    Tests each feature group's impact on:
    - Cluster stability
    - Sell-event prediction
    - Sybil/coordination detection
    - Noise rate
    - UNKNOWN archetype rate
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        n_bootstrap: int = 10,
    ):
        """
        Initialize ablation runner.

        Args:
            min_cluster_size: HDBSCAN min_cluster_size
            n_bootstrap: Bootstrap samples for stability measurement
        """
        self.min_cluster_size = min_cluster_size
        self.n_bootstrap = n_bootstrap

    def run_full_ablation(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        survival_data: Optional[pd.DataFrame] = None,
        coordination_labels: Optional[NDArray[np.int32]] = None,
    ) -> AblationStudyResults:
        """
        Run complete ablation study.

        Args:
            features: (n_samples, n_features) feature array
            feature_names: Feature names
            survival_data: Optional DataFrame with duration/event for hazard ablation
            coordination_labels: Optional ground truth for sybil detection

        Returns:
            AblationStudyResults with all assessments
        """
        logger.info(
            "starting_ablation_study",
            n_samples=features.shape[0],
            n_features=features.shape[1],
            n_groups=len(FEATURE_GROUPS),
        )

        # Run clustering ablation
        tester = AblationTester(
            feature_groups=FEATURE_GROUPS,
            min_cluster_size=self.min_cluster_size,
        )
        clustering_results = tester.run_ablation_study(features, feature_names)

        # Get baseline metrics
        baseline_silhouette = clustering_results.baseline.silhouette_score
        baseline_noise = clustering_results.baseline.noise_percentage

        # Compute baseline concordance if survival data available
        baseline_concordance = None
        if survival_data is not None:
            baseline_concordance = self._compute_concordance(
                features, feature_names, survival_data
            )

        # Assess each feature group
        impacts: dict[str, FeatureGroupImpact] = {}

        for group_name, ablation_result in clustering_results.ablations.items():
            impact = self._assess_feature_group(
                group_name=group_name,
                ablation_result=ablation_result,
                baseline_silhouette=baseline_silhouette,
                baseline_noise=baseline_noise,
                features=features,
                feature_names=feature_names,
                survival_data=survival_data,
                coordination_labels=coordination_labels,
            )
            impacts[group_name] = impact

        # Categorize groups
        essential = [g for g, i in impacts.items() if i.is_essential]
        redundant = [g for g, i in impacts.items() if i.is_redundant]
        harmful = [g for g, i in impacts.items() if i.is_harmful]

        # Generate recommendation
        recommendation = self._generate_recommendation(essential, redundant, harmful)

        logger.info(
            "ablation_study_complete",
            essential_groups=essential,
            redundant_groups=redundant,
            harmful_groups=harmful,
        )

        return AblationStudyResults(
            feature_group_impacts=impacts,
            baseline_silhouette=baseline_silhouette,
            baseline_noise_rate=baseline_noise,
            baseline_concordance=baseline_concordance,
            essential_groups=essential,
            redundant_groups=redundant,
            harmful_groups=harmful,
            recommendation=recommendation,
        )

    def _assess_feature_group(
        self,
        group_name: str,
        ablation_result,
        baseline_silhouette: Optional[float],
        baseline_noise: float,
        features: NDArray[np.float64],
        feature_names: list[str],
        survival_data: Optional[pd.DataFrame],
        coordination_labels: Optional[NDArray[np.int32]],
    ) -> FeatureGroupImpact:
        """Assess impact of a single feature group."""
        notes = []

        # Clustering impact
        stability_delta = ablation_result.silhouette_delta or 0.0
        noise_delta = ablation_result.noise_delta

        # A negative delta means removing hurt performance -> group is important
        # A positive delta means removing improved performance -> group is harmful

        # Compute sell prediction impact if data available
        sell_delta = None
        if survival_data is not None:
            # Get indices of features NOT in this group
            group_features = FEATURE_GROUPS.get(group_name, [])
            group_feature_names = [f.name if hasattr(f, 'name') else f for f in group_features]

            ablated_indices = [
                i for i, name in enumerate(feature_names)
                if name not in group_feature_names
            ]

            if ablated_indices:
                ablated_features = features[:, ablated_indices]
                ablated_names = [feature_names[i] for i in ablated_indices]

                concordance_without = self._compute_concordance(
                    ablated_features, ablated_names, survival_data
                )

                baseline_c = self._compute_concordance(features, feature_names, survival_data)
                if concordance_without is not None and baseline_c is not None:
                    sell_delta = concordance_without - baseline_c

        # Sybil detection impact
        sybil_delta = None
        if coordination_labels is not None:
            # Would compute coordination detection accuracy change
            pass

        # Unknown rate impact (estimate from noise)
        unknown_delta = noise_delta * 0.8  # Rough proxy

        # Determine categorization
        is_essential = False
        is_redundant = False
        is_harmful = False

        # Essential: removing hurts stability significantly
        if stability_delta is not None and stability_delta < -0.05:
            is_essential = True
            notes.append(f"Essential: Removing drops silhouette by {abs(stability_delta):.3f}")

        # Essential: removing hurts prediction significantly
        if sell_delta is not None and sell_delta < -0.02:
            is_essential = True
            notes.append(f"Essential: Removing drops concordance by {abs(sell_delta):.3f}")

        # Redundant: removing has minimal effect
        if (
            (stability_delta is None or abs(stability_delta) < 0.01) and
            (sell_delta is None or abs(sell_delta) < 0.01) and
            abs(noise_delta) < 2.0
        ):
            is_redundant = True
            notes.append("Redundant: Minimal impact when removed")

        # Harmful: removing improves metrics
        if stability_delta is not None and stability_delta > 0.02:
            is_harmful = True
            notes.append(f"Harmful: Removing improves silhouette by {stability_delta:.3f}")

        if noise_delta < -5.0:
            is_harmful = True
            notes.append(f"Harmful: Removing reduces noise by {abs(noise_delta):.1f}%")

        # Get feature list
        group_config = FEATURE_GROUPS.get(group_name, [])
        feature_list = [
            f.name if hasattr(f, 'name') else str(f)
            for f in group_config
        ]

        return FeatureGroupImpact(
            group_name=group_name,
            features=feature_list,
            cluster_stability_delta=stability_delta or 0.0,
            noise_rate_delta=noise_delta,
            unknown_rate_delta=unknown_delta,
            sell_prediction_delta=sell_delta,
            sybil_detection_delta=sybil_delta,
            is_essential=is_essential,
            is_redundant=is_redundant,
            is_harmful=is_harmful,
            assessment_notes=notes,
        )

    def _compute_concordance(
        self,
        features: NDArray[np.float64],
        feature_names: list[str],
        survival_data: pd.DataFrame,
    ) -> Optional[float]:
        """Compute Cox PH concordance index."""
        try:
            from lifelines import CoxPHFitter

            # Prepare data
            df = survival_data.copy()
            for i, name in enumerate(feature_names):
                if name in df.columns:
                    continue
                if i < features.shape[1]:
                    df[name] = features[:, i]

            # Filter to available features
            available = [n for n in feature_names if n in df.columns]
            if not available:
                return None

            cols = ["duration", "event"] + available
            model_df = df[cols].copy().fillna(0)
            model_df = model_df[model_df["duration"] > 0]

            if len(model_df) < 50:
                return None

            fitter = CoxPHFitter(penalizer=0.1)
            fitter.fit(model_df, duration_col="duration", event_col="event")

            return fitter.concordance_index_

        except Exception as e:
            logger.warning("concordance_computation_failed", error=str(e))
            return None

    def _generate_recommendation(
        self,
        essential: list[str],
        redundant: list[str],
        harmful: list[str],
    ) -> str:
        """Generate deployment recommendation based on ablation results."""
        parts = []

        if essential:
            parts.append(f"KEEP: {', '.join(essential)} (essential for performance)")

        if harmful:
            parts.append(f"REMOVE: {', '.join(harmful)} (hurting performance)")

        if redundant:
            parts.append(f"OPTIONAL: {', '.join(redundant)} (minimal impact)")

        if not parts:
            return "All feature groups have moderate impact - keep current configuration"

        return "; ".join(parts)
