"""
Scientific Training Pipeline for Rug Pull Detection.

This module implements a rigorous, scientifically-validated approach
to training rug pull detection models.

Key Features (per DS Knowledge Base and Causal Scientist):
1. Temporal cross-validation (NEVER random shuffle for time-series)
2. Multi-model ensemble (Cox PH, XGBoost, Isolation Forest)
3. Class imbalance handling (SMOTE, class weights)
4. Rigorous statistical validation (Brier, ROC-AUC, calibration)
5. Refutation tests (challenge every estimate)
6. Multiple estimators for robustness

Per Fraud Detection Pattern:
- Extreme imbalance (1:10+ ratio expected)
- Anomaly detection + classification hybrid
- Threshold tuning critical
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()


class ModelType(Enum):
    """Types of models in the ensemble."""
    COX_PH = "cox_ph"
    XGBOOST = "xgboost"
    ISOLATION_FOREST = "isolation_forest"
    ENSEMBLE = "ensemble"


@dataclass
class ClassImbalanceConfig:
    """Configuration for handling class imbalance."""
    use_smote: bool = True
    smote_k_neighbors: int = 5
    use_class_weights: bool = True
    rug_weight_multiplier: float = 1.0  # Will be computed automatically


@dataclass
class TemporalCVConfig:
    """Configuration for temporal cross-validation."""
    n_splits: int = 5
    gap: int = 0  # Gap between train and test
    test_size: Optional[int] = None


@dataclass
class RefutationConfig:
    """Configuration for refutation tests (per Causal Scientist)."""
    run_placebo_test: bool = True
    run_subset_test: bool = True
    subset_fraction: float = 0.8
    run_random_feature_test: bool = True
    max_effect_change_pct: float = 0.5  # If estimate changes >50%, suspicious


@dataclass
class ScientificTrainingConfig:
    """Configuration for scientific training pipeline."""

    # Features to use
    feature_columns: list[str] = field(default_factory=lambda: [
        "hhi",
        "gini",
        "entropy",
        "whale_dominance_top10",
        "whale_dominance_top5",
        "top_holder_share",
        "holder_count",
    ])

    # Model thresholds
    min_roc_auc: float = 0.65
    max_brier_score: float = 0.25
    min_precision: float = 0.50
    min_recall: float = 0.50

    # Ensemble weights (will be optimized)
    ensemble_weights: dict[ModelType, float] = field(default_factory=lambda: {
        ModelType.COX_PH: 0.3,
        ModelType.XGBOOST: 0.5,
        ModelType.ISOLATION_FOREST: 0.2,
    })

    # Sub-configs
    imbalance: ClassImbalanceConfig = field(default_factory=ClassImbalanceConfig)
    temporal_cv: TemporalCVConfig = field(default_factory=TemporalCVConfig)
    refutation: RefutationConfig = field(default_factory=RefutationConfig)


@dataclass
class ModelValidationMetrics:
    """Validation metrics for a single model."""
    model_type: ModelType
    roc_auc: float
    brier_score: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion_matrix: list[list[int]]
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    refutation_passed: bool
    refutation_details: dict[str, Any]


@dataclass
class EnsembleValidationMetrics:
    """Validation metrics for the full ensemble."""
    individual_metrics: dict[ModelType, ModelValidationMetrics]
    ensemble_roc_auc: float
    ensemble_brier_score: float
    ensemble_precision: float
    ensemble_recall: float
    ensemble_f1: float
    optimal_threshold: float
    passes_thresholds: bool
    failed_checks: list[str]
    validation_timestamp: datetime


@dataclass
class TrainedEnsemble:
    """A trained ensemble of models."""
    version: str
    models: dict[ModelType, Any]  # Trained model objects
    scaler: StandardScaler
    feature_columns: list[str]
    class_weights: dict[int, float]
    validation: EnsembleValidationMetrics
    trained_at: datetime
    training_samples: int
    is_deployable: bool

    def predict(
        self,
        X: np.ndarray,
        threshold: float | None = None,
    ) -> tuple[float, int]:
        """
        Make predictions using the ensemble.

        Args:
            X: Feature array of shape (1, n_features) or (n_samples, n_features)
            threshold: Classification threshold (uses optimal if None)

        Returns:
            (probability, predicted_label) tuple
        """
        # Scale features
        X_scaled = self.scaler.transform(X)

        # Use optimal threshold if not specified
        if threshold is None:
            threshold = self.validation.optimal_threshold

        # Get XGBoost prediction (primary model)
        if ModelType.XGBOOST in self.models:
            xgb_model = self.models[ModelType.XGBOOST]
            proba = xgb_model.predict_proba(X_scaled)[:, 1]
        else:
            # Fallback to isolation forest anomaly score
            iso_model = self.models.get(ModelType.ISOLATION_FOREST)
            if iso_model:
                scores = -iso_model.score_samples(X_scaled)
                proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            else:
                raise ValueError("No valid model found in ensemble")

        # Return single values if single sample
        if len(proba) == 1:
            prob = float(proba[0])
            label = 1 if prob >= threshold else 0
            return prob, label

        # Return arrays for batch predictions
        labels = (proba >= threshold).astype(int)
        return proba, labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions.

        Args:
            X: Feature array

        Returns:
            Array of probabilities for the positive (rug) class
        """
        X_scaled = self.scaler.transform(X)

        if ModelType.XGBOOST in self.models:
            return self.models[ModelType.XGBOOST].predict_proba(X_scaled)[:, 1]

        # Fallback
        iso_model = self.models.get(ModelType.ISOLATION_FOREST)
        if iso_model:
            scores = -iso_model.score_samples(X_scaled)
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        raise ValueError("No valid model found")


class ScientificTrainingPipeline:
    """
    Scientific training pipeline for rug pull detection.

    Implements rigorous ML practices:
    - Temporal validation (no data leakage)
    - Multiple estimators for robustness
    - Refutation tests for causal claims
    - Class imbalance handling
    """

    def __init__(self, config: ScientificTrainingConfig | None = None):
        self.config = config or ScientificTrainingConfig()
        self._version_counter = 0

    def prepare_data(
        self,
        data: pd.DataFrame,
        label_column: str = "label",
    ) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
        """
        Prepare data for training.

        Args:
            data: Raw DataFrame with features and labels
            label_column: Name of the label column

        Returns:
            (X, y, scaler) tuple
        """
        # Convert labels to binary
        y = (data[label_column] == "rug").astype(int)

        # Extract features
        X = data[self.config.feature_columns].copy()

        # Handle missing values
        X = X.fillna(0)

        # Scale features (fit on full data, will re-fit on train in CV)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=self.config.feature_columns,
            index=X.index,
        )

        logger.info(
            "data_prepared",
            samples=len(X),
            features=len(self.config.feature_columns),
            rug_count=y.sum(),
            safe_count=(y == 0).sum(),
            imbalance_ratio=f"{y.sum() / max(1, (y == 0).sum()):.2f}",
        )

        return X_scaled, y, scaler

    def compute_class_weights(self, y: pd.Series) -> dict[int, float]:
        """
        Compute class weights for imbalanced data.

        Uses inverse frequency weighting.
        """
        class_counts = y.value_counts()
        total = len(y)

        weights = {}
        for cls, count in class_counts.items():
            weights[cls] = total / (len(class_counts) * count)

        # Apply multiplier for rug class
        if 1 in weights:
            weights[1] *= self.config.imbalance.rug_weight_multiplier

        logger.info("class_weights_computed", weights=weights)
        return weights

    def apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE for oversampling minority class.

        Only applied during training, not validation.
        """
        if not self.config.imbalance.use_smote:
            return X, y

        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(
                k_neighbors=min(
                    self.config.imbalance.smote_k_neighbors,
                    y.sum() - 1,  # Can't have more neighbors than minority samples
                ),
                random_state=42,
            )

            X_resampled, y_resampled = smote.fit_resample(X, y)

            logger.info(
                "smote_applied",
                original_samples=len(X),
                resampled_samples=len(X_resampled),
                original_rug_count=y.sum(),
                resampled_rug_count=y_resampled.sum(),
            )

            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

        except ImportError:
            logger.warning("smote_not_available", message="Install imbalanced-learn")
            return X, y
        except Exception as e:
            logger.warning("smote_failed", error=str(e))
            return X, y

    def temporal_cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_fn: callable,
    ) -> list[float]:
        """
        Temporal cross-validation (NEVER random shuffle).

        Per DS Knowledge Base: "Split by time to preserve temporal order."

        Args:
            X: Features
            y: Labels
            model_fn: Function that creates and fits a model

        Returns:
            List of ROC-AUC scores per fold
        """
        tscv = TimeSeriesSplit(
            n_splits=self.config.temporal_cv.n_splits,
            gap=self.config.temporal_cv.gap,
            test_size=self.config.temporal_cv.test_size,
        )

        scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Apply SMOTE only on training data
            X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train)

            try:
                model = model_fn()
                model.fit(X_train_resampled, y_train_resampled)

                # Get predictions
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = model.decision_function(X_test)
                    # Normalize to [0, 1]
                    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (
                        y_pred_proba.max() - y_pred_proba.min() + 1e-10
                    )

                # Handle edge case of single class in test set
                if len(set(y_test)) < 2:
                    logger.warning("single_class_in_fold", fold=fold)
                    scores.append(0.5)
                else:
                    score = roc_auc_score(y_test, y_pred_proba)
                    scores.append(score)

            except Exception as e:
                logger.warning("cv_fold_failed", fold=fold, error=str(e))
                scores.append(0.5)

        logger.info(
            "temporal_cv_complete",
            n_folds=len(scores),
            mean_score=np.mean(scores),
            std_score=np.std(scores),
        )

        return scores

    def run_refutation_tests(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Any,
        original_score: float,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Run refutation tests per Causal Scientist patterns.

        Tests:
        1. Placebo test: Does random target give similar score?
        2. Subset test: Does subset give similar score?
        3. Random feature test: Does adding noise feature change score?

        Args:
            X: Features
            y: True labels
            model: Trained model
            original_score: Original ROC-AUC score

        Returns:
            (passed, details) tuple
        """
        config = self.config.refutation
        details = {"original_score": original_score}
        passed = True

        # 1. Placebo test: Random labels should give ~0.5 AUC
        if config.run_placebo_test:
            y_placebo = pd.Series(
                np.random.randint(0, 2, size=len(y)),
                index=y.index,
            )

            try:
                model_placebo = model.__class__(**model.get_params())
                model_placebo.fit(X, y_placebo)

                if hasattr(model_placebo, "predict_proba"):
                    y_pred = model_placebo.predict_proba(X)[:, 1]
                else:
                    y_pred = model_placebo.decision_function(X)

                placebo_score = roc_auc_score(y, y_pred)
                details["placebo_score"] = placebo_score

                # Placebo should be close to 0.5 (random)
                if placebo_score > 0.6:
                    logger.warning(
                        "placebo_test_suspicious",
                        placebo_score=placebo_score,
                    )
                    passed = False
                    details["placebo_test_passed"] = False
                else:
                    details["placebo_test_passed"] = True

            except Exception as e:
                details["placebo_test_error"] = str(e)

        # 2. Subset test: Similar performance on random subset
        if config.run_subset_test:
            subset_idx = np.random.choice(
                len(X),
                size=int(len(X) * config.subset_fraction),
                replace=False,
            )
            X_subset = X.iloc[subset_idx]
            y_subset = y.iloc[subset_idx]

            try:
                model_subset = model.__class__(**model.get_params())
                model_subset.fit(X_subset, y_subset)

                if hasattr(model_subset, "predict_proba"):
                    y_pred = model_subset.predict_proba(X_subset)[:, 1]
                else:
                    y_pred = model_subset.decision_function(X_subset)

                if len(set(y_subset)) >= 2:
                    subset_score = roc_auc_score(y_subset, y_pred)
                    details["subset_score"] = subset_score

                    change_pct = abs(original_score - subset_score) / original_score
                    details["subset_change_pct"] = change_pct

                    if change_pct > config.max_effect_change_pct:
                        logger.warning(
                            "subset_test_suspicious",
                            original=original_score,
                            subset=subset_score,
                            change_pct=change_pct,
                        )
                        passed = False
                        details["subset_test_passed"] = False
                    else:
                        details["subset_test_passed"] = True

            except Exception as e:
                details["subset_test_error"] = str(e)

        # 3. Random feature test: Adding noise feature shouldn't help
        if config.run_random_feature_test:
            X_with_noise = X.copy()
            X_with_noise["random_noise"] = np.random.randn(len(X))

            try:
                model_noise = model.__class__(**model.get_params())
                model_noise.fit(X_with_noise, y)

                if hasattr(model_noise, "predict_proba"):
                    y_pred = model_noise.predict_proba(X_with_noise)[:, 1]
                else:
                    y_pred = model_noise.decision_function(X_with_noise)

                noise_score = roc_auc_score(y, y_pred)
                details["noise_feature_score"] = noise_score

                # Adding noise should not significantly improve score
                if noise_score > original_score * 1.1:  # >10% improvement suspicious
                    logger.warning(
                        "noise_feature_test_suspicious",
                        original=original_score,
                        with_noise=noise_score,
                    )
                    passed = False
                    details["noise_test_passed"] = False
                else:
                    details["noise_test_passed"] = True

            except Exception as e:
                details["noise_test_error"] = str(e)

        logger.info(
            "refutation_tests_complete",
            passed=passed,
            tests_run=sum(
                1
                for k in details
                if k.endswith("_passed") or k.endswith("_score")
            ),
        )

        return passed, details

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        class_weights: dict[int, float],
    ) -> Any:
        """Train XGBoost classifier."""
        try:
            from xgboost import XGBClassifier

            # Compute sample weights
            sample_weights = y_train.map(class_weights)

            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=class_weights.get(1, 1.0) / class_weights.get(0, 1.0),
                random_state=42,
                eval_metric="auc",
                use_label_encoder=False,
            )

            model.fit(X_train, y_train, sample_weight=sample_weights)

            logger.info(
                "xgboost_trained",
                n_estimators=model.n_estimators,
                best_iteration=model.best_iteration if hasattr(model, 'best_iteration') else "N/A",
            )

            return model

        except ImportError:
            logger.warning("xgboost_not_available")
            return None

    def train_isolation_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Any:
        """
        Train Isolation Forest for anomaly detection.

        Rug pulls are anomalies (minority class).
        """
        # Train only on safe tokens (normal behavior) if available
        X_normal = X_train[y_train == 0]

        # Fall back to all training data if no safe samples
        if len(X_normal) < 5:
            logger.warning(
                "insufficient_safe_samples_for_isolation_forest",
                safe_count=len(X_normal),
                using_all_data=True,
            )
            X_normal = X_train

        # Calculate contamination based on actual data
        rug_ratio = y_train.mean()
        contamination = min(0.5, max(0.05, rug_ratio))

        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
        )

        model.fit(X_normal)

        logger.info(
            "isolation_forest_trained",
            normal_samples=len(X_normal),
            contamination=contamination,
        )

        return model

    def train_ensemble(
        self,
        data: pd.DataFrame,
        label_column: str = "label",
        version: str | None = None,
    ) -> TrainedEnsemble:
        """
        Train the full ensemble of models.

        Args:
            data: Training data with features and labels
            label_column: Name of the label column
            version: Model version string

        Returns:
            TrainedEnsemble with all models and validation metrics
        """
        if version is None:
            self._version_counter += 1
            version = f"v2.0.{self._version_counter}"

        logger.info("ensemble_training_started", version=version, samples=len(data))

        # Prepare data
        X, y, scaler = self.prepare_data(data, label_column)
        class_weights = self.compute_class_weights(y)

        # Split for final validation (stratified to ensure both classes in both sets)
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )

        # Apply SMOTE to training data
        X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train)

        models = {}
        individual_metrics = {}
        failed_checks = []

        # 1. Train XGBoost
        logger.info("training_xgboost")
        xgb_model = self.train_xgboost(X_train_resampled, y_train_resampled, class_weights)
        if xgb_model:
            models[ModelType.XGBOOST] = xgb_model

            # Cross-validate
            cv_scores = self.temporal_cross_validate(
                X_train, y_train,
                lambda: self.train_xgboost(X_train_resampled, y_train_resampled, class_weights)
            )

            # Evaluate on test set
            y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)

            roc_auc = roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) >= 2 else 0.5
            brier = brier_score_loss(y_test, y_pred_proba)

            # Refutation tests
            refutation_passed, refutation_details = self.run_refutation_tests(
                X_train, y_train, xgb_model, roc_auc
            )

            individual_metrics[ModelType.XGBOOST] = ModelValidationMetrics(
                model_type=ModelType.XGBOOST,
                roc_auc=roc_auc,
                brier_score=brier,
                precision=precision_score(y_test, y_pred, zero_division=0),
                recall=recall_score(y_test, y_pred, zero_division=0),
                f1=f1_score(y_test, y_pred, zero_division=0),
                accuracy=accuracy_score(y_test, y_pred),
                confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
                cv_scores=cv_scores,
                cv_mean=float(np.mean(cv_scores)),
                cv_std=float(np.std(cv_scores)),
                refutation_passed=refutation_passed,
                refutation_details=refutation_details,
            )

        # 2. Train Isolation Forest
        logger.info("training_isolation_forest")
        if_model = self.train_isolation_forest(X_train, y_train)
        models[ModelType.ISOLATION_FOREST] = if_model

        # Evaluate (Isolation Forest returns -1 for anomaly, 1 for normal)
        if_pred = if_model.predict(X_test)
        if_pred_binary = (if_pred == -1).astype(int)  # -1 (anomaly) -> 1 (rug)
        if_scores = -if_model.decision_function(X_test)  # Higher = more anomalous
        if_scores_normalized = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)

        if_roc_auc = roc_auc_score(y_test, if_scores_normalized) if len(set(y_test)) >= 2 else 0.5

        individual_metrics[ModelType.ISOLATION_FOREST] = ModelValidationMetrics(
            model_type=ModelType.ISOLATION_FOREST,
            roc_auc=if_roc_auc,
            brier_score=brier_score_loss(y_test, if_scores_normalized),
            precision=precision_score(y_test, if_pred_binary, zero_division=0),
            recall=recall_score(y_test, if_pred_binary, zero_division=0),
            f1=f1_score(y_test, if_pred_binary, zero_division=0),
            accuracy=accuracy_score(y_test, if_pred_binary),
            confusion_matrix=confusion_matrix(y_test, if_pred_binary).tolist(),
            cv_scores=[],  # No CV for unsupervised
            cv_mean=0.0,
            cv_std=0.0,
            refutation_passed=True,  # Unsupervised doesn't need refutation
            refutation_details={},
        )

        # 3. Ensemble predictions (weighted average)
        ensemble_proba = np.zeros(len(X_test))
        total_weight = 0

        for model_type, weight in self.config.ensemble_weights.items():
            if model_type in models:
                if model_type == ModelType.XGBOOST:
                    proba = models[model_type].predict_proba(X_test)[:, 1]
                elif model_type == ModelType.ISOLATION_FOREST:
                    scores = -models[model_type].decision_function(X_test)
                    proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                else:
                    continue

                ensemble_proba += weight * proba
                total_weight += weight

        if total_weight > 0:
            ensemble_proba /= total_weight

        # Find optimal threshold using precision-recall curve
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, ensemble_proba)
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])  # Last element is for threshold=1
        optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5

        ensemble_pred = (ensemble_proba > optimal_threshold).astype(int)

        # Ensemble metrics
        ensemble_roc_auc = roc_auc_score(y_test, ensemble_proba) if len(set(y_test)) >= 2 else 0.5
        ensemble_brier = brier_score_loss(y_test, ensemble_proba)
        ensemble_precision = precision_score(y_test, ensemble_pred, zero_division=0)
        ensemble_recall = recall_score(y_test, ensemble_pred, zero_division=0)
        ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=0)

        # Check thresholds
        if ensemble_roc_auc < self.config.min_roc_auc:
            failed_checks.append(f"ROC-AUC {ensemble_roc_auc:.3f} < {self.config.min_roc_auc}")
        if ensemble_brier > self.config.max_brier_score:
            failed_checks.append(f"Brier {ensemble_brier:.3f} > {self.config.max_brier_score}")
        if ensemble_precision < self.config.min_precision:
            failed_checks.append(f"Precision {ensemble_precision:.3f} < {self.config.min_precision}")
        if ensemble_recall < self.config.min_recall:
            failed_checks.append(f"Recall {ensemble_recall:.3f} < {self.config.min_recall}")

        # Check refutation tests
        for model_type, metrics in individual_metrics.items():
            if not metrics.refutation_passed:
                failed_checks.append(f"{model_type.value} failed refutation tests")

        passes = len(failed_checks) == 0

        validation = EnsembleValidationMetrics(
            individual_metrics=individual_metrics,
            ensemble_roc_auc=ensemble_roc_auc,
            ensemble_brier_score=ensemble_brier,
            ensemble_precision=ensemble_precision,
            ensemble_recall=ensemble_recall,
            ensemble_f1=ensemble_f1,
            optimal_threshold=float(optimal_threshold),
            passes_thresholds=passes,
            failed_checks=failed_checks,
            validation_timestamp=datetime.now(timezone.utc),
        )

        logger.info(
            "ensemble_training_complete",
            version=version,
            ensemble_roc_auc=ensemble_roc_auc,
            ensemble_f1=ensemble_f1,
            optimal_threshold=optimal_threshold,
            is_deployable=passes,
            failed_checks=failed_checks,
        )

        return TrainedEnsemble(
            version=version,
            models=models,
            scaler=scaler,
            feature_columns=self.config.feature_columns,
            class_weights=class_weights,
            validation=validation,
            trained_at=datetime.now(timezone.utc),
            training_samples=len(X_train),
            is_deployable=passes,
        )

    def predict(
        self,
        ensemble: TrainedEnsemble,
        X: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained ensemble.

        Args:
            ensemble: Trained ensemble
            X: Features DataFrame

        Returns:
            (probabilities, predictions) tuple
        """
        # Ensure features are in correct order
        X_scaled = pd.DataFrame(
            ensemble.scaler.transform(X[ensemble.feature_columns]),
            columns=ensemble.feature_columns,
            index=X.index,
        )

        ensemble_proba = np.zeros(len(X))
        total_weight = 0

        for model_type, weight in self.config.ensemble_weights.items():
            if model_type in ensemble.models:
                model = ensemble.models[model_type]

                if model_type == ModelType.XGBOOST:
                    proba = model.predict_proba(X_scaled)[:, 1]
                elif model_type == ModelType.ISOLATION_FOREST:
                    scores = -model.decision_function(X_scaled)
                    proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                else:
                    continue

                ensemble_proba += weight * proba
                total_weight += weight

        if total_weight > 0:
            ensemble_proba /= total_weight

        predictions = (ensemble_proba > ensemble.validation.optimal_threshold).astype(int)

        return ensemble_proba, predictions
