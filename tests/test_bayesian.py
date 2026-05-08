"""Tests for Bayesian risk estimation module.

This module tests the prior distributions, belief updating,
and risk belief model functionality.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from scipy import stats

from src.bayesian import (
    BetaPrior,
    GammaPrior,
    NormalPrior,
    PriorDistribution,
    create_default_priors,
    Evidence,
    EvidenceType,
    EvidenceBatch,
    BayesianUpdater,
    RiskBeliefModel,
    RiskBeliefState,
    BeliefUpdate,
    RiskEstimate,
)
from src.bayesian.updater import (
    UpdateConfig,
    create_concentration_evidence,
    create_anomaly_evidence,
    create_regime_evidence,
)


class TestBetaPrior:
    """Tests for BetaPrior distribution."""

    def test_basic_properties(self) -> None:
        """Test basic Beta distribution properties."""
        prior = BetaPrior(alpha=2, beta_=5)

        assert prior.name == "Beta(2.00, 5.00)"
        assert prior.mean == pytest.approx(2 / 7)
        assert prior.concentration == 7

    def test_mean_formula(self) -> None:
        """Test mean matches expected formula."""
        for alpha, beta in [(1, 1), (2, 2), (5, 3), (10, 2)]:
            prior = BetaPrior(alpha=alpha, beta_=beta)
            expected_mean = alpha / (alpha + beta)
            assert prior.mean == pytest.approx(expected_mean)

    def test_variance_formula(self) -> None:
        """Test variance matches expected formula."""
        prior = BetaPrior(alpha=2, beta_=5)
        ab = prior.alpha + prior.beta_
        expected_var = (prior.alpha * prior.beta_) / (ab**2 * (ab + 1))
        assert prior.variance == pytest.approx(expected_var)

    def test_mode(self) -> None:
        """Test mode calculation."""
        # Mode exists when alpha > 1 and beta > 1
        prior = BetaPrior(alpha=3, beta_=5)
        expected_mode = (3 - 1) / (3 + 5 - 2)
        assert prior.mode == pytest.approx(expected_mode)

        # Mode doesn't exist for alpha <= 1
        prior_uniform = BetaPrior(alpha=1, beta_=1)
        assert prior_uniform.mode is None

    def test_pdf_integrates_to_one(self) -> None:
        """Test PDF integrates to 1."""
        prior = BetaPrior(alpha=2, beta_=5)
        x = np.linspace(0.001, 0.999, 1000)
        dx = x[1] - x[0]
        integral = np.sum(prior.pdf(x)) * dx
        assert integral == pytest.approx(1.0, rel=0.01)

    def test_cdf_bounds(self) -> None:
        """Test CDF is bounded [0, 1]."""
        prior = BetaPrior(alpha=2, beta_=5)
        assert prior.cdf(0) == pytest.approx(0)
        assert prior.cdf(1) == pytest.approx(1)

    def test_ppf_inverse_of_cdf(self) -> None:
        """Test PPF is inverse of CDF."""
        prior = BetaPrior(alpha=2, beta_=5)
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            x = prior.ppf(q)
            assert prior.cdf(x) == pytest.approx(q, rel=1e-6)

    def test_credible_interval(self) -> None:
        """Test credible interval properties."""
        prior = BetaPrior(alpha=2, beta_=5)
        lower, upper = prior.credible_interval(0.95)

        assert lower < upper
        assert lower >= 0
        assert upper <= 1
        # Check coverage
        assert prior.cdf(upper) - prior.cdf(lower) == pytest.approx(0.95)

    def test_sample_shape(self) -> None:
        """Test sampling returns correct shape."""
        prior = BetaPrior(alpha=2, beta_=5)
        samples = prior.sample(size=1000, random_state=42)
        assert samples.shape == (1000,)
        assert np.all((samples >= 0) & (samples <= 1))

    def test_update_with_successes(self) -> None:
        """Test updating with discrete observations."""
        prior = BetaPrior(alpha=1, beta_=1)  # Uniform prior
        posterior = prior.update(successes=3, failures=7)

        assert posterior.alpha == 4
        assert posterior.beta_ == 8
        assert posterior.mean == pytest.approx(4 / 12)

    def test_update_continuous(self) -> None:
        """Test continuous evidence updating."""
        prior = BetaPrior(alpha=2, beta_=5)
        prior_mean = prior.mean

        # Positive direction should increase mean
        posterior = prior.update_continuous(evidence_strength=1.0, direction=0.8)
        assert posterior.mean > prior_mean

        # Negative direction should decrease mean
        posterior_neg = prior.update_continuous(evidence_strength=1.0, direction=-0.8)
        assert posterior_neg.mean < prior_mean

    def test_from_mean_concentration(self) -> None:
        """Test creating prior from mean and concentration."""
        prior = BetaPrior.from_mean_concentration(mean=0.3, concentration=10)

        assert prior.mean == pytest.approx(0.3)
        assert prior.concentration == pytest.approx(10)

    def test_from_mean_concentration_validation(self) -> None:
        """Test validation in from_mean_concentration."""
        with pytest.raises(ValueError, match="Mean must be in"):
            BetaPrior.from_mean_concentration(mean=1.5, concentration=10)

        with pytest.raises(ValueError, match="Concentration must be positive"):
            BetaPrior.from_mean_concentration(mean=0.5, concentration=-1)

    def test_to_dict(self) -> None:
        """Test serialization."""
        prior = BetaPrior(alpha=2, beta_=5)
        d = prior.to_dict()

        assert d["type"] == "beta"
        assert d["alpha"] == 2
        assert d["beta"] == 5
        assert "mean" in d
        assert "std" in d


class TestGammaPrior:
    """Tests for GammaPrior distribution."""

    def test_basic_properties(self) -> None:
        """Test basic Gamma distribution properties."""
        prior = GammaPrior(shape=2, rate=0.5)

        assert prior.mean == pytest.approx(4)  # shape/rate
        assert prior.variance == pytest.approx(8)  # shape/rate^2
        assert prior.scale == pytest.approx(2)

    def test_pdf_positive_support(self) -> None:
        """Test PDF has positive support."""
        prior = GammaPrior(shape=2, rate=0.5)
        assert prior.pdf(-1) == 0
        assert prior.pdf(0) >= 0
        assert prior.pdf(1) > 0

    def test_update(self) -> None:
        """Test updating with observations."""
        prior = GammaPrior(shape=2, rate=0.5)
        observations = np.array([1.0, 2.0, 3.0])
        posterior = prior.update(observations)

        assert posterior.shape == 2 + 3
        assert posterior.rate == pytest.approx(0.5 + 6.0)


class TestNormalPrior:
    """Tests for NormalPrior distribution."""

    def test_basic_properties(self) -> None:
        """Test basic Normal distribution properties."""
        prior = NormalPrior(mu=0, sigma=1)

        assert prior.mean == 0
        assert prior.variance == 1
        assert prior.std == 1

    def test_pdf_symmetry(self) -> None:
        """Test PDF is symmetric around mean."""
        prior = NormalPrior(mu=5, sigma=2)
        assert prior.pdf(3) == pytest.approx(prior.pdf(7))

    def test_update_single_observation(self) -> None:
        """Test updating with single observation."""
        prior = NormalPrior(mu=0, sigma=1)
        observations = np.array([2.0])
        posterior = prior.update(observations, known_variance=1.0)

        # Posterior mean should be between prior mean and observation
        assert 0 < posterior.mean < 2


class TestCreateDefaultPriors:
    """Tests for default prior creation."""

    def test_creates_all_priors(self) -> None:
        """Test all expected priors are created."""
        priors = create_default_priors()

        assert "rug_probability" in priors
        assert "concentration_risk" in priors
        assert "liquidity_risk" in priors
        assert "time_to_event" in priors
        assert "coordination_score" in priors

    def test_prior_types(self) -> None:
        """Test priors have correct types."""
        priors = create_default_priors()

        assert isinstance(priors["rug_probability"], BetaPrior)
        assert isinstance(priors["time_to_event"], GammaPrior)


class TestEvidence:
    """Tests for Evidence class."""

    @pytest.fixture
    def sample_evidence(self) -> Evidence:
        """Create sample evidence."""
        return Evidence(
            evidence_type=EvidenceType.CONCENTRATION_CHANGE,
            value=0.8,
            timestamp=datetime.now(timezone.utc),
            strength=0.7,
            direction=0.5,
        )

    def test_evidence_creation(self, sample_evidence: Evidence) -> None:
        """Test evidence creation."""
        assert sample_evidence.evidence_type == EvidenceType.CONCENTRATION_CHANGE
        assert sample_evidence.value == 0.8
        assert sample_evidence.strength == 0.7
        assert sample_evidence.direction == 0.5

    def test_evidence_validation(self) -> None:
        """Test evidence validation."""
        with pytest.raises(ValueError, match="strength must be in"):
            Evidence(
                evidence_type=EvidenceType.CONCENTRATION_CHANGE,
                value=0.5,
                timestamp=datetime.now(timezone.utc),
                strength=1.5,  # Invalid
            )

        with pytest.raises(ValueError, match="direction must be in"):
            Evidence(
                evidence_type=EvidenceType.CONCENTRATION_CHANGE,
                value=0.5,
                timestamp=datetime.now(timezone.utc),
                direction=2.0,  # Invalid
            )

    def test_is_risky(self, sample_evidence: Evidence) -> None:
        """Test is_risky property."""
        assert sample_evidence.is_risky  # direction > 0

        safe_evidence = Evidence(
            evidence_type=EvidenceType.CONCENTRATION_CHANGE,
            value=0.5,
            timestamp=datetime.now(timezone.utc),
            direction=-0.5,
        )
        assert not safe_evidence.is_risky

    def test_effective_strength(self, sample_evidence: Evidence) -> None:
        """Test effective_strength calculation."""
        assert sample_evidence.effective_strength == pytest.approx(0.7 * 0.5)

    def test_decay(self) -> None:
        """Test evidence decay over time."""
        old_timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
        evidence = Evidence(
            evidence_type=EvidenceType.CONCENTRATION_CHANGE,
            value=0.5,
            timestamp=old_timestamp,
            strength=1.0,
        )

        decayed = evidence.decay(half_life_hours=24.0)

        assert decayed.strength < evidence.strength
        assert decayed.metadata.get("decayed") is True


class TestEvidenceBatch:
    """Tests for EvidenceBatch class."""

    def test_batch_creation(self) -> None:
        """Test batch creation and add."""
        batch = EvidenceBatch()
        evidence = Evidence(
            evidence_type=EvidenceType.CONCENTRATION_CHANGE,
            value=0.5,
            timestamp=datetime.now(timezone.utc),
        )

        batch.add(evidence)

        assert len(batch) == 1

    def test_filter_by_type(self) -> None:
        """Test filtering by evidence type."""
        batch = EvidenceBatch()
        batch.add(Evidence(
            evidence_type=EvidenceType.CONCENTRATION_CHANGE,
            value=0.5,
            timestamp=datetime.now(timezone.utc),
        ))
        batch.add(Evidence(
            evidence_type=EvidenceType.ANOMALY_DETECTION,
            value=0.8,
            timestamp=datetime.now(timezone.utc),
        ))

        filtered = batch.filter_by_type(EvidenceType.CONCENTRATION_CHANGE)

        assert len(filtered) == 1
        assert filtered.evidences[0].evidence_type == EvidenceType.CONCENTRATION_CHANGE

    def test_net_direction(self) -> None:
        """Test net direction calculation."""
        batch = EvidenceBatch()
        batch.add(Evidence(
            evidence_type=EvidenceType.CONCENTRATION_CHANGE,
            value=0.5,
            timestamp=datetime.now(timezone.utc),
            strength=1.0,
            direction=0.8,
        ))
        batch.add(Evidence(
            evidence_type=EvidenceType.ANOMALY_DETECTION,
            value=0.5,
            timestamp=datetime.now(timezone.utc),
            strength=1.0,
            direction=-0.2,
        ))

        # Net direction should be weighted average
        expected = (0.8 + (-0.2)) / 2
        assert batch.net_direction == pytest.approx(expected)


class TestBayesianUpdater:
    """Tests for BayesianUpdater class."""

    @pytest.fixture
    def updater(self) -> BayesianUpdater:
        """Create updater with default config."""
        return BayesianUpdater()

    def test_update_beta_increases_risk(self, updater: BayesianUpdater) -> None:
        """Test that risky evidence increases posterior mean."""
        prior = BetaPrior(alpha=2, beta_=5)
        evidence = Evidence(
            evidence_type=EvidenceType.DUMP_SIGNATURE,
            value=0.8,
            timestamp=datetime.now(timezone.utc),
            strength=0.8,
            direction=0.9,
        )

        posterior = updater.update_beta(prior, evidence)

        assert posterior.mean > prior.mean

    def test_update_beta_decreases_risk(self, updater: BayesianUpdater) -> None:
        """Test that safe evidence decreases posterior mean."""
        prior = BetaPrior(alpha=5, beta_=2)
        evidence = Evidence(
            evidence_type=EvidenceType.CONCENTRATION_CHANGE,
            value=0.2,
            timestamp=datetime.now(timezone.utc),
            strength=0.8,
            direction=-0.8,
        )

        posterior = updater.update_beta(prior, evidence)

        assert posterior.mean < prior.mean

    def test_update_respects_threshold(self, updater: BayesianUpdater) -> None:
        """Test that weak evidence is ignored."""
        config = UpdateConfig(min_strength_threshold=0.5)
        updater = BayesianUpdater(config=config)

        prior = BetaPrior(alpha=2, beta_=5)
        weak_evidence = Evidence(
            evidence_type=EvidenceType.CONCENTRATION_CHANGE,
            value=0.5,
            timestamp=datetime.now(timezone.utc),
            strength=0.3,  # Below threshold
            direction=0.9,
        )

        posterior = updater.update_beta(prior, weak_evidence)

        # Should be unchanged
        assert posterior.alpha == prior.alpha
        assert posterior.beta_ == prior.beta_

    def test_information_gain_nonnegative(self, updater: BayesianUpdater) -> None:
        """Test information gain is non-negative."""
        prior = BetaPrior(alpha=2, beta_=5)
        evidence = Evidence(
            evidence_type=EvidenceType.CONCENTRATION_CHANGE,
            value=0.5,
            timestamp=datetime.now(timezone.utc),
            strength=0.7,
            direction=0.5,
        )

        info_gain = updater.compute_information_gain(prior, evidence)

        assert info_gain >= 0


class TestEvidenceFactories:
    """Tests for evidence factory functions."""

    def test_create_concentration_evidence(self) -> None:
        """Test concentration evidence creation."""
        evidence = create_concentration_evidence(old_hhi=0.3, new_hhi=0.5)

        assert evidence.evidence_type == EvidenceType.CONCENTRATION_CHANGE
        assert evidence.value == 0.5
        assert evidence.direction > 0  # Concentration increased = risky

    def test_create_anomaly_evidence(self) -> None:
        """Test anomaly evidence creation."""
        evidence = create_anomaly_evidence(anomaly_score=0.8, wallet_count=5)

        assert evidence.evidence_type == EvidenceType.ANOMALY_DETECTION
        assert evidence.value == 0.8
        assert evidence.direction > 0  # High anomaly = risky

    def test_create_regime_evidence(self) -> None:
        """Test regime transition evidence creation."""
        evidence = create_regime_evidence(
            from_regime="accumulating",
            to_regime="distributing",
            confidence=0.9,
        )

        assert evidence.evidence_type == EvidenceType.REGIME_TRANSITION
        assert evidence.direction > 0  # Accumulating -> distributing = risky


class TestRiskBeliefModel:
    """Tests for RiskBeliefModel class."""

    @pytest.fixture
    def model(self) -> RiskBeliefModel:
        """Create model with default priors."""
        return RiskBeliefModel(prior_alpha=2, prior_beta=5)

    def test_initialization(self, model: RiskBeliefModel) -> None:
        """Test model initialization."""
        assert model.alpha == 2
        assert model.beta == 5
        assert model.state.rug_probability.mean == pytest.approx(2 / 7)

    def test_update_single_evidence(self, model: RiskBeliefModel) -> None:
        """Test updating with single evidence."""
        prior_mean = model.state.rug_probability.mean

        evidence = Evidence(
            evidence_type=EvidenceType.DUMP_SIGNATURE,
            value=0.8,
            timestamp=datetime.now(timezone.utc),
            strength=0.8,
            direction=0.9,
        )

        model.update(evidence)

        assert model.state.rug_probability.mean > prior_mean
        assert model.state.total_updates == 1

    def test_update_batch_evidence(self, model: RiskBeliefModel) -> None:
        """Test updating with batch evidence."""
        batch = EvidenceBatch()
        batch.add(Evidence(
            evidence_type=EvidenceType.CONCENTRATION_CHANGE,
            value=0.7,
            timestamp=datetime.now(timezone.utc),
            strength=0.6,
            direction=0.5,
        ))
        batch.add(Evidence(
            evidence_type=EvidenceType.ANOMALY_DETECTION,
            value=0.8,
            timestamp=datetime.now(timezone.utc),
            strength=0.7,
            direction=0.6,
        ))

        model.update(batch)

        assert model.state.total_updates == 2

    def test_posterior_rug_probability(self, model: RiskBeliefModel) -> None:
        """Test posterior estimate."""
        estimate = model.posterior_rug_probability()

        assert isinstance(estimate, RiskEstimate)
        assert 0 <= estimate.mean <= 1
        assert estimate.lower_ci < estimate.mean < estimate.upper_ci
        assert estimate.confidence_level == 0.95

    def test_credible_interval(self, model: RiskBeliefModel) -> None:
        """Test credible interval calculation."""
        lower, upper = model.credible_interval(0.95)

        assert lower < upper
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1

    def test_information_gain(self, model: RiskBeliefModel) -> None:
        """Test information gain calculation."""
        evidence = Evidence(
            evidence_type=EvidenceType.CONCENTRATION_CHANGE,
            value=0.5,
            timestamp=datetime.now(timezone.utc),
            strength=0.7,
            direction=0.5,
        )

        info_gain = model.information_gain(evidence)

        assert info_gain >= 0

    def test_composite_risk_score(self, model: RiskBeliefModel) -> None:
        """Test composite risk score."""
        composite = model.composite_risk_score()

        assert isinstance(composite, RiskEstimate)
        assert 0 <= composite.mean <= 1

    def test_risk_decomposition(self, model: RiskBeliefModel) -> None:
        """Test risk decomposition."""
        decomp = model.risk_decomposition()

        assert "rug_probability" in decomp
        assert "concentration_risk" in decomp
        assert "liquidity_risk" in decomp
        assert "coordination_risk" in decomp

        for name, estimate in decomp.items():
            assert isinstance(estimate, RiskEstimate)

    def test_uncertainty_level(self, model: RiskBeliefModel) -> None:
        """Test uncertainty level categorization."""
        level = model.uncertainty_level()
        assert level in ("low", "moderate", "high", "very_high")

    def test_suggest_evidence(self, model: RiskBeliefModel) -> None:
        """Test evidence suggestion."""
        suggested = model.suggest_evidence([
            EvidenceType.CONCENTRATION_CHANGE,
            EvidenceType.ANOMALY_DETECTION,
        ])

        assert isinstance(suggested, EvidenceType)

    def test_sample_risk(self, model: RiskBeliefModel) -> None:
        """Test sampling from risk distribution."""
        samples = model.sample_risk(size=100, random_state=42)

        assert samples.shape == (100,)
        assert np.all((samples >= 0) & (samples <= 1))

    def test_reset(self, model: RiskBeliefModel) -> None:
        """Test model reset."""
        # Update model
        evidence = Evidence(
            evidence_type=EvidenceType.DUMP_SIGNATURE,
            value=0.8,
            timestamp=datetime.now(timezone.utc),
            strength=0.8,
            direction=0.9,
        )
        model.update(evidence)

        # Reset
        model.reset(prior_alpha=1, prior_beta=1)

        assert model.alpha == 1
        assert model.beta == 1
        assert model.state.total_updates == 0

    def test_to_dict(self, model: RiskBeliefModel) -> None:
        """Test serialization."""
        d = model.to_dict()

        assert "state" in d
        assert "rug_estimate" in d
        assert "composite_risk" in d
        assert "uncertainty_level" in d

    def test_from_historical_rate(self) -> None:
        """Test creating model from historical rate."""
        model = RiskBeliefModel.from_historical_rate(
            historical_rug_rate=0.3,
            sample_size=100,
        )

        # Mean should be close to historical rate
        assert model.state.rug_probability.mean == pytest.approx(0.3, rel=0.1)


class TestRiskEstimate:
    """Tests for RiskEstimate dataclass."""

    def test_properties(self) -> None:
        """Test RiskEstimate properties."""
        estimate = RiskEstimate(
            mean=0.3,
            lower_ci=0.2,
            upper_ci=0.4,
            std=0.05,
            confidence_level=0.95,
        )

        assert estimate.width == pytest.approx(0.2)
        assert estimate.relative_uncertainty == pytest.approx(0.2 / 0.3)

    def test_to_dict(self) -> None:
        """Test RiskEstimate serialization."""
        estimate = RiskEstimate(
            mean=0.3,
            lower_ci=0.2,
            upper_ci=0.4,
            std=0.05,
        )

        d = estimate.to_dict()

        assert d["mean"] == 0.3
        assert d["lower_ci"] == 0.2
        assert d["upper_ci"] == 0.4
        assert d["std"] == 0.05


class TestIntegration:
    """Integration tests for Bayesian module."""

    def test_full_belief_update_workflow(self) -> None:
        """Test complete belief update workflow."""
        # Create model with skeptical prior
        model = RiskBeliefModel(prior_alpha=1, prior_beta=3)
        initial_estimate = model.posterior_rug_probability()

        # Add evidence of increasing risk
        evidences = [
            create_concentration_evidence(0.3, 0.5),
            create_anomaly_evidence(0.7, 3),
            create_regime_evidence("stable", "distributing", 0.8),
        ]

        for evidence in evidences:
            model.update(evidence)

        final_estimate = model.posterior_rug_probability()

        # Risk should have increased
        assert final_estimate.mean > initial_estimate.mean
        # Uncertainty should have decreased (more evidence)
        assert final_estimate.width <= initial_estimate.width + 0.1  # Allow some tolerance

    def test_belief_convergence(self) -> None:
        """Test that beliefs converge with consistent evidence."""
        model = RiskBeliefModel(prior_alpha=1, prior_beta=1)  # Uniform prior

        # Add many consistent positive signals
        for _ in range(20):
            evidence = Evidence(
                evidence_type=EvidenceType.DUMP_SIGNATURE,
                value=0.9,
                timestamp=datetime.now(timezone.utc),
                strength=0.8,
                direction=0.9,
            )
            model.update(evidence)

        estimate = model.posterior_rug_probability()

        # Should converge toward high risk
        assert estimate.mean > 0.5
        # Should have low uncertainty
        assert model.uncertainty_level() in ("low", "moderate")
