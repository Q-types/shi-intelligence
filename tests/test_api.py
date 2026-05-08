"""Tests for the SHI REST API.

This module tests the FastAPI endpoints for token intelligence,
wallet profiles, and Bayesian risk updates.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.dependencies import clear_risk_models
from src.api.schemas import (
    TokenAnalysisRequest,
    ForecastRequest,
    RiskUpdateRequest,
    EvidenceInput,
    HealthResponse,
    ErrorResponse,
)


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    clear_risk_models()  # Reset state before each test
    return TestClient(app)


@pytest.fixture
def sample_mint() -> str:
    """Sample token mint address."""
    return "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


@pytest.fixture
def sample_wallet() -> str:
    """Sample wallet address."""
    return "9WzDXwBbmPdLGzGNVJzJsqGy4V4H9jF6PvfBNfyKLdNN"


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check returns healthy status."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "components" in data

    def test_health_response_structure(self, client: TestClient) -> None:
        """Test health response matches schema."""
        response = client.get("/api/v1/health")
        data = response.json()

        # Validate can be parsed as HealthResponse
        health = HealthResponse(**data)
        assert health.status in ("healthy", "degraded", "unhealthy")


class TestTokenIntelligenceEndpoints:
    """Tests for token intelligence endpoints."""

    def test_get_intelligence_basic(
        self, client: TestClient, sample_mint: str
    ) -> None:
        """Test basic intelligence retrieval."""
        response = client.get(f"/api/v1/token/{sample_mint}/intelligence")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["token_mint"] == sample_mint

    def test_get_intelligence_with_params(
        self, client: TestClient, sample_mint: str
    ) -> None:
        """Test intelligence with query parameters."""
        response = client.get(
            f"/api/v1/token/{sample_mint}/intelligence",
            params={
                "include_historical": True,
                "include_forecast": False,
                "forecast_days": 14,
            },
        )

        assert response.status_code == 200

    def test_get_intelligence_invalid_mint(self, client: TestClient) -> None:
        """Test intelligence with invalid mint address."""
        response = client.get("/api/v1/token/invalid/intelligence")

        assert response.status_code == 400

    def test_get_forecast(self, client: TestClient, sample_mint: str) -> None:
        """Test forecast endpoint."""
        response = client.get(
            f"/api/v1/token/{sample_mint}/forecast",
            params={"days": 7},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["token_mint"] == sample_mint

    def test_get_explanation(self, client: TestClient, sample_mint: str) -> None:
        """Test risk explanation endpoint."""
        response = client.get(f"/api/v1/token/{sample_mint}/explain")

        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "top_factors" in data


class TestWalletEndpoints:
    """Tests for wallet intelligence endpoints."""

    def test_get_wallet_profile(
        self, client: TestClient, sample_wallet: str
    ) -> None:
        """Test wallet profile retrieval."""
        response = client.get(f"/api/v1/wallet/{sample_wallet}/profile")

        assert response.status_code == 200
        data = response.json()
        assert data["wallet_address"] == sample_wallet
        assert "current_archetype" in data
        assert "risk_score" in data

    def test_get_wallet_profile_invalid_address(
        self, client: TestClient
    ) -> None:
        """Test wallet profile with invalid address."""
        response = client.get("/api/v1/wallet/invalid/profile")

        assert response.status_code == 400

    def test_analyze_sequence(
        self, client: TestClient, sample_wallet: str
    ) -> None:
        """Test sequence analysis endpoint."""
        response = client.get(f"/api/v1/wallet/{sample_wallet}/sequence")

        assert response.status_code == 200
        data = response.json()
        assert data["wallet"] == sample_wallet
        assert "dump_likelihood" in data
        assert "signatures_found" in data


class TestRiskBeliefEndpoints:
    """Tests for Bayesian risk estimation endpoints."""

    def test_get_risk_beliefs_initial(
        self, client: TestClient, sample_mint: str
    ) -> None:
        """Test getting initial risk beliefs."""
        response = client.get(f"/api/v1/token/{sample_mint}/risk/belief")

        assert response.status_code == 200
        data = response.json()
        assert data["token_mint"] == sample_mint
        assert "rug_probability" in data
        assert "uncertainty_level" in data
        assert data["updates_applied"] == 0

    def test_update_risk_beliefs(
        self, client: TestClient, sample_mint: str
    ) -> None:
        """Test updating risk beliefs with evidence."""
        request_data = {
            "evidences": [
                {
                    "evidence_type": "concentration_change",
                    "value": 0.7,
                    "strength": 0.8,
                    "direction": 0.6,
                },
            ],
            "reset_beliefs": False,
        }

        response = client.post(
            f"/api/v1/token/{sample_mint}/risk/update",
            json=request_data,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["updates_applied"] >= 1

    def test_update_with_multiple_evidence(
        self, client: TestClient, sample_mint: str
    ) -> None:
        """Test updating with multiple evidence items."""
        request_data = {
            "evidences": [
                {
                    "evidence_type": "concentration_change",
                    "value": 0.7,
                    "strength": 0.8,
                    "direction": 0.5,
                },
                {
                    "evidence_type": "anomaly_detection",
                    "value": 0.9,
                    "strength": 0.7,
                    "direction": 0.8,
                },
            ],
        }

        response = client.post(
            f"/api/v1/token/{sample_mint}/risk/update",
            json=request_data,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["updates_applied"] >= 2

    def test_update_with_reset(
        self, client: TestClient, sample_mint: str
    ) -> None:
        """Test updating with belief reset."""
        # First update
        client.post(
            f"/api/v1/token/{sample_mint}/risk/update",
            json={
                "evidences": [{"evidence_type": "dump_signature", "value": 0.9, "direction": 0.9}],
            },
        )

        # Reset and update
        response = client.post(
            f"/api/v1/token/{sample_mint}/risk/update",
            json={
                "evidences": [{"evidence_type": "concentration_change", "value": 0.3, "direction": -0.3}],
                "reset_beliefs": True,
                "prior_alpha": 1.0,
                "prior_beta": 1.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        # After reset and one update, should have 1 update
        assert data["updates_applied"] == 1

    def test_update_invalid_evidence_type(
        self, client: TestClient, sample_mint: str
    ) -> None:
        """Test update with invalid evidence type."""
        request_data = {
            "evidences": [
                {
                    "evidence_type": "invalid_type",
                    "value": 0.5,
                },
            ],
        }

        response = client.post(
            f"/api/v1/token/{sample_mint}/risk/update",
            json=request_data,
        )

        assert response.status_code == 400

    def test_reset_risk_beliefs(
        self, client: TestClient, sample_mint: str
    ) -> None:
        """Test resetting risk beliefs."""
        # First create some updates
        client.post(
            f"/api/v1/token/{sample_mint}/risk/update",
            json={
                "evidences": [{"evidence_type": "dump_signature", "value": 0.9, "direction": 0.9}],
            },
        )

        # Reset
        response = client.delete(
            f"/api/v1/token/{sample_mint}/risk/reset",
            params={"alpha": 2.0, "beta": 3.0},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify reset
        beliefs = client.get(f"/api/v1/token/{sample_mint}/risk/belief").json()
        assert beliefs["updates_applied"] == 0

    def test_risk_beliefs_persist(
        self, client: TestClient, sample_mint: str
    ) -> None:
        """Test that beliefs persist across requests."""
        # Update
        client.post(
            f"/api/v1/token/{sample_mint}/risk/update",
            json={
                "evidences": [{"evidence_type": "concentration_change", "value": 0.8, "direction": 0.7}],
            },
        )

        # Get beliefs
        response1 = client.get(f"/api/v1/token/{sample_mint}/risk/belief")
        mean1 = response1.json()["rug_probability"]["mean"]

        # Update again
        client.post(
            f"/api/v1/token/{sample_mint}/risk/update",
            json={
                "evidences": [{"evidence_type": "dump_signature", "value": 0.9, "direction": 0.9}],
            },
        )

        # Get beliefs again - should have changed
        response2 = client.get(f"/api/v1/token/{sample_mint}/risk/belief")
        mean2 = response2.json()["rug_probability"]["mean"]

        # Risk should have increased with positive evidence
        assert mean2 > mean1


class TestAdminEndpoints:
    """Tests for admin endpoints."""

    def test_clear_cache(self, client: TestClient, sample_mint: str) -> None:
        """Test clearing risk model cache."""
        # Create a model
        client.get(f"/api/v1/token/{sample_mint}/risk/belief")

        # Clear cache
        response = client.post("/api/v1/admin/clear-cache")

        assert response.status_code == 200
        assert response.json()["status"] == "success"


class TestSchemas:
    """Tests for request/response schemas."""

    def test_token_analysis_request_defaults(self) -> None:
        """Test TokenAnalysisRequest defaults."""
        request = TokenAnalysisRequest()

        assert request.include_historical is True
        assert request.include_forecast is False
        assert request.forecast_days == 7

    def test_evidence_input_validation(self) -> None:
        """Test EvidenceInput validation."""
        # Valid input
        evidence = EvidenceInput(
            evidence_type="concentration_change",
            value=0.5,
            strength=0.8,
            direction=0.5,
        )
        assert evidence.strength == 0.8

        # Invalid strength
        with pytest.raises(ValueError):
            EvidenceInput(
                evidence_type="concentration_change",
                value=0.5,
                strength=1.5,  # > 1
            )

        # Invalid direction
        with pytest.raises(ValueError):
            EvidenceInput(
                evidence_type="concentration_change",
                value=0.5,
                direction=2.0,  # > 1
            )

    def test_evidence_input_to_evidence(self) -> None:
        """Test converting EvidenceInput to Evidence."""
        input_data = EvidenceInput(
            evidence_type="concentration_change",
            value=0.7,
            strength=0.8,
            direction=0.5,
        )

        evidence = input_data.to_evidence()

        assert evidence.value == 0.7
        assert evidence.strength == 0.8
        assert evidence.direction == 0.5

    def test_evidence_input_invalid_type(self) -> None:
        """Test EvidenceInput with invalid type."""
        input_data = EvidenceInput(
            evidence_type="invalid_type",
            value=0.5,
        )

        with pytest.raises(ValueError, match="Unknown evidence type"):
            input_data.to_evidence()

    def test_risk_update_request_validation(self) -> None:
        """Test RiskUpdateRequest validation."""
        # Valid request
        request = RiskUpdateRequest(
            evidences=[
                EvidenceInput(evidence_type="concentration_change", value=0.5),
            ],
        )
        assert len(request.evidences) == 1

        # Empty evidences should fail
        with pytest.raises(ValueError):
            RiskUpdateRequest(evidences=[])


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_for_unknown_route(self, client: TestClient) -> None:
        """Test 404 for unknown routes."""
        response = client.get("/api/v1/unknown")
        assert response.status_code == 404

    def test_validation_error_response(
        self, client: TestClient, sample_mint: str
    ) -> None:
        """Test validation error response format."""
        # Send invalid evidence
        response = client.post(
            f"/api/v1/token/{sample_mint}/risk/update",
            json={
                "evidences": [{"evidence_type": "invalid", "value": 0.5}],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data or "detail" in data


class TestIntegration:
    """Integration tests for API workflows."""

    def test_complete_analysis_workflow(
        self, client: TestClient, sample_mint: str
    ) -> None:
        """Test complete analysis workflow."""
        # 1. Get initial intelligence
        intel_response = client.get(f"/api/v1/token/{sample_mint}/intelligence")
        assert intel_response.status_code == 200

        # 2. Get initial risk beliefs
        belief_response = client.get(f"/api/v1/token/{sample_mint}/risk/belief")
        assert belief_response.status_code == 200
        initial_mean = belief_response.json()["rug_probability"]["mean"]

        # 3. Update with evidence
        update_response = client.post(
            f"/api/v1/token/{sample_mint}/risk/update",
            json={
                "evidences": [
                    {"evidence_type": "concentration_change", "value": 0.8, "direction": 0.7},
                    {"evidence_type": "dump_signature", "value": 0.9, "direction": 0.9},
                ],
            },
        )
        assert update_response.status_code == 200

        # 4. Verify beliefs updated
        final_response = client.get(f"/api/v1/token/{sample_mint}/risk/belief")
        final_mean = final_response.json()["rug_probability"]["mean"]

        # Risk should have increased
        assert final_mean > initial_mean

    def test_wallet_analysis_workflow(
        self, client: TestClient, sample_wallet: str
    ) -> None:
        """Test wallet analysis workflow."""
        # 1. Get profile
        profile_response = client.get(f"/api/v1/wallet/{sample_wallet}/profile")
        assert profile_response.status_code == 200

        # 2. Analyze sequence
        sequence_response = client.get(f"/api/v1/wallet/{sample_wallet}/sequence")
        assert sequence_response.status_code == 200
        assert "dump_likelihood" in sequence_response.json()
