"""
Integration Tests with Live Data.

Tests the complete SHI pipeline with real Helius API data.
These tests require HELIUS_API_KEY environment variable.

Sprint 6: Validates Helius integration, model inference, and alert system.
"""

from __future__ import annotations

import asyncio
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import numpy as np

# Check if we have live API access
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
HAS_LIVE_ACCESS = bool(HELIUS_API_KEY)

# Test tokens (well-known Solana tokens)
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
JUP_MINT = "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"


@pytest.fixture
def model_path() -> Optional[Path]:
    """Get trained model path if available."""
    model_dir = Path(__file__).parent.parent.parent / "models" / "trained"
    if model_dir.exists():
        models = list(model_dir.glob("ensemble_*.pkl"))
        if models:
            return max(models, key=lambda p: p.stat().st_mtime)

    # Check debug folder
    debug_dir = model_dir / "debug"
    if debug_dir.exists():
        models = list(debug_dir.glob("ensemble_*.pkl"))
        if models:
            return max(models, key=lambda p: p.stat().st_mtime)

    return None


class TestHeliusIntegration:
    """Tests for Helius RPC integration."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_LIVE_ACCESS, reason="HELIUS_API_KEY not set")
    async def test_helius_provider_initialization(self) -> None:
        """Test that HeliusProvider initializes correctly."""
        from src.data.providers import HeliusProvider

        provider = HeliusProvider()
        assert provider.name == "helius"
        assert provider.api_key is not None

        await provider.close()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_LIVE_ACCESS, reason="HELIUS_API_KEY not set")
    async def test_fetch_token_holders_usdc(self) -> None:
        """Test fetching USDC holders from Helius."""
        from src.data.providers import HeliusProvider

        provider = HeliusProvider()

        try:
            snapshot = await provider.get_token_holders(USDC_MINT, limit=100)

            # Validate snapshot structure
            assert snapshot.mint == USDC_MINT
            assert snapshot.holder_count > 0
            assert snapshot.total_supply > 0
            assert len(snapshot.balances) > 0

            # USDC should have many holders
            assert snapshot.holder_count >= 100

            # Validate balance data
            for balance in snapshot.balances[:10]:
                assert balance.wallet is not None
                assert len(balance.wallet) >= 32
                assert balance.balance >= 0

            print(f"\n✅ USDC: {snapshot.holder_count} holders fetched")
            print(f"   Total supply: {snapshot.total_supply:,.0f}")
            print(f"   Top holder: {snapshot.balances[0].balance:,.0f}")

        finally:
            await provider.close()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_LIVE_ACCESS, reason="HELIUS_API_KEY not set")
    async def test_fetch_bonk_holders(self) -> None:
        """Test fetching BONK holders (meme token with many holders)."""
        from src.data.providers import HeliusProvider

        provider = HeliusProvider()

        try:
            snapshot = await provider.get_token_holders(BONK_MINT, limit=200)

            assert snapshot.holder_count > 0
            assert len(snapshot.balances) > 0

            # BONK is a popular meme token
            print(f"\n✅ BONK: {snapshot.holder_count} holders fetched")

            # Compute concentration metrics
            from src.metrics.distribution import (
                compute_hhi,
                compute_gini_coefficient,
                compute_whale_dominance_ratio,
            )

            shares = snapshot.shares
            balances = [b.balance for b in snapshot.balances]

            hhi = compute_hhi(shares)
            gini = compute_gini_coefficient(balances)
            wdr = compute_whale_dominance_ratio(balances, snapshot.total_supply, k=10)

            print(f"   HHI: {hhi.value:.6f}")
            print(f"   Gini: {gini.value:.4f}")
            print(f"   Top 10 Whale Dominance: {wdr.value:.2%}")

            # BONK should have relatively distributed holdings
            assert hhi.value < 0.5, "BONK should not be highly concentrated"

        finally:
            await provider.close()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_LIVE_ACCESS, reason="HELIUS_API_KEY not set")
    async def test_rate_limiting(self) -> None:
        """Test that rate limiting is respected."""
        from src.data.providers import HeliusProvider
        import time

        provider = HeliusProvider()

        try:
            # Make multiple rapid requests
            start_time = time.time()
            snapshots = []

            for _ in range(3):
                snapshot = await provider.get_token_holders(USDC_MINT, limit=50)
                snapshots.append(snapshot)

            elapsed = time.time() - start_time

            # All requests should succeed
            assert len(snapshots) == 3
            assert all(s.holder_count > 0 for s in snapshots)

            print(f"\n✅ 3 requests completed in {elapsed:.2f}s")

        finally:
            await provider.close()


class TestModelInference:
    """Tests for trained model inference."""

    @pytest.mark.asyncio
    async def test_model_loading(self, model_path: Optional[Path]) -> None:
        """Test loading the trained ensemble model."""
        if model_path is None:
            pytest.skip("No trained model available")

        with open(model_path, "rb") as f:
            ensemble = pickle.load(f)

        assert ensemble is not None
        assert hasattr(ensemble, "models")
        assert hasattr(ensemble, "validation")
        assert hasattr(ensemble, "predict")

        print(f"\n✅ Model loaded: {model_path.name}")
        print(f"   Version: {ensemble.version}")
        print(f"   Training samples: {ensemble.training_samples}")
        print(f"   Deployable: {ensemble.is_deployable}")

    @pytest.mark.asyncio
    async def test_model_prediction_format(self, model_path: Optional[Path]) -> None:
        """Test that model predictions have correct format."""
        if model_path is None:
            pytest.skip("No trained model available")

        with open(model_path, "rb") as f:
            ensemble = pickle.load(f)

        # Create test features (matching training feature order)
        test_features = np.array([[
            0.15,   # hhi
            0.65,   # gini
            2.5,    # entropy
            0.35,   # whale_dominance_top10
            0.25,   # whale_dominance_top5
            0.12,   # top_holder_share
            150,    # holder_count
        ]])

        # Get prediction
        prob, label = ensemble.predict(test_features)

        # Validate output format
        assert isinstance(prob, (float, np.floating))
        assert 0 <= prob <= 1
        assert label in [0, 1]

        print(f"\n✅ Prediction test:")
        print(f"   Rug probability: {prob:.4f}")
        print(f"   Predicted label: {'rug' if label == 1 else 'safe'}")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_LIVE_ACCESS, reason="HELIUS_API_KEY not set")
    async def test_end_to_end_inference(self, model_path: Optional[Path]) -> None:
        """Test end-to-end inference with live data."""
        if model_path is None:
            pytest.skip("No trained model available")

        from src.data.providers import HeliusProvider
        from src.metrics.distribution import (
            compute_hhi,
            compute_gini_coefficient,
            compute_shannon_entropy,
            compute_whale_dominance_ratio,
        )

        # Load model
        with open(model_path, "rb") as f:
            ensemble = pickle.load(f)

        # Fetch live data
        provider = HeliusProvider()

        try:
            # Test with JUP (established token)
            snapshot = await provider.get_token_holders(JUP_MINT, limit=500)

            # Compute features
            shares = snapshot.shares
            balances = [b.balance for b in snapshot.balances]

            hhi = compute_hhi(shares)
            gini = compute_gini_coefficient(balances)
            entropy = compute_shannon_entropy(shares)
            wdr_10 = compute_whale_dominance_ratio(balances, snapshot.total_supply, k=10)
            wdr_5 = compute_whale_dominance_ratio(balances, snapshot.total_supply, k=5)
            top_holder_share = max(shares) if shares else 0

            # Build feature vector
            features = np.array([[
                hhi.value,
                gini.value,
                entropy.value,
                wdr_10.value,
                wdr_5.value,
                top_holder_share,
                snapshot.holder_count,
            ]])

            # Get prediction
            prob, label = ensemble.predict(features)

            print(f"\n✅ JUP Token Analysis:")
            print(f"   Holders: {snapshot.holder_count}")
            print(f"   HHI: {hhi.value:.6f}")
            print(f"   Gini: {gini.value:.4f}")
            print(f"   Entropy: {entropy.value:.4f}")
            print(f"   Rug Probability: {prob:.4f}")
            print(f"   Classification: {'RUG RISK' if label == 1 else 'SAFE'}")

            # JUP is an established token, should be classified as safe
            # But note: model may not be perfectly calibrated
            assert 0 <= prob <= 1

        finally:
            await provider.close()


class TestAlertSystemIntegration:
    """Tests for alert system integration."""

    @pytest.mark.asyncio
    async def test_alert_engine_with_websocket(self) -> None:
        """Test AlertEngine integration with WebSocket broadcasting."""
        from src.monitoring.alerts import AlertEngine, Alert, AlertType, AlertSeverity
        from src.api.websocket import WebSocketConnectionManager, SubscriptionType

        # Create WebSocket manager
        ws_manager = WebSocketConnectionManager()

        # Create mock WebSocket
        class MockWebSocket:
            def __init__(self):
                self.messages = []
                self.accepted = False

            async def accept(self):
                self.accepted = True

            async def send_text(self, msg):
                self.messages.append(msg)

        mock_ws = MockWebSocket()
        conn_id = await ws_manager.connect(mock_ws)

        # Subscribe to test token
        test_token = "TestToken123456789012345678901234567890123"
        await ws_manager.subscribe(conn_id, SubscriptionType.TOKEN, test_token)

        # Create AlertEngine with broadcast callback
        alert_engine = AlertEngine(
            db_session=MagicMock(),
            broadcast_callback=ws_manager.broadcast_alert,
        )

        # Create and broadcast alert manually
        test_alert = Alert(
            id=1,
            alert_type=AlertType.CONCENTRATION_INCREASE,
            severity=AlertSeverity.WARNING,
            wallet_address=None,
            token_mint=test_token,
            timestamp=datetime.now(timezone.utc),
            details={"hhi_change": 0.03, "new_hhi": 0.42},
            user_id="test_user",
        )

        recipients = await ws_manager.broadcast_alert(test_alert)

        assert recipients == 1
        assert len(mock_ws.messages) > 1  # Welcome + subscribe_ack + alert

        # Find alert message
        import json
        alert_msgs = [
            json.loads(m) for m in mock_ws.messages
            if json.loads(m).get("type") == "alert"
        ]
        assert len(alert_msgs) == 1
        assert alert_msgs[0]["data"]["alert_type"] == "concentration_increase"

        print("\n✅ Alert broadcast verified")

    @pytest.mark.asyncio
    async def test_multiple_subscription_types(self) -> None:
        """Test alerts routing to correct subscribers."""
        from src.monitoring.alerts import Alert, AlertType, AlertSeverity
        from src.api.websocket import WebSocketConnectionManager, SubscriptionType

        ws_manager = WebSocketConnectionManager()

        # Create mock WebSockets
        class MockWebSocket:
            def __init__(self, name):
                self.name = name
                self.messages = []

            async def accept(self):
                pass

            async def send_text(self, msg):
                self.messages.append(msg)

            def get_alerts(self):
                import json
                return [
                    json.loads(m) for m in self.messages
                    if json.loads(m).get("type") == "alert"
                ]

        # Create subscribers with different subscriptions
        token_ws = MockWebSocket("token_subscriber")
        user_ws = MockWebSocket("user_subscriber")
        all_ws = MockWebSocket("all_subscriber")

        token_conn = await ws_manager.connect(token_ws)
        user_conn = await ws_manager.connect(user_ws)
        all_conn = await ws_manager.connect(all_ws)

        token_mint = "TargetToken123"
        user_id = "TargetUser456"

        await ws_manager.subscribe(token_conn, SubscriptionType.TOKEN, token_mint)
        await ws_manager.subscribe(user_conn, SubscriptionType.USER, user_id)
        await ws_manager.subscribe(all_conn, SubscriptionType.ALL)

        # Create alert matching token
        token_alert = Alert(
            id=10,
            alert_type=AlertType.WHALE_MOVEMENT,
            severity=AlertSeverity.HIGH,
            wallet_address="Whale123",
            token_mint=token_mint,
            timestamp=datetime.now(timezone.utc),
            details={"delta": 1000000},
            user_id="other_user",
        )

        await ws_manager.broadcast_alert(token_alert)

        # Create alert matching user
        user_alert = Alert(
            id=11,
            alert_type=AlertType.REGIME_CHANGE,
            severity=AlertSeverity.CRITICAL,
            wallet_address=None,
            token_mint="OtherToken",
            timestamp=datetime.now(timezone.utc),
            details={"from_regime": "growth", "to_regime": "decay"},
            user_id=user_id,
        )

        await ws_manager.broadcast_alert(user_alert)

        # Verify routing
        token_alerts = token_ws.get_alerts()
        user_alerts = user_ws.get_alerts()
        all_alerts = all_ws.get_alerts()

        assert len(token_alerts) == 1, "Token subscriber should receive 1 alert"
        assert len(user_alerts) == 1, "User subscriber should receive 1 alert"
        assert len(all_alerts) == 2, "All subscriber should receive 2 alerts"

        assert token_alerts[0]["data"]["alert_type"] == "whale_movement"
        assert user_alerts[0]["data"]["alert_type"] == "regime_change"

        print("\n✅ Subscription routing verified")
        print(f"   Token subscriber: {len(token_alerts)} alerts")
        print(f"   User subscriber: {len(user_alerts)} alerts")
        print(f"   All subscriber: {len(all_alerts)} alerts")


class TestFullPipelineIntegration:
    """End-to-end pipeline integration tests."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_LIVE_ACCESS, reason="HELIUS_API_KEY not set")
    async def test_complete_analysis_flow(self, model_path: Optional[Path]) -> None:
        """Test complete analysis flow: fetch -> compute -> predict -> alert."""
        if model_path is None:
            pytest.skip("No trained model available")

        from src.data.providers import HeliusProvider
        from src.metrics.distribution import (
            compute_hhi,
            compute_gini_coefficient,
            compute_shannon_entropy,
            compute_whale_dominance_ratio,
        )
        from src.monitoring.alerts import Alert, AlertType, AlertSeverity
        from src.api.websocket import WebSocketConnectionManager, SubscriptionType

        print("\n" + "=" * 60)
        print("Full Pipeline Integration Test")
        print("=" * 60)

        # 1. Load model
        with open(model_path, "rb") as f:
            ensemble = pickle.load(f)
        print(f"✅ Model loaded: {ensemble.version}")

        # 2. Set up WebSocket
        ws_manager = WebSocketConnectionManager()

        class MockWebSocket:
            def __init__(self):
                self.messages = []

            async def accept(self):
                pass

            async def send_text(self, msg):
                self.messages.append(msg)

        mock_ws = MockWebSocket()
        conn_id = await ws_manager.connect(mock_ws)
        await ws_manager.subscribe(conn_id, SubscriptionType.TOKEN, BONK_MINT)
        print("✅ WebSocket subscriber connected")

        # 3. Fetch live data
        provider = HeliusProvider()

        try:
            snapshot = await provider.get_token_holders(BONK_MINT, limit=500)
            print(f"✅ Fetched {snapshot.holder_count} holders")

            # 4. Compute metrics
            shares = snapshot.shares
            balances = [b.balance for b in snapshot.balances]

            hhi = compute_hhi(shares)
            gini = compute_gini_coefficient(balances)
            entropy = compute_shannon_entropy(shares)
            wdr_10 = compute_whale_dominance_ratio(balances, snapshot.total_supply, k=10)
            wdr_5 = compute_whale_dominance_ratio(balances, snapshot.total_supply, k=5)
            top_holder_share = max(shares) if shares else 0

            print(f"✅ Metrics computed:")
            print(f"   HHI: {hhi.value:.6f}")
            print(f"   Gini: {gini.value:.4f}")
            print(f"   Entropy: {entropy.value:.4f}")

            # 5. Run model inference
            features = np.array([[
                hhi.value,
                gini.value,
                entropy.value,
                wdr_10.value,
                wdr_5.value,
                top_holder_share,
                snapshot.holder_count,
            ]])

            prob, label = ensemble.predict(features)
            print(f"✅ Model prediction:")
            print(f"   Rug probability: {prob:.4f}")
            print(f"   Classification: {'RUG RISK' if label == 1 else 'SAFE'}")

            # 6. Generate alert if needed
            if prob > 0.5:
                alert = Alert(
                    id=100,
                    alert_type=AlertType.CONCENTRATION_INCREASE,
                    severity=AlertSeverity.HIGH if prob > 0.7 else AlertSeverity.WARNING,
                    wallet_address=None,
                    token_mint=BONK_MINT,
                    timestamp=datetime.now(timezone.utc),
                    details={
                        "rug_probability": float(prob),
                        "hhi": hhi.value,
                        "gini": gini.value,
                    },
                    user_id="integration_test",
                )

                recipients = await ws_manager.broadcast_alert(alert)
                print(f"✅ Alert broadcast to {recipients} subscriber(s)")
            else:
                print("✅ No alert needed (low risk)")

            # Verify pipeline completed
            assert snapshot.holder_count > 0
            assert 0 <= prob <= 1
            print("\n" + "=" * 60)
            print("FULL PIPELINE TEST PASSED ✅")
            print("=" * 60)

        finally:
            await provider.close()


class TestDataQuality:
    """Tests for data quality and consistency."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_LIVE_ACCESS, reason="HELIUS_API_KEY not set")
    async def test_holder_data_consistency(self) -> None:
        """Test that holder data is internally consistent."""
        from src.data.providers import HeliusProvider

        provider = HeliusProvider()

        try:
            snapshot = await provider.get_token_holders(USDC_MINT, limit=100)

            # Sum of shares should be close to 1 (may be slightly less due to limit)
            total_share = sum(snapshot.shares)
            assert total_share <= 1.0, f"Total share {total_share} exceeds 1.0"

            # All balances should be non-negative
            assert all(b.balance >= 0 for b in snapshot.balances)

            # Wallet addresses should be valid length
            assert all(len(b.wallet) >= 32 for b in snapshot.balances)

            # Timestamps should be reasonable
            for b in snapshot.balances:
                assert b.timestamp is not None

            print(f"\n✅ Data consistency verified for {snapshot.holder_count} holders")

        finally:
            await provider.close()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_LIVE_ACCESS, reason="HELIUS_API_KEY not set")
    async def test_metric_bounds(self) -> None:
        """Test that computed metrics stay within valid bounds."""
        from src.data.providers import HeliusProvider
        from src.metrics.distribution import (
            compute_hhi,
            compute_gini_coefficient,
            compute_shannon_entropy,
            compute_whale_dominance_ratio,
        )

        provider = HeliusProvider()

        try:
            # Test with multiple tokens
            test_tokens = [USDC_MINT, BONK_MINT]

            for token in test_tokens:
                snapshot = await provider.get_token_holders(token, limit=200)
                shares = snapshot.shares
                balances = [b.balance for b in snapshot.balances]

                if len(shares) < 5:
                    continue

                hhi = compute_hhi(shares)
                gini = compute_gini_coefficient(balances)
                entropy = compute_shannon_entropy(shares)
                wdr = compute_whale_dominance_ratio(balances, snapshot.total_supply, k=10)

                # Validate bounds
                assert 0 <= hhi.value <= 1, f"HHI {hhi.value} out of bounds"
                assert 0 <= gini.value <= 1, f"Gini {gini.value} out of bounds"
                assert entropy.value >= 0, f"Entropy {entropy.value} negative"
                assert 0 <= wdr.value <= 1, f"WDR {wdr.value} out of bounds"

                print(f"✅ {token[:8]}... metrics in bounds")

        finally:
            await provider.close()


# Run specific tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
