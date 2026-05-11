#!/usr/bin/env python3
"""
Test WebSocket Alert Delivery.

Verifies the WebSocket connection manager and alert broadcasting work correctly.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.websocket import (
    WebSocketConnectionManager,
    SubscriptionType,
    WebSocketMessage,
)
from src.monitoring.alerts import (
    Alert,
    AlertType,
    AlertSeverity,
    AlertEngine,
)


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.messages: list[str] = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, message: str):
        if self.closed:
            raise RuntimeError("WebSocket is closed")
        self.messages.append(message)

    async def receive_text(self) -> str:
        # Simulate receiving a message
        await asyncio.sleep(0.1)
        return json.dumps({"action": "ping"})

    def get_received_alerts(self) -> list[dict]:
        """Parse received messages and return alerts."""
        alerts = []
        for msg in self.messages:
            data = json.loads(msg)
            if data.get("type") == "alert":
                alerts.append(data)
        return alerts


async def test_connection_manager():
    """Test WebSocket connection management."""
    print("\n" + "=" * 60)
    print("Test: WebSocket Connection Manager")
    print("=" * 60)

    manager = WebSocketConnectionManager()

    # Create mock clients
    client1 = MockWebSocket("client1")
    client2 = MockWebSocket("client2")
    client3 = MockWebSocket("client3")

    # Connect clients
    conn_id1 = await manager.connect(client1)
    conn_id2 = await manager.connect(client2)
    conn_id3 = await manager.connect(client3)

    print(f"Connected 3 clients: {conn_id1}, {conn_id2}, {conn_id3}")

    stats = manager.get_statistics()
    assert stats["total_connections"] == 3, f"Expected 3 connections, got {stats['total_connections']}"
    print(f"Connection count: {stats['total_connections']} ✓")

    # Subscribe clients to different tokens
    token1 = "Token1Mint123456789"
    token2 = "Token2Mint987654321"

    await manager.subscribe(conn_id1, SubscriptionType.TOKEN, token1)
    await manager.subscribe(conn_id2, SubscriptionType.TOKEN, token1)  # Same token as client1
    await manager.subscribe(conn_id3, SubscriptionType.TOKEN, token2)  # Different token

    stats = manager.get_statistics()
    print(f"Token subscriptions: {stats['token_subscriptions']} ✓")
    assert stats["token_subscriptions"] == 3

    # Test token-specific broadcast
    test_alert = Alert(
        id=1,
        alert_type=AlertType.WHALE_MOVEMENT,
        severity=AlertSeverity.HIGH,
        wallet_address="WalletABC123",
        token_mint=token1,
        timestamp=datetime.now(timezone.utc),
        details={"delta": 1000000, "pct_of_supply": 0.05},
        user_id="user1",
    )

    recipients = await manager.broadcast_alert(test_alert)
    print(f"Broadcast to token1: {recipients} recipients ✓")
    assert recipients == 2, f"Expected 2 recipients, got {recipients}"

    # Verify client1 and client2 received the alert
    client1_alerts = client1.get_received_alerts()
    client2_alerts = client2.get_received_alerts()
    client3_alerts = client3.get_received_alerts()

    assert len(client1_alerts) == 1, f"Client1 should have 1 alert, got {len(client1_alerts)}"
    assert len(client2_alerts) == 1, f"Client2 should have 1 alert, got {len(client2_alerts)}"
    assert len(client3_alerts) == 0, f"Client3 should have 0 alerts, got {len(client3_alerts)}"
    print("Alert routing verified ✓")

    # Test disconnect
    await manager.disconnect(conn_id1)
    stats = manager.get_statistics()
    assert stats["total_connections"] == 2
    print(f"Disconnection handled: {stats['total_connections']} connections remaining ✓")

    print("\n✅ WebSocket Connection Manager tests passed!")


async def test_alert_engine_integration():
    """Test AlertEngine with WebSocket broadcasting."""
    print("\n" + "=" * 60)
    print("Test: AlertEngine WebSocket Integration")
    print("=" * 60)

    manager = WebSocketConnectionManager()

    # Create mock clients
    client = MockWebSocket("test_client")
    conn_id = await manager.connect(client)

    # Subscribe to a token
    token_mint = "TestTokenMint123"
    await manager.subscribe(conn_id, SubscriptionType.TOKEN, token_mint)
    print(f"Client subscribed to {token_mint[:16]}...")

    # Create AlertEngine with WebSocket broadcast callback
    # Note: In production, db_session would be a real session
    class MockSession:
        pass

    alert_engine = AlertEngine(
        db_session=MockSession(),  # type: ignore
        broadcast_callback=manager.broadcast_alert,
    )

    print("AlertEngine configured with broadcast callback ✓")

    # Verify broadcast callback is set
    assert alert_engine._broadcast_callback is not None
    print("Broadcast callback verified ✓")

    # Create a test alert directly (simulating what the engine would produce)
    test_alert = Alert(
        id=100,
        alert_type=AlertType.CONCENTRATION_INCREASE,
        severity=AlertSeverity.WARNING,
        wallet_address=None,
        token_mint=token_mint,
        timestamp=datetime.now(timezone.utc),
        details={"hhi_change": 0.03, "new_hhi": 0.45},
        user_id="test_user",
    )

    # Broadcast manually to verify
    recipients = await manager.broadcast_alert(test_alert)
    print(f"Manual broadcast: {recipients} recipient(s) ✓")

    # Check client received the alert
    alerts = client.get_received_alerts()
    assert len(alerts) == 1, f"Expected 1 alert, got {len(alerts)}"
    assert alerts[0]["data"]["alert_type"] == "concentration_increase"
    print("Alert type verified ✓")
    assert alerts[0]["data"]["severity"] == "warning"
    print("Alert severity verified ✓")

    # Check broadcast stats
    stats = alert_engine.get_broadcast_stats()
    print(f"Broadcast stats: {stats}")

    print("\n✅ AlertEngine WebSocket Integration tests passed!")


async def test_subscription_types():
    """Test different subscription types."""
    print("\n" + "=" * 60)
    print("Test: Subscription Types")
    print("=" * 60)

    manager = WebSocketConnectionManager()

    # Create clients
    token_client = MockWebSocket("token_client")
    user_client = MockWebSocket("user_client")
    all_client = MockWebSocket("all_client")

    token_conn = await manager.connect(token_client)
    user_conn = await manager.connect(user_client)
    all_conn = await manager.connect(all_client)

    # Different subscription types
    token_mint = "TargetToken123"
    user_id = "TargetUser456"

    await manager.subscribe(token_conn, SubscriptionType.TOKEN, token_mint)
    await manager.subscribe(user_conn, SubscriptionType.USER, user_id)
    await manager.subscribe(all_conn, SubscriptionType.ALL)

    print("Subscriptions configured:")
    print(f"  - Token client subscribed to {token_mint[:12]}...")
    print(f"  - User client subscribed to {user_id}")
    print(f"  - All client subscribed to ALL alerts")

    # Create alert matching token
    token_alert = Alert(
        id=200,
        alert_type=AlertType.ANOMALY_SPIKE,
        severity=AlertSeverity.CRITICAL,
        wallet_address=None,
        token_mint=token_mint,
        timestamp=datetime.now(timezone.utc),
        details={"anomaly_count": 10},
        user_id="different_user",  # Not our subscribed user
    )

    recipients = await manager.broadcast_alert(token_alert)
    print(f"\nToken alert broadcast: {recipients} recipients")

    # Token client and all client should receive
    assert len(token_client.get_received_alerts()) == 1, "Token client should receive"
    assert len(user_client.get_received_alerts()) == 0, "User client should NOT receive"
    assert len(all_client.get_received_alerts()) == 1, "All client should receive"
    print("Token subscription verified ✓")

    # Create alert matching user
    user_alert = Alert(
        id=201,
        alert_type=AlertType.REGIME_CHANGE,
        severity=AlertSeverity.HIGH,
        wallet_address=None,
        token_mint="DifferentToken",
        timestamp=datetime.now(timezone.utc),
        details={"from_regime": "growth", "to_regime": "decay"},
        user_id=user_id,  # Our subscribed user
    )

    recipients = await manager.broadcast_alert(user_alert)
    print(f"User alert broadcast: {recipients} recipients")

    # User client and all client should receive
    assert len(user_client.get_received_alerts()) == 1, "User client should receive"
    assert len(all_client.get_received_alerts()) == 2, "All client should have 2 total"
    print("User subscription verified ✓")

    # Verify stats
    stats = manager.get_statistics()
    print(f"\nFinal stats: {stats}")
    assert stats["unique_tokens_watched"] == 1
    assert stats["unique_users_subscribed"] == 1
    assert stats["all_subscriptions"] == 1

    print("\n✅ Subscription Types tests passed!")


async def test_unsubscribe():
    """Test unsubscription."""
    print("\n" + "=" * 60)
    print("Test: Unsubscription")
    print("=" * 60)

    manager = WebSocketConnectionManager()

    client = MockWebSocket("unsub_client")
    conn_id = await manager.connect(client)

    token_mint = "UnsubTestToken"
    await manager.subscribe(conn_id, SubscriptionType.TOKEN, token_mint)

    # Verify subscription
    stats = manager.get_statistics()
    assert stats["token_subscriptions"] == 1
    print(f"Subscribed: {stats['token_subscriptions']} subscription ✓")

    # Unsubscribe
    result = await manager.unsubscribe(conn_id, SubscriptionType.TOKEN, token_mint)
    assert result is True
    print("Unsubscribe returned True ✓")

    # Verify no longer subscribed
    stats = manager.get_statistics()
    assert stats["token_subscriptions"] == 0
    print(f"After unsubscribe: {stats['token_subscriptions']} subscriptions ✓")

    # Broadcast should not reach client
    test_alert = Alert(
        id=300,
        alert_type=AlertType.WHALE_MOVEMENT,
        severity=AlertSeverity.INFO,
        wallet_address="Wallet123",
        token_mint=token_mint,
        timestamp=datetime.now(timezone.utc),
        details={},
        user_id="user",
    )

    # Clear previous messages (welcome + subscribe_ack)
    client.messages.clear()

    recipients = await manager.broadcast_alert(test_alert)
    assert recipients == 0, f"Expected 0 recipients, got {recipients}"
    assert len(client.get_received_alerts()) == 0
    print("Unsubscribed client did not receive alert ✓")

    print("\n✅ Unsubscription tests passed!")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("SHI WebSocket Alert Delivery Tests")
    print("=" * 60)

    try:
        await test_connection_manager()
        await test_alert_engine_integration()
        await test_subscription_types()
        await test_unsubscribe()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
