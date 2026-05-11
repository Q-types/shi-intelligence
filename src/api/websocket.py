"""
WebSocket Alert Delivery for SHI.

Real-time alert streaming via WebSocket connections.
Supports per-token and per-user subscriptions with heartbeat.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from ..monitoring.alerts import Alert, AlertType, AlertSeverity
from ..core.types import TokenMint

logger = structlog.get_logger()


class SubscriptionType(Enum):
    """Types of WebSocket subscriptions."""
    TOKEN = "token"  # Subscribe to specific token alerts
    USER = "user"  # Subscribe to all user alerts
    ALL = "all"  # Subscribe to all alerts (admin only)


@dataclass
class WebSocketMessage:
    """WebSocket message format."""
    type: str  # "alert", "heartbeat", "subscribe_ack", "error"
    timestamp: str
    data: Dict[str, Any]

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type,
            "timestamp": self.timestamp,
            "data": self.data,
        })


@dataclass
class ClientSubscription:
    """Active subscription for a client."""
    subscription_type: SubscriptionType
    filter_value: Optional[str]  # Token mint or user ID
    subscribed_at: datetime


class WebSocketConnectionManager:
    """
    Manages WebSocket connections for real-time alert delivery.

    Features:
    - Per-token subscriptions
    - Per-user subscriptions
    - Heartbeat to keep connections alive
    - Automatic cleanup on disconnect
    """

    HEARTBEAT_INTERVAL = 30  # seconds

    def __init__(self):
        """Initialize WebSocket connection manager."""
        # Active connections: connection_id -> WebSocket
        self._connections: Dict[str, WebSocket] = {}

        # Subscriptions by connection
        self._subscriptions: Dict[str, List[ClientSubscription]] = {}

        # Reverse indices for efficient broadcasting
        self._token_subscribers: Dict[TokenMint, Set[str]] = {}  # token -> connection_ids
        self._user_subscribers: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self._all_subscribers: Set[str] = set()  # admin connections

        # Heartbeat tasks
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}

        # Connection counter
        self._connection_counter = 0

    def _generate_connection_id(self) -> str:
        """Generate unique connection ID."""
        self._connection_counter += 1
        return f"ws-{self._connection_counter}-{datetime.now(timezone.utc).timestamp()}"

    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection

        Returns:
            Connection ID for tracking
        """
        await websocket.accept()

        connection_id = self._generate_connection_id()
        self._connections[connection_id] = websocket
        self._subscriptions[connection_id] = []

        # Start heartbeat
        self._heartbeat_tasks[connection_id] = asyncio.create_task(
            self._heartbeat_loop(connection_id)
        )

        logger.info(
            "websocket_connected",
            connection_id=connection_id,
            total_connections=len(self._connections),
        )

        # Send welcome message
        await self._send_message(
            connection_id,
            WebSocketMessage(
                type="connected",
                timestamp=datetime.now(timezone.utc).isoformat(),
                data={
                    "connection_id": connection_id,
                    "message": "Connected to SHI Alert Stream",
                    "heartbeat_interval": self.HEARTBEAT_INTERVAL,
                },
            ),
        )

        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            connection_id: Connection to disconnect
        """
        # Cancel heartbeat
        if connection_id in self._heartbeat_tasks:
            self._heartbeat_tasks[connection_id].cancel()
            try:
                await self._heartbeat_tasks[connection_id]
            except asyncio.CancelledError:
                pass
            del self._heartbeat_tasks[connection_id]

        # Remove from reverse indices
        if connection_id in self._all_subscribers:
            self._all_subscribers.discard(connection_id)

        for token_mint, subscribers in list(self._token_subscribers.items()):
            subscribers.discard(connection_id)
            if not subscribers:
                del self._token_subscribers[token_mint]

        for user_id, subscribers in list(self._user_subscribers.items()):
            subscribers.discard(connection_id)
            if not subscribers:
                del self._user_subscribers[user_id]

        # Remove subscriptions and connection
        self._subscriptions.pop(connection_id, None)
        self._connections.pop(connection_id, None)

        logger.info(
            "websocket_disconnected",
            connection_id=connection_id,
            total_connections=len(self._connections),
        )

    async def subscribe(
        self,
        connection_id: str,
        subscription_type: SubscriptionType,
        filter_value: Optional[str] = None,
    ) -> bool:
        """
        Add a subscription for a connection.

        Args:
            connection_id: Connection ID
            subscription_type: Type of subscription
            filter_value: Token mint or user ID

        Returns:
            True if subscribed successfully
        """
        if connection_id not in self._connections:
            return False

        subscription = ClientSubscription(
            subscription_type=subscription_type,
            filter_value=filter_value,
            subscribed_at=datetime.now(timezone.utc),
        )

        self._subscriptions[connection_id].append(subscription)

        # Update reverse indices
        if subscription_type == SubscriptionType.TOKEN and filter_value:
            if filter_value not in self._token_subscribers:
                self._token_subscribers[filter_value] = set()
            self._token_subscribers[filter_value].add(connection_id)

        elif subscription_type == SubscriptionType.USER and filter_value:
            if filter_value not in self._user_subscribers:
                self._user_subscribers[filter_value] = set()
            self._user_subscribers[filter_value].add(connection_id)

        elif subscription_type == SubscriptionType.ALL:
            self._all_subscribers.add(connection_id)

        logger.info(
            "websocket_subscribed",
            connection_id=connection_id,
            subscription_type=subscription_type.value,
            filter_value=filter_value,
        )

        # Send acknowledgment
        await self._send_message(
            connection_id,
            WebSocketMessage(
                type="subscribe_ack",
                timestamp=datetime.now(timezone.utc).isoformat(),
                data={
                    "subscription_type": subscription_type.value,
                    "filter_value": filter_value,
                    "status": "subscribed",
                },
            ),
        )

        return True

    async def unsubscribe(
        self,
        connection_id: str,
        subscription_type: SubscriptionType,
        filter_value: Optional[str] = None,
    ) -> bool:
        """
        Remove a subscription for a connection.

        Args:
            connection_id: Connection ID
            subscription_type: Type of subscription
            filter_value: Token mint or user ID

        Returns:
            True if unsubscribed successfully
        """
        if connection_id not in self._connections:
            return False

        # Remove from subscription list
        self._subscriptions[connection_id] = [
            s for s in self._subscriptions[connection_id]
            if not (s.subscription_type == subscription_type and s.filter_value == filter_value)
        ]

        # Update reverse indices
        if subscription_type == SubscriptionType.TOKEN and filter_value:
            if filter_value in self._token_subscribers:
                self._token_subscribers[filter_value].discard(connection_id)

        elif subscription_type == SubscriptionType.USER and filter_value:
            if filter_value in self._user_subscribers:
                self._user_subscribers[filter_value].discard(connection_id)

        elif subscription_type == SubscriptionType.ALL:
            self._all_subscribers.discard(connection_id)

        logger.info(
            "websocket_unsubscribed",
            connection_id=connection_id,
            subscription_type=subscription_type.value,
            filter_value=filter_value,
        )

        return True

    async def broadcast_alert(self, alert: Alert) -> int:
        """
        Broadcast an alert to all relevant subscribers.

        Args:
            alert: Alert to broadcast

        Returns:
            Number of connections that received the alert
        """
        recipients: Set[str] = set()

        # Add all subscribers
        recipients.update(self._all_subscribers)

        # Add token subscribers
        if alert.token_mint in self._token_subscribers:
            recipients.update(self._token_subscribers[alert.token_mint])

        # Add user subscribers
        if alert.user_id and alert.user_id in self._user_subscribers:
            recipients.update(self._user_subscribers[alert.user_id])

        if not recipients:
            logger.debug("broadcast_no_recipients", alert_type=alert.alert_type.value)
            return 0

        # Build alert message
        message = WebSocketMessage(
            type="alert",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data={
                "alert_id": alert.id,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "token_mint": alert.token_mint,
                "wallet_address": alert.wallet_address,
                "timestamp": alert.timestamp.isoformat(),
                "details": alert.details,
                "message": alert.get_message(),
            },
        )

        # Send to all recipients
        sent_count = 0
        failed_connections = []

        for connection_id in recipients:
            try:
                await self._send_message(connection_id, message)
                sent_count += 1
            except Exception as e:
                logger.warning(
                    "broadcast_send_failed",
                    connection_id=connection_id,
                    error=str(e),
                )
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)

        logger.info(
            "alert_broadcast_complete",
            alert_type=alert.alert_type.value,
            recipients=sent_count,
            failed=len(failed_connections),
        )

        return sent_count

    async def _send_message(
        self,
        connection_id: str,
        message: WebSocketMessage,
    ) -> None:
        """
        Send a message to a specific connection.

        Args:
            connection_id: Target connection
            message: Message to send
        """
        websocket = self._connections.get(connection_id)
        if websocket:
            await websocket.send_text(message.to_json())

    async def _heartbeat_loop(self, connection_id: str) -> None:
        """
        Send periodic heartbeats to keep connection alive.

        Args:
            connection_id: Connection to heartbeat
        """
        while connection_id in self._connections:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)

                if connection_id not in self._connections:
                    break

                await self._send_message(
                    connection_id,
                    WebSocketMessage(
                        type="heartbeat",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        data={"status": "alive"},
                    ),
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(
                    "heartbeat_failed",
                    connection_id=connection_id,
                    error=str(e),
                )
                break

    async def handle_message(
        self,
        connection_id: str,
        message: str,
    ) -> None:
        """
        Handle incoming message from a client.

        Args:
            connection_id: Connection that sent the message
            message: Raw message string
        """
        try:
            data = json.loads(message)
            action = data.get("action")

            if action == "subscribe":
                sub_type = SubscriptionType(data.get("type", "token"))
                filter_value = data.get("filter")
                await self.subscribe(connection_id, sub_type, filter_value)

            elif action == "unsubscribe":
                sub_type = SubscriptionType(data.get("type", "token"))
                filter_value = data.get("filter")
                await self.unsubscribe(connection_id, sub_type, filter_value)

            elif action == "ping":
                await self._send_message(
                    connection_id,
                    WebSocketMessage(
                        type="pong",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        data={},
                    ),
                )

            else:
                await self._send_message(
                    connection_id,
                    WebSocketMessage(
                        type="error",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        data={"error": f"Unknown action: {action}"},
                    ),
                )

        except json.JSONDecodeError:
            await self._send_message(
                connection_id,
                WebSocketMessage(
                    type="error",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    data={"error": "Invalid JSON message"},
                ),
            )

        except Exception as e:
            logger.error(
                "message_handling_error",
                connection_id=connection_id,
                error=str(e),
            )
            await self._send_message(
                connection_id,
                WebSocketMessage(
                    type="error",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    data={"error": "Internal error processing message"},
                ),
            )

    def get_statistics(self) -> Dict:
        """
        Get WebSocket connection statistics.

        Returns:
            Dict with connection stats
        """
        return {
            "total_connections": len(self._connections),
            "token_subscriptions": sum(len(s) for s in self._token_subscribers.values()),
            "user_subscriptions": sum(len(s) for s in self._user_subscribers.values()),
            "all_subscriptions": len(self._all_subscribers),
            "unique_tokens_watched": len(self._token_subscribers),
            "unique_users_subscribed": len(self._user_subscribers),
        }


# Global connection manager instance
ws_manager = WebSocketConnectionManager()
