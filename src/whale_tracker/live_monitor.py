"""Live monitoring with Streamlit st.fragment for real-time updates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

logger = structlog.get_logger()

# Note: Streamlit imports are done inside methods to avoid import errors
# when this module is used outside of Streamlit context


@dataclass
class MonitoringSession:
    """Tracks a live monitoring session."""

    session_id: str
    token_mint: str
    wallet_count: int
    refresh_interval: int | None  # None = manual only
    started_at: datetime
    last_refresh: datetime | None = None
    refresh_count: int = 0
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "token_mint": self.token_mint,
            "wallet_count": self.wallet_count,
            "refresh_interval": self.refresh_interval,
            "started_at": self.started_at.isoformat(),
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "refresh_count": self.refresh_count,
            "is_active": self.is_active,
        }


class LiveMonitor:
    """Manages live monitoring state and Streamlit fragments."""

    # Refresh interval options
    REFRESH_OPTIONS: dict[str, int | None] = {
        "30 seconds": 30,
        "1 minute": 60,
        "5 minutes": 300,
        "Manual only": None,
    }

    @staticmethod
    def init_session_state() -> None:
        """
        Initialize Streamlit session state for live monitoring.

        Call this at the start of your Streamlit app.
        """
        import streamlit as st

        if "streaming" not in st.session_state:
            st.session_state.streaming = False
        if "refresh_seconds" not in st.session_state:
            st.session_state.refresh_seconds = 60
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = None
        if "refresh_count" not in st.session_state:
            st.session_state.refresh_count = 0
        if "tracked_wallets" not in st.session_state:
            st.session_state.tracked_wallets = []
        if "whale_profiles" not in st.session_state:
            st.session_state.whale_profiles = []

    @staticmethod
    def get_refresh_interval() -> int | None:
        """
        Get the current refresh interval based on session state.

        Returns:
            Refresh interval in seconds, or None if streaming is disabled
        """
        import streamlit as st

        if st.session_state.get("streaming", False):
            return st.session_state.get("refresh_seconds", 60)
        return None

    @staticmethod
    def render_controls() -> int | None:
        """
        Render streaming control UI in sidebar.

        Returns:
            Current refresh interval (or None if disabled)
        """
        import streamlit as st

        st.sidebar.subheader("Live Monitoring")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            is_streaming = st.session_state.get("streaming", False)
            button_text = "Stop" if is_streaming else "Start"
            button_type = "secondary" if is_streaming else "primary"

            if st.button(button_text, type=button_type, use_container_width=True):
                st.session_state.streaming = not is_streaming
                if st.session_state.streaming:
                    st.session_state.refresh_count = 0
                    logger.info("streaming_started")
                else:
                    logger.info("streaming_stopped")
                st.rerun()

        with col2:
            # Get current selection index
            options = list(LiveMonitor.REFRESH_OPTIONS.keys())
            current_seconds = st.session_state.get("refresh_seconds", 60)

            # Find matching option
            current_index = 1  # Default: 1 minute
            for i, (name, seconds) in enumerate(LiveMonitor.REFRESH_OPTIONS.items()):
                if seconds == current_seconds:
                    current_index = i
                    break

            selected = st.selectbox(
                "Refresh",
                options=options,
                index=current_index,
                disabled=st.session_state.get("streaming", False),
                label_visibility="collapsed",
            )
            st.session_state.refresh_seconds = LiveMonitor.REFRESH_OPTIONS[selected]

        # Status display
        if st.session_state.get("streaming", False):
            interval = st.session_state.get("refresh_seconds", 60)
            st.sidebar.success(f"Streaming every {interval}s")

            refresh_count = st.session_state.get("refresh_count", 0)
            last_refresh = st.session_state.get("last_refresh")

            if last_refresh:
                st.sidebar.caption(
                    f"Refreshes: {refresh_count} | "
                    f"Last: {last_refresh.strftime('%H:%M:%S')}"
                )
            else:
                st.sidebar.caption(f"Refreshes: {refresh_count}")
        else:
            st.sidebar.info("Streaming paused")

        return LiveMonitor.get_refresh_interval()

    @staticmethod
    def record_refresh() -> None:
        """Record that a refresh occurred."""
        import streamlit as st

        st.session_state.last_refresh = datetime.now(timezone.utc)
        st.session_state.refresh_count = st.session_state.get("refresh_count", 0) + 1

    @staticmethod
    def create_live_fragment(
        data_fetcher: Callable[[], Any],
        renderer: Callable[[Any], None],
    ) -> Callable:
        """
        Create a live-updating fragment function.

        This is a helper to create the pattern:
        ```
        @st.fragment(run_every=refresh_interval)
        def live_dashboard():
            data = fetch_data()
            render_data(data)
        ```

        Args:
            data_fetcher: Function that fetches fresh data
            renderer: Function that renders the data

        Returns:
            Fragment function to be decorated with @st.fragment
        """
        import streamlit as st

        def fragment_fn():
            # Record refresh
            LiveMonitor.record_refresh()

            # Fetch and render
            try:
                data = data_fetcher()
                renderer(data)
            except Exception as e:
                logger.error("fragment_error", error=str(e))
                st.error(f"Error updating data: {e}")

        return fragment_fn


def render_streaming_indicator() -> None:
    """Render a visual indicator when streaming is active."""
    import streamlit as st

    if st.session_state.get("streaming", False):
        interval = st.session_state.get("refresh_seconds", 60)
        refresh_count = st.session_state.get("refresh_count", 0)

        st.markdown(f"""
        <div style="
            position: fixed;
            top: 60px;
            right: 20px;
            background: rgba(76, 175, 80, 0.9);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            z-index: 999;
            display: flex;
            align-items: center;
            gap: 8px;
        ">
            <span style="
                width: 8px;
                height: 8px;
                background: white;
                border-radius: 50%;
                animation: pulse 1s infinite;
            "></span>
            LIVE ({interval}s) | #{refresh_count}
        </div>
        <style>
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
        </style>
        """, unsafe_allow_html=True)
