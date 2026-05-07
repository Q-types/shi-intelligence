"""
Demo: Temporal Foundation (Sprint 1)

Demonstrates the new temporal intelligence capabilities:
1. Metric trajectory tracking over time
2. Derivative calculations (dHHI/dt, dGini/dt)
3. Trend detection (centralizing vs decentralizing)
4. HMM-based regime detection
5. Capital flow forecasting
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from src.temporal.trajectories import TrajectoryTracker, MetricPoint
from src.temporal.regimes import HolderRegimeDetector, create_rule_based_regime
from src.temporal.forecasting import (
    CapitalFlowForecaster,
    FlowFeatures,
    extract_flow_features_from_snapshots,
)

console = Console()


def generate_synthetic_token_data(
    n_days: int = 90,
    regime_type: str = "distribution",
) -> list[dict]:
    """
    Generate synthetic token metric snapshots.

    Args:
        n_days: Number of days to generate
        regime_type: "distribution", "accumulation", "decay"

    Returns:
        List of metric snapshots
    """
    base_time = datetime(2026, 1, 1)
    snapshots = []

    for i in range(n_days):
        # Different patterns for different regimes
        if regime_type == "distribution":
            # Centralizing: HHI/Gini increasing
            hhi = 0.1 + (i * 0.003) + np.random.randn() * 0.005
            gini = 0.4 + (i * 0.004) + np.random.randn() * 0.01
            churn = 0.02 + np.random.randn() * 0.005
            whale_dom = 0.3 + (i * 0.002)
            holders = 1000 - (i * 2)

        elif regime_type == "accumulation":
            # Decentralizing: HHI/Gini decreasing
            hhi = 0.3 - (i * 0.002) + np.random.randn() * 0.005
            gini = 0.6 - (i * 0.003) + np.random.randn() * 0.01
            churn = 0.015 + np.random.randn() * 0.005
            whale_dom = 0.4 - (i * 0.001)
            holders = 800 + (i * 5)

        elif regime_type == "decay":
            # High churn, holders leaving
            hhi = 0.2 + (i * 0.001) + np.random.randn() * 0.01
            gini = 0.5 + np.random.randn() * 0.02
            churn = 0.05 + (i * 0.001) + np.random.randn() * 0.01
            whale_dom = 0.3 + np.random.randn() * 0.02
            holders = 1000 - (i * 10)

        else:
            # Stable
            hhi = 0.15 + np.random.randn() * 0.01
            gini = 0.5 + np.random.randn() * 0.02
            churn = 0.02 + np.random.randn() * 0.005
            whale_dom = 0.3 + np.random.randn() * 0.01
            holders = 1000 + np.random.randn() * 50

        snapshots.append(
            {
                "timestamp": base_time + timedelta(days=i),
                "hhi": max(0, min(1, hhi)),
                "gini": max(0, min(1, gini)),
                "churn_rate": max(0, churn),
                "whale_dominance": max(0, min(1, whale_dom)),
                "holder_count": int(max(0, holders)),
                "coordination_score": 0.3 + np.random.randn() * 0.1,
            }
        )

    return snapshots


def demo_trajectory_tracking():
    """Demonstrate trajectory tracking and derivatives."""
    console.print("\n[bold cyan]Demo 1: Metric Trajectory Tracking[/bold cyan]\n")

    # Generate data for distributing token
    snapshots = generate_synthetic_token_data(60, regime_type="distribution")

    # Track trajectories
    tracker = TrajectoryTracker()
    trajectories = tracker.compute_multi_metric_trajectory(snapshots, window_days=30)

    # Display results
    table = Table(title="Metric Trajectories (Last 30 Days)")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Velocity (per day)", style="yellow")
    table.add_column("Acceleration", style="magenta")
    table.add_column("Trend", style="blue")

    for metric_name, traj in trajectories.items():
        table.add_row(
            metric_name,
            f"{traj.mean:.4f}",
            f"{traj.velocity:.6f}",
            f"{traj.acceleration:.6f}",
            traj.trend.value,
        )

    console.print(table)

    # Extract regime signals
    if "hhi" in trajectories and "gini" in trajectories:
        signals = tracker.detect_regime_signals(
            trajectories["hhi"],
            trajectories["gini"],
            trajectories.get("churn_rate"),
        )

        console.print("\n[bold]Regime Signals:[/bold]")
        for signal, value in signals.items():
            console.print(f"  {signal}: {value}")

    return trajectories


def demo_regime_detection(trajectories):
    """Demonstrate regime detection."""
    console.print("\n[bold cyan]Demo 2: Holder Regime Detection[/bold cyan]\n")

    hhi_traj = trajectories["hhi"]
    gini_traj = trajectories["gini"]

    # Rule-based regime
    regime = create_rule_based_regime(
        dhhi_dt=hhi_traj.velocity,
        dgini_dt=gini_traj.velocity,
        dchurn_dt=trajectories.get("churn_rate").velocity if "churn_rate" in trajectories else 0.0,
        coordination_score=0.3,
        hhi_trend=hhi_traj.trend,
    )

    console.print(f"[bold]Detected Regime (Rule-Based):[/bold] {regime.value}")
    console.print(f"  dHHI/dt: {hhi_traj.velocity:.6f}")
    console.print(f"  dGini/dt: {gini_traj.velocity:.6f}")
    console.print(f"  Trend: {hhi_traj.trend.value}")

    # HMM-based (requires training data)
    console.print("\n[bold]HMM-Based Detection:[/bold]")
    console.print("  Training HMM detector on synthetic data...")

    detector = HolderRegimeDetector(n_iter=50)

    # Generate training data for multiple tokens
    training_sequences = []
    for regime_type in ["distribution", "accumulation", "decay"]:
        train_snapshots = generate_synthetic_token_data(40, regime_type=regime_type)
        train_tracker = TrajectoryTracker()
        train_trajs = train_tracker.compute_multi_metric_trajectory(train_snapshots)

        # Extract features
        features = []
        for i in range(len(train_snapshots) - 10):
            window = train_snapshots[i : i + 10]
            window_trajs = train_tracker.compute_multi_metric_trajectory(window)

            feat = detector.extract_features_from_trajectories(
                window_trajs["hhi"],
                window_trajs["gini"],
                window_trajs.get("churn_rate"),
                coordination_score=0.3,
            )
            features.append(feat)

        training_sequences.append(np.vstack(features))

    detector.fit(training_sequences)

    # Predict on current data
    test_features = detector.extract_features_from_trajectories(
        hhi_traj, gini_traj, trajectories.get("churn_rate"), coordination_score=0.3
    )

    regime_state = detector.predict_regime(test_features.reshape(1, -1))

    console.print(f"  Detected Regime: {regime_state.regime.value}")
    console.print(f"  Confidence: {regime_state.confidence:.2%}")
    console.print(f"  Transition Probability: {regime_state.transition_probability:.2%}")


def demo_capital_flow_forecasting():
    """Demonstrate capital flow forecasting."""
    console.print("\n[bold cyan]Demo 3: Capital Flow Forecasting[/bold cyan]\n")

    # Generate snapshots
    snapshots = generate_synthetic_token_data(90, regime_type="distribution")

    # Extract flow features
    features = extract_flow_features_from_snapshots(snapshots[-48:], lookback_hours=24)

    if features:
        console.print("[bold]Current Flow Features:[/bold]")
        console.print(f"  dHHI/dt: {features.dhhi_dt:.6f}")
        console.print(f"  dGini/dt: {features.dgini_dt:.6f}")
        console.print(f"  New holders rate: {features.new_holders_rate:.2f}/hour")
        console.print(f"  Exiting holders rate: {features.exiting_holders_rate:.2f}/hour")
        console.print(f"  Whale accumulation rate: {features.whale_accumulation_rate:.6f}")

    # Train forecaster
    console.print("\n[bold]Training Capital Flow Forecaster...[/bold]")

    forecaster = CapitalFlowForecaster()

    # Generate training data
    training_features = []
    training_flows = []

    for i in range(40, 80):
        window = snapshots[i - 40 : i]
        feat = extract_flow_features_from_snapshots(window, lookback_hours=24)
        if feat:
            training_features.append(feat)
            # Synthetic flow (correlated with holder change)
            flow = feat.new_holders_rate - feat.exiting_holders_rate
            training_flows.append(flow)

    forecaster.fit(training_features, training_flows)

    # Forecast
    if features:
        forecast = forecaster.forecast(features, horizon_hours=24)

        console.print(f"\n[bold]24-Hour Capital Flow Forecast:[/bold]")
        console.print(f"  Predicted Net Flow: {forecast.predicted_net_flow:.2f}")
        console.print(
            f"  95% CI: [{forecast.confidence_interval_lower:.2f}, "
            f"{forecast.confidence_interval_upper:.2f}]"
        )
        console.print(
            f"  Liquidity Stress Probability: {forecast.liquidity_stress_probability:.2%}"
        )

        console.print(f"\n[bold]Top Contributing Features:[/bold]")
        for feature, importance in forecast.top_features.items():
            console.print(f"  {feature}: {importance:.4f}")


def plot_trajectories(snapshots):
    """Plot metric trajectories."""
    timestamps = [s["timestamp"] for s in snapshots]
    hhi_values = [s["hhi"] for s in snapshots]
    gini_values = [s["gini"] for s in snapshots]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # HHI trajectory
    ax1.plot(timestamps, hhi_values, "b-", linewidth=2, label="HHI")
    ax1.set_ylabel("HHI", fontsize=12)
    ax1.set_title("HHI Trajectory Over Time", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Gini trajectory
    ax2.plot(timestamps, gini_values, "g-", linewidth=2, label="Gini")
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Gini Coefficient", fontsize=12)
    ax2.set_title("Gini Trajectory Over Time", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("/tmp/shi_temporal_demo.png", dpi=150)
    console.print("\n[green]Plot saved to /tmp/shi_temporal_demo.png[/green]")


def main():
    """Run all demos."""
    console.print("[bold magenta]SHI Temporal Foundation Demo (Sprint 1)[/bold magenta]")
    console.print("=" * 60)

    # Demo 1: Trajectory tracking
    trajectories = demo_trajectory_tracking()

    # Demo 2: Regime detection
    demo_regime_detection(trajectories)

    # Demo 3: Capital flow forecasting
    demo_capital_flow_forecasting()

    # Generate plots
    console.print("\n[bold cyan]Generating Visualizations...[/bold cyan]")
    snapshots = generate_synthetic_token_data(90, regime_type="distribution")
    plot_trajectories(snapshots)

    console.print("\n[bold green]✓ All demos completed successfully![/bold green]")
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("  1. Run database migration: alembic upgrade head")
    console.print("  2. Integrate with existing SHI pipeline")
    console.print("  3. Add real-time snapshot collection")
    console.print("  4. Proceed to Sprint 2: Graph Intelligence")


if __name__ == "__main__":
    main()
