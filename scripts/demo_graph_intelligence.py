"""
Demo Script for Sprint 2: Graph Intelligence.

Demonstrates:
1. Node2Vec graph embeddings
2. Wallet similarity detection and Sybil cluster discovery
3. Dynamic network metrics tracking
4. Anomaly detection with Isolation Forest
"""

from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from src.core.types import FundingEdge
from src.graph import (
    FundingGraph,
    GraphEmbedder,
    EmbeddingConfig,
    WalletSimilarityDetector,
    DynamicNetworkAnalyzer,
    WalletAnomalyDetector,
    AnomalyConfig,
)

console = Console()


def create_demo_graph() -> FundingGraph:
    """Create demo funding graph with coordinated and normal wallets."""
    graph = FundingGraph()

    console.print("\n[bold cyan]Creating Demo Funding Graph...[/bold cyan]")

    # Coordinated Cluster 1: MasterFunder -> Bot1, Bot2, Bot3, Bot4
    coordinated_edges_1 = [
        FundingEdge(
            "MasterFunder_A",
            f"Bot_{i}",
            1000000 + i * 10000,
            datetime.now() - timedelta(hours=24 - i),
            f"sig_bot_{i}",
        )
        for i in range(1, 5)
    ]

    # Coordinated Cluster 2: MasterFunder_B -> Sybil1, Sybil2, Sybil3
    coordinated_edges_2 = [
        FundingEdge(
            "MasterFunder_B",
            f"Sybil_{i}",
            800000,
            datetime.now() - timedelta(hours=12),
            f"sig_sybil_{i}",
        )
        for i in range(1, 4)
    ]

    # Normal wallets: diverse funding patterns
    normal_edges = [
        FundingEdge(
            f"Funder_{i}",
            f"Wallet_{i}",
            500000 + i * 50000,
            datetime.now() - timedelta(hours=i),
            f"sig_normal_{i}",
        )
        for i in range(1, 11)
    ]

    # Super whale: funds many wallets (potential anomaly)
    whale_edges = [
        FundingEdge(
            "SuperWhale",
            f"Recipient_{i}",
            2000000,
            datetime.now() - timedelta(hours=i),
            f"sig_whale_{i}",
        )
        for i in range(1, 8)
    ]

    # Add some cross-connections
    cross_edges = [
        FundingEdge("Bot_1", "Wallet_5", 300000, datetime.now(), "sig_cross_1"),
        FundingEdge("Sybil_1", "Sybil_2", 100000, datetime.now(), "sig_cross_2"),
    ]

    all_edges = coordinated_edges_1 + coordinated_edges_2 + normal_edges + whale_edges + cross_edges
    graph.add_edges_from_list(all_edges)

    table = Table(title="Graph Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Nodes (Wallets)", str(graph.num_vertices))
    table.add_row("Edges (Funding Relationships)", str(graph.num_edges))
    console.print(table)

    return graph


def demo_embeddings(graph: FundingGraph) -> GraphEmbedder:
    """Demonstrate Node2Vec embeddings."""
    console.print("\n[bold cyan]1. Node2Vec Graph Embeddings[/bold cyan]")

    config = EmbeddingConfig(
        dimensions=32, walk_length=20, num_walks=100, p=1.0, q=1.0, workers=2
    )

    embedder = GraphEmbedder(config=config)

    console.print("Fitting Node2Vec (this may take a moment)...")
    embeddings = embedder.fit_transform(graph, embedding_id="demo_v1")

    console.print(f"✓ Generated embeddings for {len(embeddings)} wallets")

    # Show example embeddings
    table = Table(title="Sample Embeddings (first 5 dimensions)")
    table.add_column("Wallet", style="cyan")
    table.add_column("Dimensions", style="magenta")

    for wallet in list(embeddings.keys())[:5]:
        vector = embeddings[wallet].vector
        dims_str = ", ".join([f"{v:.3f}" for v in vector[:5]])
        table.add_row(wallet, f"[{dims_str}, ...]")

    console.print(table)

    # Show similar wallets
    console.print("\n[yellow]Finding similar wallets to 'Bot_1':[/yellow]")
    similar = embedder.find_similar_wallets("Bot_1", k=5, min_similarity=0.5)

    table = Table(title="Wallets Similar to Bot_1")
    table.add_column("Wallet", style="cyan")
    table.add_column("Similarity", style="green")

    for wallet, sim in similar:
        table.add_row(wallet, f"{sim:.4f}")

    console.print(table)

    return embedder


def demo_similarity_detection(graph: FundingGraph, embedder: GraphEmbedder):
    """Demonstrate wallet similarity and Sybil detection."""
    console.print("\n[bold cyan]2. Wallet Similarity & Sybil Detection[/bold cyan]")

    detector = WalletSimilarityDetector(
        embedder=embedder, graph=graph, embedding_weight=0.7, structural_weight=0.3
    )

    # Find coordinated pairs
    console.print("\n[yellow]Finding coordinated wallet pairs...[/yellow]")
    test_wallets = ["Bot_1", "Bot_2", "Bot_3", "Bot_4", "Sybil_1", "Sybil_2", "Wallet_1"]
    pairs = detector.find_coordinated_pairs(test_wallets, min_similarity=0.6, top_k=10)

    table = Table(title="Coordinated Wallet Pairs")
    table.add_column("Wallet 1", style="cyan")
    table.add_column("Wallet 2", style="cyan")
    table.add_column("Combined Sim", style="green")
    table.add_column("Structural Sim", style="yellow")
    table.add_column("Coordinated", style="red")

    for score in pairs:
        table.add_row(
            score.wallet1,
            score.wallet2,
            f"{score.combined_similarity:.3f}",
            f"{score.structural_similarity:.3f}",
            "✓" if score.is_coordinated else "✗",
        )

    console.print(table)

    # Detect Sybil clusters
    console.print("\n[yellow]Detecting Sybil clusters...[/yellow]")
    all_wallets = [w for w in graph._wallet_set if not w.startswith("Funder")]
    clusters = detector.detect_sybil_clusters(
        all_wallets, similarity_threshold=0.6, min_cluster_size=2
    )

    for cluster in clusters:
        panel_content = f"""
Cluster ID: {cluster.cluster_id}
Wallets: {', '.join(list(cluster.wallets)[:10])}{'...' if len(cluster.wallets) > 10 else ''}
Size: {len(cluster.wallets)}
Mean Similarity: {cluster.mean_similarity:.3f}
Sybil Probability: {cluster.sybil_probability:.2%}
Shared Funders: {len(cluster.shared_funders)}
        """

        color = "red" if cluster.sybil_probability > 0.7 else "yellow"
        console.print(Panel(panel_content, title=f"[{color}]Cluster {cluster.cluster_id}[/{color}]"))

        # Analyze cluster behavior
        analysis = detector.analyze_cluster_behavior(cluster)
        console.print(f"  Dominant Funder: {analysis['dominant_funder']}")
        console.print(f"  Funder Coverage: {analysis['funder_coverage']:.2%}")


def demo_dynamic_metrics(graph: FundingGraph):
    """Demonstrate dynamic network metrics."""
    console.print("\n[bold cyan]3. Dynamic Network Metrics[/bold cyan]")

    analyzer = DynamicNetworkAnalyzer()

    # Simulate network evolution
    console.print("\n[yellow]Simulating network evolution over time...[/yellow]")

    # T0: Initial state
    snapshot1 = analyzer.compute_snapshot(graph, timestamp=datetime.now() - timedelta(days=2))

    # T1: Add some edges (growth)
    new_edges = [
        FundingEdge("NewFunder_1", "NewWallet_1", 1000000, datetime.now(), "sig_new_1"),
        FundingEdge("NewFunder_2", "NewWallet_2", 1000000, datetime.now(), "sig_new_2"),
    ]
    graph.add_edges_from_list(new_edges)
    snapshot2 = analyzer.compute_snapshot(graph, timestamp=datetime.now() - timedelta(days=1))

    # T2: Current state
    snapshot3 = analyzer.compute_snapshot(graph, timestamp=datetime.now())

    # Display snapshots
    table = Table(title="Network Evolution")
    table.add_column("Metric", style="cyan")
    table.add_column("T-2 days", style="yellow")
    table.add_column("T-1 day", style="yellow")
    table.add_column("Today", style="green")

    metrics = [
        ("Nodes", "num_nodes"),
        ("Edges", "num_edges"),
        ("Density", "density"),
        ("Modularity", "modularity"),
        ("Centralization", "centralization"),
        ("Communities", "num_communities"),
    ]

    for metric_name, attr in metrics:
        v1 = getattr(snapshot1, attr)
        v2 = getattr(snapshot2, attr)
        v3 = getattr(snapshot3, attr)

        if isinstance(v1, float):
            table.add_row(metric_name, f"{v1:.4f}", f"{v2:.4f}", f"{v3:.4f}")
        else:
            table.add_row(metric_name, str(v1), str(v2), str(v3))

    console.print(table)

    # Compute dynamics
    dynamics = analyzer.compute_dynamics(window_size=3)
    if dynamics:
        console.print("\n[yellow]Network Dynamics (Velocities):[/yellow]")
        rprint(f"  Density Velocity: [green]{dynamics.velocity_density:+.6f}[/green] per day")
        rprint(
            f"  Modularity Velocity: [green]{dynamics.velocity_modularity:+.6f}[/green] per day"
        )
        rprint(
            f"  Centralization Velocity: [green]{dynamics.velocity_centralization:+.6f}[/green] per day"
        )
        rprint(f"  Is Fragmenting: [red]{dynamics.is_fragmenting}[/red]")
        rprint(f"  Is Consolidating: [blue]{dynamics.is_consolidating}[/blue]")

    # Network health
    health = analyzer.get_network_health_score()
    if health:
        health_color = "green" if health > 0.7 else "yellow" if health > 0.4 else "red"
        console.print(f"\nNetwork Health Score: [{health_color}]{health:.2%}[/{health_color}]")


def demo_anomaly_detection(graph: FundingGraph, embedder: GraphEmbedder):
    """Demonstrate anomaly detection."""
    console.print("\n[bold cyan]4. Wallet Anomaly Detection[/bold cyan]")

    config = AnomalyConfig(contamination=0.1, n_estimators=100, threshold=-0.5)
    detector = WalletAnomalyDetector(embedder=embedder, graph=graph, config=config)

    all_wallets = list(graph._wallet_set)

    console.print("\n[yellow]Training Isolation Forest...[/yellow]")
    detector.fit(all_wallets, include_embeddings=True)
    console.print("✓ Model trained")

    # Find anomalies
    console.print("\n[yellow]Finding anomalous wallets...[/yellow]")
    anomalies = detector.find_anomalies(all_wallets, top_k=10, include_embeddings=True)

    table = Table(title="Top Anomalous Wallets")
    table.add_column("Wallet", style="cyan")
    table.add_column("Anomaly Score", style="red")
    table.add_column("Confidence", style="yellow")
    table.add_column("Top Feature", style="magenta")

    for anomaly in anomalies:
        top_feature = max(
            anomaly.feature_contributions.items(), key=lambda x: x[1], default=("N/A", 0)
        )
        table.add_row(
            anomaly.wallet,
            f"{anomaly.score:.4f}",
            f"{anomaly.confidence:.2%}",
            f"{top_feature[0]} ({top_feature[1]:.2%})",
        )

    console.print(table)

    # Distribution statistics
    distribution = detector.get_anomaly_distribution(all_wallets, include_embeddings=True)

    console.print("\n[yellow]Anomaly Distribution:[/yellow]")
    rprint(f"  Total Wallets: {distribution['total_wallets']}")
    rprint(
        f"  Anomalous: [red]{distribution['anomalous_count']}[/red] ({distribution['anomalous_percentage']:.1f}%)"
    )
    rprint(f"  Mean Score: {distribution['mean_score']:.4f}")
    rprint(f"  Std Dev: {distribution['std_score']:.4f}")
    rprint(f"  Range: [{distribution['min_score']:.4f}, {distribution['max_score']:.4f}]")


def main():
    """Run complete demo."""
    console.print(
        Panel.fit(
            "[bold cyan]SHI Sprint 2: Graph Intelligence Demo[/bold cyan]\n"
            "Demonstrates Node2Vec, Similarity Detection, Network Dynamics, and Anomaly Detection",
            border_style="cyan",
        )
    )

    # Create demo graph
    graph = create_demo_graph()

    # Demo 1: Embeddings
    embedder = demo_embeddings(graph)

    # Demo 2: Similarity Detection
    demo_similarity_detection(graph, embedder)

    # Demo 3: Dynamic Metrics
    demo_dynamic_metrics(graph)

    # Demo 4: Anomaly Detection
    demo_anomaly_detection(graph, embedder)

    console.print(
        "\n[bold green]✓ Demo Complete![/bold green]\n"
        "[yellow]Sprint 2 features demonstrated successfully.[/yellow]\n"
    )


if __name__ == "__main__":
    main()
