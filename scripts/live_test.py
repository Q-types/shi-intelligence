#!/usr/bin/env python3
"""
Live Token Analysis Script for SHI.

Tests the full analysis pipeline on a real Solana token.

Usage:
    python scripts/live_test.py <token_mint>
    python scripts/live_test.py  # Uses USDC as default

Examples:
    # USDC
    python scripts/live_test.py EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v

    # BONK
    python scripts/live_test.py DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263

    # JTO
    python scripts/live_test.py jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()


async def test_token_analysis(mint: str) -> None:
    """Run full analysis on a token."""
    from src.data.client import SolanaDataClient
    from src.metrics import (
        compute_hhi,
        compute_shannon_entropy,
        compute_gini_coefficient,
        compute_whale_dominance_ratio,
    )
    from src.data.price_provider import JupiterPriceProvider

    print("\n" + "=" * 70)
    print(f"  SHI LIVE TOKEN ANALYSIS")
    print(f"  Token: {mint}")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70 + "\n")

    # Initialize clients
    client = SolanaDataClient()
    price_provider = JupiterPriceProvider()

    try:
        # 1. Fetch price data
        print("[1/4] Fetching price data from Jupiter...")
        try:
            price_data = await price_provider.get_price(mint)
            if price_data:
                print(f"      Price: ${price_data.price_usd:.6f}")
                if price_data.price_change_24h_pct:
                    change_emoji = "📈" if price_data.price_change_24h_pct > 0 else "📉"
                    print(f"      24h Change: {change_emoji} {price_data.price_change_24h_pct:+.2f}%")
                print(f"      Confidence: {price_data.confidence}")
            else:
                print("      Price: Not available")
        except Exception as e:
            print(f"      Price fetch failed: {e}")

        # 2. Fetch holder data
        print("\n[2/4] Fetching holder data from Helius...")
        snapshot = await client.get_token_holders(mint, limit=5000)
        print(f"      Holders fetched: {snapshot.holder_count:,}")
        print(f"      Total supply: {snapshot.total_supply:,.2f}")

        # 3. Compute distribution metrics
        print("\n[3/4] Computing distribution metrics...")
        shares = snapshot.shares
        balances = [b.balance for b in snapshot.balances]

        hhi = compute_hhi(shares)
        entropy = compute_shannon_entropy(shares)
        gini = compute_gini_coefficient(balances)
        wdr = compute_whale_dominance_ratio(balances, snapshot.total_supply)

        # 4. Display results
        print("\n[4/4] Analysis Results")
        print("-" * 50)

        # Holder Stats
        print("\n📊 HOLDER STATISTICS")
        print(f"   Total Holders: {snapshot.holder_count:,}")
        print(f"   Total Supply:  {snapshot.total_supply:,.2f}")

        # Top Holders
        print("\n🐋 TOP 10 HOLDERS")
        top_10 = sorted(snapshot.balances, key=lambda x: x.balance, reverse=True)[:10]
        cumulative_pct = 0.0
        for i, holder in enumerate(top_10, 1):
            pct = (holder.balance / snapshot.total_supply) * 100
            cumulative_pct += pct
            wallet_short = f"{holder.wallet[:6]}...{holder.wallet[-4:]}"
            print(f"   {i:2}. {wallet_short}  {pct:6.2f}%  (cum: {cumulative_pct:.2f}%)")

        # Distribution Metrics
        print("\n📈 DISTRIBUTION METRICS")

        # HHI interpretation
        hhi_risk = "🟢 Low" if hhi.value < 0.1 else "🟡 Medium" if hhi.value < 0.25 else "🔴 High"
        print(f"   HHI (concentration):     {hhi.value:.6f}  {hhi_risk}")

        # Entropy interpretation
        entropy_risk = "🟢 High diversity" if entropy.value > 4 else "🟡 Moderate" if entropy.value > 2 else "🔴 Low diversity"
        print(f"   Shannon Entropy:         {entropy.value:.4f}     {entropy_risk}")

        # Gini interpretation
        gini_risk = "🟢 Equal" if gini.value < 0.5 else "🟡 Unequal" if gini.value < 0.8 else "🔴 Very unequal"
        print(f"   Gini Coefficient:        {gini.value:.4f}     {gini_risk}")

        # Whale dominance
        wdr_risk = "🟢 Distributed" if wdr.value < 0.3 else "🟡 Moderate" if wdr.value < 0.5 else "🔴 Whale dominated"
        print(f"   Whale Dominance (Top10): {wdr.value:.2%}    {wdr_risk}")

        # Overall risk assessment
        print("\n⚠️  RISK ASSESSMENT")
        risk_score = (
            (1 if hhi.value > 0.25 else 0.5 if hhi.value > 0.1 else 0) +
            (1 if entropy.value < 2 else 0.5 if entropy.value < 4 else 0) +
            (1 if gini.value > 0.8 else 0.5 if gini.value > 0.5 else 0) +
            (1 if wdr.value > 0.5 else 0.5 if wdr.value > 0.3 else 0)
        ) / 4

        if risk_score > 0.7:
            print("   🔴 HIGH RISK - Heavy concentration, potential manipulation")
        elif risk_score > 0.4:
            print("   🟡 MEDIUM RISK - Some concentration, monitor large holders")
        else:
            print("   🟢 LOW RISK - Well distributed, healthy holder base")

        print(f"   Risk Score: {risk_score:.2f}/1.00")

        print("\n" + "=" * 70)
        print("  Analysis complete. All outputs are observational and probabilistic.")
        print("=" * 70 + "\n")

    finally:
        await client.close()
        await price_provider.close()


async def main():
    # Default to USDC if no mint provided
    if len(sys.argv) > 1:
        mint = sys.argv[1]
    else:
        # USDC
        mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        print("No mint provided, using USDC as default")

    await test_token_analysis(mint)


if __name__ == "__main__":
    asyncio.run(main())
