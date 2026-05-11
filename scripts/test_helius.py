#!/usr/bin/env python3
"""
Test Helius API Integration.

Verifies that the Helius API key works and can fetch token data.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


async def test_helius_connection():
    """Test basic Helius API connectivity."""
    from src.data.providers import HeliusProvider
    from src.core.config import settings

    print("=" * 60)
    print("SHI Helius Integration Test")
    print("=" * 60)

    # Check API key
    api_key = settings.helius_api_key
    if not api_key:
        print("ERROR: HELIUS_API_KEY not set in .env")
        return False

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print()

    # Initialize provider
    provider = HeliusProvider()
    print(f"Provider: {provider.name}")
    print(f"RPC URL: {provider.rpc_url[:50]}...")
    print()

    # Test 1: Fetch BONK token holders (small, established token)
    print("Test 1: Fetching BONK token holders (limit 100)...")
    bonk_mint = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"

    try:
        snapshot = await provider.get_token_holders(bonk_mint, limit=100)
        print(f"  SUCCESS: Fetched {snapshot.holder_count} holders")
        print(f"  Total Supply: {snapshot.total_supply:,}")
        print(f"  Timestamp: {snapshot.timestamp}")

        # Show top 5 holders
        sorted_balances = sorted(snapshot.balances, key=lambda x: x.balance, reverse=True)
        print("\n  Top 5 Holders:")
        for i, bal in enumerate(sorted_balances[:5], 1):
            pct = (bal.balance / snapshot.total_supply) * 100
            print(f"    {i}. {bal.wallet[:8]}...{bal.wallet[-4:]} - {pct:.2f}%")

    except Exception as e:
        print(f"  FAILED: {e}")
        await provider.close()
        return False

    print()

    # Test 2: Fetch wallet metadata
    print("Test 2: Fetching wallet metadata...")
    test_wallet = sorted_balances[0].wallet if sorted_balances else None

    if test_wallet:
        try:
            metadata = await provider.get_wallet_metadata(test_wallet)
            print(f"  SUCCESS: Got metadata for {test_wallet[:8]}...")
            print(f"  First seen: {metadata.first_seen_at}")
            print(f"  Funded by: {metadata.funded_by[:8] if metadata.funded_by else 'Unknown'}...")
        except Exception as e:
            print(f"  WARNING: {e}")

    print()

    # Test 3: Test with a known rug pull token
    print("Test 3: Fetching LIBRA token (known rug pull)...")
    libra_mint = "Hz1XePA2vukqFBcf9P7VJ3AsMKoTXyPn3s21dNvGrHnd"

    try:
        libra_snapshot = await provider.get_token_holders(libra_mint, limit=50)
        print(f"  SUCCESS: Fetched {libra_snapshot.holder_count} holders")
        print(f"  Total Supply: {libra_snapshot.total_supply:,}")
    except Exception as e:
        print(f"  INFO: {e} (token may be inactive)")

    await provider.close()

    print()
    print("=" * 60)
    print("Helius Integration Test Complete!")
    print("=" * 60)
    return True


async def test_full_pipeline():
    """Test full analysis pipeline with real data."""
    from src.data.client import SolanaDataClient
    from src.metrics.distribution import compute_hhi, compute_gini_coefficient

    print("\n" + "=" * 60)
    print("Full Pipeline Test")
    print("=" * 60)

    client = SolanaDataClient()

    # Fetch BONK holders
    bonk_mint = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
    print(f"\nFetching {bonk_mint[:8]}... holders...")

    try:
        snapshot = await client.get_token_holders(bonk_mint, limit=1000)
        print(f"Fetched {snapshot.holder_count} holders")

        # Compute metrics
        shares = snapshot.shares
        balances = [b.balance for b in snapshot.balances]
        hhi = compute_hhi(shares)
        gini = compute_gini_coefficient(balances)

        print(f"\nMetrics:")
        print(f"  HHI: {hhi.value:.6f}")
        print(f"  Gini: {gini.value:.4f}")
        print(f"  Top holder share: {max(shares):.2%}")

    except Exception as e:
        print(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

    await client.close()


if __name__ == "__main__":
    print("Starting Helius Integration Tests...\n")

    success = asyncio.run(test_helius_connection())

    if success:
        asyncio.run(test_full_pipeline())
