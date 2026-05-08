"""
Main entry point for Solana Holder Intelligence.

Usage:
    python -m src.main --mode bot      # Run Telegram bot
    python -m src.main --mode analyze  # Run single analysis
"""

from __future__ import annotations

import asyncio
import argparse
import sys

import structlog

from .core.config import settings
from .telegram.bot import SHIBot

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer() if settings.log_level == "DEBUG" else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


async def run_bot() -> None:
    """Run the Telegram bot."""
    logger.info("starting_shi_bot", version="0.1.0")

    bot = SHIBot()

    try:
        await bot.start()
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("shutdown_requested")
    finally:
        await bot.stop()


async def run_analysis(mint: str) -> None:
    """Run single token analysis."""
    from .data.client import SolanaDataClient
    from .metrics import (
        compute_hhi,
        compute_shannon_entropy,
        compute_gini_coefficient,
        compute_whale_dominance_ratio,
    )

    logger.info("running_analysis", mint=mint)

    async with SolanaDataClient() as client:
        # Fetch holders
        snapshot = await client.get_token_holders(mint)
        logger.info("holders_fetched", count=snapshot.holder_count)

        # Compute metrics
        shares = snapshot.shares
        balances = [b.balance for b in snapshot.balances]

        hhi = compute_hhi(shares)
        entropy = compute_shannon_entropy(shares)
        gini = compute_gini_coefficient(balances)
        wdr = compute_whale_dominance_ratio(balances, snapshot.total_supply)

        # Print results
        print(f"\nAnalysis for {mint}")
        print("=" * 60)
        print(f"Holders: {snapshot.holder_count:,}")
        print(f"Total Supply: {snapshot.total_supply:,}")
        print()
        print("Distribution Metrics:")
        print(f"  HHI: {hhi.value:.6f}")
        print(f"  Shannon Entropy: {entropy.value:.4f}")
        print(f"  Gini Coefficient: {gini.value:.4f}")
        print(f"  Whale Dominance (Top 10): {wdr.value:.2%}")
        print()
        print("Note: All outputs are observational and probabilistic.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Solana Holder Intelligence",
    )
    parser.add_argument(
        "--mode",
        choices=["bot", "analyze"],
        default="bot",
        help="Run mode: bot (Telegram) or analyze (single token)",
    )
    parser.add_argument(
        "--mint",
        type=str,
        help="Token mint address (required for analyze mode)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if args.mode == "bot":
        asyncio.run(run_bot())
    elif args.mode == "analyze":
        if not args.mint:
            print("Error: --mint required for analyze mode")
            sys.exit(1)
        asyncio.run(run_analysis(args.mint))


if __name__ == "__main__":
    main()
