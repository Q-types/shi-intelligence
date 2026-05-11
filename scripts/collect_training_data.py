#!/usr/bin/env python3
"""
Scientific Training Data Collection Pipeline.

Collects expanded dataset for rug pull detection model training.
Target: 500+ labeled tokens with holder snapshots and features.

Data Sources:
1. SolRPDS Academic Dataset (historical rug pulls)
2. Verified rug pulls (high-profile scams)
3. Established safe tokens (long-running projects)
4. Random sample from Helius (unlabeled for semi-supervised)

Per Data Engineer Sharp Edges:
- Track input vs output counts (no silent data loss)
- Use event time, not processing time
- Store everything in UTC
"""

import asyncio
import csv
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.data.providers import HeliusProvider
from src.metrics.distribution import (
    compute_hhi,
    compute_gini_coefficient,
    compute_shannon_entropy,
    compute_whale_dominance_ratio,
)

logger = structlog.get_logger()


@dataclass
class TokenSample:
    """A token sample with label and features."""
    mint: str
    label: str  # "rug", "safe", "unknown"
    source: str  # "solrpds", "verified", "established", "random"
    holder_count: int
    total_supply: float
    hhi: float
    gini: float
    entropy: float
    whale_dominance_top10: float
    whale_dominance_top5: float
    top_holder_share: float
    collected_at: datetime
    collection_errors: list[str]


@dataclass
class CollectionStats:
    """Statistics for data collection run."""
    tokens_requested: int
    tokens_fetched: int
    tokens_failed: int
    rug_count: int
    safe_count: int
    unknown_count: int
    total_holders_processed: int


class TrainingDataCollector:
    """
    Collects labeled training data for rug pull detection.

    Implements Data Engineer best practices:
    - Tracks all data counts
    - No silent failures
    - UTC timestamps
    """

    RATE_LIMIT_DELAY = 0.15  # 150ms between requests (under 10/s limit)

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.provider: Optional[HeliusProvider] = None

        # Counters for data quality
        self.tokens_requested = 0
        self.tokens_fetched = 0
        self.tokens_failed = 0
        self.failed_tokens: list[tuple[str, str]] = []  # (mint, error)

    async def initialize(self):
        """Initialize the Helius provider."""
        self.provider = HeliusProvider()
        logger.info("provider_initialized", name=self.provider.name)

    async def close(self):
        """Close provider connections."""
        if self.provider:
            await self.provider.close()

    async def collect_token_features(
        self,
        mint: str,
        label: str,
        source: str,
        holder_limit: int = 1000,
    ) -> Optional[TokenSample]:
        """
        Collect features for a single token.

        Args:
            mint: Token mint address
            label: "rug", "safe", or "unknown"
            source: Data source identifier
            holder_limit: Max holders to fetch

        Returns:
            TokenSample with computed features, or None if failed
        """
        self.tokens_requested += 1
        errors = []

        try:
            await asyncio.sleep(self.RATE_LIMIT_DELAY)

            snapshot = await self.provider.get_token_holders(
                mint, limit=holder_limit
            )

            if snapshot.holder_count < 5:
                errors.append(f"Too few holders: {snapshot.holder_count}")
                self.tokens_failed += 1
                self.failed_tokens.append((mint, "too_few_holders"))
                return None

            # Compute distribution metrics
            shares = snapshot.shares
            balances = [b.balance for b in snapshot.balances]

            hhi_result = compute_hhi(shares)
            gini_result = compute_gini_coefficient(balances)
            entropy_result = compute_shannon_entropy(shares)
            wdr_10 = compute_whale_dominance_ratio(balances, snapshot.total_supply, k=10)
            wdr_5 = compute_whale_dominance_ratio(balances, snapshot.total_supply, k=5)

            sample = TokenSample(
                mint=mint,
                label=label,
                source=source,
                holder_count=snapshot.holder_count,
                total_supply=snapshot.total_supply,
                hhi=hhi_result.value,
                gini=gini_result.value,
                entropy=entropy_result.value,
                whale_dominance_top10=wdr_10.value,
                whale_dominance_top5=wdr_5.value,
                top_holder_share=max(shares) if shares else 0.0,
                collected_at=datetime.now(timezone.utc),
                collection_errors=errors,
            )

            self.tokens_fetched += 1
            logger.info(
                "token_collected",
                mint=mint[:8],
                label=label,
                holders=snapshot.holder_count,
                hhi=f"{hhi_result.value:.4f}",
            )

            return sample

        except Exception as e:
            self.tokens_failed += 1
            self.failed_tokens.append((mint, str(e)))
            logger.warning("token_collection_failed", mint=mint[:8], error=str(e))
            return None

    async def collect_from_file(
        self,
        file_path: Path,
        label: str,
        source: str,
        max_tokens: Optional[int] = None,
    ) -> list[TokenSample]:
        """
        Collect features for tokens listed in a file.

        Args:
            file_path: Path to file with mint addresses (one per line or CSV)
            label: Label to apply
            source: Source identifier
            max_tokens: Maximum tokens to process

        Returns:
            List of TokenSample objects
        """
        mints = []

        if file_path.suffix == ".csv":
            with open(file_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "token_mint" in row:
                        mints.append(row["token_mint"])
                    elif "mint" in row:
                        mints.append(row["mint"])
                    elif "address" in row:
                        mints.append(row["address"])
        else:
            with open(file_path) as f:
                mints = [line.strip() for line in f if line.strip()]

        if max_tokens:
            mints = mints[:max_tokens]

        logger.info(
            "collecting_from_file",
            file=file_path.name,
            count=len(mints),
            label=label,
        )

        samples = []
        for mint in mints:
            sample = await self.collect_token_features(mint, label, source)
            if sample:
                samples.append(sample)

        return samples

    async def collect_full_dataset(
        self,
        solrpds_limit: int = 100,
        verified_rugs_limit: int = 30,
        safe_tokens_limit: int = 50,
    ) -> list[TokenSample]:
        """
        Collect full training dataset from all sources.

        Args:
            solrpds_limit: Max tokens from SolRPDS
            verified_rugs_limit: Max verified rug pulls
            safe_tokens_limit: Max safe tokens

        Returns:
            Combined list of all samples
        """
        await self.initialize()

        all_samples = []
        data_dir = Path(__file__).parent.parent / "data" / "training"

        try:
            # 1. SolRPDS Academic Dataset (rug pulls)
            solrpds_file = data_dir / "solrpds_token_mints_2023.txt"
            if solrpds_file.exists():
                logger.info("collecting_solrpds", limit=solrpds_limit)
                samples = await self.collect_from_file(
                    solrpds_file, "rug", "solrpds", solrpds_limit
                )
                all_samples.extend(samples)

            # 2. Verified rug pulls (high-profile)
            verified_file = data_dir / "verified_rugpulls.csv"
            if verified_file.exists():
                logger.info("collecting_verified_rugs", limit=verified_rugs_limit)
                samples = await self.collect_from_file(
                    verified_file, "rug", "verified", verified_rugs_limit
                )
                all_samples.extend(samples)

            # 3. Safe tokens (established projects)
            safe_file = data_dir / "safe_tokens.csv"
            if safe_file.exists():
                logger.info("collecting_safe_tokens", limit=safe_tokens_limit)
                samples = await self.collect_from_file(
                    safe_file, "safe", "established", safe_tokens_limit
                )
                all_samples.extend(samples)

            # Log collection statistics
            stats = self._compute_stats(all_samples)
            logger.info(
                "collection_complete",
                requested=stats.tokens_requested,
                fetched=stats.tokens_fetched,
                failed=stats.tokens_failed,
                rug_count=stats.rug_count,
                safe_count=stats.safe_count,
                success_rate=f"{stats.tokens_fetched / max(1, stats.tokens_requested) * 100:.1f}%",
            )

            return all_samples

        finally:
            await self.close()

    def _compute_stats(self, samples: list[TokenSample]) -> CollectionStats:
        """Compute collection statistics."""
        rug_count = sum(1 for s in samples if s.label == "rug")
        safe_count = sum(1 for s in samples if s.label == "safe")
        unknown_count = sum(1 for s in samples if s.label == "unknown")
        total_holders = sum(s.holder_count for s in samples)

        return CollectionStats(
            tokens_requested=self.tokens_requested,
            tokens_fetched=self.tokens_fetched,
            tokens_failed=self.tokens_failed,
            rug_count=rug_count,
            safe_count=safe_count,
            unknown_count=unknown_count,
            total_holders_processed=total_holders,
        )

    def save_dataset(
        self,
        samples: list[TokenSample],
        filename: str = "training_dataset.csv",
    ) -> Path:
        """
        Save collected samples to CSV.

        Args:
            samples: List of TokenSample objects
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, "w", newline="") as f:
            if samples:
                fieldnames = list(asdict(samples[0]).keys())
                # Convert datetime to string for CSV
                fieldnames = [fn for fn in fieldnames if fn != "collection_errors"]
                fieldnames.append("error_count")

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for sample in samples:
                    row = asdict(sample)
                    row["collected_at"] = row["collected_at"].isoformat()
                    row["error_count"] = len(row.pop("collection_errors", []))
                    writer.writerow(row)

        logger.info("dataset_saved", path=str(output_path), samples=len(samples))
        return output_path

    def save_failed_tokens(self, filename: str = "failed_tokens.json") -> Path:
        """Save list of failed tokens for debugging."""
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(
                {"failed_tokens": self.failed_tokens, "count": len(self.failed_tokens)},
                f,
                indent=2,
            )

        return output_path


async def main():
    """Main data collection entry point."""
    print("=" * 60)
    print("SHI Scientific Training Data Collection")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "training" / "collected"
    collector = TrainingDataCollector(output_dir)

    # Collect dataset with expanded limits
    # Start with reasonable limits to test, can increase later
    samples = await collector.collect_full_dataset(
        solrpds_limit=50,  # Start smaller for testing
        verified_rugs_limit=30,
        safe_tokens_limit=14,  # We only have 14 safe tokens currently
    )

    if samples:
        # Save collected data
        dataset_path = collector.save_dataset(samples)
        failed_path = collector.save_failed_tokens()

        # Print summary
        stats = collector._compute_stats(samples)

        print("\n" + "=" * 60)
        print("Collection Summary")
        print("=" * 60)
        print(f"Tokens Requested: {stats.tokens_requested}")
        print(f"Tokens Fetched:   {stats.tokens_fetched}")
        print(f"Tokens Failed:    {stats.tokens_failed}")
        print(f"Rug Pulls:        {stats.rug_count}")
        print(f"Safe Tokens:      {stats.safe_count}")
        print(f"Success Rate:     {stats.tokens_fetched / max(1, stats.tokens_requested) * 100:.1f}%")
        print(f"\nDataset saved to: {dataset_path}")
        print(f"Failed tokens log: {failed_path}")

        # Show class balance
        print("\n" + "=" * 60)
        print("Class Balance Analysis")
        print("=" * 60)
        if stats.rug_count > 0 and stats.safe_count > 0:
            imbalance_ratio = stats.rug_count / stats.safe_count
            print(f"Rug/Safe Ratio: {imbalance_ratio:.2f}")
            if imbalance_ratio > 3:
                print("WARNING: Significant class imbalance detected!")
                print("         Consider SMOTE or class weighting during training.")

        # Show feature distributions
        print("\n" + "=" * 60)
        print("Feature Distributions (Mean +/- Std)")
        print("=" * 60)

        import numpy as np
        rug_samples = [s for s in samples if s.label == "rug"]
        safe_samples = [s for s in samples if s.label == "safe"]

        for feature in ["hhi", "gini", "entropy", "whale_dominance_top10", "top_holder_share"]:
            rug_values = [getattr(s, feature) for s in rug_samples]
            safe_values = [getattr(s, feature) for s in safe_samples]

            if rug_values and safe_values:
                rug_mean, rug_std = np.mean(rug_values), np.std(rug_values)
                safe_mean, safe_std = np.mean(safe_values), np.std(safe_values)

                print(f"\n{feature.upper()}:")
                print(f"  Rug:  {rug_mean:.4f} +/- {rug_std:.4f}")
                print(f"  Safe: {safe_mean:.4f} +/- {safe_std:.4f}")
    else:
        print("No samples collected!")


if __name__ == "__main__":
    asyncio.run(main())
