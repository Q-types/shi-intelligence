"""
Candidate Block Construction for Coordination Detection.

Instead of comparing every wallet pair naively (O(n²)), we use blocking
strategies to create candidate groups and only compute detailed similarity
within these groups.

Blocking strategies:
1. Same upstream funder
2. Funding within broad time window
3. Similar position size bucket
4. Same token entry window
5. Previous co-participation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence, Callable
from collections import defaultdict
from enum import Enum
import math

import structlog

from .features import WalletContext

logger = structlog.get_logger()


class BlockingStrategy(Enum):
    """Available blocking strategies."""

    SAME_FUNDER = "same_funder"
    FUNDING_TIME_WINDOW = "funding_time_window"
    POSITION_SIZE_BUCKET = "position_size_bucket"
    TOKEN_ENTRY_WINDOW = "token_entry_window"
    CO_PARTICIPATION = "co_participation"


@dataclass
class CandidateBlock:
    """A block of candidate wallets for detailed comparison."""

    block_id: str
    strategy: BlockingStrategy
    wallets: list[str]

    # Block metadata
    funder: Optional[str] = None  # For SAME_FUNDER
    time_window_start: Optional[datetime] = None  # For time-based blocks
    time_window_end: Optional[datetime] = None
    size_bucket: Optional[str] = None  # For POSITION_SIZE_BUCKET
    shared_token: Optional[str] = None  # For CO_PARTICIPATION

    @property
    def size(self) -> int:
        return len(self.wallets)

    @property
    def pair_count(self) -> int:
        """Number of unique pairs in this block."""
        n = len(self.wallets)
        return n * (n - 1) // 2


@dataclass
class BlockingResult:
    """Result of blocking operation."""

    blocks: list[CandidateBlock]
    total_wallets: int
    total_pairs_naive: int  # O(n²) pairs
    total_pairs_blocked: int  # Pairs after blocking
    reduction_factor: float
    strategies_used: list[BlockingStrategy]


def _block_by_funder(
    contexts: dict[str, WalletContext],
    min_block_size: int = 2,
) -> list[CandidateBlock]:
    """Block wallets by shared funder."""
    funder_to_wallets: dict[str, list[str]] = defaultdict(list)

    for addr, ctx in contexts.items():
        for funder in ctx.funders:
            funder_to_wallets[funder].append(addr)

    blocks = []
    for funder, wallets in funder_to_wallets.items():
        if len(wallets) >= min_block_size:
            blocks.append(
                CandidateBlock(
                    block_id=f"funder_{funder[:8]}",
                    strategy=BlockingStrategy.SAME_FUNDER,
                    wallets=wallets,
                    funder=funder,
                )
            )

    return blocks


def _block_by_funding_time(
    contexts: dict[str, WalletContext],
    window_hours: float = 24.0,
    min_block_size: int = 2,
) -> list[CandidateBlock]:
    """Block wallets by funding time window."""
    # Sort wallets by funding time
    wallets_with_time = [
        (addr, ctx.earliest_funding_time)
        for addr, ctx in contexts.items()
        if ctx.earliest_funding_time is not None
    ]

    if not wallets_with_time:
        return []

    wallets_with_time.sort(key=lambda x: x[1])

    blocks = []
    window_delta = timedelta(hours=window_hours)
    block_num = 0

    i = 0
    while i < len(wallets_with_time):
        # Start a new window
        window_start = wallets_with_time[i][1]
        window_end = window_start + window_delta
        window_wallets = []

        # Collect all wallets in this window
        j = i
        while j < len(wallets_with_time) and wallets_with_time[j][1] <= window_end:
            window_wallets.append(wallets_with_time[j][0])
            j += 1

        if len(window_wallets) >= min_block_size:
            blocks.append(
                CandidateBlock(
                    block_id=f"funding_time_{block_num}",
                    strategy=BlockingStrategy.FUNDING_TIME_WINDOW,
                    wallets=window_wallets,
                    time_window_start=window_start,
                    time_window_end=window_end,
                )
            )
            block_num += 1

        # Move to next non-overlapping window
        i = j if j > i else i + 1

    return blocks


def _block_by_position_size(
    contexts: dict[str, WalletContext],
    num_buckets: int = 10,
    min_block_size: int = 2,
) -> list[CandidateBlock]:
    """Block wallets by position size bucket (log scale)."""
    # Get wallets with position sizes
    wallets_with_size = [
        (addr, ctx.position_size)
        for addr, ctx in contexts.items()
        if ctx.position_size > 0
    ]

    if not wallets_with_size:
        return []

    # Compute log-scale bucket boundaries
    sizes = [s for _, s in wallets_with_size]
    min_size = min(sizes)
    max_size = max(sizes)

    if max_size <= min_size:
        # All same size - one block
        wallets = [w for w, _ in wallets_with_size]
        if len(wallets) >= min_block_size:
            return [
                CandidateBlock(
                    block_id="size_bucket_0",
                    strategy=BlockingStrategy.POSITION_SIZE_BUCKET,
                    wallets=wallets,
                    size_bucket="uniform",
                )
            ]
        return []

    # Log-scale boundaries
    log_min = math.log10(min_size)
    log_max = math.log10(max_size)
    bucket_width = (log_max - log_min) / num_buckets

    bucket_to_wallets: dict[int, list[str]] = defaultdict(list)
    for addr, size in wallets_with_size:
        bucket = min(int((math.log10(size) - log_min) / bucket_width), num_buckets - 1)
        bucket_to_wallets[bucket].append(addr)

    blocks = []
    for bucket, wallets in bucket_to_wallets.items():
        if len(wallets) >= min_block_size:
            log_low = log_min + bucket * bucket_width
            log_high = log_min + (bucket + 1) * bucket_width
            blocks.append(
                CandidateBlock(
                    block_id=f"size_bucket_{bucket}",
                    strategy=BlockingStrategy.POSITION_SIZE_BUCKET,
                    wallets=wallets,
                    size_bucket=f"10^{log_low:.1f}-10^{log_high:.1f}",
                )
            )

    return blocks


def _block_by_entry_time(
    contexts: dict[str, WalletContext],
    window_hours: float = 1.0,
    min_block_size: int = 2,
) -> list[CandidateBlock]:
    """Block wallets by first buy time window."""
    wallets_with_time = [
        (addr, ctx.first_buy_time)
        for addr, ctx in contexts.items()
        if ctx.first_buy_time is not None
    ]

    if not wallets_with_time:
        return []

    wallets_with_time.sort(key=lambda x: x[1])

    blocks = []
    window_delta = timedelta(hours=window_hours)
    block_num = 0

    i = 0
    while i < len(wallets_with_time):
        window_start = wallets_with_time[i][1]
        window_end = window_start + window_delta
        window_wallets = []

        j = i
        while j < len(wallets_with_time) and wallets_with_time[j][1] <= window_end:
            window_wallets.append(wallets_with_time[j][0])
            j += 1

        if len(window_wallets) >= min_block_size:
            blocks.append(
                CandidateBlock(
                    block_id=f"entry_time_{block_num}",
                    strategy=BlockingStrategy.TOKEN_ENTRY_WINDOW,
                    wallets=window_wallets,
                    time_window_start=window_start,
                    time_window_end=window_end,
                )
            )
            block_num += 1

        i = j if j > i else i + 1

    return blocks


def _block_by_co_participation(
    contexts: dict[str, WalletContext],
    min_shared_tokens: int = 2,
    min_block_size: int = 2,
) -> list[CandidateBlock]:
    """Block wallets by previous co-participation in tokens."""
    # Build token -> wallets map
    token_to_wallets: dict[str, set[str]] = defaultdict(set)
    for addr, ctx in contexts.items():
        for token in ctx.tokens_traded:
            token_to_wallets[token].add(addr)

    # Find wallet pairs that co-participated in multiple tokens
    pair_tokens: dict[tuple[str, str], set[str]] = defaultdict(set)
    for token, wallets in token_to_wallets.items():
        wallet_list = list(wallets)
        for i, w1 in enumerate(wallet_list):
            for w2 in wallet_list[i + 1:]:
                pair = tuple(sorted([w1, w2]))
                pair_tokens[pair].add(token)

    # Build clusters from pairs with enough shared tokens
    strong_pairs = {
        pair: tokens
        for pair, tokens in pair_tokens.items()
        if len(tokens) >= min_shared_tokens
    }

    if not strong_pairs:
        return []

    # Union-find to cluster wallets
    all_wallets = set()
    for (w1, w2) in strong_pairs:
        all_wallets.add(w1)
        all_wallets.add(w2)

    parent = {w: w for w in all_wallets}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for (w1, w2) in strong_pairs:
        union(w1, w2)

    # Group by root
    cluster_wallets: dict[str, list[str]] = defaultdict(list)
    for wallet in all_wallets:
        root = find(wallet)
        cluster_wallets[root].append(wallet)

    blocks = []
    for root, wallets in cluster_wallets.items():
        if len(wallets) >= min_block_size:
            # Find common tokens for this cluster
            common_tokens = set(contexts[wallets[0]].tokens_traded)
            for w in wallets[1:]:
                common_tokens &= contexts[w].tokens_traded

            blocks.append(
                CandidateBlock(
                    block_id=f"co_participation_{root[:8]}",
                    strategy=BlockingStrategy.CO_PARTICIPATION,
                    wallets=wallets,
                    shared_token=list(common_tokens)[0] if common_tokens else None,
                )
            )

    return blocks


def create_candidate_blocks(
    contexts: dict[str, WalletContext],
    strategies: Optional[list[BlockingStrategy]] = None,
    min_block_size: int = 2,
    max_pairs_per_block: int = 10000,
    funding_window_hours: float = 24.0,
    entry_window_hours: float = 1.0,
    position_size_buckets: int = 10,
    min_shared_tokens: int = 2,
) -> BlockingResult:
    """
    Create candidate blocks using specified strategies.

    Args:
        contexts: Dict of wallet address -> WalletContext
        strategies: Blocking strategies to use (default: all)
        min_block_size: Minimum wallets per block
        max_pairs_per_block: Split large blocks to avoid O(n²) explosion
        funding_window_hours: Window size for funding time blocking
        entry_window_hours: Window size for entry time blocking
        position_size_buckets: Number of buckets for position size blocking
        min_shared_tokens: Minimum shared tokens for co-participation blocking

    Returns:
        BlockingResult with candidate blocks
    """
    if strategies is None:
        strategies = list(BlockingStrategy)

    all_blocks = []

    for strategy in strategies:
        if strategy == BlockingStrategy.SAME_FUNDER:
            blocks = _block_by_funder(contexts, min_block_size)
        elif strategy == BlockingStrategy.FUNDING_TIME_WINDOW:
            blocks = _block_by_funding_time(contexts, funding_window_hours, min_block_size)
        elif strategy == BlockingStrategy.POSITION_SIZE_BUCKET:
            blocks = _block_by_position_size(contexts, position_size_buckets, min_block_size)
        elif strategy == BlockingStrategy.TOKEN_ENTRY_WINDOW:
            blocks = _block_by_entry_time(contexts, entry_window_hours, min_block_size)
        elif strategy == BlockingStrategy.CO_PARTICIPATION:
            blocks = _block_by_co_participation(contexts, min_shared_tokens, min_block_size)
        else:
            continue

        all_blocks.extend(blocks)

    # Split large blocks
    final_blocks = []
    for block in all_blocks:
        if block.pair_count <= max_pairs_per_block:
            final_blocks.append(block)
        else:
            # Split into smaller blocks
            max_size = int(math.sqrt(2 * max_pairs_per_block)) + 1
            wallets = block.wallets
            for i in range(0, len(wallets), max_size):
                chunk = wallets[i:i + max_size]
                if len(chunk) >= min_block_size:
                    final_blocks.append(
                        CandidateBlock(
                            block_id=f"{block.block_id}_part{i // max_size}",
                            strategy=block.strategy,
                            wallets=chunk,
                            funder=block.funder,
                            time_window_start=block.time_window_start,
                            time_window_end=block.time_window_end,
                            size_bucket=block.size_bucket,
                            shared_token=block.shared_token,
                        )
                    )

    # Calculate statistics
    n = len(contexts)
    naive_pairs = n * (n - 1) // 2

    # Count unique pairs across all blocks
    all_pairs = set()
    for block in final_blocks:
        wallets = block.wallets
        for i, w1 in enumerate(wallets):
            for w2 in wallets[i + 1:]:
                all_pairs.add(tuple(sorted([w1, w2])))

    blocked_pairs = len(all_pairs)
    reduction = (naive_pairs - blocked_pairs) / naive_pairs if naive_pairs > 0 else 0

    result = BlockingResult(
        blocks=final_blocks,
        total_wallets=n,
        total_pairs_naive=naive_pairs,
        total_pairs_blocked=blocked_pairs,
        reduction_factor=reduction,
        strategies_used=strategies,
    )

    logger.info(
        "blocking_complete",
        blocks_created=len(final_blocks),
        total_wallets=n,
        naive_pairs=naive_pairs,
        blocked_pairs=blocked_pairs,
        reduction_factor=f"{reduction:.2%}",
    )

    return result
