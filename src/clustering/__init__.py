"""
Wallet Archetype Clustering for SHI.

Assigns wallets to behavioral archetypes:
- Snipers
- Long-Term Accumulators
- Coordinated Cluster Members
- Liquidity Actors
- Exchange-Linked
- Dormant Whales

These archetype definitions are FIXED per PDR.
"""

from .archetypes import (
    ARCHETYPES,
    Archetype,
    assign_archetype,
    cluster_wallets,
)

__all__ = [
    "ARCHETYPES",
    "Archetype",
    "assign_archetype",
    "cluster_wallets",
]
