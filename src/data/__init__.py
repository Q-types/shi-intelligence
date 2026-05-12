"""
Data Source Abstraction Layer for SHI.

Provides provider-agnostic access to Solana blockchain data.

Features:
- Provider abstraction (Helius / RPC / alternative indexers)
- Schema validation layer
- Retry + exponential backoff logic
- Rate limit handling
- Data provenance logging
- Checksum validation for critical datasets
- Detection of partial ingestion
- Fail-safe behavior on incomplete data
- Query budgeting and cost tracking
- Caching layer for repeated token queries
- Price data integration (Jupiter API)
"""

from .providers import DataProvider, HeliusProvider, RPCProvider
from .client import SolanaDataClient
from .cache import QueryCache
from .price_provider import JupiterPriceProvider, PriceData

__all__ = [
    "DataProvider",
    "HeliusProvider",
    "RPCProvider",
    "SolanaDataClient",
    "QueryCache",
    "JupiterPriceProvider",
    "PriceData",
]
