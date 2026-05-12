"""
Data Provider Abstraction.

Provider-agnostic interface for Solana data ingestion.
Supports Helius, direct RPC, and alternative indexers.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..core.config import settings
from ..core.types import (
    TokenMint,
    WalletAddress,
    TokenBalance,
    WalletMetadata,
    FundingEdge,
    HolderSnapshot,
)

logger = structlog.get_logger()


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: float = 60.0):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s")


class PartialDataError(Exception):
    """Raised when data ingestion is incomplete."""

    def __init__(self, message: str, received: int, expected: int):
        self.received = received
        self.expected = expected
        super().__init__(f"{message}: received {received}/{expected}")


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...

    @abstractmethod
    async def get_token_holders(
        self,
        mint: TokenMint,
        *,
        limit: int | None = None,
    ) -> HolderSnapshot:
        """
        Fetch all holders for a token.

        Args:
            mint: Token mint address
            limit: Optional limit on holders (for sampling)

        Returns:
            HolderSnapshot with all holder balances
        """
        ...

    @abstractmethod
    async def get_wallet_metadata(
        self,
        wallet: WalletAddress,
    ) -> WalletMetadata:
        """
        Fetch metadata for a wallet.

        Args:
            wallet: Wallet address

        Returns:
            WalletMetadata including funding source
        """
        ...

    @abstractmethod
    async def get_funding_edges(
        self,
        wallets: list[WalletAddress],
    ) -> AsyncIterator[FundingEdge]:
        """
        Fetch funding relationships for wallets.

        Args:
            wallets: List of wallet addresses

        Yields:
            FundingEdge for each funding transfer
        """
        ...

    @abstractmethod
    async def get_historical_balances(
        self,
        wallet: WalletAddress,
        mint: TokenMint,
        *,
        since: datetime | None = None,
    ) -> list[TokenBalance]:
        """
        Fetch historical balance snapshots.

        Args:
            wallet: Wallet address
            mint: Token mint
            since: Optional start time

        Returns:
            List of TokenBalance snapshots over time
        """
        ...


class HeliusProvider(DataProvider):
    """Helius API provider implementation with DAS API support."""

    # Helius rate limits (free tier: 10 req/s, 100k credits/month)
    RATE_LIMIT_PER_SECOND = 10

    def __init__(
        self,
        api_key: str | None = None,
        rpc_url: str | None = None,
    ):
        self.api_key = api_key or settings.helius_api_key
        # Build RPC URL with API key
        if self.api_key:
            self.rpc_url = f"https://mainnet.helius-rpc.com/?api-key={self.api_key}"
            self.das_url = f"https://mainnet.helius-rpc.com/?api-key={self.api_key}"
        else:
            self.rpc_url = rpc_url or settings.helius_rpc_url
            self.das_url = self.rpc_url
        self._client: httpx.AsyncClient | None = None
        self._query_count = 0
        self._query_cost = 0.0
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "helius"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
            )
        return self._client

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        min_interval = 1.0 / self.RATE_LIMIT_PER_SECOND
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _track_query(self, cost: float = 1.0) -> None:
        """Track query for budgeting."""
        self._query_count += 1
        self._query_cost += cost
        logger.debug(
            "query_tracked",
            provider=self.name,
            total_queries=self._query_count,
            total_cost=self._query_cost,
        )

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
    )
    async def _request(
        self,
        method: str,
        params: dict | list | None = None,
    ) -> dict:
        """Make RPC request with retry logic and rate limiting."""
        await self._rate_limit()
        client = await self._get_client()

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or [],
        }

        self._track_query()

        response = await client.post(self.rpc_url, json=payload)

        # Handle rate limiting
        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", 60))
            logger.warning("rate_limited", provider=self.name, retry_after=retry_after)
            raise RateLimitError(retry_after)

        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise Exception(f"RPC error: {data['error']}")

        return data["result"]

    async def _das_request(
        self,
        method: str,
        params: dict,
    ) -> dict:
        """Make Helius DAS API request."""
        await self._rate_limit()
        client = await self._get_client()

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }

        self._track_query()

        response = await client.post(self.das_url, json=payload)

        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", 60))
            logger.warning("rate_limited_das", provider=self.name, retry_after=retry_after)
            raise RateLimitError(retry_after)

        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise Exception(f"DAS API error: {data['error']}")

        return data.get("result", {})

    async def get_token_holders(
        self,
        mint: TokenMint,
        *,
        limit: int | None = None,
    ) -> HolderSnapshot:
        """Fetch token holders via Helius DAS API."""
        logger.info("fetching_holders", mint=mint, provider=self.name)

        # Use Helius getTokenAccounts DAS API
        all_accounts = []
        page = 1

        while True:
            params = {
                "mint": mint,
                "page": page,
                "limit": 1000,  # Max per page for Helius
                "displayOptions": {
                    "showZeroBalance": False,
                },
            }

            try:
                result = await self._das_request("getTokenAccounts", params)
            except Exception as e:
                logger.warning("das_api_failed_trying_rpc", error=str(e))
                # Fallback to standard RPC method
                return await self._get_token_holders_rpc(mint, limit=limit)

            accounts = result.get("token_accounts", [])
            if not accounts:
                break

            all_accounts.extend(accounts)

            logger.debug(
                "fetched_page",
                page=page,
                count=len(accounts),
                total=len(all_accounts),
            )

            # Check if we have more pages
            if len(accounts) < 1000 or (limit and len(all_accounts) >= limit):
                break

            page += 1

            # Safety limit to prevent infinite loops
            if page > 100:
                logger.warning("pagination_limit_reached", pages=page)
                break

        # Apply limit if specified
        if limit and len(all_accounts) > limit:
            all_accounts = all_accounts[:limit]

        # Convert to TokenBalance objects
        now = datetime.now(timezone.utc)
        balances = []
        total_supply = 0

        for acc in all_accounts:
            try:
                balance = TokenBalance(
                    wallet=acc["owner"],
                    mint=mint,
                    balance=int(acc["amount"]),
                    decimals=acc.get("decimals", 9),
                    timestamp=now,
                )
                balances.append(balance)
                total_supply += balance.balance
            except Exception as e:
                logger.warning("invalid_account_skipped", error=str(e), account=acc)
                continue

        # Ensure we have at least some supply
        if total_supply == 0:
            total_supply = 1  # Avoid division by zero

        snapshot = HolderSnapshot(
            mint=mint,
            timestamp=now,
            total_supply=total_supply,
            holder_count=len(balances),
            balances=balances,
        )

        # Log provenance
        logger.info(
            "holders_fetched",
            mint=mint,
            holder_count=snapshot.holder_count,
            total_supply=total_supply,
            checksum=self._compute_checksum(snapshot),
        )

        return snapshot

    async def _get_token_holders_rpc(
        self,
        mint: TokenMint,
        *,
        limit: int | None = None,
    ) -> HolderSnapshot:
        """Fallback: Fetch holders via standard RPC getProgramAccounts."""
        logger.info("fetching_holders_rpc_fallback", mint=mint)

        client = await self._get_client()

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getProgramAccounts",
            "params": [
                "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                {
                    "encoding": "jsonParsed",
                    "filters": [
                        {"dataSize": 165},
                        {"memcmp": {"offset": 0, "bytes": mint}},
                    ],
                },
            ],
        }

        response = await client.post(self.rpc_url, json=payload)
        response.raise_for_status()
        result = response.json().get("result", [])

        now = datetime.now(timezone.utc)
        balances = []
        total_supply = 0

        for acc in result:
            try:
                info = acc["account"]["data"]["parsed"]["info"]
                balance = TokenBalance(
                    wallet=info["owner"],
                    mint=mint,
                    balance=int(info["tokenAmount"]["amount"]),
                    decimals=info["tokenAmount"]["decimals"],
                    timestamp=now,
                )
                balances.append(balance)
                total_supply += balance.balance
            except (KeyError, TypeError) as e:
                logger.warning("invalid_rpc_account", error=str(e))
                continue

        if limit:
            balances = balances[:limit]

        if total_supply == 0:
            total_supply = 1

        return HolderSnapshot(
            mint=mint,
            timestamp=now,
            total_supply=total_supply,
            holder_count=len(balances),
            balances=balances,
        )

    async def get_wallet_metadata(
        self,
        wallet: WalletAddress,
    ) -> WalletMetadata:
        """Fetch wallet metadata including funding source via Helius parsed transactions."""
        first_seen = datetime.now(timezone.utc)
        funded_by = None
        first_funded_at = None

        try:
            # Get first transactions to find funder
            result = await self._request(
                "getSignaturesForAddress",
                [wallet, {"limit": 10}],  # Get recent signatures
            )

            if result:
                # Sort by blockTime to get earliest
                sorted_sigs = sorted(result, key=lambda x: x.get("blockTime", 0))
                first_sig = sorted_sigs[0] if sorted_sigs else None

                if first_sig and first_sig.get("blockTime"):
                    first_seen = datetime.fromtimestamp(
                        first_sig["blockTime"],
                        tz=timezone.utc,
                    )

                    # Use Helius enhanced transaction parsing
                    tx = await self._request(
                        "getTransaction",
                        [first_sig["signature"], {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                    )

                    if tx:
                        funded_by, first_funded_at = self._parse_funding_source(
                            tx, wallet, first_seen
                        )

        except Exception as e:
            logger.warning("wallet_metadata_fetch_failed", wallet=wallet, error=str(e))

        return WalletMetadata(
            address=wallet,
            funded_by=funded_by,
            first_funded_at=first_funded_at,
            first_seen_at=first_seen,
        )

    def _parse_funding_source(
        self,
        tx: dict,
        target_wallet: str,
        tx_time: datetime,
    ) -> tuple[str | None, datetime | None]:
        """Parse transaction to find who funded this wallet."""
        try:
            message = tx.get("transaction", {}).get("message", {})
            instructions = message.get("instructions", [])

            # Look for SOL transfer or token transfer to target wallet
            for instr in instructions:
                parsed = instr.get("parsed", {})
                if isinstance(parsed, dict):
                    instr_type = parsed.get("type", "")
                    info = parsed.get("info", {})

                    # System program transfer
                    if instr_type == "transfer":
                        if info.get("destination") == target_wallet:
                            source = info.get("source")
                            if source and source != target_wallet:
                                return source, tx_time

                    # Create account (someone funded this account)
                    if instr_type == "createAccount":
                        if info.get("newAccount") == target_wallet:
                            source = info.get("source")
                            if source:
                                return source, tx_time

            # Check account keys for fee payer as fallback
            account_keys = message.get("accountKeys", [])
            if account_keys:
                # First account is usually the fee payer
                fee_payer = account_keys[0]
                if isinstance(fee_payer, dict):
                    fee_payer = fee_payer.get("pubkey", "")
                if fee_payer and fee_payer != target_wallet:
                    return fee_payer, tx_time

        except Exception as e:
            logger.debug("funding_source_parse_failed", error=str(e))

        return None, None

    async def get_funding_edges(
        self,
        wallets: list[WalletAddress],
    ) -> AsyncIterator[FundingEdge]:
        """Fetch funding relationships for wallets."""
        logger.info("fetching_funding_edges", wallet_count=len(wallets))

        for i, wallet in enumerate(wallets):
            try:
                # Get signatures for this wallet
                sigs = await self._request(
                    "getSignaturesForAddress",
                    [wallet, {"limit": 20}],
                )

                if not sigs:
                    continue

                # Sort by time to get earliest
                sorted_sigs = sorted(sigs, key=lambda x: x.get("blockTime", 0))

                for sig_info in sorted_sigs[:5]:  # Check first 5 transactions
                    sig = sig_info.get("signature", "")
                    if not sig:
                        continue

                    try:
                        tx = await self._request(
                            "getTransaction",
                            [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                        )

                        if not tx:
                            continue

                        # Parse for funding transfer
                        edge = self._parse_funding_edge(tx, wallet, sig)
                        if edge:
                            yield edge
                            break  # Found the funding edge for this wallet

                    except Exception as e:
                        logger.debug("tx_parse_failed", signature=sig, error=str(e))
                        continue

            except Exception as e:
                logger.warning("funding_edge_fetch_failed", wallet=wallet, error=str(e))
                continue

            # Log progress
            if (i + 1) % 100 == 0:
                logger.info("funding_edges_progress", processed=i + 1, total=len(wallets))

    def _parse_funding_edge(
        self,
        tx: dict,
        target_wallet: str,
        signature: str,
    ) -> FundingEdge | None:
        """Parse a transaction to extract funding edge."""
        try:
            block_time = tx.get("blockTime", 0)
            if not block_time:
                return None

            timestamp = datetime.fromtimestamp(block_time, tz=timezone.utc)
            message = tx.get("transaction", {}).get("message", {})
            instructions = message.get("instructions", [])

            for instr in instructions:
                parsed = instr.get("parsed", {})
                if not isinstance(parsed, dict):
                    continue

                instr_type = parsed.get("type", "")
                info = parsed.get("info", {})

                # SOL transfer
                if instr_type == "transfer":
                    if info.get("destination") == target_wallet:
                        source = info.get("source")
                        lamports = info.get("lamports", 0)
                        if source and source != target_wallet:
                            # Validate signature format (87-88 base58 chars)
                            if len(signature) >= 87:
                                return FundingEdge(
                                    source=source,
                                    target=target_wallet,
                                    amount_lamports=lamports,
                                    timestamp=timestamp,
                                    signature=signature,
                                )

                # Create account
                if instr_type == "createAccount":
                    if info.get("newAccount") == target_wallet:
                        source = info.get("source")
                        lamports = info.get("lamports", 0)
                        if source and len(signature) >= 87:
                            return FundingEdge(
                                source=source,
                                target=target_wallet,
                                amount_lamports=lamports,
                                timestamp=timestamp,
                                signature=signature,
                            )

        except Exception as e:
            logger.debug("funding_edge_parse_failed", error=str(e))

        return None

    async def get_historical_balances(
        self,
        wallet: WalletAddress,
        mint: TokenMint,
        *,
        since: datetime | None = None,
    ) -> list[TokenBalance]:
        """Fetch historical balance snapshots."""
        # This would require indexed historical data
        # Placeholder implementation
        logger.warning(
            "historical_balances_not_implemented",
            wallet=wallet,
            mint=mint,
        )
        return []

    def _compute_checksum(self, snapshot: HolderSnapshot) -> str:
        """Compute checksum for data validation."""
        data = f"{snapshot.mint}:{snapshot.holder_count}:{snapshot.total_supply}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class RPCProvider(DataProvider):
    """Direct Solana RPC provider (fallback)."""

    def __init__(self, rpc_url: str | None = None):
        self.rpc_url = rpc_url or settings.solana_rpc_url
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "rpc"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.rpc_url,
                timeout=30.0,
            )
        return self._client

    async def get_token_holders(
        self,
        mint: TokenMint,
        *,
        limit: int | None = None,
    ) -> HolderSnapshot:
        """Fetch holders via RPC getProgramAccounts."""
        logger.info("fetching_holders_rpc", mint=mint)

        client = await self._get_client()

        # This is expensive - prefer Helius
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getProgramAccounts",
            "params": [
                "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token program
                {
                    "encoding": "jsonParsed",
                    "filters": [
                        {"dataSize": 165},
                        {"memcmp": {"offset": 0, "bytes": mint}},
                    ],
                },
            ],
        }

        response = await client.post("", json=payload)
        response.raise_for_status()
        result = response.json()["result"]

        now = datetime.now(timezone.utc)
        balances = []
        total_supply = 0

        for acc in result:
            info = acc["account"]["data"]["parsed"]["info"]
            balance = TokenBalance(
                wallet=info["owner"],
                mint=mint,
                balance=int(info["tokenAmount"]["amount"]),
                decimals=info["tokenAmount"]["decimals"],
                timestamp=now,
            )
            balances.append(balance)
            total_supply += balance.balance

        if limit:
            balances = balances[:limit]

        return HolderSnapshot(
            mint=mint,
            timestamp=now,
            total_supply=total_supply,
            holder_count=len(balances),
            balances=balances,
        )

    async def get_wallet_metadata(self, wallet: WalletAddress) -> WalletMetadata:
        """Fetch wallet metadata via RPC."""
        return WalletMetadata(
            address=wallet,
            first_seen_at=datetime.now(timezone.utc),
        )

    async def get_funding_edges(
        self,
        wallets: list[WalletAddress],
    ) -> AsyncIterator[FundingEdge]:
        """Not efficiently supported by basic RPC."""
        logger.warning("funding_edges_not_supported_by_rpc")
        return
        yield  # Make this a generator

    async def get_historical_balances(
        self,
        wallet: WalletAddress,
        mint: TokenMint,
        *,
        since: datetime | None = None,
    ) -> list[TokenBalance]:
        """Not supported by basic RPC."""
        return []

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
