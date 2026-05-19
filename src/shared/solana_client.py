"""Solana Client Adapter - Interface to Solana RPC/Helius APIs.

Wraps existing SHI infrastructure or provides standalone implementation.
Includes rate limiting, retries, and error handling.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

# Add parent SHI path for imports
sys.path.insert(0, str(__file__).rsplit("/sweenee", 1)[0])

logger = structlog.get_logger()

# Default configuration
DEFAULT_RPC_URL = "https://api.mainnet-beta.solana.com"
HELIUS_RPC_TEMPLATE = "https://mainnet.helius-rpc.com/?api-key={}"
RATE_LIMIT_PER_SECOND = 5  # Conservative for Helius free tier
MAX_RETRIES = 3
RETRY_DELAY = 2.0


@dataclass
class TokenBalance:
    """Token balance for a wallet."""

    wallet_address: str
    token_mint: str
    raw_amount: int
    decimals: int
    ui_amount: float
    fetched_at: datetime

    @classmethod
    def zero(cls, wallet: str, mint: str, decimals: int = 6) -> "TokenBalance":
        """Create a zero balance."""
        return cls(
            wallet_address=wallet,
            token_mint=mint,
            raw_amount=0,
            decimals=decimals,
            ui_amount=0.0,
            fetched_at=datetime.now(timezone.utc),
        )


@dataclass
class TransactionSignature:
    """Transaction signature with metadata."""

    signature: str
    slot: int
    block_time: datetime | None
    err: dict | None = None

    @property
    def explorer_url(self) -> str:
        """Solscan URL for this transaction."""
        return f"https://solscan.io/tx/{self.signature}"


class SolanaClient:
    """Async Solana RPC client with rate limiting and retries."""

    def __init__(
        self,
        rpc_url: str | None = None,
        helius_api_key: str | None = None,
        rate_limit: int = RATE_LIMIT_PER_SECOND,
        max_retries: int = MAX_RETRIES,
    ):
        # Determine RPC URL
        self.helius_api_key = helius_api_key or os.getenv("HELIUS_API_KEY", "")
        if self.helius_api_key:
            self.rpc_url = HELIUS_RPC_TEMPLATE.format(self.helius_api_key)
        else:
            self.rpc_url = rpc_url or os.getenv("SOLANA_RPC_URL", DEFAULT_RPC_URL)

        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self._request_times: list[float] = []
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _rate_limit(self):
        """Enforce rate limiting."""
        now = asyncio.get_event_loop().time()
        # Remove old timestamps
        self._request_times = [t for t in self._request_times if now - t < 1.0]

        if len(self._request_times) >= self.rate_limit:
            wait_time = 1.0 - (now - self._request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self._request_times.append(now)

    async def _rpc_call(
        self, method: str, params: list[Any], retries: int = 0
    ) -> dict[str, Any]:
        """Make an RPC call with retries."""
        await self._rate_limit()

        client = await self._get_client()
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }

        try:
            response = await client.post(self.rpc_url, json=payload)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                error = result["error"]
                logger.warning(
                    "rpc_error",
                    method=method,
                    error=error,
                )
                # Retry on rate limit errors
                if error.get("code") == 429 and retries < self.max_retries:
                    await asyncio.sleep(RETRY_DELAY * (retries + 1))
                    return await self._rpc_call(method, params, retries + 1)
                raise RuntimeError(f"RPC error: {error}")

            return result.get("result", {})

        except httpx.HTTPStatusError as e:
            if retries < self.max_retries:
                await asyncio.sleep(RETRY_DELAY * (retries + 1))
                return await self._rpc_call(method, params, retries + 1)
            raise
        except Exception as e:
            logger.error("rpc_call_failed", method=method, error=str(e))
            if retries < self.max_retries:
                await asyncio.sleep(RETRY_DELAY * (retries + 1))
                return await self._rpc_call(method, params, retries + 1)
            raise

    async def get_token_balance(
        self, wallet: str, mint: str, decimals: int = 6
    ) -> TokenBalance:
        """Get token balance for a wallet.

        Uses getTokenAccountsByOwner to find all token accounts for the mint.
        """
        try:
            result = await self._rpc_call(
                "getTokenAccountsByOwner",
                [
                    wallet,
                    {"mint": mint},
                    {"encoding": "jsonParsed"},
                ],
            )

            accounts = result.get("value", [])
            total_raw = 0
            actual_decimals = decimals

            for account in accounts:
                parsed = account.get("account", {}).get("data", {}).get("parsed", {})
                info = parsed.get("info", {})
                token_amount = info.get("tokenAmount", {})

                total_raw += int(token_amount.get("amount", 0))
                actual_decimals = int(token_amount.get("decimals", decimals))

            ui_amount = total_raw / (10**actual_decimals)

            return TokenBalance(
                wallet_address=wallet,
                token_mint=mint,
                raw_amount=total_raw,
                decimals=actual_decimals,
                ui_amount=ui_amount,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.warning(
                "balance_fetch_failed",
                wallet=wallet[:8],
                mint=mint[:8],
                error=str(e),
            )
            return TokenBalance.zero(wallet, mint, decimals)

    async def get_token_balances_batch(
        self, wallets: list[str], mint: str, decimals: int = 6
    ) -> list[TokenBalance]:
        """Get token balances for multiple wallets."""
        tasks = [self.get_token_balance(w, mint, decimals) for w in wallets]
        return await asyncio.gather(*tasks)

    async def get_signatures_for_address(
        self, address: str, limit: int = 100, before: str | None = None
    ) -> list[TransactionSignature]:
        """Get recent transaction signatures for an address."""
        params: list[Any] = [address, {"limit": limit}]
        if before:
            params[1]["before"] = before

        try:
            result = await self._rpc_call("getSignaturesForAddress", params)

            signatures = []
            for sig_info in result:
                block_time = None
                if sig_info.get("blockTime"):
                    block_time = datetime.fromtimestamp(
                        sig_info["blockTime"], tz=timezone.utc
                    )

                signatures.append(
                    TransactionSignature(
                        signature=sig_info["signature"],
                        slot=sig_info.get("slot", 0),
                        block_time=block_time,
                        err=sig_info.get("err"),
                    )
                )

            return signatures

        except Exception as e:
            logger.warning(
                "signatures_fetch_failed",
                address=address[:8],
                error=str(e),
            )
            return []

    async def get_transaction(self, signature: str) -> dict[str, Any] | None:
        """Get parsed transaction details."""
        try:
            result = await self._rpc_call(
                "getTransaction",
                [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
            )
            return result
        except Exception as e:
            logger.warning(
                "transaction_fetch_failed",
                signature=signature[:16],
                error=str(e),
            )
            return None


# Singleton instance
_client: SolanaClient | None = None


def get_client() -> SolanaClient:
    """Get or create the Solana client singleton."""
    global _client
    if _client is None:
        _client = SolanaClient()
    return _client
