"""
Exit Event Classifier (Sprint 9).

Classifies token balance decreases into specific exit types to enable
accurate realised PnL computation.

HARD RULES:
1. Balance decrease alone is NOT a sell
2. Realised PnL requires sell confidence
3. LP actions must NOT be treated as sells
4. Transfers must NOT generate realised PnL unless later sale observed
5. CEX deposits are uncertain exits unless sale can be inferred
6. Low reliability PnL must NOT display precise values
7. All classifications must include confidence and evidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


# ============================================================================
# Exit Event Types
# ============================================================================


class ExitEventType(str, Enum):
    """Classification of token balance decrease events."""

    DEX_SELL = "dex_sell"  # Swap on DEX with quote asset received
    TRANSFER_OUT = "transfer_out"  # Simple transfer to another wallet
    CEX_DEPOSIT = "cex_deposit"  # Transfer to known/suspected CEX address
    LP_ADD = "lp_add"  # Add liquidity to pool (LP tokens minted)
    LP_REMOVE = "lp_remove"  # Remove liquidity from pool
    BURN = "burn"  # Token burned (sent to burn address)
    BRIDGE = "bridge"  # Cross-chain bridge transfer
    WALLET_MIGRATION = "wallet_migration"  # Transfer to related/owned wallet
    PROGRAM_INTERACTION = "program_interaction"  # Unknown program interaction
    UNKNOWN_EXIT = "unknown_exit"  # Cannot classify with confidence


# ============================================================================
# Known Addresses and Program IDs
# ============================================================================


# DEX Program IDs
DEX_PROGRAMS = {
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "jupiter_v6",
    "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB": "jupiter_v4",
    "JUP3c2Uh3WA4Ng34tw6kPd2G4C5BB21Xo36Je1s32Ph": "jupiter_v3",
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "raydium_amm",
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK": "raydium_clmm",
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "orca_whirlpool",
    "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "orca_v2",
    "DjVE6JNiYqPL2QXyCUUh8rNjHrbz9hXHNYt99MQ59qw1": "orca_v1",
    "srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX": "openbook",
    "PhoeNiXZ8ByJGLkxNfZRnkUfjvmuYqLR89jjFHGqdXY": "phoenix",
}

# LP/AMM Program IDs
LP_PROGRAMS = {
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "raydium_amm",
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK": "raydium_clmm",
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "orca_whirlpool",
    "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "orca_v2",
    "MERLuDFBMmsHnsBPZw2sDQZHvXFMwp8EdjudcU2HKky": "mercurial",
    "SSwpkEEcbUqx4vtoEByFjSkhKdCT862DNVb52nZg1UZ": "saber",
}

# Bridge Program IDs
BRIDGE_PROGRAMS = {
    "wormDTUJ6AWPNvk59vGQbDvGJmqbDTdgWgAqcLBCgUb": "wormhole",
    "worm2ZoG2kUd4vFXhvjh93UUH596ayRfgQ2MgjNMTth": "wormhole_v2",
    "DeBr1pTRLNxMKVaQJR4i5sNRNNcV8K5bRPpgqjxh5gZQ": "debridge",
    "3u8hJUVTA4jH1wYAyUur7FFZVQ8H635K3tSHHF4ssjQ5": "allbridge",
}

# Known CEX deposit addresses (partial list - would be expanded)
KNOWN_CEX_ADDRESSES = {
    # Binance
    "2ojv9BAiHUrvsm9gxDe7fJSzbNZSJcxZvf8dqmWGHG8S": "binance",
    "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9": "binance",
    # FTX (defunct but historical)
    "CuieVDEDtLo7FypA9SbLM9saXFdb1dsshEkyErMqkRQq": "ftx",
    # Coinbase
    "H8sMJSCQxfKiFTCfDR3DUMLPwcRbM61LGFJ8N4dK3WjS": "coinbase",
    # Kraken
    "GJRs4FwHtemZ5ZE9x3FNvJ8TMwitKTh21yxdRPqn7npE": "kraken",
    # OKX
    "5VCwKtCXgCJ6kit5FybXjvriW3xELsFDhYrPSqtJNmcD": "okx",
}

# Burn addresses
BURN_ADDRESSES = {
    "1nc1nerator11111111111111111111111111111111": "burn",
    "11111111111111111111111111111111": "system_program",
}

# Token Program IDs
TOKEN_PROGRAMS = {
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA": "token_program",
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb": "token_2022",
}


# ============================================================================
# Data Structures
# ============================================================================


@dataclass(frozen=True)
class ExitEvidence:
    """Evidence supporting an exit classification."""

    # Transaction context
    signature: str
    slot: int
    block_time: datetime | None

    # Token movement
    token_mint: str
    token_amount: int  # Raw amount (pre-decimals)
    token_decimals: int

    # Program detection
    program_ids_detected: tuple[str, ...]
    dex_detected: str | None
    lp_program_detected: str | None
    bridge_detected: str | None

    # Quote asset movement
    sol_change_lamports: int  # Positive = received, negative = sent
    has_quote_asset_received: bool
    quote_asset_mint: str | None
    quote_asset_amount: int | None

    # Destination analysis
    destination_address: str | None
    destination_is_known_cex: bool
    destination_cex_name: str | None
    destination_is_burn_address: bool
    destination_is_high_fan_in: bool  # Many wallets send to this address

    # LP token movement
    lp_token_minted: bool
    lp_token_burned: bool
    lp_token_amount: int | None

    # Related wallet signals
    destination_shares_funder: bool
    destination_has_same_token: bool
    rapid_followup_detected: bool  # Token appears at destination quickly


@dataclass(frozen=True)
class ExitEventClassification:
    """Complete classification of an exit event."""

    # Classification result
    exit_type: ExitEventType
    confidence: float  # 0.0-1.0

    # Evidence
    evidence: ExitEvidence

    # Derived metrics
    sell_confidence_score: float  # Specific confidence this is a true sell
    pnl_computable: bool  # Whether realised PnL can be computed

    # Related wallet (for migrations/transfers)
    downstream_address: str | None
    downstream_wallet_type: str | None  # "cex", "wallet", "pool", "burn", "unknown"

    # Reasoning
    classification_reason: str
    confidence_factors: tuple[str, ...]  # What contributed to confidence


@dataclass
class ExitClassifierConfig:
    """Configuration for exit classifier."""

    # Confidence thresholds
    min_sell_confidence_for_pnl: float = 0.7
    min_transfer_confidence: float = 0.5
    min_lp_confidence: float = 0.6

    # Detection thresholds
    min_sol_movement_for_swap: int = 10_000_000  # 0.01 SOL in lamports
    high_fan_in_threshold: int = 100  # Addresses with >100 incoming transfers

    # Feature flags
    use_cex_detection: bool = True
    use_migration_detection: bool = True
    use_lp_detection: bool = True
    use_bridge_detection: bool = True


# ============================================================================
# Exit Event Classifier
# ============================================================================


class ExitEventClassifier:
    """
    Classifies token balance decreases into specific exit types.

    This enables accurate realised PnL computation by distinguishing
    true sells from transfers, LP actions, CEX deposits, etc.
    """

    def __init__(
        self,
        config: ExitClassifierConfig | None = None,
        cex_addresses: dict[str, str] | None = None,
        high_fan_in_addresses: set[str] | None = None,
    ):
        """
        Initialize the classifier.

        Args:
            config: Classifier configuration
            cex_addresses: Additional CEX addresses (address -> cex_name)
            high_fan_in_addresses: Known high fan-in addresses
        """
        self._config = config or ExitClassifierConfig()
        self._cex_addresses = {**KNOWN_CEX_ADDRESSES, **(cex_addresses or {})}
        self._high_fan_in_addresses = high_fan_in_addresses or set()

    def classify(
        self,
        wallet_address: str,
        token_mint: str,
        token_amount: int,
        token_decimals: int,
        tx_data: dict[str, Any],
        related_wallet_info: dict[str, Any] | None = None,
    ) -> ExitEventClassification:
        """
        Classify a token balance decrease event.

        Args:
            wallet_address: Source wallet address
            token_mint: Token mint being exited
            token_amount: Raw token amount (negative for exits)
            token_decimals: Token decimals
            tx_data: Full transaction data from RPC
            related_wallet_info: Optional info about destination wallet

        Returns:
            Complete classification with confidence and evidence
        """
        # Build evidence from transaction data
        evidence = self._extract_evidence(
            wallet_address=wallet_address,
            token_mint=token_mint,
            token_amount=abs(token_amount),
            token_decimals=token_decimals,
            tx_data=tx_data,
            related_wallet_info=related_wallet_info,
        )

        # Run classification pipeline
        exit_type, confidence, reason, factors = self._classify_exit(evidence)

        # Compute sell confidence
        sell_confidence = self._compute_sell_confidence(evidence, exit_type)

        # Determine if PnL is computable
        pnl_computable = (
            exit_type == ExitEventType.DEX_SELL
            and sell_confidence >= self._config.min_sell_confidence_for_pnl
        )

        # Determine downstream wallet type
        downstream_type = self._get_downstream_type(evidence)

        return ExitEventClassification(
            exit_type=exit_type,
            confidence=confidence,
            evidence=evidence,
            sell_confidence_score=sell_confidence,
            pnl_computable=pnl_computable,
            downstream_address=evidence.destination_address,
            downstream_wallet_type=downstream_type,
            classification_reason=reason,
            confidence_factors=tuple(factors),
        )

    def _extract_evidence(
        self,
        wallet_address: str,
        token_mint: str,
        token_amount: int,
        token_decimals: int,
        tx_data: dict[str, Any],
        related_wallet_info: dict[str, Any] | None,
    ) -> ExitEvidence:
        """Extract evidence from transaction data."""
        # Basic transaction info
        signature = tx_data.get("transaction", {}).get("signatures", [""])[0]
        slot = tx_data.get("slot", 0)
        block_time_unix = tx_data.get("blockTime")
        block_time = (
            datetime.fromtimestamp(block_time_unix, tz=timezone.utc)
            if block_time_unix
            else None
        )

        # Extract program IDs from account keys
        account_keys = (
            tx_data.get("transaction", {})
            .get("message", {})
            .get("accountKeys", [])
        )
        program_ids = []
        for key in account_keys:
            key_str = key.get("pubkey") if isinstance(key, dict) else key
            if key_str:
                program_ids.append(key_str)

        # Detect DEX, LP, Bridge programs
        dex_detected = None
        lp_program_detected = None
        bridge_detected = None

        for prog_id in program_ids:
            if prog_id in DEX_PROGRAMS and not dex_detected:
                dex_detected = DEX_PROGRAMS[prog_id]
            if prog_id in LP_PROGRAMS and not lp_program_detected:
                lp_program_detected = LP_PROGRAMS[prog_id]
            if prog_id in BRIDGE_PROGRAMS and not bridge_detected:
                bridge_detected = BRIDGE_PROGRAMS[prog_id]

        # Analyze SOL movement
        meta = tx_data.get("meta", {})
        pre_balances = meta.get("preBalances", [])
        post_balances = meta.get("postBalances", [])

        sol_change = 0
        wallet_idx = None
        for idx, key in enumerate(account_keys):
            key_str = key.get("pubkey") if isinstance(key, dict) else key
            if key_str == wallet_address:
                wallet_idx = idx
                break

        if wallet_idx is not None and wallet_idx < len(pre_balances) and wallet_idx < len(post_balances):
            sol_change = post_balances[wallet_idx] - pre_balances[wallet_idx]

        has_quote_received = sol_change > self._config.min_sol_movement_for_swap

        # Find destination address from token transfers
        destination_address = None
        post_token_balances = meta.get("postTokenBalances", [])
        pre_token_balances = meta.get("preTokenBalances", [])

        # Look for wallet that received this token
        for post_bal in post_token_balances:
            if post_bal.get("mint") == token_mint:
                owner = post_bal.get("owner", "")
                if owner and owner != wallet_address:
                    # Check if this is an increase
                    pre_amount = 0
                    for pre_bal in pre_token_balances:
                        if pre_bal.get("mint") == token_mint and pre_bal.get("owner") == owner:
                            pre_amount = int(pre_bal.get("uiTokenAmount", {}).get("amount", 0))
                            break
                    post_amount = int(post_bal.get("uiTokenAmount", {}).get("amount", 0))
                    if post_amount > pre_amount:
                        destination_address = owner
                        break

        # Check destination characteristics
        destination_is_cex = destination_address in self._cex_addresses if destination_address else False
        destination_cex_name = self._cex_addresses.get(destination_address) if destination_is_cex else None
        destination_is_burn = destination_address in BURN_ADDRESSES if destination_address else False
        destination_is_high_fan_in = destination_address in self._high_fan_in_addresses if destination_address else False

        # Check for LP token movements
        lp_token_minted = False
        lp_token_burned = False
        lp_token_amount = None

        # Look for LP token mint/burn patterns
        for post_bal in post_token_balances:
            owner = post_bal.get("owner", "")
            mint = post_bal.get("mint", "")
            if owner == wallet_address and mint != token_mint:
                # Check if this could be an LP token
                post_amount = int(post_bal.get("uiTokenAmount", {}).get("amount", 0))
                pre_amount = 0
                for pre_bal in pre_token_balances:
                    if pre_bal.get("mint") == mint and pre_bal.get("owner") == owner:
                        pre_amount = int(pre_bal.get("uiTokenAmount", {}).get("amount", 0))
                        break
                if post_amount > pre_amount:
                    # Wallet received tokens (could be LP tokens)
                    if lp_program_detected:
                        lp_token_minted = True
                        lp_token_amount = post_amount - pre_amount
                elif post_amount < pre_amount:
                    if lp_program_detected:
                        lp_token_burned = True
                        lp_token_amount = pre_amount - post_amount

        # Related wallet signals
        shares_funder = False
        has_same_token = False
        rapid_followup = False

        if related_wallet_info:
            shares_funder = related_wallet_info.get("shares_funder", False)
            has_same_token = related_wallet_info.get("has_same_token", False)
            rapid_followup = related_wallet_info.get("rapid_followup", False)

        return ExitEvidence(
            signature=signature,
            slot=slot,
            block_time=block_time,
            token_mint=token_mint,
            token_amount=token_amount,
            token_decimals=token_decimals,
            program_ids_detected=tuple(program_ids),
            dex_detected=dex_detected,
            lp_program_detected=lp_program_detected,
            bridge_detected=bridge_detected,
            sol_change_lamports=sol_change,
            has_quote_asset_received=has_quote_received,
            quote_asset_mint=None,  # Would need more parsing for specific quote
            quote_asset_amount=sol_change if has_quote_received else None,
            destination_address=destination_address,
            destination_is_known_cex=destination_is_cex,
            destination_cex_name=destination_cex_name,
            destination_is_burn_address=destination_is_burn,
            destination_is_high_fan_in=destination_is_high_fan_in,
            lp_token_minted=lp_token_minted,
            lp_token_burned=lp_token_burned,
            lp_token_amount=lp_token_amount,
            destination_shares_funder=shares_funder,
            destination_has_same_token=has_same_token,
            rapid_followup_detected=rapid_followup,
        )

    def _classify_exit(
        self,
        evidence: ExitEvidence,
    ) -> tuple[ExitEventType, float, str, list[str]]:
        """
        Classify exit based on evidence.

        Returns:
            (exit_type, confidence, reason, confidence_factors)
        """
        factors = []

        # Priority 1: BURN
        if evidence.destination_is_burn_address:
            factors.append("destination_is_burn_address")
            return ExitEventType.BURN, 0.95, "Token sent to burn address", factors

        # Priority 2: LP_ADD (token out + LP token minted)
        if evidence.lp_token_minted and evidence.lp_program_detected:
            factors.append("lp_token_minted")
            factors.append(f"lp_program:{evidence.lp_program_detected}")
            confidence = 0.9
            return ExitEventType.LP_ADD, confidence, "LP tokens minted in same transaction", factors

        # Priority 3: LP_REMOVE (might have token out + other tokens in)
        if evidence.lp_token_burned and evidence.lp_program_detected:
            factors.append("lp_token_burned")
            factors.append(f"lp_program:{evidence.lp_program_detected}")
            return ExitEventType.LP_REMOVE, 0.85, "LP tokens burned in same transaction", factors

        # Priority 4: DEX_SELL (swap with quote received)
        if evidence.dex_detected and evidence.has_quote_asset_received:
            factors.append(f"dex_detected:{evidence.dex_detected}")
            factors.append("quote_asset_received")
            confidence = 0.9
            if evidence.sol_change_lamports > 100_000_000:  # > 0.1 SOL
                factors.append("significant_sol_received")
                confidence = 0.95
            return ExitEventType.DEX_SELL, confidence, f"DEX swap via {evidence.dex_detected} with SOL received", factors

        # Priority 5: BRIDGE
        if evidence.bridge_detected:
            factors.append(f"bridge_detected:{evidence.bridge_detected}")
            return ExitEventType.BRIDGE, 0.85, f"Bridge transfer via {evidence.bridge_detected}", factors

        # Priority 6: CEX_DEPOSIT
        if evidence.destination_is_known_cex:
            factors.append(f"known_cex:{evidence.destination_cex_name}")
            return ExitEventType.CEX_DEPOSIT, 0.9, f"Transfer to known CEX: {evidence.destination_cex_name}", factors

        if evidence.destination_is_high_fan_in and not evidence.dex_detected:
            factors.append("high_fan_in_destination")
            return ExitEventType.CEX_DEPOSIT, 0.6, "Transfer to high fan-in address (likely CEX)", factors

        # Priority 7: WALLET_MIGRATION
        if evidence.destination_shares_funder or evidence.rapid_followup_detected:
            factors_list = []
            confidence = 0.5

            if evidence.destination_shares_funder:
                factors_list.append("shares_funder")
                confidence += 0.2

            if evidence.rapid_followup_detected:
                factors_list.append("rapid_followup")
                confidence += 0.15

            if evidence.destination_has_same_token:
                factors_list.append("destination_has_token")
                confidence += 0.1

            factors.extend(factors_list)
            return ExitEventType.WALLET_MIGRATION, min(confidence, 0.85), "Possible wallet migration", factors

        # Priority 8: PROGRAM_INTERACTION (unknown program but not a simple transfer)
        unknown_programs = [
            p for p in evidence.program_ids_detected
            if p not in DEX_PROGRAMS
            and p not in LP_PROGRAMS
            and p not in BRIDGE_PROGRAMS
            and p not in TOKEN_PROGRAMS
        ]
        if unknown_programs and not evidence.destination_address:
            factors.append("unknown_program_interaction")
            return ExitEventType.PROGRAM_INTERACTION, 0.5, "Interaction with unknown program", factors

        # Priority 9: TRANSFER_OUT (simple transfer)
        if evidence.destination_address:
            factors.append("has_destination_address")
            if not evidence.dex_detected:
                factors.append("no_dex_detected")
            return ExitEventType.TRANSFER_OUT, 0.7, "Simple token transfer to another wallet", factors

        # Priority 10: UNKNOWN_EXIT
        return ExitEventType.UNKNOWN_EXIT, 0.3, "Cannot classify exit with confidence", factors

    def _compute_sell_confidence(
        self,
        evidence: ExitEvidence,
        exit_type: ExitEventType,
    ) -> float:
        """
        Compute confidence that this exit is a true sell.

        This is the critical score for determining if realised PnL
        should be computed.
        """
        if exit_type != ExitEventType.DEX_SELL:
            # Non-sells have low sell confidence by definition
            return 0.1

        score = 0.0

        # Evidence factors for sell confidence
        if evidence.dex_detected:
            score += 0.25

        if evidence.has_quote_asset_received:
            score += 0.25

        if evidence.sol_change_lamports > self._config.min_sol_movement_for_swap:
            score += 0.15
            # More SOL = more confidence
            if evidence.sol_change_lamports > 100_000_000:  # > 0.1 SOL
                score += 0.1
            if evidence.sol_change_lamports > 1_000_000_000:  # > 1 SOL
                score += 0.1

        # Negative evidence
        if evidence.destination_is_known_cex:
            score -= 0.3  # Could be CEX deposit, not swap

        if evidence.lp_token_minted:
            score -= 0.4  # LP action, not sell

        if evidence.destination_shares_funder:
            score -= 0.2  # Could be migration

        # Ensure bounds
        return max(0.0, min(1.0, score))

    def _get_downstream_type(self, evidence: ExitEvidence) -> str | None:
        """Determine the type of downstream address."""
        if not evidence.destination_address:
            return None

        if evidence.destination_is_burn_address:
            return "burn"

        if evidence.destination_is_known_cex:
            return "cex"

        if evidence.lp_program_detected:
            return "pool"

        if evidence.destination_shares_funder:
            return "related_wallet"

        return "wallet"


# ============================================================================
# Sell Confidence Scorer (Detailed)
# ============================================================================


class SellConfidenceScorer:
    """
    Computes detailed sell confidence score for exit events.

    Uses multiple evidence factors to determine how confident we are
    that a token balance decrease represents a true sale.
    """

    def __init__(
        self,
        min_confidence_for_pnl: float = 0.7,
    ):
        self._min_confidence_for_pnl = min_confidence_for_pnl

    def compute_score(
        self,
        classification: ExitEventClassification,
    ) -> tuple[float, bool, dict[str, float]]:
        """
        Compute detailed sell confidence score.

        Args:
            classification: Exit event classification

        Returns:
            (score, pnl_computable, score_breakdown)
        """
        evidence = classification.evidence

        # Score breakdown by factor
        breakdown = {
            "swap_instruction": 0.0,
            "dex_program": 0.0,
            "quote_received": 0.0,
            "token_decrease": 0.0,
            "counterparty_type": 0.0,
            "lp_token_movement": 0.0,
            "destination_type": 0.0,
            "cex_address": 0.0,
            "transaction_route": 0.0,
        }

        # Factor 1: Swap instruction detected
        if evidence.dex_detected:
            breakdown["swap_instruction"] = 0.15
            breakdown["dex_program"] = 0.15

        # Factor 2: Quote asset received
        if evidence.has_quote_asset_received:
            breakdown["quote_received"] = 0.20
            if evidence.sol_change_lamports > 100_000_000:
                breakdown["quote_received"] = 0.25

        # Factor 3: Token balance decreased
        if evidence.token_amount > 0:
            breakdown["token_decrease"] = 0.10

        # Factor 4: Counterparty type (negative if suspicious)
        if evidence.destination_is_known_cex:
            breakdown["cex_address"] = -0.20
        elif evidence.destination_is_high_fan_in:
            breakdown["destination_type"] = -0.10

        # Factor 5: LP token movement (strong negative)
        if evidence.lp_token_minted:
            breakdown["lp_token_movement"] = -0.30
        elif evidence.lp_token_burned:
            breakdown["lp_token_movement"] = -0.20

        # Factor 6: Transaction route
        if not evidence.bridge_detected and evidence.dex_detected:
            breakdown["transaction_route"] = 0.10

        # Sum up
        total_score = sum(breakdown.values())
        total_score = max(0.0, min(1.0, total_score))

        pnl_computable = (
            classification.exit_type == ExitEventType.DEX_SELL
            and total_score >= self._min_confidence_for_pnl
        )

        return total_score, pnl_computable, breakdown


# ============================================================================
# PnL Reliability Scorer
# ============================================================================


class PnLReliabilityScorer:
    """
    Computes PnL reliability score for realised PnL estimates.

    Based on:
    - Sell confidence
    - Price confidence
    - Liquidity confidence
    - Accounting lot quality
    - Transfer ambiguity
    - Event completeness
    """

    def __init__(
        self,
        weight_sell_confidence: float = 0.25,
        weight_price_confidence: float = 0.20,
        weight_liquidity_confidence: float = 0.15,
        weight_lot_quality: float = 0.15,
        weight_transfer_ambiguity: float = 0.15,
        weight_event_completeness: float = 0.10,
    ):
        self._weights = {
            "sell_confidence": weight_sell_confidence,
            "price_confidence": weight_price_confidence,
            "liquidity_confidence": weight_liquidity_confidence,
            "lot_quality": weight_lot_quality,
            "transfer_ambiguity": weight_transfer_ambiguity,
            "event_completeness": weight_event_completeness,
        }

    def compute_reliability(
        self,
        sell_confidence: float,
        entry_price_confidence: float,
        exit_price_confidence: float,
        liquidity_confidence: float,
        lot_count: int,
        has_transfer_ambiguity: bool,
        event_completeness: float,
    ) -> tuple[float, str, dict[str, float]]:
        """
        Compute PnL reliability score.

        Args:
            sell_confidence: Confidence this is a true sell
            entry_price_confidence: Confidence in entry price
            exit_price_confidence: Confidence in exit price
            liquidity_confidence: Confidence in liquidity at exit
            lot_count: Number of cost basis lots used
            has_transfer_ambiguity: Whether transfers are involved
            event_completeness: How complete the event data is (0-1)

        Returns:
            (reliability_score, display_mode, component_scores)
        """
        # Compute component scores
        price_confidence = min(entry_price_confidence, exit_price_confidence)

        # Lot quality: more lots = slightly lower quality (more uncertainty)
        lot_quality = max(0.5, 1.0 - (lot_count - 1) * 0.1)

        # Transfer ambiguity penalty
        transfer_score = 0.3 if has_transfer_ambiguity else 1.0

        components = {
            "sell_confidence": sell_confidence,
            "price_confidence": price_confidence,
            "liquidity_confidence": liquidity_confidence,
            "lot_quality": lot_quality,
            "transfer_ambiguity": transfer_score,
            "event_completeness": event_completeness,
        }

        # Weighted sum
        reliability = sum(
            components[k] * self._weights[k]
            for k in self._weights
        )

        # Determine display mode
        if reliability >= 0.7:
            display_mode = "precise"
        elif reliability >= 0.4:
            display_mode = "range"
        else:
            display_mode = "unavailable"

        return reliability, display_mode, components


# ============================================================================
# Transfer Chain Detection (Sprint 9 - Task 3)
# ============================================================================


@dataclass(frozen=True)
class TransferChainResult:
    """Result of transfer chain detection."""

    likely_migration: bool
    related_wallet_candidate: str | None
    migration_confidence: float
    chain_length: int
    evidence_factors: tuple[str, ...]
    chain_wallets: tuple[str, ...]  # All wallets in the chain


@dataclass
class TransferChainConfig:
    """Configuration for transfer chain detection."""

    # Time thresholds
    rapid_followup_seconds: int = 300  # 5 minutes
    chain_detection_window_hours: int = 24

    # Confidence thresholds
    min_migration_confidence: float = 0.6
    shared_funder_weight: float = 0.3
    same_token_held_weight: float = 0.2
    rapid_movement_weight: float = 0.25
    no_quote_received_weight: float = 0.15
    same_behavior_weight: float = 0.1

    # Chain limits
    max_chain_depth: int = 5


class TransferChainDetector:
    """
    Detects wallet migration and internal transfer chains.

    Identifies when a token balance decrease is part of a wallet migration
    rather than a true exit, by analyzing:
    - Transfer patterns between related wallets
    - Shared funders (same initial funding source)
    - Same token holding patterns post-transfer
    - Rapid movement timing
    - Absence of quote asset (no swap, just movement)
    """

    def __init__(
        self,
        config: TransferChainConfig | None = None,
    ):
        self._config = config or TransferChainConfig()

    async def detect_migration(
        self,
        source_wallet: str,
        destination_wallet: str,
        token_mint: str,
        transfer_timestamp: datetime,
        wallet_info_provider: "WalletInfoProvider | None" = None,
    ) -> TransferChainResult:
        """
        Detect if a transfer is part of a wallet migration.

        Args:
            source_wallet: Wallet sending tokens
            destination_wallet: Wallet receiving tokens
            token_mint: Token being transferred
            transfer_timestamp: When the transfer occurred
            wallet_info_provider: Optional provider for wallet info lookups

        Returns:
            TransferChainResult with migration assessment
        """
        evidence_factors = []
        confidence = 0.0

        # Factor 1: No quote asset received (critical for migration vs sell)
        # If source wallet received quote, it's more likely a sell
        no_quote_info = await self._check_no_quote_received(
            source_wallet, transfer_timestamp, wallet_info_provider
        )
        if no_quote_info["no_quote"]:
            confidence += self._config.no_quote_received_weight
            evidence_factors.append("no_quote_received")

        # Factor 2: Shared funder
        shares_funder, funder_address = await self._check_shared_funder(
            source_wallet, destination_wallet, wallet_info_provider
        )
        if shares_funder:
            confidence += self._config.shared_funder_weight
            evidence_factors.append(f"shared_funder:{funder_address[:8]}...")

        # Factor 3: Destination holds same token (after transfer)
        dest_has_token = await self._check_destination_has_token(
            destination_wallet, token_mint, transfer_timestamp, wallet_info_provider
        )
        if dest_has_token:
            confidence += self._config.same_token_held_weight
            evidence_factors.append("destination_holds_token")

        # Factor 4: Rapid movement (token moves through quickly)
        rapid_movement = await self._check_rapid_movement(
            destination_wallet, token_mint, transfer_timestamp, wallet_info_provider
        )
        if rapid_movement["is_rapid"]:
            confidence += self._config.rapid_movement_weight
            evidence_factors.append(
                f"rapid_movement:{rapid_movement.get('seconds', 0)}s"
            )

        # Factor 5: Similar behavior patterns
        same_behavior = await self._check_same_behavior(
            source_wallet, destination_wallet, wallet_info_provider
        )
        if same_behavior:
            confidence += self._config.same_behavior_weight
            evidence_factors.append("similar_behavior")

        # Detect chain depth
        chain_wallets = await self._trace_chain(
            source_wallet, destination_wallet, token_mint, wallet_info_provider
        )

        # Determine if this is likely a migration
        likely_migration = confidence >= self._config.min_migration_confidence

        return TransferChainResult(
            likely_migration=likely_migration,
            related_wallet_candidate=destination_wallet if likely_migration else None,
            migration_confidence=min(1.0, confidence),
            chain_length=len(chain_wallets),
            evidence_factors=tuple(evidence_factors),
            chain_wallets=tuple(chain_wallets),
        )

    async def _check_no_quote_received(
        self,
        wallet: str,
        timestamp: datetime,
        provider: "WalletInfoProvider | None",
    ) -> dict[str, Any]:
        """Check if wallet received any quote asset (SOL/USDC) around the transfer."""
        if provider is None:
            return {"no_quote": True, "uncertain": True}

        try:
            quote_received = await provider.get_quote_received(wallet, timestamp)
            return {"no_quote": not quote_received, "uncertain": False}
        except Exception:
            return {"no_quote": True, "uncertain": True}

    async def _check_shared_funder(
        self,
        source: str,
        destination: str,
        provider: "WalletInfoProvider | None",
    ) -> tuple[bool, str | None]:
        """Check if wallets share the same initial funder."""
        if provider is None:
            return False, None

        try:
            source_funder = await provider.get_initial_funder(source)
            dest_funder = await provider.get_initial_funder(destination)

            if source_funder and dest_funder and source_funder == dest_funder:
                return True, source_funder

            # Also check if source funded destination or vice versa
            if source_funder == destination or dest_funder == source:
                return True, source if dest_funder == source else destination

            return False, None
        except Exception:
            return False, None

    async def _check_destination_has_token(
        self,
        destination: str,
        token_mint: str,
        transfer_timestamp: datetime,
        provider: "WalletInfoProvider | None",
    ) -> bool:
        """Check if destination wallet holds the token after transfer."""
        if provider is None:
            return False

        try:
            # Check balance shortly after transfer
            check_time = transfer_timestamp.replace(
                second=transfer_timestamp.second + 60
            )
            balance = await provider.get_token_balance(destination, token_mint, check_time)
            return balance is not None and balance > 0
        except Exception:
            return False

    async def _check_rapid_movement(
        self,
        destination: str,
        token_mint: str,
        transfer_timestamp: datetime,
        provider: "WalletInfoProvider | None",
    ) -> dict[str, Any]:
        """Check if token moves out of destination quickly (intermediate hop)."""
        if provider is None:
            return {"is_rapid": False}

        try:
            # Look for outgoing transfer of same token within threshold
            next_movement = await provider.get_next_token_movement(
                destination, token_mint, transfer_timestamp
            )

            if next_movement:
                time_diff = (next_movement["timestamp"] - transfer_timestamp).total_seconds()
                if time_diff <= self._config.rapid_followup_seconds:
                    return {
                        "is_rapid": True,
                        "seconds": time_diff,
                        "next_destination": next_movement.get("destination"),
                    }

            return {"is_rapid": False}
        except Exception:
            return {"is_rapid": False}

    async def _check_same_behavior(
        self,
        source: str,
        destination: str,
        provider: "WalletInfoProvider | None",
    ) -> bool:
        """Check if wallets exhibit similar behavior patterns."""
        if provider is None:
            return False

        try:
            source_behavior = await provider.get_behavior_profile(source)
            dest_behavior = await provider.get_behavior_profile(destination)

            if source_behavior and dest_behavior:
                # Compare trading patterns, timing, token preferences
                similarity = self._compute_behavior_similarity(
                    source_behavior, dest_behavior
                )
                return similarity > 0.7

            return False
        except Exception:
            return False

    def _compute_behavior_similarity(
        self,
        profile_a: dict[str, Any],
        profile_b: dict[str, Any],
    ) -> float:
        """Compute similarity between two behavior profiles."""
        # Simple similarity based on available metrics
        similarity_scores = []

        # Trading frequency similarity
        freq_a = profile_a.get("trades_per_week", 0)
        freq_b = profile_b.get("trades_per_week", 0)
        if freq_a > 0 and freq_b > 0:
            freq_sim = min(freq_a, freq_b) / max(freq_a, freq_b)
            similarity_scores.append(freq_sim)

        # Average hold time similarity
        hold_a = profile_a.get("avg_hold_hours", 0)
        hold_b = profile_b.get("avg_hold_hours", 0)
        if hold_a > 0 and hold_b > 0:
            hold_sim = min(hold_a, hold_b) / max(hold_a, hold_b)
            similarity_scores.append(hold_sim)

        # Position size similarity
        size_a = profile_a.get("avg_position_size_sol", 0)
        size_b = profile_b.get("avg_position_size_sol", 0)
        if size_a > 0 and size_b > 0:
            size_sim = min(size_a, size_b) / max(size_a, size_b)
            similarity_scores.append(size_sim)

        if not similarity_scores:
            return 0.0

        return sum(similarity_scores) / len(similarity_scores)

    async def _trace_chain(
        self,
        source: str,
        destination: str,
        token_mint: str,
        provider: "WalletInfoProvider | None",
    ) -> list[str]:
        """Trace the full chain of transfers for this token."""
        chain = [source, destination]

        if provider is None:
            return chain

        try:
            current = destination
            for _ in range(self._config.max_chain_depth - 2):
                next_dest = await provider.get_next_transfer_destination(
                    current, token_mint
                )
                if next_dest and next_dest not in chain:
                    chain.append(next_dest)
                    current = next_dest
                else:
                    break

            return chain
        except Exception:
            return chain


class WalletInfoProvider:
    """
    Interface for providing wallet information for transfer chain detection.

    This is an abstract interface - concrete implementations would query
    the actual data sources (RPC, database, etc.).
    """

    async def get_initial_funder(self, wallet: str) -> str | None:
        """Get the wallet that initially funded this wallet with SOL."""
        raise NotImplementedError

    async def get_token_balance(
        self, wallet: str, token_mint: str, timestamp: datetime
    ) -> int | None:
        """Get token balance at a specific time."""
        raise NotImplementedError

    async def get_next_token_movement(
        self, wallet: str, token_mint: str, after_timestamp: datetime
    ) -> dict[str, Any] | None:
        """Get the next movement of a token from this wallet."""
        raise NotImplementedError

    async def get_behavior_profile(self, wallet: str) -> dict[str, Any] | None:
        """Get the behavior profile for a wallet."""
        raise NotImplementedError

    async def get_next_transfer_destination(
        self, wallet: str, token_mint: str
    ) -> str | None:
        """Get the destination of the next transfer of this token."""
        raise NotImplementedError

    async def get_quote_received(self, wallet: str, timestamp: datetime) -> bool:
        """Check if wallet received quote asset (SOL/USDC) around timestamp."""
        raise NotImplementedError


# ============================================================================
# LP Action Separation (Sprint 9 - Task 4)
# ============================================================================


@dataclass(frozen=True)
class LPActionResult:
    """Result of LP action detection."""

    is_lp_action: bool
    action_type: str | None  # "add_liquidity", "remove_liquidity", "stake", "unstake"
    lp_program: str | None
    lp_token_mint: str | None
    lp_token_amount: int | None
    pool_address: str | None
    confidence: float
    evidence_factors: tuple[str, ...]


class LPActionDetector:
    """
    Detects LP (liquidity provider) actions that should NOT be treated as sells.

    LP actions include:
    - Adding liquidity (token out, LP token in)
    - Removing liquidity (LP token out, tokens in)
    - Staking LP tokens
    - Unstaking LP tokens

    HARD RULE: LP actions must NOT be treated as sells for PnL purposes.
    """

    # Known LP token patterns (partial mints that indicate LP tokens)
    LP_TOKEN_SUFFIXES = ["LP", "lp"]

    def __init__(self):
        self._lp_programs = LP_PROGRAMS.copy()

    def detect_lp_action(
        self,
        evidence: ExitEvidence,
        additional_token_movements: list[dict[str, Any]] | None = None,
    ) -> LPActionResult:
        """
        Detect if an exit event is an LP action.

        Args:
            evidence: Exit evidence from classifier
            additional_token_movements: Other token movements in same tx

        Returns:
            LPActionResult with LP action assessment
        """
        factors = []
        confidence = 0.0
        action_type = None
        lp_token_mint = None
        lp_token_amount = None
        pool_address = None

        # Check 1: LP program detected
        if evidence.lp_program_detected:
            confidence += 0.3
            factors.append(f"lp_program:{evidence.lp_program_detected}")
            pool_address = self._find_pool_address(evidence)

        # Check 2: LP token minted (add liquidity)
        if evidence.lp_token_minted:
            confidence += 0.4
            factors.append("lp_token_minted")
            action_type = "add_liquidity"
            lp_token_amount = evidence.lp_token_amount

            # If we have the mint info, record it
            if additional_token_movements:
                for movement in additional_token_movements:
                    if movement.get("direction") == "in" and self._looks_like_lp_token(
                        movement.get("mint", "")
                    ):
                        lp_token_mint = movement.get("mint")
                        break

        # Check 3: LP token burned (remove liquidity)
        if evidence.lp_token_burned:
            confidence += 0.35
            factors.append("lp_token_burned")
            action_type = "remove_liquidity"
            lp_token_amount = evidence.lp_token_amount

        # Check 4: No quote asset received (distinguishes from swap)
        if not evidence.has_quote_asset_received and evidence.lp_program_detected:
            confidence += 0.15
            factors.append("no_direct_quote_received")

        # Check 5: Destination is pool/program, not wallet
        if not evidence.destination_address and evidence.lp_program_detected:
            confidence += 0.1
            factors.append("tokens_to_program")

        is_lp_action = confidence >= 0.5

        return LPActionResult(
            is_lp_action=is_lp_action,
            action_type=action_type if is_lp_action else None,
            lp_program=evidence.lp_program_detected,
            lp_token_mint=lp_token_mint,
            lp_token_amount=lp_token_amount,
            pool_address=pool_address,
            confidence=min(1.0, confidence),
            evidence_factors=tuple(factors),
        )

    def _find_pool_address(self, evidence: ExitEvidence) -> str | None:
        """Try to identify the pool address from evidence."""
        # Pool would typically be one of the program accounts
        for prog_id in evidence.program_ids_detected:
            if prog_id in self._lp_programs:
                continue  # Skip the program itself
            # Heuristic: could be a PDA derived from the LP program
            # In practice, would need more sophisticated detection
        return None

    def _looks_like_lp_token(self, mint: str) -> bool:
        """Check if a token mint looks like an LP token."""
        # In practice, would check against known LP token registry
        # or detect via token metadata
        for suffix in self.LP_TOKEN_SUFFIXES:
            if mint.endswith(suffix):
                return True
        return False


# ============================================================================
# CEX Deposit Detection (Sprint 9 - Task 5)
# ============================================================================


@dataclass(frozen=True)
class CEXDepositResult:
    """Result of CEX deposit detection."""

    is_cex_deposit: bool
    cex_name: str | None
    deposit_address: str | None
    detection_method: str  # "known_address", "fan_in_pattern", "exchange_label", "uncertain"
    confidence: float
    evidence_factors: tuple[str, ...]


@dataclass
class CEXDetectionConfig:
    """Configuration for CEX deposit detection."""

    # Fan-in thresholds
    high_fan_in_threshold: int = 100  # Addresses with >100 incoming transfers
    very_high_fan_in_threshold: int = 500

    # Confidence weights
    known_address_confidence: float = 0.95
    high_fan_in_confidence: float = 0.6
    very_high_fan_in_confidence: float = 0.75
    exchange_label_confidence: float = 0.85


class CEXDepositDetector:
    """
    Detects transfers to centralized exchanges (CEX).

    CEX deposits are uncertain exits because:
    - The wallet owner may sell on the CEX
    - Or they may just hold on the CEX
    - Or transfer to another wallet from CEX

    HARD RULE: CEX deposits are uncertain exits unless sale can be inferred.
    """

    def __init__(
        self,
        config: CEXDetectionConfig | None = None,
        known_cex_addresses: dict[str, str] | None = None,
        exchange_labels: dict[str, str] | None = None,
    ):
        self._config = config or CEXDetectionConfig()
        self._known_addresses = {**KNOWN_CEX_ADDRESSES, **(known_cex_addresses or {})}
        self._exchange_labels = exchange_labels or {}

    def detect_cex_deposit(
        self,
        destination_address: str,
        fan_in_count: int | None = None,
        address_label: str | None = None,
    ) -> CEXDepositResult:
        """
        Detect if a transfer destination is a CEX deposit address.

        Args:
            destination_address: Address receiving tokens
            fan_in_count: Number of unique addresses that have sent to this address
            address_label: Optional label from address labeling service

        Returns:
            CEXDepositResult with detection assessment
        """
        factors = []

        # Method 1: Known CEX address
        if destination_address in self._known_addresses:
            cex_name = self._known_addresses[destination_address]
            factors.append(f"known_address:{cex_name}")
            return CEXDepositResult(
                is_cex_deposit=True,
                cex_name=cex_name,
                deposit_address=destination_address,
                detection_method="known_address",
                confidence=self._config.known_address_confidence,
                evidence_factors=tuple(factors),
            )

        # Method 2: Exchange label from external service
        if address_label:
            label_lower = address_label.lower()
            for exchange_keyword in ["binance", "coinbase", "ftx", "kraken", "okx", "bybit", "kucoin"]:
                if exchange_keyword in label_lower:
                    factors.append(f"label:{address_label}")
                    return CEXDepositResult(
                        is_cex_deposit=True,
                        cex_name=exchange_keyword,
                        deposit_address=destination_address,
                        detection_method="exchange_label",
                        confidence=self._config.exchange_label_confidence,
                        evidence_factors=tuple(factors),
                    )

        # Method 3: High fan-in pattern
        if fan_in_count is not None:
            if fan_in_count >= self._config.very_high_fan_in_threshold:
                factors.append(f"very_high_fan_in:{fan_in_count}")
                return CEXDepositResult(
                    is_cex_deposit=True,
                    cex_name=None,  # Unknown CEX
                    deposit_address=destination_address,
                    detection_method="fan_in_pattern",
                    confidence=self._config.very_high_fan_in_confidence,
                    evidence_factors=tuple(factors),
                )
            elif fan_in_count >= self._config.high_fan_in_threshold:
                factors.append(f"high_fan_in:{fan_in_count}")
                return CEXDepositResult(
                    is_cex_deposit=True,
                    cex_name=None,  # Unknown CEX
                    deposit_address=destination_address,
                    detection_method="fan_in_pattern",
                    confidence=self._config.high_fan_in_confidence,
                    evidence_factors=tuple(factors),
                )

        # Not detected as CEX
        return CEXDepositResult(
            is_cex_deposit=False,
            cex_name=None,
            deposit_address=destination_address,
            detection_method="uncertain",
            confidence=0.0,
            evidence_factors=tuple(factors) if factors else ("no_cex_indicators",),
        )

    def add_known_address(self, address: str, cex_name: str) -> None:
        """Add a known CEX address to the detector."""
        self._known_addresses[address] = cex_name

    def get_known_cex_names(self) -> set[str]:
        """Get the set of known CEX names."""
        return set(self._known_addresses.values())


# ============================================================================
# Factory Function
# ============================================================================


def create_exit_classifier(
    config: ExitClassifierConfig | None = None,
    additional_cex_addresses: dict[str, str] | None = None,
) -> ExitEventClassifier:
    """Create an exit classifier with default or custom configuration."""
    return ExitEventClassifier(
        config=config,
        cex_addresses=additional_cex_addresses,
    )


def create_transfer_chain_detector(
    config: TransferChainConfig | None = None,
) -> TransferChainDetector:
    """Create a transfer chain detector with default or custom configuration."""
    return TransferChainDetector(config=config)


def create_lp_action_detector() -> LPActionDetector:
    """Create an LP action detector."""
    return LPActionDetector()


def create_cex_deposit_detector(
    config: CEXDetectionConfig | None = None,
    additional_cex_addresses: dict[str, str] | None = None,
) -> CEXDepositDetector:
    """Create a CEX deposit detector with default or custom configuration."""
    return CEXDepositDetector(
        config=config,
        known_cex_addresses=additional_cex_addresses,
    )
