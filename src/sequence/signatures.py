"""Pre-dump signature detection in wallet action sequences.

This module provides detection of behavioral signatures that indicate
potential rug pull or dump patterns based on historical analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

import numpy as np
import numpy.typing as npt
import structlog

from src.core.types import WalletAddress
from .encoder import ActionSequence, WalletActionEncoder, WalletActionType

logger = structlog.get_logger()


class SignatureType(Enum):
    """Types of dump signatures."""

    RAPID_ACCUMULATE_DUMP = "rapid_accumulate_dump"
    SLOW_BLEED = "slow_bleed"
    PUMP_AND_DUMP = "pump_and_dump"
    COORDINATED_EXIT = "coordinated_exit"
    LP_DRAIN = "lp_drain"
    FUNDED_DIRECT_SELL = "funded_direct_sell"
    CYCLIC_WASH = "cyclic_wash"


@dataclass(frozen=True)
class DumpSignature:
    """Definition of a pre-dump behavioral signature.

    Attributes
    ----------
    signature_type : SignatureType
        Type of dump signature.
    pattern : tuple[WalletActionType, ...]
        Expected action pattern.
    description : str
        Human-readable description.
    severity : str
        Risk severity: low, medium, high, critical.
    base_score : float
        Base risk score for this signature (0 to 1).
    timing_weight : float
        How much timing matters (0 to 1).
    """

    signature_type: SignatureType
    pattern: tuple[WalletActionType, ...]
    description: str
    severity: str
    base_score: float
    timing_weight: float

    @property
    def pattern_str(self) -> str:
        """Return pattern as readable string."""
        return " -> ".join(a.value for a in self.pattern)


@dataclass(frozen=True)
class SignatureMatch:
    """Result of signature matching for a wallet.

    Attributes
    ----------
    wallet : WalletAddress
        Wallet that matched.
    signature : DumpSignature
        Matched signature definition.
    confidence : float
        Confidence of match (0 to 1).
    risk_score : float
        Computed risk score (0 to 1).
    matched_positions : tuple[int, ...]
        Positions in sequence where pattern was found.
    context : dict[str, float]
        Additional context metrics.
    """

    wallet: WalletAddress
    signature: DumpSignature
    confidence: float
    risk_score: float
    matched_positions: tuple[int, ...]
    context: dict[str, float] = field(default_factory=dict)

    @property
    def is_high_risk(self) -> bool:
        """Check if this is a high-risk match."""
        return self.risk_score >= 0.7


@dataclass
class SignatureConfig:
    """Configuration for signature detection.

    Attributes
    ----------
    min_confidence : float
        Minimum confidence threshold for matches.
    enable_fuzzy_matching : bool
        Allow approximate pattern matching.
    fuzzy_tolerance : int
        Max allowed mismatches in fuzzy mode.
    recency_weight : float
        Weight given to recent patterns vs older ones.
    """

    min_confidence: float = 0.5
    enable_fuzzy_matching: bool = True
    fuzzy_tolerance: int = 1
    recency_weight: float = 0.7


class DumpSignatureDetector:
    """Detect pre-dump behavioral signatures.

    This class identifies wallet behaviors that historically
    precede rug pulls or coordinated dumps.

    Parameters
    ----------
    config : SignatureConfig | None
        Detection configuration.

    Examples
    --------
    >>> detector = DumpSignatureDetector()
    >>> matches = detector.detect(sequence)
    >>> score = detector.score_dump_likelihood(sequence)
    """

    # Pre-defined dump signatures based on common rug patterns
    KNOWN_SIGNATURES: tuple[DumpSignature, ...] = (
        DumpSignature(
            signature_type=SignatureType.FUNDED_DIRECT_SELL,
            pattern=(
                WalletActionType.FUNDED,
                WalletActionType.SWAP_SELL,
            ),
            description="Wallet receives funding and immediately sells - classic dump pattern",
            severity="critical",
            base_score=0.9,
            timing_weight=0.8,
        ),
        DumpSignature(
            signature_type=SignatureType.RAPID_ACCUMULATE_DUMP,
            pattern=(
                WalletActionType.SWAP_BUY,
                WalletActionType.SWAP_BUY,
                WalletActionType.SWAP_SELL,
                WalletActionType.SWAP_SELL,
            ),
            description="Quick accumulation followed by rapid selling",
            severity="high",
            base_score=0.75,
            timing_weight=0.6,
        ),
        DumpSignature(
            signature_type=SignatureType.LP_DRAIN,
            pattern=(
                WalletActionType.LP_ADD,
                WalletActionType.IDLE,
                WalletActionType.LP_REMOVE,
                WalletActionType.SWAP_SELL,
            ),
            description="Add LP, wait, remove LP, then sell - liquidity drain",
            severity="critical",
            base_score=0.85,
            timing_weight=0.5,
        ),
        DumpSignature(
            signature_type=SignatureType.PUMP_AND_DUMP,
            pattern=(
                WalletActionType.SWAP_BUY,
                WalletActionType.SWAP_BUY,
                WalletActionType.SWAP_BUY,
                WalletActionType.SWAP_SELL,
                WalletActionType.SWAP_SELL,
                WalletActionType.SWAP_SELL,
            ),
            description="Multiple buys to pump price, then aggressive selling",
            severity="high",
            base_score=0.8,
            timing_weight=0.7,
        ),
        DumpSignature(
            signature_type=SignatureType.SLOW_BLEED,
            pattern=(
                WalletActionType.IDLE,
                WalletActionType.SWAP_SELL,
                WalletActionType.IDLE,
                WalletActionType.SWAP_SELL,
            ),
            description="Periodic selling with idle periods - slow dump",
            severity="medium",
            base_score=0.6,
            timing_weight=0.4,
        ),
        DumpSignature(
            signature_type=SignatureType.CYCLIC_WASH,
            pattern=(
                WalletActionType.SWAP_BUY,
                WalletActionType.TRANSFER_OUT,
                WalletActionType.TRANSFER_IN,
                WalletActionType.SWAP_SELL,
            ),
            description="Buy, transfer around, sell - wash trading pattern",
            severity="high",
            base_score=0.7,
            timing_weight=0.5,
        ),
        DumpSignature(
            signature_type=SignatureType.COORDINATED_EXIT,
            pattern=(
                WalletActionType.IDLE,
                WalletActionType.IDLE,
                WalletActionType.SWAP_SELL,
                WalletActionType.SWAP_SELL,
                WalletActionType.TRANSFER_OUT,
            ),
            description="Long idle then sudden sell and exit - coordinated dump",
            severity="critical",
            base_score=0.85,
            timing_weight=0.9,
        ),
    )

    def __init__(self, config: SignatureConfig | None = None) -> None:
        """Initialize detector with configuration."""
        self.config = config or SignatureConfig()
        logger.info(
            "dump_signature_detector_initialized",
            num_signatures=len(self.KNOWN_SIGNATURES),
            min_confidence=self.config.min_confidence,
        )

    def detect(
        self,
        sequence: ActionSequence,
        signatures: Sequence[DumpSignature] | None = None,
    ) -> list[SignatureMatch]:
        """Detect dump signatures in a sequence.

        Parameters
        ----------
        sequence : ActionSequence
            Action sequence to analyze.
        signatures : Sequence[DumpSignature] | None
            Signatures to check. Uses KNOWN_SIGNATURES if None.

        Returns
        -------
        list[SignatureMatch]
            List of matched signatures.
        """
        if len(sequence.actions) == 0:
            return []

        signatures = signatures or self.KNOWN_SIGNATURES
        matches: list[SignatureMatch] = []

        for sig in signatures:
            match_result = self._match_signature(sequence, sig)
            if match_result is not None:
                matches.append(match_result)

        # Sort by risk score
        matches.sort(key=lambda m: m.risk_score, reverse=True)

        logger.debug(
            "signature_detection_complete",
            wallet=sequence.wallet,
            sequence_length=len(sequence.actions),
            matches_found=len(matches),
        )

        return matches

    def detect_batch(
        self,
        sequences: Sequence[ActionSequence],
    ) -> dict[WalletAddress, list[SignatureMatch]]:
        """Detect signatures across multiple sequences.

        Parameters
        ----------
        sequences : Sequence[ActionSequence]
            Sequences to analyze.

        Returns
        -------
        dict[WalletAddress, list[SignatureMatch]]
            Mapping from wallet to matches.
        """
        results: dict[WalletAddress, list[SignatureMatch]] = {}

        for seq in sequences:
            matches = self.detect(seq)
            if matches:
                results[seq.wallet] = matches

        logger.info(
            "batch_detection_complete",
            total_sequences=len(sequences),
            wallets_with_matches=len(results),
        )

        return results

    def score_dump_likelihood(self, sequence: ActionSequence) -> float:
        """Score overall dump likelihood for a sequence.

        Combines all signature matches into a single risk score.

        Parameters
        ----------
        sequence : ActionSequence
            Action sequence to score.

        Returns
        -------
        float
            Dump likelihood score (0 to 1).
        """
        if len(sequence.actions) == 0:
            return 0.0

        matches = self.detect(sequence)

        if not matches:
            # Still check for individual risky actions
            return self._compute_base_risk(sequence)

        # Combine match scores
        weighted_scores: list[float] = []

        for match in matches:
            # Weight by confidence and recency
            recency_factor = self._compute_recency_factor(
                match.matched_positions, len(sequence.actions)
            )
            weighted_score = (
                match.risk_score * match.confidence * recency_factor
            )
            weighted_scores.append(weighted_score)

        # Use noisy-or combination for multiple matches
        if len(weighted_scores) == 1:
            combined = weighted_scores[0]
        else:
            # P(at least one) = 1 - prod(1 - P_i)
            combined = 1 - np.prod([1 - s for s in weighted_scores])

        return float(min(1.0, combined))

    def _match_signature(
        self, sequence: ActionSequence, signature: DumpSignature
    ) -> SignatureMatch | None:
        """Match a single signature against a sequence.

        Parameters
        ----------
        sequence : ActionSequence
            Sequence to check.
        signature : DumpSignature
            Signature to match.

        Returns
        -------
        SignatureMatch | None
            Match result or None if no match.
        """
        pattern_len = len(signature.pattern)
        seq_actions = sequence.actions

        if len(seq_actions) < pattern_len:
            return None

        # Try exact matching first
        positions = self._find_exact_matches(seq_actions, signature.pattern)

        # Try fuzzy matching if enabled and no exact matches
        if not positions and self.config.enable_fuzzy_matching:
            positions, mismatch_count = self._find_fuzzy_matches(
                seq_actions, signature.pattern
            )
            if mismatch_count > self.config.fuzzy_tolerance:
                positions = []

        if not positions:
            return None

        # Calculate confidence based on match quality and position
        confidence = self._calculate_match_confidence(
            sequence, signature, positions
        )

        if confidence < self.config.min_confidence:
            return None

        # Calculate risk score
        risk_score = self._calculate_risk_score(
            sequence, signature, positions, confidence
        )

        return SignatureMatch(
            wallet=sequence.wallet,
            signature=signature,
            confidence=confidence,
            risk_score=risk_score,
            matched_positions=tuple(positions),
            context={
                "sequence_length": float(len(seq_actions)),
                "pattern_length": float(pattern_len),
                "match_count": float(len(positions)),
            },
        )

    def _find_exact_matches(
        self,
        actions: tuple[WalletActionType, ...],
        pattern: tuple[WalletActionType, ...],
    ) -> list[int]:
        """Find exact pattern matches.

        Parameters
        ----------
        actions : tuple[WalletActionType, ...]
            Action sequence.
        pattern : tuple[WalletActionType, ...]
            Pattern to find.

        Returns
        -------
        list[int]
            Starting positions of matches.
        """
        positions: list[int] = []
        pattern_len = len(pattern)

        for i in range(len(actions) - pattern_len + 1):
            if actions[i : i + pattern_len] == pattern:
                positions.append(i)

        return positions

    def _find_fuzzy_matches(
        self,
        actions: tuple[WalletActionType, ...],
        pattern: tuple[WalletActionType, ...],
    ) -> tuple[list[int], int]:
        """Find approximate pattern matches.

        Parameters
        ----------
        actions : tuple[WalletActionType, ...]
            Action sequence.
        pattern : tuple[WalletActionType, ...]
            Pattern to find.

        Returns
        -------
        tuple[list[int], int]
            Starting positions and minimum mismatch count.
        """
        positions: list[int] = []
        pattern_len = len(pattern)
        min_mismatches = pattern_len + 1

        for i in range(len(actions) - pattern_len + 1):
            mismatches = sum(
                1 for j in range(pattern_len)
                if actions[i + j] != pattern[j]
            )

            if mismatches <= self.config.fuzzy_tolerance:
                if mismatches < min_mismatches:
                    min_mismatches = mismatches
                    positions = [i]
                elif mismatches == min_mismatches:
                    positions.append(i)

        return positions, min_mismatches

    def _calculate_match_confidence(
        self,
        sequence: ActionSequence,
        signature: DumpSignature,
        positions: list[int],
    ) -> float:
        """Calculate confidence score for a match.

        Parameters
        ----------
        sequence : ActionSequence
            Full sequence.
        signature : DumpSignature
            Matched signature.
        positions : list[int]
            Match positions.

        Returns
        -------
        float
            Confidence score (0 to 1).
        """
        # Base confidence from match count
        match_confidence = min(len(positions) / 2, 1.0)

        # Position factor - matches near end are more significant
        if positions:
            last_pos = max(positions)
            seq_len = len(sequence.actions)
            position_factor = (last_pos + len(signature.pattern)) / seq_len
        else:
            position_factor = 0.5

        # Length factor - longer sequences need more evidence
        length_factor = min(1.0, len(signature.pattern) / max(len(sequence.actions) / 5, 1))

        # Combine factors
        confidence = (
            0.4 * match_confidence
            + 0.4 * position_factor
            + 0.2 * length_factor
        )

        return float(min(1.0, confidence))

    def _calculate_risk_score(
        self,
        sequence: ActionSequence,
        signature: DumpSignature,
        positions: list[int],
        confidence: float,
    ) -> float:
        """Calculate risk score for a match.

        Parameters
        ----------
        sequence : ActionSequence
            Full sequence.
        signature : DumpSignature
            Matched signature.
        positions : list[int]
            Match positions.
        confidence : float
            Match confidence.

        Returns
        -------
        float
            Risk score (0 to 1).
        """
        # Base score from signature
        base = signature.base_score

        # Recency factor
        recency = self._compute_recency_factor(positions, len(sequence.actions))

        # Timing weight
        timing = signature.timing_weight * recency

        # Combine
        risk = base * (1 - signature.timing_weight) + timing
        risk *= confidence

        return float(min(1.0, risk))

    def _compute_recency_factor(
        self, positions: list[int] | tuple[int, ...], seq_len: int
    ) -> float:
        """Compute how recent the matches are.

        Parameters
        ----------
        positions : list[int] | tuple[int, ...]
            Match positions.
        seq_len : int
            Sequence length.

        Returns
        -------
        float
            Recency factor (0 to 1, higher = more recent).
        """
        if not positions or seq_len == 0:
            return 0.5

        # Use latest position
        latest = max(positions)
        recency = (latest + 1) / seq_len

        # Apply recency weight
        weighted = (
            recency * self.config.recency_weight
            + (1 - self.config.recency_weight) * 0.5
        )

        return float(weighted)

    def _compute_base_risk(self, sequence: ActionSequence) -> float:
        """Compute base risk from individual actions.

        Parameters
        ----------
        sequence : ActionSequence
            Action sequence.

        Returns
        -------
        float
            Base risk score (0 to 1).
        """
        if len(sequence.actions) == 0:
            return 0.0

        # Count risky action ratios
        sell_count = sum(
            1 for a in sequence.actions
            if a in (WalletActionType.SWAP_SELL, WalletActionType.TRANSFER_OUT)
        )
        buy_count = sum(
            1 for a in sequence.actions
            if a in (WalletActionType.SWAP_BUY, WalletActionType.FUNDED)
        )

        total = len(sequence.actions)
        sell_ratio = sell_count / total

        # High sell ratio without corresponding buys is risky
        if buy_count > 0:
            sell_buy_ratio = sell_count / buy_count
        else:
            sell_buy_ratio = sell_count * 2  # Penalize selling without buying

        base_risk = 0.3 * sell_ratio + 0.2 * min(sell_buy_ratio / 3, 1.0)

        return float(min(1.0, base_risk))

    def get_signature_by_type(
        self, sig_type: SignatureType
    ) -> DumpSignature | None:
        """Get a signature definition by type.

        Parameters
        ----------
        sig_type : SignatureType
            Signature type to retrieve.

        Returns
        -------
        DumpSignature | None
            Signature definition or None if not found.
        """
        for sig in self.KNOWN_SIGNATURES:
            if sig.signature_type == sig_type:
                return sig
        return None

    def add_custom_signature(self, signature: DumpSignature) -> None:
        """Add a custom signature for detection.

        Note: This modifies the class-level KNOWN_SIGNATURES.

        Parameters
        ----------
        signature : DumpSignature
            Custom signature to add.
        """
        # Create new tuple with added signature
        DumpSignatureDetector.KNOWN_SIGNATURES = (
            *self.KNOWN_SIGNATURES, signature
        )
        logger.info(
            "custom_signature_added",
            signature_type=signature.signature_type.value,
            pattern=signature.pattern_str,
        )
