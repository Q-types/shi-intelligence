"""Wallet action sequence modeling for behavioral analysis.

This module provides tools for encoding wallet actions as sequences,
detecting behavioral patterns, and identifying pre-dump signatures.
"""

from __future__ import annotations

from .encoder import WalletActionEncoder, WalletActionType, ActionSequence
from .patterns import SequencePatternDetector, Motif, BehaviorCluster
from .signatures import DumpSignatureDetector, DumpSignature, SignatureMatch, SignatureType

__all__ = [
    # Encoder
    "WalletActionEncoder",
    "WalletActionType",
    "ActionSequence",
    # Pattern detection
    "SequencePatternDetector",
    "Motif",
    "BehaviorCluster",
    # Signature detection
    "DumpSignatureDetector",
    "DumpSignature",
    "SignatureMatch",
    "SignatureType",
]
