"""
Analysis Pipeline for SHI.

Orchestrates the full analysis flow:
1. Data ingestion
2. Feature engineering
3. Metrics computation
4. Graph analysis
5. Archetype clustering
6. Risk scoring

All outputs are deterministic and versioned.
"""

from .features import FeatureEngineer
from .metrics_pipeline import MetricsPipeline
from .orchestrator import AnalysisOrchestrator

__all__ = [
    "FeatureEngineer",
    "MetricsPipeline",
    "AnalysisOrchestrator",
]
