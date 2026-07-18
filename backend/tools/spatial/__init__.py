"""Automatic new-home spatial-region production.

The public API deliberately starts from model observations, not from the
hand-authored hero ``RegionManifest`` fixture.  Trusted observations may be
projected to the existing downstream contract only after the configured
coverage gate passes.
"""

from backend.tools.spatial.producer import (
    CandidateStatus,
    CoverageStatus,
    GateStatus,
    PowerState,
    SpatialCandidate,
    SpatialCandidateManifest,
    SpatialMetrics,
    SpatialObservation,
    SpatialProducerConfig,
    SpatialProductionResult,
    load_observations_jsonl,
    produce_spatial_regions,
    write_spatial_outputs,
)

__all__ = [
    "CandidateStatus",
    "CoverageStatus",
    "GateStatus",
    "PowerState",
    "SpatialCandidate",
    "SpatialCandidateManifest",
    "SpatialMetrics",
    "SpatialObservation",
    "SpatialProducerConfig",
    "SpatialProductionResult",
    "load_observations_jsonl",
    "produce_spatial_regions",
    "write_spatial_outputs",
]
