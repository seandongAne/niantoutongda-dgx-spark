"""Automatic new-home spatial-region production.

The public API deliberately starts from model observations, not from the
hand-authored hero ``RegionManifest`` fixture.  Trusted observations may be
projected to the existing downstream contract only after the configured
coverage gate passes.
"""

from backend.tools.spatial.adjudication import (
    ADJUDICATED_REGION_MANIFEST_FILENAME,
    ADJUDICATION_MANIFEST_FILENAME,
    ADJUDICATION_METRICS_FILENAME,
    ADJUDICATION_NORMALIZED_HASH_FILENAME,
    ReviewerKind,
    SpatialAdjudicationManifest,
    SpatialAdjudicationMetrics,
    SpatialAdjudicationResult,
    VisualAdjudicationDecision,
    VisualAdjudicationReview,
    VisualDecisionStatus,
    VisualFrameEvidence,
    VisualOperation,
    adjudicate_spatial_regions,
    load_visual_adjudication,
    remove_stale_adjudicated_regions,
    write_spatial_adjudication_outputs,
)

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
    "ADJUDICATED_REGION_MANIFEST_FILENAME",
    "ADJUDICATION_MANIFEST_FILENAME",
    "ADJUDICATION_METRICS_FILENAME",
    "ADJUDICATION_NORMALIZED_HASH_FILENAME",
    "CandidateStatus",
    "CoverageStatus",
    "GateStatus",
    "PowerState",
    "ReviewerKind",
    "SpatialAdjudicationManifest",
    "SpatialAdjudicationMetrics",
    "SpatialAdjudicationResult",
    "SpatialCandidate",
    "SpatialCandidateManifest",
    "SpatialMetrics",
    "SpatialObservation",
    "SpatialProducerConfig",
    "SpatialProductionResult",
    "VisualAdjudicationDecision",
    "VisualAdjudicationReview",
    "VisualDecisionStatus",
    "VisualFrameEvidence",
    "VisualOperation",
    "adjudicate_spatial_regions",
    "load_observations_jsonl",
    "load_visual_adjudication",
    "produce_spatial_regions",
    "remove_stale_adjudicated_regions",
    "write_spatial_adjudication_outputs",
    "write_spatial_outputs",
]
