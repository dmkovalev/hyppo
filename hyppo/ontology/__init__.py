"""OWL 2 DL reasoning rules for the Hyppo virtual experiment platform.

This package implements 16 OWL DL rules (no SWRL, no arithmetic) that extend
the base ontology defined in ``hyppo.core._base``.  Rules cover:

- Core classification and structural axioms (rules 1-6)
- Provenance and versioning (rules 7-8)
- Workflow validation (rules 9-10)
- Quality gates for hypothesis lattice pruning (rules 11-12)
- Multi-experiment sharing detection (rule 13)
- Model compatibility and data-format checks (rules 14-15)
- Lifecycle state management (rule 16)

Additionally, ``oil_constraints`` provides Python-layer physical validators
that cannot be expressed in pure OWL DL.
"""

from .consistency import (
    ConsistencyResult,
    Status,
    check_consistency,
)
from .core_rules import (
    CompleteExperiment,
    DataDrivenHypothesis,
    DataDrivenModel,
    DomainOntology,
    HybridHypothesis,
    HybridModel,
    InvalidHypothesis,
    MaterialBalanceHypothesis,
    NeuroDomainOntology,
    OilDomainOntology,
    PhysicsHypothesis,
    PhysicsModel,
    PredictionSourceHypothesis,
    StaleHypothesis,
    UncomputedHypothesis,
    ValidHypothesis,
    has_dependency,
    has_for_ontology,
)
from .lifecycle import (
    ActiveHypothesis,
    ArchivedHypothesis,
    BlockingDeprecation,
    ConflictingHypothesis,
    DeprecatedHypothesis,
    DraftHypothesis,
    FreshHypothesis,
    PreferredHypothesis,
    apply_pydantic_to_ontology,
    refresh_hypothesis,
)
from .model_compatibility import (
    Dataset,
    DatasetNotInConfig,
    FormatMismatch,
    GraphConsumer,
    GraphFormat,
    ModelConfig,
    ModelWithDatasetNeed,
    TimeSeriesFormat,
    TimeSeriesProducer,
    feedsInto,
    hasAccessibleDataset,
    hasAvailableDataset,
    usedInConfig,
)
from .multi_experiment import (
    Experiment,
    SharedHypothesis,
    usesHypothesis,
)
from .provenance import (
    DerivedStaleRun,
    ExperimentRun,
    HypothesisVersion,
    HypothesisWithStaleAncestor,
    ObsoleteVersion,
    StaleRun,
    run_depends_on_hypothesis,
    superseded_by,
    uses_hypothesis,
    uses_hypothesis_version,
    version_of,
)
from .quality_gates import (
    HighQuality,
    LowQuality,
    PromisingRoute,
    PrunableSubtree,
    hasAncestor,
    hasDescendant,
)
from .workflow_validation import (
    ConflictFreeTask,
    ConflictingTask,
    OrphanHypothesis,
    WorkflowTask,
    hasHypothesis,
)

__all__ = [
    "CompleteExperiment",
    "DataDrivenHypothesis",
    "DataDrivenModel",
    "DomainOntology",
    "HybridHypothesis",
    "HybridModel",
    "InvalidHypothesis",
    "MaterialBalanceHypothesis",
    "NeuroDomainOntology",
    "OilDomainOntology",
    "PhysicsHypothesis",
    "PhysicsModel",
    "PredictionSourceHypothesis",
    "StaleHypothesis",
    "UncomputedHypothesis",
    "ValidHypothesis",
    "has_dependency",
    "has_for_ontology",
    "DerivedStaleRun",
    "ExperimentRun",
    "HypothesisVersion",
    "HypothesisWithStaleAncestor",
    "ObsoleteVersion",
    "StaleRun",
    "run_depends_on_hypothesis",
    "superseded_by",
    "uses_hypothesis",
    "uses_hypothesis_version",
    "version_of",
    "ConflictFreeTask",
    "ConflictingTask",
    "OrphanHypothesis",
    "WorkflowTask",
    "hasHypothesis",
    "HighQuality",
    "LowQuality",
    "PrunableSubtree",
    "PromisingRoute",
    "hasAncestor",
    "hasDescendant",
    "Experiment",
    "SharedHypothesis",
    "usesHypothesis",
    "Dataset",
    "DatasetNotInConfig",
    "FormatMismatch",
    "GraphConsumer",
    "GraphFormat",
    "ModelConfig",
    "ModelWithDatasetNeed",
    "TimeSeriesFormat",
    "TimeSeriesProducer",
    "feedsInto",
    "hasAccessibleDataset",
    "hasAvailableDataset",
    "usedInConfig",
    "ActiveHypothesis",
    "ArchivedHypothesis",
    "BlockingDeprecation",
    "ConflictingHypothesis",
    "DeprecatedHypothesis",
    "DraftHypothesis",
    "FreshHypothesis",
    "PreferredHypothesis",
    "apply_pydantic_to_ontology",
    "refresh_hypothesis",
    "ConsistencyResult",
    "Status",
    "check_consistency",
]
