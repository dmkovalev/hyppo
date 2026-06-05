"""Layer-2 marker protocol for OWL DL rules requiring CWA (rules 2, 9, 11, 13, 15).

The four-step protocol for each rule r_k (k ∈ {2, 9, 11, 13, 15}):

  Step 1 — CWA evidence collection.
      Programmatically check a closed-world condition that the OWL reasoner
      cannot verify under OWA (negation / universal quantifier / cardinality).

  Step 2 — Marker assertion.
      When the condition holds, append the marker class to the individual's
      ``is_a`` list via owlready2 (``individual.is_a.append(MarkerClass)``).

  Step 3 — Reasoner invocation.
      Call ``sync_reasoner_hermit()`` to check consistency with the asserted
      markers and run deductive closure over layer-1 rules.

  Step 4 — Validation / rollback.
      If the reasoner finds an inconsistency (OwlReadyInconsistentOntologyError),
      retract the marker and emit a warning.

Layer ordering guarantee (from the dissertation §3.3.2):
  Pydantic (layer 3) → Markers (layer 2) → HermiT (layer 1).

Public API
----------
apply_markers(ontology, *, run_hermit=True) -> MarkerReport
    Apply all five marker rules to the current ABox of *ontology*.

apply_rule_2(ontology, *, run_hermit=True)  -> list[str]
apply_rule_9(ontology, *, run_hermit=True)  -> list[str]
apply_rule_11(ontology, *, run_hermit=True) -> list[str]
apply_rule_13(ontology, *, run_hermit=True) -> list[str]
apply_rule_15(ontology, *, run_hermit=True) -> list[str]
    Each returns the list of individual IRIs that received the marker.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# owlready2 guard (same pattern as consistency.py)
# ---------------------------------------------------------------------------
try:
    from owlready2 import (
        OwlReadyInconsistentOntologyError,
        sync_reasoner_hermit,
    )

    _HERMIT_AVAILABLE = True
except ImportError:  # pragma: no cover - environment guard
    _HERMIT_AVAILABLE = False

    class OwlReadyInconsistentOntologyError(Exception):  # type: ignore[misc]
        pass


# ---------------------------------------------------------------------------
# Lazy imports of marker classes (avoid circular import at module load time)
# ---------------------------------------------------------------------------

def _import_marker_classes():
    """Return a namespace dict with all required marker classes."""
    from hyppo.ontology.core_rules import CompleteExperiment
    from hyppo.ontology.workflow_validation import OrphanHypothesis, WorkflowTask, hasHypothesis
    from hyppo.ontology.quality_gates import PrunableSubtree, LowQuality, hasDescendant
    from hyppo.ontology.multi_experiment import SharedHypothesis, Experiment, usesHypothesis
    from hyppo.ontology.model_compatibility import (
        DatasetNotInConfig,
        ModelWithDatasetNeed,
        usedInConfig,
        hasAvailableDataset,
    )
    from hyppo.core._base import Hypothesis
    return dict(
        CompleteExperiment=CompleteExperiment,
        OrphanHypothesis=OrphanHypothesis,
        WorkflowTask=WorkflowTask,
        hasHypothesis=hasHypothesis,
        PrunableSubtree=PrunableSubtree,
        LowQuality=LowQuality,
        hasDescendant=hasDescendant,
        SharedHypothesis=SharedHypothesis,
        Experiment=Experiment,
        usesHypothesis=usesHypothesis,
        DatasetNotInConfig=DatasetNotInConfig,
        ModelWithDatasetNeed=ModelWithDatasetNeed,
        usedInConfig=usedInConfig,
        hasAvailableDataset=hasAvailableDataset,
        Hypothesis=Hypothesis,
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MarkerReport:
    """Summary of all markers asserted / rolled back during one run."""

    rule2_marked: list[str] = field(default_factory=list)
    """IRIs of VirtualExperiment individuals marked CompleteExperiment."""

    rule9_marked: list[str] = field(default_factory=list)
    """IRIs of Hypothesis individuals marked OrphanHypothesis."""

    rule11_marked: list[str] = field(default_factory=list)
    """IRIs of Hypothesis individuals marked PrunableSubtree."""

    rule13_marked: list[str] = field(default_factory=list)
    """IRIs of Hypothesis individuals marked SharedHypothesis."""

    rule15_marked: list[str] = field(default_factory=list)
    """IRIs of Model individuals marked DatasetNotInConfig."""

    rolled_back: list[str] = field(default_factory=list)
    """IRIs of individuals whose markers were retracted due to inconsistency."""

    hermit_skipped: bool = False
    """True when HermiT was not invoked (owlready2/Java unavailable)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_marker(individual, marker_cls, ontology, *, run_hermit: bool) -> bool:
    """Step 2-4: assert marker, optionally run HermiT, roll back on contradiction.

    Returns True if the marker was successfully asserted and retained.
    """
    if marker_cls in individual.is_a:
        return True  # already marked — idempotent

    # Step 2: assert marker
    individual.is_a.append(marker_cls)
    log.debug("Marker asserted: %s ∈ %s", individual.iri, marker_cls.__name__)

    if not run_hermit:
        return True

    if not _HERMIT_AVAILABLE:
        log.debug("HermiT unavailable — skipping consistency check for %s", individual.iri)
        return True

    # Step 3: run reasoner
    try:
        with ontology:
            sync_reasoner_hermit(infer_property_values=False)
    except OwlReadyInconsistentOntologyError as exc:
        # Step 4: rollback
        try:
            individual.is_a.remove(marker_cls)
        except ValueError:
            pass
        warnings.warn(
            f"Marker {marker_cls.__name__} on {individual.iri} caused "
            f"inconsistency and was retracted: {exc}",
            stacklevel=3,
        )
        log.warning(
            "Marker rolled back: %s ∈ %s (inconsistency: %s)",
            individual.iri, marker_cls.__name__, exc,
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Rule 2: CompleteExperiment
# ---------------------------------------------------------------------------

def apply_rule_2(ontology, *, run_hermit: bool = True) -> list[str]:
    """Mark VirtualExperiment individuals that are fully specified (Rule 2).

    CWA condition: the experiment has at least one model AND at least one
    dataset linked via ``has_for_model`` (and ``has_for_configuration`` as
    proxy for dataset in the current ontology schema, matching the dissertation
    definition of «linked to at least one model and dataset»).

    The dissertation (§3.3.2, Rule 2) states the marker is set procedurally
    after verifying completeness; HermiT can then classify the individual as
    ``CompleteExperiment`` via the equivalent_to restriction in core_rules.py.
    """
    ns = _import_marker_classes()
    CompleteExperiment = ns["CompleteExperiment"]

    from hyppo.core._base import VirtualExperiment

    marked: list[str] = []
    for ve in VirtualExperiment.instances():
        # Step 1: CWA evidence — all five required slots must be non-empty
        has_onto = bool(ve.has_for_ontology)
        has_wf = bool(ve.has_for_workflow)
        has_hyp = bool(ve.has_for_hypothesis)
        has_model = bool(ve.has_for_model)
        has_cfg = bool(ve.has_for_configuration)
        if has_onto and has_wf and has_hyp and has_model and has_cfg:
            ok = _assert_marker(ve, CompleteExperiment, ontology, run_hermit=run_hermit)
            if ok:
                marked.append(ve.iri)
    return marked


# ---------------------------------------------------------------------------
# Rule 9: OrphanHypothesis
# ---------------------------------------------------------------------------

def apply_rule_9(ontology, *, run_hermit: bool = True) -> list[str]:
    """Mark hypothesis individuals not referenced by any WorkflowTask (Rule 9).

    CWA condition: no WorkflowTask individual has the hypothesis in its
    ``hasHypothesis`` property.  This is the SPARQL-style closed-world check
    described in the dissertation (§3.3.2, Rule 9 / OrphanHypothesis).
    """
    ns = _import_marker_classes()
    OrphanHypothesis = ns["OrphanHypothesis"]
    WorkflowTask = ns["WorkflowTask"]
    Hypothesis = ns["Hypothesis"]

    # Step 1: collect all hypotheses referenced by at least one task
    referenced: set = set()
    for task in WorkflowTask.instances():
        for h in task.hasHypothesis:
            referenced.add(h)

    marked: list[str] = []
    for h in Hypothesis.instances():
        if h not in referenced:
            ok = _assert_marker(h, OrphanHypothesis, ontology, run_hermit=run_hermit)
            if ok:
                marked.append(h.iri)
    return marked


# ---------------------------------------------------------------------------
# Rule 11: PrunableSubtree
# ---------------------------------------------------------------------------

def apply_rule_11(ontology, *, run_hermit: bool = True) -> list[str]:
    """Mark low-quality hypotheses whose entire descendant subtree is also low-quality (Rule 11).

    CWA condition (universal quantifier ∀ has_leaf.LowScoreHypothesis):
    a hypothesis is a PrunableSubtree root if:
      1. it is already classified as LowQuality, AND
      2. every descendant reachable via ``hasDescendant`` is also LowQuality.

    The ∀ quantifier requires CWA — the reasoner cannot verify absence under OWA.
    """
    ns = _import_marker_classes()
    PrunableSubtree = ns["PrunableSubtree"]
    LowQuality = ns["LowQuality"]
    Hypothesis = ns["Hypothesis"]

    marked: list[str] = []
    for h in LowQuality.instances():
        # Step 1: CWA — all transitive descendants must be LowQuality
        descendants = _collect_descendants(h)
        if all(isinstance(d, LowQuality) or LowQuality in d.is_a for d in descendants):
            ok = _assert_marker(h, PrunableSubtree, ontology, run_hermit=run_hermit)
            if ok:
                marked.append(h.iri)
    return marked


def _collect_descendants(root) -> list:
    """BFS over hasDescendant links to collect all descendants."""
    from hyppo.ontology.quality_gates import hasDescendant as _hasDescendant  # noqa: F401
    visited: list = []
    queue = list(root.hasDescendant)
    seen: set = {root}
    while queue:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        visited.append(node)
        queue.extend(node.hasDescendant)
    return visited


# ---------------------------------------------------------------------------
# Rule 13: SharedHypothesis
# ---------------------------------------------------------------------------

def apply_rule_13(ontology, *, run_hermit: bool = True) -> list[str]:
    """Mark hypotheses used by two or more distinct experiments (Rule 13).

    CWA condition (minimum cardinality on inverse ``usesHypothesis``):
    count the number of Experiment individuals that link to this hypothesis
    via ``usesHypothesis``.  If count ≥ 2 → SharedHypothesis.

    HermiT does not reliably infer minimum cardinality on inverse properties
    for individuals, so this check is procedural (dissertation §3.3.2, Rule 13).
    """
    ns = _import_marker_classes()
    SharedHypothesis = ns["SharedHypothesis"]
    Experiment = ns["Experiment"]
    Hypothesis = ns["Hypothesis"]

    # Step 1: build inverse index  hypothesis -> list[Experiment]
    hyp_to_exps: dict = {}
    for exp in Experiment.instances():
        for h in exp.usesHypothesis:
            hyp_to_exps.setdefault(h, []).append(exp)

    marked: list[str] = []
    for h in Hypothesis.instances():
        exps = hyp_to_exps.get(h, [])
        if len(exps) >= 2:
            ok = _assert_marker(h, SharedHypothesis, ontology, run_hermit=run_hermit)
            if ok:
                marked.append(h.iri)
    return marked


# ---------------------------------------------------------------------------
# Rule 15: DatasetNotInConfig
# ---------------------------------------------------------------------------

def apply_rule_15(ontology, *, run_hermit: bool = True) -> list[str]:
    """Mark models needing a dataset that have no accessible dataset (Rule 15).

    CWA condition (negation ¬hasAccessibleDataset.some(Dataset)):
    a ModelWithDatasetNeed individual has no accessible dataset when:
      - it has no ``usedInConfig`` link, OR
      - all linked ModelConfig instances have no ``hasAvailableDataset`` links.

    The negation requires CWA — the reasoner cannot infer absence under OWA.
    The dissertation (§3.3.2, Rule 15) calls the marker ``DatasetMissing`` /
    ``DatasetNotInConfig``.
    """
    ns = _import_marker_classes()
    DatasetNotInConfig = ns["DatasetNotInConfig"]
    ModelWithDatasetNeed = ns["ModelWithDatasetNeed"]

    marked: list[str] = []
    for m in ModelWithDatasetNeed.instances():
        # Step 1: CWA — check that no config provides a dataset
        has_accessible = False
        for cfg in m.usedInConfig:
            if cfg.hasAvailableDataset:
                has_accessible = True
                break
        # Also check inferred hasAccessibleDataset (property chain result)
        if not has_accessible and m.hasAccessibleDataset:
            has_accessible = True

        if not has_accessible:
            ok = _assert_marker(m, DatasetNotInConfig, ontology, run_hermit=run_hermit)
            if ok:
                marked.append(m.iri)
    return marked


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------

def apply_markers(ontology, *, run_hermit: bool = True) -> MarkerReport:
    """Apply all five layer-2 marker rules to the current ABox.

    Runs rules in the order: 2, 9, 11, 13, 15.  Each rule is applied
    independently; HermiT (if enabled) is called per-individual assertion so
    that a contradiction in one marker does not block the others.

    Parameters
    ----------
    ontology
        owlready2 ontology object carrying the ABox individuals.
    run_hermit
        If False, skip HermiT invocations (useful in CI without Java).

    Returns
    -------
    MarkerReport
        Per-rule lists of marked IRIs and rollback events.
    """
    report = MarkerReport()

    if run_hermit and not _HERMIT_AVAILABLE:
        log.warning("apply_markers: owlready2 not available — HermiT skipped")
        report.hermit_skipped = True
        run_hermit = False

    report.rule2_marked = apply_rule_2(ontology, run_hermit=run_hermit)
    report.rule9_marked = apply_rule_9(ontology, run_hermit=run_hermit)
    report.rule11_marked = apply_rule_11(ontology, run_hermit=run_hermit)
    report.rule13_marked = apply_rule_13(ontology, run_hermit=run_hermit)
    report.rule15_marked = apply_rule_15(ontology, run_hermit=run_hermit)

    log.info(
        "apply_markers complete: rule2=%d rule9=%d rule11=%d rule13=%d rule15=%d",
        len(report.rule2_marked),
        len(report.rule9_marked),
        len(report.rule11_marked),
        len(report.rule13_marked),
        len(report.rule15_marked),
    )
    return report
