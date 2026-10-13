#!/usr/bin/env python
"""RL clash verdict versus full DL consistency verdict (chapter 2, thm:integrity).

The integrity control decides consistency by the OWL 2 RL closure: a clash
derived by the RL rules means the ontology is inconsistent (soundness). The
dissertation claims no completeness of this test with respect to DL. This
script checks, empirically, whether the two verdicts agree on the ontologies
of the three domains of chapter 4 and on variants with injected defects.

For every domain (petro Norne, astro BGM, neuro HCP; ABox exactly as in
reasoning_time.py) and every variant the ABox is rebuilt from scratch, then

* RL verdict  -- ``rl_materialize`` returns a non-empty error list (clash);
* DL verdict  -- the full DL reasoner (``sync_reasoner``) raises
  ``OwlReadyInconsistentOntologyError``.

Variants: the nominal ABox; defects that must clash (disjoint domain ontology
classes, two disjoint lifecycle states, two disjoint model kinds, two distinct
implementing models of one hypothesis under a functional property, a model
individual also typed as a hypothesis); defects of the life cycle that must
not clash (orphan hypothesis, missing dataset, empty task and unrealised
hypothesis, invariant violation marker, REFUTED status).

Usage:  PYTHONPATH=. python examples/research/chapter4/rl_vs_dl_consistency.py
Output: examples/research/chapter4/results/rl_vs_dl_consistency.json
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from owlready2 import AllDifferent, OwlReadyInconsistentOntologyError, sync_reasoner  # noqa: E402

import hyppo.ontology.core_rules as cr  # noqa: E402
import hyppo.ontology.lifecycle as lc  # noqa: E402
import hyppo.ontology.model_compatibility as mc  # noqa: E402
import hyppo.ontology.workflow_validation as wv  # noqa: E402
from hyppo.adapters._spec import chain_edges  # noqa: E402
from hyppo.core._base import is_implemented_by_model  # noqa: E402
from hyppo.core._base import virtual_experiment_onto as onto  # noqa: E402
from hyppo.ontology.consistency import rl_materialize  # noqa: E402

from examples.research.chapter4._domain_run import (  # noqa: E402
    build_abox,
    derive_graph,
    reset_abox,
)
from examples.research.chapter4.reasoning_time import domains  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "rl_vs_dl_consistency.json"


# ---------------------------------------------------------------------------
# injected defects: callback(abox, key) mutates the freshly built ABox
# ---------------------------------------------------------------------------
def _first(ab):
    return next(iter(ab.hyp))


def disjoint_domain(ab, key):
    other = cr.NeuroDomainOntology if key.startswith("petro") else cr.OilDomainOntology
    ab.domain.is_a.append(other)


def disjoint_lifecycle(ab, key):
    ab.hyp[_first(ab)].is_a.append(lc.DeprecatedHypothesis)  # Active and Deprecated


def disjoint_model_kind(ab, key):
    m = ab.model[_first(ab)]
    for c in (cr.PhysicsModel, cr.DataDrivenModel):
        if c not in m.is_a:
            m.is_a.append(c)


def two_models_functional(ab, key):
    n = _first(ab)
    extra = type(ab.model[n])(f"{ab.model[n].name}_duplicate")
    # second value of the functional property, asserted at triple level
    onto._add_obj_triple_spo(ab.hyp[n].storid, is_implemented_by_model.storid, extra.storid)
    AllDifferent([ab.model[n], extra])


def model_typed_as_hypothesis(ab, key):
    ab.model[_first(ab)].is_a.append(cr.Hypothesis)


def orphan_hypothesis(ab, key):
    h = cr.Hypothesis(f"{key}_orphan")
    h.is_a.append(lc.DraftHypothesis)
    ab.ve.has_for_hypothesis.append(h)


def dataset_missing(ab, key):
    for cfg in mc.ModelConfig.instances():
        cfg.hasAvailableDataset = []


def ic_conditions(ab, key):
    names = list(ab.hyp)
    ab.hyp[names[-1]].is_implemented_by_model = None
    ab.task[0].hasHypothesis = []


def invariant_marker(ab, key):
    ab.hyp[_first(ab)].is_a.append(cr.InvalidHypothesis)


def refuted_status(ab, key):
    cr.apply_epistemic_status(ab.hyp[_first(ab)], "REFUTED")


VARIANTS = {
    "nominal": (None, False),
    "disjoint_domain": (disjoint_domain, True),
    "disjoint_lifecycle_states": (disjoint_lifecycle, True),
    "disjoint_model_kinds": (disjoint_model_kind, True),
    "two_models_functional_alldifferent": (two_models_functional, True),
    "model_typed_as_hypothesis": (model_typed_as_hypothesis, True),
    "orphan_hypothesis": (orphan_hypothesis, False),
    "dataset_missing": (dataset_missing, False),
    "empty_task_unrealised_hypothesis": (ic_conditions, False),
    "invariant_violation_marker": (invariant_marker, False),
    "refuted_status": (refuted_status, False),
}


def build(key, d, g):
    reset_abox()
    return build_abox(key.split("_")[1].lower(), d["specs"], d["tasks"], g.edges,
                      domain_cls=d["domain_cls"], quantity_cls=d["quantity_cls"],
                      vocabulary=d["vocabulary"], dataset=d["dataset"], needs_dataset=d["needs"])


def rl_verdict() -> tuple[bool, list[str]]:
    _g, errors = rl_materialize(onto.world)
    return not errors, errors


def dl_verdict() -> bool:
    try:
        sync_reasoner(infer_property_values=True, debug=0)
        return True
    except OwlReadyInconsistentOntologyError:
        return False


def run() -> dict:
    out = {"platform": f"{platform.system()} {platform.release()}, Python {platform.python_version()}",
           "rl": "OWL 2 RL closure, clash = non-empty error list",
           "dl": "full DL reasoner, inconsistency exception",
           "domains": {}}
    total = agree = 0
    for key, d in domains().items():
        g = derive_graph(d["specs"], d["tasks"], chain_edges(len(d["tasks"])))
        res = {}
        for name, (inject, expect_clash) in VARIANTS.items():
            ab = build(key, d, g)
            if inject is not None:
                with onto:
                    inject(ab, key)
            rl_ok, errors = rl_verdict()
            dl_ok = dl_verdict()
            res[name] = {"expected_inconsistent": expect_clash,
                         "rl_consistent": rl_ok, "dl_consistent": dl_ok,
                         "agree": rl_ok == dl_ok,
                         "rl_errors": [e[:160] for e in errors[:3]]}
            total += 1
            agree += rl_ok == dl_ok
        reset_abox()
        out["domains"][key] = res
    out["n_cases"], out["n_agree"] = total, agree
    out["all_agree"] = total == agree
    return out


def main() -> None:
    out = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out["domains"].items():
        for name, r in v.items():
            print(f"  {k:12s} {name:38s} RL {'ok   ' if r['rl_consistent'] else 'CLASH'} "
                  f"DL {'ok   ' if r['dl_consistent'] else 'INCONS'} "
                  f"{'agree' if r['agree'] else 'DISAGREE'}")
    print(f"agree {out['n_agree']}/{out['n_cases']}; wrote {OUT}")


if __name__ == "__main__":
    main()
