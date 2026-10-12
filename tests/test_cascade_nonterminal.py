"""Theorem thm:integrity (IV): the cascade is non-terminal and derived.

* a hypothesis carrying a (stale) manual ValidHypothesis marker whose ancestor
  becomes invalid is marked StaleHypothesis by RL materialisation WITHOUT a
  clash (ValidHypothesis is no longer disjoint with StaleHypothesis);
* assign_valid then removes the stale Valid marker and marks the rest valid.
"""

import itertools

from owlready2 import get_ontology

_counter = itertools.count()


def _onto():
    from hyppo.core._base import virtual_experiment_onto  # noqa: F401
    import hyppo.ontology.core_rules as cr  # registers rule-4 classes

    return virtual_experiment_onto, cr


def test_valid_marker_does_not_block_cascade():
    onto, cr = _onto()
    from hyppo.core._base import Hypothesis
    from hyppo.ontology.consistency import rl_materialize

    k = next(_counter)
    with onto:
        root = Hypothesis(f"nt_root_{k}")
        mid = Hypothesis(f"nt_mid_{k}")
        leaf = Hypothesis(f"nt_leaf_{k}")
        mid.derived_by = [root]
        leaf.derived_by = [mid]
        leaf.is_a.append(cr.ValidHypothesis)   # stale manual marker
        root.is_a.append(cr.InvalidHypothesis)

    _graph, errors = rl_materialize(onto.world)
    assert not errors, errors  # no clash between Valid and the derived Stale

    marked = cr.materialize_cascade(onto)
    assert {mid, leaf} <= set(marked) | {h for h in (mid, leaf) if cr.StaleHypothesis in h.is_a}
    assert cr.StaleHypothesis in leaf.is_a and cr.StaleHypothesis in mid.is_a
    assert cr.StaleHypothesis not in root.is_a

    valid = cr.assign_valid([root, mid, leaf])
    assert cr.ValidHypothesis not in leaf.is_a  # stale Valid marker removed
    assert leaf not in valid and mid not in valid
    assert root not in valid  # the invalidated hypothesis itself is not valid
