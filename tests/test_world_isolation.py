"""World isolation for the virtual-experiment ontology.

The schema, rules and oil-domain classes are definable into a fresh owlready2
``World`` via :func:`hyppo.core.create_ve_world`, and
:func:`hyppo.adapters.norne_adapter.build_oil_virtual_experiment` allocates a
new isolated world per call. These tests pin the isolation contract:

- re-callable build (no fixed-IRI ``sqlite3.IntegrityError``);
- each world carries its own Stage-B consistency verdict;
- individuals classify against their own world's classes (owlready2
  ``isinstance`` is storid-based, hence world-local — a fresh individual is
  NOT ``isinstance`` of the module-global class);
- the module-global ``virtual_experiment_onto`` path is untouched.

Stage A runs Java-free (``stage_a_engine="owlrl"``) for determinism.
"""

from __future__ import annotations

from hyppo.adapters.norne_adapter import build_oil_virtual_experiment
from hyppo.core import create_ve_world, virtual_experiment_onto
from hyppo.core._base import Hypothesis as GlobalHypothesis
from hyppo.ontology.consistency import Status, check_consistency

_GOOD_LATTICE = {0: {1}, 1: {2}, 2: set()}
_CYCLIC_LATTICE = {0: {1}, 1: {0}}


def test_double_build_no_integrity_error():
    """Two builds in one process succeed and yield distinct isolated worlds."""
    ve1 = build_oil_virtual_experiment()
    ve2 = build_oil_virtual_experiment()

    assert ve1["world"] is not ve2["world"]
    assert len(ve1["hypotheses"]) == 6
    assert len(ve2["hypotheses"]) == 6


def test_two_worlds_independent_stage_b_verdicts():
    """Each world produces its own Stage-B verdict without cross-talk."""
    ve1 = build_oil_virtual_experiment()
    ve2 = build_oil_virtual_experiment()

    r1 = check_consistency(
        ve1["virtual_experiment"], ve1["onto"], _GOOD_LATTICE, stage_a_engine="owlrl"
    )
    r2 = check_consistency(
        ve2["virtual_experiment"], ve2["onto"], _CYCLIC_LATTICE, stage_a_engine="owlrl"
    )

    assert r1.ok is True and r1.status == Status.OK
    assert r2.ok is False and r2.status == Status.C3_VIOLATED


def test_create_ve_world_stage_b_independent_of_global():
    """create_ve_world yields a fresh World whose ontology passes Stage B,
    independently of the module-global default world."""
    world, onto, _ns = create_ve_world()

    res = check_consistency(None, onto, _GOOD_LATTICE, stage_a_engine="owlrl")

    assert res.ok is True and res.status == Status.OK
    assert onto.world is world
    assert world is not virtual_experiment_onto.world


def test_fresh_build_individuals_are_isinstance_of_world_ns():
    """Fresh-world individuals classify against their own world's ns classes
    but NOT against the module-global class (storid-based, world-local)."""
    ve = build_oil_virtual_experiment()
    ns = ve["ns"]
    h = ve["hypotheses_map"]["h_CRM"]

    assert isinstance(h, ns.Hypothesis)
    assert isinstance(h.is_implemented_by_model, ns.PhysicsModel)
    # Cross-world isinstance is False: the fresh individual does not belong to
    # the module-global (default-world) Hypothesis class.
    assert not isinstance(h, GlobalHypothesis)


def test_module_global_path_unaffected():
    """A fresh build does not touch the module-global schema objects."""
    before = GlobalHypothesis
    ve = build_oil_virtual_experiment()

    assert GlobalHypothesis is before
    assert GlobalHypothesis.namespace.world is virtual_experiment_onto.world
    # The fresh world's Hypothesis is a distinct class object.
    assert ve["ns"].Hypothesis is not GlobalHypothesis
