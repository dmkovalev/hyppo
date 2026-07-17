"""Java-free Stage A fallback (owlrl OWL 2 RL closure) for Algorithm 3.

These tests exercise the pure-Python degradation path of ``check_consistency``'s
Stage A: they must run without a Java runtime. They complement — never replace —
the HermiT golden claims in ``test_golden_claims.py`` (the 48-claim contract).
"""

from __future__ import annotations

import owlready2
import pytest

from hyppo.ontology import consistency as C
from hyppo.ontology.consistency import Status, check_consistency

_GOOD_LATTICE = {0: {1}, 1: {2}, 2: set()}


def _fresh_world(two_models: bool):
    """Isolated owlready2 World mirroring the golden alg3 Stage A construction.

    ``two_models=True`` forces a FunctionalProperty to hold two AllDifferent
    models (the C2 contradiction); ``False`` leaves the hypothesis model-less
    (the OWA case that must stay consistent).
    """
    from owlready2 import (
        AllDifferent,
        FunctionalProperty,
        ObjectProperty,
        Thing,
        World,
    )

    w = World()
    onto = w.get_ontology("http://owlrl.test/alg3_stage_a.owl")
    with onto:

        class GModel(Thing):
            pass

        class GHypothesis(Thing):
            pass

        class g_implemented_by(ObjectProperty, FunctionalProperty):
            domain = [GHypothesis]
            range = [GModel]

        GHypothesis.is_a.append(g_implemented_by.some(GModel))
        m1, m2 = GModel("m1"), GModel("m2")
        AllDifferent([m1, m2])
        h = GHypothesis("h")
        if two_models:
            h.g_implemented_by = m1
            h.is_a.append(onto.g_implemented_by.value(m2))
    return w, onto


def test_owlrl_detects_functional_violation_c2():
    """C2 via owlrl: a functional property bound to two distinct models is
    inconsistent, classified into the C2 status — matching HermiT's verdict."""
    _w, onto = _fresh_world(two_models=True)
    res = check_consistency(None, onto, _GOOD_LATTICE, stage_a_engine="owlrl")
    assert not res.ok
    assert res.status == Status.C2_VIOLATED
    assert res.details["stage_a"] == "inconsistent"
    assert res.details["stage_a_engine"] == "owlrl"


def test_owlrl_owa_keeps_modelless_hypothesis_consistent():
    """OWA via owlrl: a model-less hypothesis is *consistent* (the existential
    may be met by an unknown model) — Stage A passes, marker layer's job."""
    _w, onto = _fresh_world(two_models=False)
    res = check_consistency(
        None,
        onto,
        _GOOD_LATTICE,
        stage_a_engine="owlrl",
        artefacts={0: {"out": {"a"}}, 1: {"in": {"a"}, "out": {"b"}}, 2: {"in": {"b"}}},
    )
    assert res.ok and res.status == Status.OK
    assert res.details["stage_a"] == "passed"
    assert res.details["stage_a_engine"] == "owlrl"


def test_forced_owlrl_runs_without_java(monkeypatch):
    """stage_a_engine='owlrl' never touches HermiT/Java even if it were broken."""

    def _boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("HermiT must not be invoked in forced owlrl mode")

    monkeypatch.setattr(C, "sync_reasoner_hermit", _boom)
    _w, onto = _fresh_world(two_models=True)
    res = check_consistency(None, onto, _GOOD_LATTICE, stage_a_engine="owlrl")
    assert not res.ok and res.status == Status.C2_VIOLATED


def test_auto_uses_owlrl_by_default():
    """auto: the OWL 2 RL closure is the default engine (RL + IC design, §2.4) —
    no Java / DL reasoner required; a functional-property clash is still caught."""
    _w, onto = _fresh_world(two_models=True)
    res = check_consistency(None, onto, _GOOD_LATTICE, stage_a_engine="auto")
    assert not res.ok and res.status == Status.C2_VIOLATED
    assert res.details["stage_a_engine"] == "owlrl"


def test_auto_owlrl_stays_consistent_for_owa():
    """The RL default preserves the verdict: a model-less hypothesis stays OK."""
    _w, onto = _fresh_world(two_models=False)
    res = check_consistency(None, onto, _GOOD_LATTICE, stage_a_engine="auto")
    assert res.ok and res.status == Status.OK
    assert res.details["stage_a_engine"] == "owlrl"


def test_all_different_expansion_two_members():
    """AllDifferent normalisation: a 2-element set yields exactly 1 pair."""
    from rdflib import OWL, RDF, BNode, Graph, URIRef
    from rdflib.collection import Collection

    g = Graph()
    a, b = URIRef("urn:a"), URIRef("urn:b")
    adiff = BNode()
    g.add((adiff, RDF.type, OWL.AllDifferent))
    Collection(g, (lst := BNode()), [a, b])
    g.add((adiff, OWL.distinctMembers, lst))

    added = C._expand_all_different(g)
    assert added == 1
    assert (a, OWL.differentFrom, b) in g or (b, OWL.differentFrom, a) in g


def test_all_different_expansion_three_members():
    """AllDifferent normalisation: a 3-element set yields exactly 3 pairs."""
    from rdflib import OWL, RDF, BNode, Graph, URIRef
    from rdflib.collection import Collection

    g = Graph()
    a, b, c = URIRef("urn:a"), URIRef("urn:b"), URIRef("urn:c")
    adiff = BNode()
    g.add((adiff, RDF.type, OWL.AllDifferent))
    Collection(g, (lst := BNode()), [a, b, c])
    g.add((adiff, OWL.members, lst))

    added = C._expand_all_different(g)
    assert added == 3
    diff = set(g.subject_objects(OWL.differentFrom))
    assert len(diff) == 3


def test_run_hermit_false_still_skips_stage_a():
    """Backward compatibility: run_hermit=False skips Stage A entirely."""
    res = check_consistency(None, None, _GOOD_LATTICE, run_hermit=False)
    assert res.details["stage_a"] == "skipped"
    assert res.details["stage_a_engine"] == "skipped"


def test_owlrl_stage_a_helper_direct():
    """Direct unit on the closure helper: contradiction -> non-empty errors."""
    w, _onto = _fresh_world(two_models=True)
    consistent, errors = C._run_owlrl_stage_a(w)
    assert not consistent and errors

    w2, _onto2 = _fresh_world(two_models=False)
    consistent2, errors2 = C._run_owlrl_stage_a(w2)
    assert consistent2 and not errors2


def test_owlready2_import_present():
    """Guard: the fallback imports must actually be available in the env."""
    assert owlready2 is not None
    assert C._OWLRL_AVAILABLE is True
    with pytest.raises(ValueError):
        check_consistency(None, None, _GOOD_LATTICE, stage_a_engine="bogus")
