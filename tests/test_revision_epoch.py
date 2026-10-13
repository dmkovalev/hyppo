"""Revision epochs in the chained keys (world semantics of thm:safe).

Marks are per hypothesis, records per chain key. Without an epoch of the
revised source in the key, a record computed BEFORE a revision of a source u
under another parameter value survives the marking (the rewrite after the
revision replaces only the record with the same key) and is read again when the
parameter returns to the old value:

    run 1: a computed with p1          -> record R1 (key K1)
    run 2: a computed with p2, its source b becomes SUPERSEDED -> a marked
    run 3: a recomputed with p2, mark cleared
    run 4: parameter back to p1        -> R1 has the current key: stale read

With the epoch eps_b in the key of every consumer of b, R1 is never read again.
A hypothesis that is not a descendant of the revised b keeps reusing its
old-parameter record (no over-invalidation).
"""

from __future__ import annotations

from hyppo.manager import Manager

HYPS = ["b", "b2", "a", "h", "c", "z"]
EDGES = [("b", "a"), ("a", "h"), ("c", "z")]  # z, c: unrelated to b


def _models(aic_b2, calls):
    def mk(h, metrics):
        def f(cfg):
            calls.append(h)
            return dict(metrics)
        return f

    return {
        "b": mk("b", {"r2": 0.9, "aic": 10.0}),
        "b2": mk("b2", {"r2": 0.9, "aic": aic_b2}),
        "a": mk("a", {"r2": 0.9}),
        "h": mk("h", {"r2": 0.9}),
        "c": mk("c", {"r2": 0.9}),
        "z": mk("z", {"r2": 0.9}),
    }


def _run(mgr, p, aic_b2, b2_version):
    calls: list[str] = []
    cfg = {h: {} for h in HYPS}
    cfg["a"] = {"p": p}
    cfg["z"] = {"p": p}
    res = mgr.orchestrate(
        hypotheses=HYPS,
        workflow_edges=EDGES,
        models=_models(aic_b2, calls),
        config=cfg,
        model_ids={h: f"{h}@1" for h in HYPS} | {"b2": f"b2@{b2_version}"},
        competes={"b": {"b2"}, "b2": {"b"}},
    )
    return sorted(calls), res


def _runs_1_to_3(mgr):
    calls, res = _run(mgr, "p1", aic_b2=50.0, b2_version=1)   # run 1: R1 for a, z
    assert calls == sorted(HYPS) and res["b"]["epistemic_status"] == "SUPPORTED"
    calls, res = _run(mgr, "p2", aic_b2=-5.0, b2_version=2)   # run 2: b revised
    assert res["b"]["epistemic_status"] == "SUPERSEDED"
    assert calls == ["a", "b2", "h", "z"]
    assert mgr.pending_stale == {"a", "h"}
    calls, _ = _run(mgr, "p2", aic_b2=-5.0, b2_version=2)     # run 3: a, h worked off
    assert calls == ["a", "h"] and mgr.pending_stale == set()


def test_old_parameter_record_not_read_after_revision(tmp_path):
    mgr = Manager(db_path=tmp_path / "e.db")
    _runs_1_to_3(mgr)
    calls, _ = _run(mgr, "p1", aic_b2=-5.0, b2_version=2)     # run 4: back to p1
    mgr.close()
    # a's p1 record predates the revision of b: it must be recomputed (and h
    # with it); z is not a consumer of b and reuses its p1 record from run 1
    assert calls == ["a", "h"]


def test_old_parameter_record_not_read_after_restart(tmp_path):
    db = tmp_path / "r.db"
    mgr = Manager(db_path=db)
    _runs_1_to_3(mgr)
    mgr.close()
    mgr2 = Manager(db_path=db)                                 # new session
    calls, _ = _run(mgr2, "p1", aic_b2=-5.0, b2_version=2)
    mgr2.close()
    assert calls == ["a", "h"]


def test_no_new_event_no_extra_recompute(tmp_path):
    """The epoch changes only on a revision event: a persisting SUPERSEDED status
    does not keep invalidating the descendants run after run."""
    mgr = Manager(db_path=tmp_path / "n.db")
    _runs_1_to_3(mgr)
    _run(mgr, "p1", aic_b2=-5.0, b2_version=2)                 # run 4
    calls, _ = _run(mgr, "p1", aic_b2=-5.0, b2_version=2)     # run 5: nothing changed
    calls_p2, _ = _run(mgr, "p2", aic_b2=-5.0, b2_version=2)  # run 6: p2 after revision
    mgr.close()
    assert calls == [] and calls_p2 == []                      # both records post-revision


def test_epochs_persisted(tmp_path):
    db = tmp_path / "p.db"
    mgr = Manager(db_path=db)
    _runs_1_to_3(mgr)
    # b2 is SUPERSEDED by b in run 1, b by b2 in run 2: one event each
    assert mgr.epochs == {"b": 1, "b2": 1}
    mgr.close()
    mgr2 = Manager(db_path=db)
    assert mgr2.epochs == {"b": 1, "b2": 1}
    mgr2.close()
