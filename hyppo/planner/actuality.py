"""Actuality of a stored result (definition def:12 of the dissertation).

A stored result of hypothesis ``h`` is *actual* for the target configuration iff
(1) the induced subgraph of its sources is the same, (2) ``h`` and every source
are implemented by the same models, (3) the projection of the configuration onto
the parameters of ``h`` and its sources is the same, and (4) neither ``h`` nor a
source is marked ``StaleHypothesis`` by the ontology layer.

(1)-(3) are checked by ONE key comparison with a *chained* cache key (as in Nix /
Bazel): the key of ``h`` hashes its own configuration, its model identifier and
the keys of its direct predecessors, so any change of a source's model, version or
parameters changes the keys of all its descendants. (4) covers revisions that
touch neither models nor configurations (e.g. a source SUPERSEDED by a rival,
witness delta_1' of prop:hidden_stale) and is checked separately.

Revision epochs: a hypothesis g carries an epoch eps_g, incremented on every
revision event of g (its status became REFUTED / SUPERSEDED). The key of h
includes ``(g, K(g), eps_g)`` for every direct predecessor g, so every record of
a consumer of g made before the revision has a different key and is never read
again, in any parameter context. The marks of (4) are then only a fast path
within the run sequence; safety across runs does not depend on them. With
eps_g = 0 the entry is the bare digest of K(g), so keys of graphs without
revision events are unchanged.

Writers (the runner) and readers (the planner / manager) must use the same keys:
compute them with :func:`chained_keys` and pass them as the cache configuration.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def chained_keys(
    nodes_topo: Iterable,
    edges: Iterable[tuple],
    configs: Mapping | None = None,
    model_ids: Mapping | None = None,
    epochs: Mapping | None = None,
) -> dict:
    """Chained cache configurations ``{h: key_dict}`` in one topological pass.

    ``key_dict = {"cfg": own config, "model": model id, "up": {pred: digest}}``;
    O(|V| + |E|). ``nodes_topo`` must list every node after its predecessors;
    ``edges`` are derived_by edges ``(source, dependent)``. ``epochs`` are the
    revision epochs eps_g of the predecessors (absent or 0: ``digest``,
    else ``digest@eps_g``)."""
    configs = configs or {}
    model_ids = model_ids or {}
    epochs = epochs or {}
    preds: dict = {}
    for u, v in edges:
        preds.setdefault(v, set()).add(u)
    keys: dict = {}
    for h in nodes_topo:
        up = {
            str(p): _digest(keys[p]) + (f"@{epochs[p]}" if epochs.get(p) else "")
            for p in sorted(preds.get(h, ()), key=str)
            if p in keys
        }
        keys[h] = {"cfg": configs.get(h, {}), "model": model_ids.get(h), "up": up}
    return keys


def is_marked_stale(hypothesis) -> bool:
    """Condition (4) of def:12: the ontology layer marked the hypothesis stale."""
    try:
        from hyppo.ontology.core_rules import StaleHypothesis
    except Exception:  # pragma: no cover - ontology layer unavailable
        return False
    return StaleHypothesis in getattr(hypothesis, "is_a", ())
