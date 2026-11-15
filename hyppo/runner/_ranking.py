"""Comparable-group ranking used by Runner's status protocol.

The caller supplies only rivalry edges between models evaluated on the same
observations and response. Semantic comparability cannot be inferred from scores.
"""
from __future__ import annotations

import math

from hyppo.comparison import combined_ranking


def status_ranking(results, competes, weights=None):
    """Return effective AIC/AICc and combined scores for evaluated rivals.

    Metrics: r2, aic, optionally bic/stat; n and k are sample size and total
    likelihood parameter count (including estimated variance). If any group
    member has n/k < 40, use AICc for every member, supplied as aicc or computed
    from n,k. Legacy groups without n,k use the explicitly supplied AIC.
    """
    weights = {'aic': 1.0} if weights is None else dict(weights)
    if (not weights or set(weights) - {'r2', 'aic', 'bic', 'stat'}
            or any(not math.isfinite(v) or v < 0 for v in weights.values())
            or not math.isclose(sum(weights.values()), 1.0)):
        raise ValueError('ranking_weights must be nonnegative and sum to one')
    weights = {key: value for key, value in weights.items() if value > 0}
    metrics = {
        h: r['metrics'] for h, r in results.items()
        if r['status'] == 'SUCCESS' and r['metrics'].get('r2') is not None
        and r['metrics'].get('aic') is not None
    }
    adjacency = {h: set() for h in metrics}
    for h, rivals in competes.items():
        for rival in rivals:
            if h in metrics and rival in metrics:
                adjacency[h].add(rival)
                adjacency[rival].add(h)
    effective, scores = {}, {}
    pending = set(metrics)
    while pending:
        seed = pending.pop()
        group, stack = {seed}, [seed]
        while stack:
            for h in adjacency[stack.pop()]:
                if h not in group:
                    group.add(h)
                    pending.discard(h)
                    stack.append(h)
        if len(group) < 2:
            continue
        for h in group:
            m = metrics[h]
            if ('n' in m) != ('k' in m):
                raise ValueError('provide both n and k for AICc selection')
            if 'n' in m and (not math.isfinite(m['n']) or not math.isfinite(m['k'])
                             or m['n'] <= 0 or m['k'] < 0):
                raise ValueError('invalid n or k')
        sizes = {metrics[h]['n'] for h in group if 'n' in metrics[h]}
        if len(sizes) > 1:
            raise ValueError('rivals must use the same observations')
        corrected = any(
            m.get('k', 0) > 0 and m['n'] / m['k'] < 40
            for m in (metrics[h] for h in group)
        )
        values = {}
        for h in group:
            m = metrics[h]
            aic = m['aic']
            if corrected:
                if 'n' in m and m['n'] <= m['k'] + 1:
                    raise ValueError('AICc requires n > k + 1')
                if 'aicc' in m:
                    aic = m['aicc']
                elif 'n' in m:
                    aic += 2 * m['k'] * (m['k'] + 1) / (m['n'] - m['k'] - 1)
                else:
                    raise ValueError('AICc comparison requires aicc or n,k for every rival')
            effective[h] = aic
            values[h] = {}
            for key in weights:
                value = aic if key == 'aic' else m.get(key)
                if value is None or not math.isfinite(value):
                    raise ValueError(f'missing or nonfinite ranking metric: {key}')
                values[h][key] = -value if key in {'aic', 'bic'} else value
            if not math.isfinite(aic):
                raise ValueError('nonfinite information criterion')
        scores.update(combined_ranking(values, weights))
    return effective, scores
