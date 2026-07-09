"""Epistemic status of a hypothesis (Section 2, part2.tex:570-588).

A hypothesis carries an *epistemic status* distinct from its *execution* status
(SUCCESS/FAILED/SKIPPED, see :mod:`hyppo.runner`). The epistemic status records the
scientific verdict after a hypothesis has been evaluated against data:

* ``PROPOSED``   -- created, not yet evaluated;
* ``SUPPORTED``  -- quality metric clears the threshold (R^2 >= theta_sup);
* ``REFUTED``    -- quality metric below the threshold (R^2 < theta_sup);
* ``SUPERSEDED`` -- a competing hypothesis is decisively better
  (Delta AIC = AIC(h) - AIC(h') > theta_aic), regardless of R^2.

The transition is a pure function so it can be tested in isolation; the
:class:`hyppo.runner.Runner` is the only place that gathers the inputs (R^2, AIC,
competitors) and writes the resulting status. Defaults follow the dissertation:
``theta_sup = 0.7`` (Burnham-Anderson conservative threshold) and ``theta_aic = 10``
(Delta AIC > 10 == no empirical support for the worse model).
"""

from __future__ import annotations

from enum import Enum


class EpistemicStatus(Enum):
    """Scientific verdict on a hypothesis after evaluation against data.

    See the module docstring for the meaning of each member and the
    thresholds (``theta_sup``, ``theta_aic``) used to derive them.
    """

    PROPOSED = "PROPOSED"
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    SUPERSEDED = "SUPERSEDED"


def evaluate_status(
    r2: float | None,
    own_aic: float | None = None,
    best_competitor_aic: float | None = None,
    *,
    theta_sup: float = 0.7,
    theta_aic: float = 10.0,
) -> EpistemicStatus:
    """Return the epistemic status of a hypothesis from its evaluation metrics.

    Args:
        r2: coefficient of determination on the test data; ``None`` if the
            hypothesis has not been evaluated yet.
        own_aic: AIC of this hypothesis (needed only for the SUPERSEDED check).
        best_competitor_aic: smallest AIC among competing hypotheses (``competes``),
            or ``None`` if there are no evaluated competitors.
        theta_sup: R^2 support threshold (default 0.7).
        theta_aic: Delta AIC threshold for being superseded (default 10).

    Precedence (part2.tex:584): SUPERSEDED dominates -- a hypothesis decisively
    beaten by a competitor is superseded even if its own R^2 clears the threshold.

    Returns:
        EpistemicStatus: ``PROPOSED`` if ``r2`` is ``None``; else ``SUPERSEDED``
        if decisively beaten by a competitor; else ``SUPPORTED`` if
        ``r2 >= theta_sup``; else ``REFUTED``.
    """
    if r2 is None:
        return EpistemicStatus.PROPOSED
    if (
        own_aic is not None
        and best_competitor_aic is not None
        and own_aic - best_competitor_aic > theta_aic
    ):
        return EpistemicStatus.SUPERSEDED
    if r2 >= theta_sup:
        return EpistemicStatus.SUPPORTED
    return EpistemicStatus.REFUTED
