"""Shared metrics type (Definition 10, Chapter 2).

A single ``Metrics`` shape used across the runner, comparison, and
metadata-repository modules instead of ad-hoc ``dict`` annotations.
"""

from __future__ import annotations

from typing import TypedDict


class Metrics(TypedDict, total=False):
    """Model-evaluation metrics keyed by name.

    All keys optional: a model function may report any subset (e.g. only
    ``r2``, or ``aic``/``bic`` together). Runtime representation stays a
    plain ``dict`` with string keys; this type only documents the shape.
    """

    r2: float
    aic: float
    bic: float
    mse: float
