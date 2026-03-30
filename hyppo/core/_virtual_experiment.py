"""VirtualExperiment — formal specification of a virtual experiment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class VirtualExperiment:
    """Formal representation of a virtual experiment as tuple <O, H, M, R, W, C>.

    Corresponds to Definition 1 from Chapter 2 of the dissertation.
    """

    experiment_id: str
    hypotheses: list[str] = field(default_factory=list)
    models: dict[str, Callable] = field(default_factory=dict)
    workflow_edges: list[tuple[str, str]] = field(default_factory=list)
    config: dict[str, dict] = field(default_factory=dict)
    description: str = ""
