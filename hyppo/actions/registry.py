"""@action decorator and global registry — verbatim of the wfonto pattern."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Set, Type

from pydantic import BaseModel

from hyppo.actions.types import AgentRole, TrustLevel


@dataclass(frozen=True)
class ActionSpec:
    kind: str
    trust: TrustLevel
    inputs: Type[BaseModel]
    outputs: Type[BaseModel]
    allowed_roles: frozenset[AgentRole]
    fn: Callable
    docstring: str
    requires_audit: bool = False


ACTION_REGISTRY: "OrderedDict[str, ActionSpec]" = OrderedDict()


def action(
    *,
    kind: str,
    trust: TrustLevel,
    inputs: Type[BaseModel],
    outputs: Type[BaseModel],
    allowed_roles: Set[AgentRole],
    requires_audit: bool = False,
):
    """Decorator that registers a function as a typed Action."""
    def wrap(fn: Callable):
        if kind in ACTION_REGISTRY:
            raise ValueError(f"Action {kind!r} already registered")
        spec = ActionSpec(
            kind=kind,
            trust=trust,
            inputs=inputs,
            outputs=outputs,
            allowed_roles=frozenset(allowed_roles),
            fn=fn,
            docstring=(fn.__doc__ or "").strip(),
            requires_audit=requires_audit,
        )
        ACTION_REGISTRY[kind] = spec
        return fn
    return wrap


def get_action(kind: str) -> ActionSpec:
    if kind not in ACTION_REGISTRY:
        raise KeyError(f"No action registered for kind={kind!r}")
    return ACTION_REGISTRY[kind]


def clear_registry():
    """Test-only helper."""
    ACTION_REGISTRY.clear()
