"""Sanity tests for the hyppo action registry."""

import pytest
from pydantic import BaseModel

from hyppo.actions.registry import (
    ACTION_REGISTRY,
    action,
    clear_registry,
    get_action,
)
from hyppo.actions.types import AgentRole, TrustLevel


@pytest.fixture(autouse=True)
def _preserve_action_registry():
    """Snapshot ACTION_REGISTRY before each test; restore on teardown.

    These tests call clear_registry() and register synthetic OrderA/OrderB
    fixtures. Without this fixture, suite-wide ordering after this file
    would see a 1- or 2-entry registry instead of the 8 production actions
    (test_mcp_server_factory et al. would fail).
    """
    snapshot = dict(ACTION_REGISTRY)
    yield
    ACTION_REGISTRY.clear()
    ACTION_REGISTRY.update(snapshot)


class _FooIn(BaseModel):
    x: int


class _FooOut(BaseModel):
    y: int


def test_register_via_decorator_populates_registry():
    clear_registry()

    @action(
        kind="DemoSquare",
        trust=TrustLevel.SAFE,
        inputs=_FooIn,
        outputs=_FooOut,
        allowed_roles={AgentRole.ReservoirEngineer},
    )
    def _square(payload: _FooIn) -> _FooOut:  # noqa: ANN001
        return _FooOut(y=payload.x**2)

    spec = get_action("DemoSquare")
    assert spec.trust is TrustLevel.SAFE
    assert spec.requires_audit is False
    assert AgentRole.ReservoirEngineer in spec.allowed_roles
    assert spec.fn(_FooIn(x=4)).y == 16


def test_duplicate_kind_raises():
    clear_registry()

    @action(
        kind="UniqueDemo",
        trust=TrustLevel.SAFE,
        inputs=_FooIn,
        outputs=_FooOut,
        allowed_roles={AgentRole.Coordinator},
    )
    def _a(_):  # noqa: ANN001
        return _FooOut(y=0)

    with pytest.raises(ValueError, match="already registered"):

        @action(
            kind="UniqueDemo",
            trust=TrustLevel.SAFE,
            inputs=_FooIn,
            outputs=_FooOut,
            allowed_roles={AgentRole.Coordinator},
        )
        def _b(_):  # noqa: ANN001
            return _FooOut(y=1)


def test_unknown_kind_lookup_raises():
    clear_registry()
    with pytest.raises(KeyError):
        get_action("DoesNotExist")


def test_decorator_keeps_insertion_order():
    clear_registry()

    @action(
        kind="OrderA",
        trust=TrustLevel.SAFE,
        inputs=_FooIn,
        outputs=_FooOut,
        allowed_roles={AgentRole.Auditor},
    )
    def _a(_):  # noqa: ANN001
        return _FooOut(y=0)

    @action(
        kind="OrderB",
        trust=TrustLevel.SAFE,
        inputs=_FooIn,
        outputs=_FooOut,
        allowed_roles={AgentRole.Auditor},
    )
    def _b(_):  # noqa: ANN001
        return _FooOut(y=0)

    names = list(ACTION_REGISTRY.keys())
    assert names.index("OrderA") < names.index("OrderB")
