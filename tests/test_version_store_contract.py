"""Contract sanity for hyppo.versioning.version_store.

These tests do NOT hit Postgres. They check the module exposes the right
async coroutine functions with the right names — later action tests
monkeypatch them with AsyncMocks.
"""
import asyncio
import inspect

import pytest


EXPECTED_FUNCTIONS = {
    "insert_hypothesis_version",
    "find_latest_active",
    "select_version_by_id",
    "select_versions_by_kind",
    "select_superseding_versions",
    "select_runs_for_version",
    "upsert_run_link",
}


def test_module_exports_expected_async_functions():
    from hyppo.versioning import version_store

    missing = set()
    not_coro = set()
    for name in EXPECTED_FUNCTIONS:
        fn = getattr(version_store, name, None)
        if fn is None:
            missing.add(name)
        elif not asyncio.iscoroutinefunction(fn):
            not_coro.add(name)
    assert not missing, f"missing exports: {missing}"
    assert not not_coro, f"not async: {not_coro}"


def test_wfworker_grpc_address_helper_is_present():
    """Mirror wfonto.mcp.wfworker_client — gives bridge clients a uniform
    way to read connection config, even though we won't use gRPC."""
    from hyppo.versioning import version_store
    assert hasattr(version_store, "database_url"), (
        "Expected a database_url() helper that reads DATABASE_URL env var"
    )
