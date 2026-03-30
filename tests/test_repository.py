"""Tests for MetadataRepository."""
import pytest


def test_save_and_load_result(tmp_path):
    from hyppo.metadata_repository import MetadataRepository
    repo = MetadataRepository(db_path=tmp_path / "test.db")
    repo.save_result("h1", {"lr": 0.01}, {"r2": 0.85}, "SUCCESS")
    result = repo.load_result("h1", {"lr": 0.01})
    assert result is not None
    assert result["metrics"]["r2"] == 0.85
    assert result["status"] == "SUCCESS"
    repo.close()


def test_has_result(tmp_path):
    from hyppo.metadata_repository import MetadataRepository
    repo = MetadataRepository(db_path=tmp_path / "test.db")
    assert not repo.has_result("h1", {"lr": 0.01})
    repo.save_result("h1", {"lr": 0.01}, {"r2": 0.9}, "SUCCESS")
    assert repo.has_result("h1", {"lr": 0.01})
    repo.close()


def test_find_nearest_lattice(tmp_path):
    from hyppo.metadata_repository import MetadataRepository
    repo = MetadataRepository(db_path=tmp_path / "test.db")
    repo.save_lattice("l1", nodes={"h1", "h2"}, edges={("h1", "h2")})
    nearest = repo.find_nearest_lattice(
        nodes={"h1", "h2", "h3"}, edges={("h1", "h2"), ("h2", "h3")}
    )
    assert nearest is not None
    assert nearest["lattice_id"] == "l1"
    assert nearest["distance"] == 2  # 1 extra node + 1 extra edge
    repo.close()


def test_in_memory_repo():
    from hyppo.metadata_repository import MetadataRepository
    repo = MetadataRepository()  # :memory:
    repo.save_result("h1", {}, {"r2": 0.5}, "SUCCESS")
    assert repo.has_result("h1", {})
    repo.close()
