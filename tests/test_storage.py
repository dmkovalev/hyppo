import pytest

from hyppo.storage import Database


def test_save_unpicklable_raises(tmp_path):
    Database.set_root(str(tmp_path))
    unpicklable = (x for x in range(3))  # generators cannot be pickled
    with pytest.raises(Exception):
        Database.save(unpicklable, "unpicklable_object")


def test_failed_save_leaves_no_artifact(tmp_path):
    Database.set_root(str(tmp_path))
    unpicklable = (x for x in range(3))
    with pytest.raises(Exception):
        Database.save(unpicklable, "broken")
    assert not (tmp_path / "broken.pickle").exists()
    assert not (tmp_path / "broken.pickle.tmp").exists()
    assert Database.load("broken") is None


def test_failed_save_keeps_previous_version(tmp_path):
    Database.set_root(str(tmp_path))
    Database.save([1, 2, 3], "stable")
    with pytest.raises(Exception):
        Database.save((x for x in range(3)), "stable")
    loaded = Database.load("stable")
    assert loaded is not None and loaded.obj == [1, 2, 3]


def test_save_load_roundtrip(tmp_path):
    Database.set_root(str(tmp_path))
    Database.save([1, 2, 3], "my_object")
    loaded = Database.load("my_object")
    assert loaded.obj == [1, 2, 3]


def test_description_preserved(tmp_path):
    Database.set_root(str(tmp_path))
    Database.save({"a": 1}, "described", description="A dict with one key")
    loaded = Database.load("described")
    assert loaded.description == "A dict with one key"


def test_load_missing_returns_none(tmp_path):
    Database.set_root(str(tmp_path))
    result = Database.load("does_not_exist")
    assert result is None


def test_subdirectory_storage(tmp_path):
    Database.set_root(str(tmp_path))
    Database.save("nested value", "nested_object", storage="sub_dir")
    assert (tmp_path / "sub_dir" / "nested_object.pickle").exists()
    loaded = Database.load("nested_object", storage="sub_dir")
    assert loaded.obj == "nested value"
