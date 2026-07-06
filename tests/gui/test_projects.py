from hyppo.gui.projects import ProjectStore


def test_create_and_list(tmp_path):
    store = ProjectStore(db_path=str(tmp_path / "p.db"))
    pid = store.create(name="demo", description="d")
    assert store.get(pid)["name"] == "demo"
    assert [p["id"] for p in store.list()] == [pid]


def test_delete(tmp_path):
    store = ProjectStore(db_path=str(tmp_path / "p.db"))
    pid = store.create(name="x")
    store.delete(pid)
    assert store.get(pid) is None
