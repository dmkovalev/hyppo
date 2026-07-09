"""Local filesystem object store based on cloudpickle.

Usage:
    from hyppo.storage import Database

    Database.set_root('/tmp/my_db')
    Database.save([1, 2, 3], 'my_object', storage='sub_dir', description='numbers')
    wrapped = Database.load('my_object', storage='sub_dir')
    wrapped.obj  # -> [1, 2, 3]
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cloudpickle as pickle

logger = logging.getLogger(__name__)


class Pickled:
    """Wrapper for objects persisted through Storage.

    Supported hooks:
    - save_data_hook (called before the object is saved)
    - load_data_hook (called after the object is loaded)

    Note:
        By default the save time is recorded in `when_saved`.

    Args:
        obj (Any): Object to store.

    Attributes:
        when_saved (datetime): Time the object was saved.
        description (str): Description of the stored object.
    """

    def __init__(self, obj: Any):
        self.obj = obj
        self.when_saved: datetime = datetime.now()
        self.description: Optional[dict] = None

    def save_data_hook(self, **kwargs: Optional[dict]) -> None:
        """Hook called before saving the object."""
        self.when_saved = datetime.now()
        self.description = kwargs.get("description")

    def load_data_hook(self) -> None:
        """Hook called after loading the object."""
        if self.description:
            logger.info("Description: %s", self.description)


class Database:
    """Static-method-only class implementing an object database (Storage) on local disk.

    The core functionality saves objects into a subdirectory (or nested
    subdirectories) relative to the root stored in the `root` attribute.
    The subdirectory is passed as an argument to the load/save methods.

    Note:
        Using cloudpickle allows saving almost any object.
        By default all files use the `.pickle` extension.

    Attributes:
        root (Path): Root directory objects are resolved against.
    """

    root: Path = Path("./")

    @classmethod
    def set_root(cls, root: str) -> None:
        """Set the database root used to resolve objects.

        Args:
            root (str): Database root directory.
        """
        cls.root = Path(root)

    @classmethod
    def save(
        cls,
        obj: Any,
        filename: Union[str, Path],
        storage: Union[str, Path] = "",
        **kwargs: Dict[str, Any],
    ) -> None:
        """Save an object to the database.

        Note:
            Required subfolders are created automatically.

        Args:
            filename (str): Name to save under. Saved as `{filename}.pickle`.
            obj (Any): Object to save.
            storage (Union[str, Path], optional): Folder or sequence of nested folders
                relative to the root to save the object into. If not given, the object
                is saved into the root.
            kwargs (Dict[str, Any], optional): Extra arguments stored on the wrapper
                object, e.g. `description`.

        Note:
            Pickling failures are logged, then re-raised.
        """
        filename = Path(filename)

        logger.debug("Current directory: %s", Path.cwd())
        logger.debug("Database root: %s", cls.root)
        logger.debug("Storage: %s", storage or "./")

        cls.root.mkdir(parents=True, exist_ok=True)
        if not filename.is_absolute():
            (cls.root / storage / filename).parent.mkdir(parents=True, exist_ok=True)
        filename = Path(f"{filename}.pickle")

        try:
            pickled_obj = Pickled(obj=obj)
            pickled_obj.save_data_hook(**kwargs)
            with open(cls.root / storage / filename, "wb") as f:
                pickle.dump(pickled_obj, f)
        except Exception:
            logger.exception(
                "Object type does not match the database. Object type: %s", type(obj)
            )
            raise

    @classmethod
    def load(
        cls, filename: Union[str, Path], storage: Union[str, Path] = ""
    ) -> Optional[Pickled]:
        """Load an object from the database.

        Args:
            filename (str): File name.
            storage (Union[str, Path], optional): Folder or sequence of nested folders
                relative to the root to load from. If not given, the object is loaded
                from the root.

        Returns:
            Optional[Pickled]: The wrapped stored object, or None if not found.
        """
        try:
            filename = Path(f"{filename}.pickle")
            logger.debug("Loading from: %s", cls.root / storage / filename)
            with open(cls.root / storage / filename, "rb") as f:
                obj = pickle.load(f)
            obj.load_data_hook()
            return obj
        except FileNotFoundError:
            logger.warning("Object is missing from the database.")
            return None

    @classmethod
    def delete(cls, filename: Union[str, Path], storage: Union[str, Path] = ""):
        """Delete an object from the database.

        Args:
            filename (str): Path to the file relative to the database root.
            storage (Union[str, Path], optional): Folder or sequence of nested folders
                relative to the root to delete from.
        """
        try:
            filename = Path(f"{filename}.pickle")
            (cls.root / storage / filename).unlink()
        except FileNotFoundError:
            logger.warning("Object is missing from the database.")

    @classmethod
    def all_storages(cls, storage: Union[str, Path] = "") -> List[str]:
        """Return all subfolders (storages) contained in {root}/{storage}.

        Args:
            storage (Union[str, Path], optional): Folder or sequence of nested folders
                relative to the root to search in.

        Returns:
            List[str]: List of subfolder names.
        """
        return [p.stem for p in (cls.root / storage).glob("*") if p.is_dir()]

    @classmethod
    def get_all_names(
        cls, storage: Union[str, Path] = "", ext: str = "pickle"
    ) -> List[str]:
        """Return the names of all files contained in {root}/{storage}.

        Args:
            storage (Union[str, Path], optional): Folder or sequence of nested folders
                relative to the root to search in.
            ext (str): File extension. Defaults to 'pickle'.

        Returns:
            List[str]: List of all object names in the database.
        """
        return [f.stem for f in cls.root.glob(f"{storage}/*.{ext}") if f.is_file()]

    @classmethod
    def load_all(cls, storage: Union[str, Path] = "") -> List[Pickled]:
        """Load all objects contained in {root}/{storage}.

        Args:
            storage (Union[str, Path], optional): Folder or sequence of nested folders
                relative to the root to search in.

        Note:
            Access the `obj` attribute to get the actual stored object.

        Returns:
            List[Pickled]: List of wrapped stored objects.
        """
        # get_all_names lists existing files, so load never returns None here.
        objs = [
            cls.load(f, storage=storage) for f in cls.get_all_names(storage=storage)
        ]
        return objs  # type: ignore[return-value]


db = Database
