import cloudpickle as pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Union, Optional


class Pickled:
    """Обертка для объектов, сохраняющихся в Storage
    Поддерживаемые хуки:
    - save_data_hook (вызывается перед сохранением объекта)
    - load_data_hook (вызывается после загрузки объекта)
    Note:
        По умолчанию в атрибуты записывается время сохрвнения (`when_saved`).
    Args:
        obj (Any): Сохраняемый объект.
    Attributes:
        when_saved (datetime): Время сохранения объекта
        description (str): Описание сохраняемого файла.
    TODO:
        * Добавить сохранение всех kwargs, а не только description.
    """

    def __init__(self, obj: Any):
        self.obj = obj
        self.when_saved: datetime = datetime.now()
        self.description = None

    def save_data_hook(self, **kwargs: Optional[dict]) -> None:
        """Хук, вызывающийся до сохранения объекта."""
        self.when_saved = datetime.now()
        self.description = kwargs.get('description')

    def load_data_hook(self) -> None:
        """Хук, вызывающийся после загрузки объекта."""
        if self.description:
            print(f'Описание: {self.description}')


class Database:
    """Класс только со статическими методами, реализующий объектную базу данных (Storage) in local filesystem.
    Основной функционал реализует сохранение объектов в поддиректор(ию/ии)
    относительно корня, хранящегося в атрибуте `root`. Поддиректория передатется
    в качестве атрибута в методы загрузки и сохранения.
    Note:
        Использование cloudpickle позволяет сохранять практически любой объект.
        По умолчанию расширение всех файлов это `.pickle`
    Attributes:
        root (Path): Корень, относительно которого будут искаться объекты.
        debug (Bool): Выводить ли справочную информацию.
        debug_print_margin (str): Отступ при печати справочной информации.
    Examples:
        >>> db.set_root('/tmp')
        >>> a = [1, 2, 3]
        >>> db.save(a, 'my_object', storage='sub_dir', description='Список с числами')
        Текущая директория: /home/alice/Documents/projects/geost_animation
        Корень базы данных: /tmp
        Хранилище: sub_dir/
        >>> db.load('my_object', storage='sub_dir').obj
        Loading from: /tmp/sub_dir/my_object.pickle
        Описание: Очень важный список
        Out: [1, 2, 3]
    """

    root: Path = Path('./')
    debug: bool = True
    debug_print_margin: str = '\t'

    @classmethod
    def set_root(cls, root: str) -> None:
        """Задавание корня базы данных, относительно которого будет осуществляться
        поиск.
        Args:
            root (str): Корень базы данных.
        """
        cls.root = Path(root)

    @classmethod
    def save(cls, obj: Any, filename: str, storage: Union[str, Path] = '', **kwargs: Dict[str, Any]) -> None:
        """Метод сохранения объекта в базу данных.
        Note:
            Необходимые подпапки создаются автоматически.
        Args:
            filename (str): С каким имененм сохранять. Шаблон сохраненного файла `{filename}.pickle`.
            obj (Any): Объект, который необходимо сохранить.
            storage (Union[str, Path], optional): Папка или последовательности вложенных папок относительно корня,
                куда сохранить объект. Если параметр не указан, объект сохранится в корень.
            kwargs (Dict[str, Any], optional): Дополнительные аргументы, сохранящиеся в объект-обертку.
                Например, `description`.
        """
        filename = Path(filename)

        if cls.debug:
            print(f'{cls.debug_print_margin}Текущая директория: {Path.cwd()}')
            print(f'{cls.debug_print_margin}Корень базы данных: {cls.root}')
            print(f'{cls.debug_print_margin}Хранилище: {storage or "./"}')

        cls.root.mkdir(parents=True, exist_ok=True)
        if not filename.is_absolute():
            (cls.root / storage / filename).parent.mkdir(parents=True, exist_ok=True)
        filename = Path(f'{filename}.pickle')

        try:
            pickled_obj = Pickled(obj=obj)
            pickled_obj.save_data_hook(**kwargs)
            with open(cls.root / storage / filename, 'wb') as f:
                pickle.dump(pickled_obj, f)
        except Exception as e:
            print('Тип данных не соответвует базе данных.')
            print('Тип объекта: ', type(obj))
            print('Исключение:', e)

    @classmethod
    def load(cls, filename: str, storage: Union[str, Path] = '') -> Pickled:
        """Метод для загрузки объекта из базы данных.
        Args:
            filename (str): Название файла.
            storage (Union[str, Path], optional): Папка или последовательности вложенных папок
                относительно корня, откуда загружать объект. Если параметр не указан, объект сохранится
                в корень.
        Returns:
            Pickled: обернутый сохраненный объект.
        """
        try:
            filename = Path(f'{filename}.pickle')
            print(f'Loading from: {cls.root / storage / filename}') if cls.debug else None
            with open(cls.root / storage / filename, 'rb') as f:
                obj = pickle.load(f)
            obj.load_data_hook()
            return obj
        except FileNotFoundError:
            print('Объект отсутствует в базе данных.')

    @classmethod
    def delete(cls, filename: str, storage: Union[str, Path] = ''):
        """Метод для удаления объекта из базы данных.
        Parameters:
            filename (str): Путь к файлу относительно корня базы данных
            storage (Union[str, Path], optional): Папка или последовательности вложенных папок
                относительно корня, откуда удалять объект.
        """
        try:
            filename = Path(f'{filename}.pickle')
            (cls.root / storage / filename).unlink()
        except FileNotFoundError:
            print('Объект отсутствует в базе данных.')

    @classmethod
    def all_storages(cls, storage: Union[str, Path] = '') -> List[str]:
        """Метод, возвращающий список всех `подпапок` (storages), содержащихся в {root}/{storage}.
        Args:
            storage (Union[str, Path], optional): Папка или последовательности вложенных папок
                относительно корня, где иискать подпапки.
        Returns:
            List[str]: Список подпапок в виде массива строк.
        """
        return [p.stem for p in (cls.root / storage).glob('*') if p.is_dir()]

    @classmethod
    def get_all_names(cls, storage: Union[str, Path] = '', ext: str = 'pickle') -> List[str]:
        """Метод, возвращающий список имен `файлов` содержащихся в {root}/{storage}.

        Args:
            storage (Union[str, Path], optional): Папка или последовательности вложенных папок
                относительно корня, где иискать файлы.
            ext (str): Расширение файлов. Значение по умолчанию - 'pickle'.
        Returns:
            List[str]: Список всех названий объектов в базе данных в виде массива строк.
        """
        return [f.stem for f in cls.root.glob(f'{storage}/*.{ext}') if f.is_file()]

    @classmethod
    def load_all(cls, storage: Union[str, Path] = '') -> List[Pickled]:
        """Метод, загружающий все объекты из базы данных, содержащихся в {root}/{storage}.
        Args:
            storage (Union[str, Path], optional): Папка или последовательности вложенных папок
                относительно корня, где иискать файлы.
        Note:
            Чтобы получить сам сохраенный объект, необходимо обратиться к атрибуту `obj`.
        Returns:
            List[Pickled]: Список обернутых сохраенных объектов (типа Pickled).
        """
        objs = [cls.load(f, storage=storage) for f in cls.get_all_names(storage=storage)]
        return objs


db = Database