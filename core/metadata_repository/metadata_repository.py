import cloudpickle as pickle
from storage.core.pickled import Pickled
from pathlib import Path
from typing import Dict, Any, List, Union

from datetime import datetime


class Database:
    root: Path = Path('./')
    debug: bool = True
    debug_print_margin: str = '\t'

    @classmethod
    def set_root(cls, root: str) -> None:
        cls.root = Path(root)

    @classmethod
    def _save(cls, obj: Any, filename: str, storage: str = '', **kwargs: Dict[str, Any]) -> None:

        filename = Path(filename)

        if cls.debug:
            print(f'{cls.debug_print_margin}Текущая директория: {Path.cwd()}')
            print(f'{cls.debug_print_margin}Корень базы данных: {cls.root}')
            print(f'{cls.debug_print_margin}Хранилище: {storage}')

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
    def load(db, type, id):
        db.connect()
        try:
            print(f'Loading from: {cls.root / storage / filename}') if cls.debug else None
            with open(cls.root / storage / filename, 'rb') as f:
                return pickle.load(f)

            if type == 'hypothesis':
                artefact = db._load_hypothesis(id)
            elif type == 'model':
                artefact = db._load_model(id)
            elif type == 'ontology':
                artefact =  db._load_ontology(id)
            elif type == 'workflow':
                artefact =  db._load_workflow(id)
            elif type == 'lattice':
                artefact = db._load_lattice(id)

            return artefact
        except FileNotFoundError:
            print('Объект отсутствует в базе данных.')

    @classmethod
    def save(db, artefact) -> None:
        db.connect()
        _save(artefact)
        return

    @classmethod
    def _all_storages(cls) -> List[str]:
        """
        Метод, отдающий список `баз данных` содержащихся в корневой базе данных

        Returns: List[str]
        --------
            Список `баз данных` в виде массива строк
        """
        return [p.stem for p in cls.root.glob('*') if p.is_dir()]

    @classmethod
    def _get_all_names(cls, storage: Union[str, Path] = '', ext: str = 'pickle') -> List[str]:
        """
        Метод, отдающий список `баз данных` содержащихся в корневой базе данных

        Parameters:
        -----------
        ext: str
            Расширение файлов
        Returns: List[str]
        --------
            Список всех названий объектов в базе данных в виде массива строк
        """
        return [f.stem for f in cls.root.glob(f'{storage}/*.{ext}') if f.is_file()]

    @classmethod
    def _load_all(cls) -> List[Pickled]:

        objs = [cls.load(f) for f in cls.get_all_names()]
        return objs


db = Database

def load_yaml(filepath, type):
    pass


def save_yaml(filepath, artefact):
    pass


def create_yaml(is_csv=True, is_cut=True, name='dataset.csv'):

#     data = [pd.read_csv(path) for path in data_path]

    db._return()

    dataset.loc[:, 'datetime'] = \
        dataset.datetime.apply(_adjust_date)

    modified_data = []
    for df, path in zip(data, [os.path.basename(x) for x in data_path]):
        df.time = pd.to_datetime(df.time,
                                    format='%Y-%m-%d %H:%M:%S')
        # format='%d-%b-%y %I.%M.%S.%f %p')
        df.columns = map(str.lower, df.columns)
        modified_data.append(df)

    output = pd.concat(modified_data, sort=False)

    if is_csv:
        output.to_csv('../data/' + name + '.csv', sep=',', index=False)

    return output


def _create_params(interval,
                   step,
                   start_date=None,
                   end_date=None,
                   from_db=False,
                   from_file=True):

    if from_db:
        engine = (
            create_engine(
                'postgresql://postgres:localhost@localhost:54321/metadata_repository'))

        db_connection = engine.connect()
        all_df = pd.read_sql("select * from metadata_repository", db_connection)
    else:
        if from_file:
            all_df = pd.read_csv('../data/metadata_repository.csv',
                                 sep=',',
                                 parse_dates=['time'])
        else:
            all_df = _create_csv()

    params = []

    for name, group in all_df.groupby('main_column'):
        group.set_index(['time'], inplace=True)
        group['main_column'] = name
        start_date = group.index.min()
        end_date = group.index.max()
        while start_date + pd.Timedelta(interval, unit='m') < end_date:
            params.append([group[(group.index > start_date) &
                                  (group.index < start_date +
                                   pd.Timedelta(interval,
                                                unit='m'))]
                           ])
            start_date += pd.Timedelta(step, unit='m')

    return params

