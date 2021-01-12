import os
import sys
import glob
import configparser
import subprocess
import socket
from multiprocessing.pool import Pool
import shutil

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter, medfilt

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import NullType
from pyspark.sql import SparkSession
from pyspark import SparkConf
import happybase
from gplearn.genetic import SymbolicRegressor

conf = SparkConf()

conf.setAppName('Test')
conf.set("spark.hadoop.yarn.resourcemanager.hostname", '172.17.14.17')
conf.set("spark.hadoop.yarn.resourcemanager.address", '172.17.14.17:8050')
conf.set('spark.executor.memoryOverhead', '10GB')
conf.set('spark.executor.instances', '12')
spark = SparkSession.builder.config(conf=conf).getOrCreate()


def read_from_hbase(dataset_name, atlas_name, table_name):
    connection = happybase.Connection()
    table = connection.table(table_name)

    data = []
    for key, data in table.scan():
        data.append(key, data)

    return data


@pandas_udf("region string, program string", PandasUDFType.GROUPED_MAP)
def extract_ROI_spark(data):
    people_path = data['path']

    X = data.drop(['target'], axis=1)
    feature_names = X.columns
    feature_names = data.drop(['target'], axis=1).columns
    y = data['target'].values
    split = int(0.7 * X.shape[0])
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]
    est_gp = SymbolicRegressor(population_size=1000,
                               tournament_size=20,
                               generations=150, stopping_criteria=0.001,
                               const_range=(-1, 1),
                               p_crossover=0.7, p_subtree_mutation=0.12,
                               p_hoist_mutation=0.06, p_point_mutation=0.12,
                               p_point_replace=1,
                               init_depth=(6, 10),
                               function_set=('mul', 'sub', 'div', 'add', 'cos'),
                               max_samples=0.9,
                               verbose=1,
                               metric='mse',
                               parsimony_coefficient=0.0005,
                               random_state=0,
                               n_jobs=1)

    est_gp.fit(X_train, y_train)

    j = 0
    for region in feature_names:
        if int(region[1:]) != j:
            break
            j += 1

    temp = pd.DataFrame([[str(j), est_gp._program]], columns=['region', 'prorgam'])
    return temp


if __name__ == '__main__':
    config_path = './config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)

    data_path = config['DATA']['data_path']
    atlas_name = config['ATLAS']['name']
    remove_confounds = bool(config['ATLAS']['remove_confounds'])
    n_jobs = int(config['GENERAL']['n_jobs'])

    data = read_from_hbase('HCP', 'harvord-axford', 'neuro_test')
    data_spark = spark.createDataFrame(data)

    df1 = data_spark.groupby("people_number").apply(extract_ROI_spark)
    print(df1.toPandas()['path'][0])