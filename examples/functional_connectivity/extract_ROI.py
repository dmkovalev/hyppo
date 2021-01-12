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

from nilearn.input_data import NiftiLabelsMasker
from nilearn import image
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
import nibabel
import zipfile
import yaml

from scipy.signal import savgol_filter, medfilt

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import NullType
from pyspark.sql import SparkSession
from pyspark import SparkConf
import happybase

conf = SparkConf()

conf.setAppName('Test')
conf.set("spark.hadoop.yarn.resourcemanager.hostname", '172.17.14.17')
conf.set("spark.hadoop.yarn.resourcemanager.address", '172.17.14.17:8050')
conf.set('spark.executor.memoryOverhead', '10GB')
conf.set('spark.executor.instances', '12')
spark = SparkSession.builder.config(conf=conf).getOrCreate()


def get_ROI(path, masker, delete_confounds):
    if delete_confounds:
        confounds = image.high_variance_confounds(path, n_confounds=5)
        return masker.transform(path, confounds=confounds)
    else:
        return masker.transform(path)


def get_masker(atlas_name, atlas_path):
    if atlas_name == 'harvard_oxford':
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', data_dir=atlas_path)
    else:
        raise OSError('there is no atlas {0}'.format(atlas_name))
    atlas_filename = atlas.maps

    masker = NiftiLabelsMasker(labels_img=atlas_filename,
                               memory='nilearn_cache')
    masker.fit()
    return masker


def extract_archive_HCP(archive_path, path_to_save):
    archive = zipfile.ZipFile(archive_path)
    people_number = int(archive_path.split('/')[-1].split('_')[0])
    rest_number = archive_path.split('/')[-1].split('_')[3]
    path_in_archive = '{0}/MNINonLinear/Results/rfMRI_{1}_{2}/rfMRI_{1}_{2}.nii.gz'
    pathes = []
    for file in archive.namelist():
        if file.startswith(path_in_archive.format(people_number, rest_number, 'LR')) or file.startswith(
                path_in_archive.format(people_number, rest_number, 'RL')):
            archive.extract(file, path_to_save)
            pathes.append(path_to_save + '/' + str(file))
    return pathes


def write_to_hbase(data, atlas, table_name, person, person_number):
    connection = happybase.Connection()
    table = connection.table(table_name)

    data = data.to_json().encode('utf-8')
    row = 'HCP_' + person
    row = row.encode('utf-8')
    table.put(row, {b'data:person': person_number.encode('utf-8'),
                    b'data:dataset': b'HCP',
                    b'data:atlas': atlas.encode('utf-8'),
                    b'data:roi': data})


@pandas_udf("path string, people_number long", PandasUDFType.GROUPED_MAP)
def extract_ROI_spark(data):
    people_path = data['path']
    people_number = np.unique(data['people_number'])[0]

    try:
        os.mkdir('/tmp/nilearn_data')
    except Exception as e:
        a = 1

    atlas_path = '/data/test/nilearn_data'
    masker = get_masker(atlas_name, atlas_path)
    j = 0

    for person in zip(people_path, [people_number] * data.shape[1]):
        pathes = extract_archive_HCP(person[0], '/tmp/nilearn_data')
        j += 1

        for person_path in pathes:
            data_person = get_ROI(person_path, masker, True)
            columns = ['x' + str(i) for i in range(data_person.shape[1])]

            data_person = pd.DataFrame(data_person, columns=columns)
            save_path = config['ATLAS']['path_to_write'] + str(people_number) + '_{}.csv'.format(j)
            j += 1
            if config['ATLAS']['where_to_write'] == 'hdd':
                data_person.to_csv(save_path, index=False)
            elif config['ATLAS']['where_to_write'] == 'base':
                write_to_hbase(data_person, config['ATLAS']['name'],
                               config['DATABASE']['table_name'], str(person), str(person) + str(j))

            os.remove(person_path)
    shutil.rmtree('/tmp/nilearn_data/' + str(people_number))
    temp = pd.DataFrame([[pathes[0], people_number]], columns=['path', 'people_number'])
    return temp



def get_hyperparameters(job_id):
    # Update file name with correct path
    with open("./hyperparams.yml", 'r') as stream:
        hyper_param_set = yaml.load(stream)

    return hyper_param_set[job_id - 1]["hyperparam_set"]


def run_ROI_kubernetes():
    config_path = os.environ['CONFIG_PATH']
    job_id = int(os.environ['JOB_ID'])

    config = configparser.ConfigParser()
    config.read(config_path)
    data_path = config['GENERAL']['data_path']
    atlas_name = config['ATLAS']['atlas_name']
    remove_confounds = bool(config['ATLAS']['remove_confounds'])
    n_jobs = int(config['GENERAL']['n_jobs'])

    data = pd.read_csv(data_path)
    people_path = data['path']
    people_number = data['people_number']

    person_path = data.loc[job_id]['path']
    person_number = data.loc[job_id]['people_number']
    masker = get_masker(atlas_name)
    data = get_ROI(person_path, masker, remove_confounds)
    columns = ['x' + str(i) for i in range(data.shape[1])]
    data = pd.DataFrame(data, columns=columns)
    save_path = config['ATLAS']['path_to_write'] + str(person_number) + '.csv'
    if config['ATLAS']['where_to_write'] == 'hdd':
        data.to_csv(save_path, index=False)
    else:
        # TODO database
        pass


def extract_ROI_kubernetes(number_of_person):
    bashCommand = "bash create_jobs.sh {0}".format(number_of_person)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

    bashCommand = "kubectl create -f hyperparam-jobs-spec"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()

if __name__ == '__main__':
    config_path = './config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)

    data_path = config['DATA']['data_path']
    atlas_name = config['ATLAS']['name']
    remove_confounds = bool(config['ATLAS']['remove_confounds'])
    n_jobs = int(config['GENERAL']['n_jobs'])

    masker = get_masker(atlas_name, None)

    data = pd.read_csv(data_path)
    # data = data[:3]
    print(data.shape)
    data_spark = spark.createDataFrame(data)

    df1 = data_spark.groupby("people_number").apply(extract_ROI_spark)
    print(df1.toPandas()['path'][0])
