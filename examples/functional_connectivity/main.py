import os
import sys
import glob
from ve_manager import virtual_experiment_manager
from ve_runner import virtual_experiment_runner

from coa_constructor  import coa_graph

if __name__ == '__main__':

    config_path = './config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)

    data_path = config['DATA']['data_path']
    atlas_name = config['ATLAS']['name']
    remove_confounds = bool(config['ATLAS']['remove_confounds'])
    n_jobs = int(config['GENERAL']['n_jobs'])

    specification = open("ve_spec.txt")
    atl_spec = open("atl_spec.txt")
    atl_spec_model = open("atl_spec_model.txt")
    conn_spec = open("conn_spec.txt")
    gender_spec = open("gender_spec.txt")

    ve = create_new_virtual_experiment(specification)

    atlas_hypothesis = Hypothesis(atl_spec)
    add(ve, 'hypothesis', atlas_hypothesis)

    atlas_model = Model(atl_spec_model):
    add(ve, 'Model', atlas_model)

    ve._add_relation('atlas_hypothesis', 'atlas_model')

    connectivity_hypothesis = Hypothesis(conn_spec)
    dataset = read_from_hbase('HCP', 'harvard-oxford', 'neuro_test')

    linear_connectivity_model = \
        generate_hypothesis_from_data(dataset, type='linear')
    nonlinear_connectivity_model = \
        generate_hypothesis_from_data(dataset)

    add(ve, 'hypothesis', connectivity_hypothesis)
    add(ve, 'model', linear_connectivity_model)
    add(ve, 'model', nonlinear_connectivity_model)

    ve._add_relation('connectivity_hypothesis', 'linear_connectivity_model')
    ve._add_relation('connectivity_hypothesis', 'nonlinear_connectivity_model')

    gender_hypothesis = Hypothesis(gender_spec)
    gender_model_men = coa_graph(nonlinear_connectivity_model)
    gender_model_women = coa_graph(nonlinear_connectivity_model)
    add(ve, 'hypothesis', gender_hypothesis)
    add(ve, 'model', gender_model_men)
    add(ve, 'model', gender_model_women)

    ve._add_relation('gender_hypothesis', 'gender_model_men')
    ve._add_relation('gender_hypothesis', 'gender_model_women')

    lattice = construct_lattice(ve)
    add(ve, 'lattice', lattice)

    run_virtual_experiment(ve)

