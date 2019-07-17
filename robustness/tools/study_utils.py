from collections import namedtuple
import json
import os
import numpy as np
import itertools

def load_config(path):
    with open(path) as config_file:
        config = json.load(config_file)
    if os.path.exists('job_parameters.json'):
        with open('job_parameters.json') as config_file:
            param_config = json.load(config_file)
        for k in param_config.keys():
            assert k in config.keys()
        config.update(param_config)
    return config

def get_config(args, path):

	args_dict = vars(args)
	args_dict = {k: args_dict[k] for k in args_dict if args_dict[k] is not None}

	json_dict = load_config(path)
	json_dict.update(args_dict)

	return namedtuple('Struct', json_dict.keys())(*json_dict.values())

def dict_product(d):
    '''
    Implementing itertools.product for dictionaries.
    E.g. {"a": [1,4],  "b": [2,3]} -> [{"a":1, "b":2}, {"a":1,"b":3} ..]
    Inputs:
    - d, a dictionary {key: [list of possible values]}
    Returns;
    - A list of dictionaries with every possible configuration
    '''
    keys = d.keys()
    vals = d.values()
    prod_values = list(itertools.product(*vals))
    all_dicts = map(lambda x: dict(zip(keys, x)), prod_values)
    return all_dicts

def get_study_config(params, rules):

    filt_params = dict_product(params)

    for rule in rules:
        filt_params = list(filter(rule, filt_params))
    return filt_params
