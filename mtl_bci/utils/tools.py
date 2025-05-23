import numpy as np
import torch
import datetime
import csv
import os
import copy
import importlib
import random
  
class DotDict(dict):
    """
    Recursive DotDict:
    a dictionary that supports dot.notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value
    
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, d):
        pass
            
    def __deepcopy__(self, memo=None):
        return DotDict(copy.deepcopy(dict(self), memo=memo))
    
def get_datetime():
    t = datetime.datetime.now()
    return t.strftime('%Y%m%d')

def set_random_seed(seed):
    """
    Set all the random initializations with a given seed

    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def write_log(filepath, name='test.csv', data=[], mode='w'):
    '''
    filepath: path to save
    data: list of data
    mode: a = update data to file, w = write a new file
    '''
    try:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        with open(os.path.join(filepath, name), mode) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(data)
    except IOError:
        raise Exception('I/O error')


def load_config(filepath, config_id=None, to_dotdict=False):
    """load experimeantal config from .py file 

    Args:
        filepath (str): path to load .py file
        config_id (str, optional): config id. Defaults to None.
        to_dotdict (bool, optional): convert to DotDict notation. Defaults to False

    Returns:
        dict: params
    """
    print('... loading config params from:', filepath, "--- config id: " + config_id if config_id else "")
    spec = importlib.util.spec_from_file_location(filepath.replace("/", ".").removesuffix(".py"), filepath)
    Module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Module)
    params = Module.config
    if to_dotdict:
        params = DotDict(params)
    return params


def load_config_function(filepath, dataset, monitor, classes=[0, 1], adaptive=None, to_dotdict=False, data_name="filter.zarr"):
    """load experimeantal config from .py file 

    Args:
        filepath (str): path to load .py file
        config_id (str, optional): config id. Defaults to None.
        to_dotdict (bool, optional): convert to DotDict notation. Defaults to False

    Returns:
        dict: params
    """

    spec = importlib.util.spec_from_file_location(filepath.replace("/", ".").removesuffix(".py"), filepath)
    Module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Module)
    params = Module.config(dataset, monitor, classes, adaptive, data_name)
    if to_dotdict:
        params = DotDict(params)
    return params