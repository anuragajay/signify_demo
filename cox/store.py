import pandas as pd
import torch as ch
import numpy as np
import dill as pickle
from uuid import uuid4
from .utils import *
import os
import warnings
from tensorboardX import SummaryWriter

TABLE_OBJECT_DIR = '.table_objects'
SAVE_DIR = 'save'
STORE_BASENAME = 'store.h5'
TENSORBOARD_DIR = 'tensorboard'
COX_DATA_KEY = 'COX_DATA'
PICKLE = '__pickle__'
OBJECT = '__object__'
PYTORCH_STATE = '__pytorch_state__'

pd.set_option('io.hdf.default_format','table')
from pandas.io.pytables import PerformanceWarning
warnings.simplefilter(action="ignore", category=PerformanceWarning)

class Store():
    '''
    Serializes and saves data from experiment runs
    Directly saves: int, float, torch scalar, string
    Saves and links: np.array, torch tensor, python object (via pickle)
    Errors: python list, python tuple
    '''

    OBJECT = OBJECT
    PICKLE = PICKLE
    PYTORCH_STATE = PYTORCH_STATE
    def __init__(self, storage_folder, exp_id=None, new=False, mode='a'):
        if not exp_id:
            exp_id = str(uuid4())

        exp_path = os.path.join(storage_folder, exp_id)
        if os.path.exists(exp_path) and new:
            raise ValueError("This experiment has already been run.")

        if not os.path.exists(exp_path):
            mkdirp(exp_path)
            print('Logging in: %s' % os.path.abspath(exp_path))

        # Start HDF file
        self.store = pd.HDFStore(os.path.join(exp_path, STORE_BASENAME), mode=mode)

        # Setup
        self.exp_id = exp_id
        self.path = os.path.abspath(exp_path)
        self.save_dir = os.path.join(exp_path, SAVE_DIR)
        self.tb_dir = os.path.join(exp_path, TENSORBOARD_DIR)

        # Where to save table objects
        self._table_object_dir = os.path.join(exp_path, TABLE_OBJECT_DIR)
        
        # API: http://tensorboardx.readthedocs.io/en/latest/tutorial.html#what-is-tensorboard-x
        if mode != 'r':
            self.tensorboard = SummaryWriter(self.tb_dir)
            mkdirp(self.save_dir)
            mkdirp(self._table_object_dir)
            mkdirp(self.tb_dir)

        self.tables = Table.tables_from_store(self.store, self._table_object_dir)
        self.keys = self.tables.keys()

    def close(self):
        self.store.close()

    def __str__(self):
        s = []
        for table_name, table in self.tables.items():
            s.append('-- Table: %s --' % table_name)
            s.append(str(table))
            s.append('')

        return '\n'.join(s)

    def get_table(self, table_id):
        return self.tables[table_id]

    def __getitem__(self, table_id):
        return self.get_table(table_id)

    def add_table(self, table_name, schema):
        '''
        Add a new table to the experiment.
        Inputs:
        - table_name, a name for the table
        - schema, a {name: type} dict for the schema of the table
        '''
        table = Table(table_name, schema, self._table_object_dir, self.store)
        self.tables[table_name] = table
        return table

    def log_table_and_tb(self, table_name, *args, summary_type='scalar'):
        if len(args) == 2:
            name, value = args
            update_dict = {
                name:value
            }
        else:
            update_dict = args[0]

        table = self.tables[table_name]
        update_dict = _clean_dict(update_dict, table.schema)
            
        tb_func = getattr(self.tensorboard, 'add_%s' % summary_type)
        iteration = table.nrows

        for name, value in update_dict.items():
            tb_func('/'.join([table_name, name]), value, iteration)
        
        table.update_row(update_dict)

    def save_with_name(self, v, name, v_type=PICKLE):
        fname = os.path.join(self.save_dir, name)
        valid_vtypes = [PICKLE, PYTORCH_STATE]
        if v_type == PICKLE:
            with open(fname, 'wb') as f:
                pickle.dump(v, f)
        elif v_type == PYTORCH_STATE:
            if 'state_dict' in dir(v):
                v = v.state_dict()
            ch.save(v, fname, pickle_module=pickle)
        else:
            raise ValueError(f"Invalid obj_type {str(v_type)}: must be one of {valid_vtypes}")

class Table():
    '''
    A class representing a single storer table, to be written to by
    the experiment.
    Exposes the function:
    - create_row
    - update_row
    '''

    def tables_from_store(store, table_obj_dir):
        tables = {}
        for key in store.keys():
            storer = store.get_storer(key)
            if COX_DATA_KEY in storer.attrs:
                data = storer.attrs[COX_DATA_KEY]
                name = data['name']
                table = Table(name, data['schema'], table_obj_dir, store,
                              has_initialized=True) 
                tables[name] = table

        return tables

    def __str__(self):
        s = str(self.df)
        if len(s.split('\n')) > 5:
            s = str(self.df[:4]) + '\n ... (%s rows hidden)' % self.df.shape[0]
        return s

    def __init__(self, name, schema, table_obj_dir, store,
                 has_initialized=False, data_columns=[]):
        '''
        Create a new Table object
        Inputs:
        - Experiment ID, a unique identifier for all the tables in a group
        - Schema, a {key: type} dictionary for the table
        '''
        self._name = name
        self._schema = schema
        self._HDFStore = store
        self._curr_row_data = None
        self._table_obj_dir = table_obj_dir
        self._has_initialized = has_initialized

        self._create_row()

    @property
    def df(self):
        if self._has_initialized:
            return self._HDFStore[self._name]
        else:
            return pd.DataFrame(columns=self._schema.keys())

    @property
    def schema(self):
        return dict(self._schema)

    @property
    def nrows(self):
        if self._has_initialized:
            return self._HDFStore.get_storer(self._name).nrows
        else:
            return 0

    def _initialize_nonempty_table(self):
        self._HDFStore.get_storer(self._name).attrs[COX_DATA_KEY] = {
            'schema':self._schema,
            'name':self._name,
        }

        self._has_initialized = True

    def append_row(self, data):
        '''
        Convenience function that lets you write row if you have all your data
        at once
        '''
        self.update_row(data)
        self.flush_row()

    def _create_row(self):
        assert self._curr_row_data is None

        curr_row_dict = {s: None for s in self._schema}
        self._curr_row_data = curr_row_dict
        
    def update_row(self, data):
        '''
        Update a row in the data store. If the row id is the same as the 
        last one used, we consider it an edit, otherwise we flush the 
        last row and start a new row with the given values. Currently
        supports python primitives [int, float, str, bool], and their 
        numpy equivalents. All other files are pickled (with dill), and 
        references to the pickle files are stored.
        Inputs:
        - row_id, the name of the row we are adding to
        - data, {key: value} of data to save to the current row
        '''
        # Data sanity checks
        assert self._curr_row_data is not None
        assert len(set(data.keys())) == len(data.keys())

        if any([k not in self._schema for k in data]):
            raise ValueError("Got keys that are undeclared in schema")

        for k, v in data.items():
            v_type = self._schema[k]
            if v_type == OBJECT:
                to_store = obj_to_string(v)
            elif v_type == PICKLE or v_type == PYTORCH_STATE:
                uid = str(uuid4())
                fname = os.path.join(self._table_obj_dir, uid)
                if v_type == PICKLE:
                    with open(fname, 'wb') as f:
                        pickle.dump(v, f)
                else:
                    if 'state_dict' in dir(v):
                        v = v.state_dict()
                    ch.save(v, fname, pickle_module=pickle)
                to_store = uid
            else:
                to_store = v_type(v)
                assert to_store is not None

            self._curr_row_data[k] = to_store

    def get_pickle(self, uid):
        fname = os.path.join(self._table_obj_dir, uid)
        with open(fname, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def get_state_dict(self, uid, **kwargs):
        fname = os.path.join(self._table_obj_dir, uid)
        kwargs['pickle_module'] = pickle
        return ch.load(fname, **kwargs)

    def get_object(self, s):
        return string_to_obj(s)

    def flush_row(self, debug=False):
        '''
        Flush current row that we are updating. At the end, create a new empty
        row we can update with update_row()
        '''
        self._curr_row_data = _clean_dict(self._curr_row_data, self._schema)

        for k in self._schema:
            try:
                assert self._curr_row_data[k] is not None
            except:
                dne = not (k in self._curr_row_data)
                if dne:
                    msg = 'Col %s does not exist!' % k
                else:
                    msg = 'Col %s is None!' % k
                
                raise ValueError(msg)

        for k, v in self._curr_row_data.items():
            self._curr_row_data[k] = [v]

        df = pd.DataFrame(self._curr_row_data)
        if debug:
            print(df)

        try:
            nrows = self._HDFStore.get_storer(self._name).nrows
        except:
            nrows = 0

        df.index += nrows
        self._HDFStore.append(self._name, df, table=True)

        if not self._has_initialized:
             self._initialize_nonempty_table()

        self._curr_row_data = None
        self._create_row()

def schema_from_dict(d, alternative=OBJECT):
    natural_types = set([int, str, float, bool])
    schema = {}
    for k, v in d.items():
        t = type(v)
        if t in natural_types:
            schema[k] = t
        else:
            schema[k] = alternative

    return schema

def _clean_dict(d, schema):
    d = dict(d)
    for k, v in d.items():
        v_type = schema[k]
        if v_type in [int, float, bool]:
            if type(v) == ch.Tensor or type(v) == np.ndarray:
                if v.shape == ():
                    v = v_type(v)
                    d[k] = v
    return d
