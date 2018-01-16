
import os
import numpy as np
import functools


# https://danijar.com/structuring-your-tensorflow-models/
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def batch_iteration_indices(N, batch_size):
    end = int(np.ceil(float(N) / float(batch_size)))
    for i in xrange(end):
        a = i*batch_size
        e = i*batch_size+batch_size
        e = e if e <= N else N
        yield (a, e)

def get_dataset_path(workspace_path):
    return os.path.join(
        workspace_path, 
        'dataset',
    )

def get_checkpoint_dir(workspace_path, experiment_name, experiment_group=''):
    return os.path.join(
        workspace_path, 
        'experiments',
        experiment_group,
        experiment_name, 
        'checkpoints'
    )

def get_log_dir(workspace_path, experiment_name, experiment_group=''):
    return os.path.join(
        workspace_path, 
        'experiments',
        experiment_group,
        experiment_name
    )

def get_checkpoint_basefilename(workspace_path, experiment_name, experiment_group=''):
    return os.path.join(
        workspace_path, 
        'experiments', 
        experiment_group,
        experiment_name, 
        'checkpoints',
        'chkpt'
    )



def get_config_file_path(workspace_path, experiment_name, experiment_group=''):
    return os.path.join(
        workspace_path, 
        'cfg',
        experiment_group,
        '{}.cfg'.format(experiment_name)
    )

def get_eval_config_file_path(workspace_path):
    return os.path.join(
        workspace_path, 
        'cfg_eval',
        'eval.cfg'
    )

def get_eval_dir(workspace_path, experiment_group, experiment_name, evaluation_name, dataset_name, cam_type):
    return os.path.join(
        workspace_path, 
        'experiments',
        experiment_group,
        experiment_name,
        'eval',
        evaluation_name,
        dataset_name + '_' + cam_type
    )



