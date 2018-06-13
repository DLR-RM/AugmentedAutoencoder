
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

def get_checkpoint_dir(log_dir):
    return os.path.join(
        log_dir, 
        'checkpoints'
    )

def get_log_dir(workspace_path, experiment_name, experiment_group=''):
    return os.path.join(
        workspace_path, 
        'experiments',
        experiment_group,
        experiment_name
    )

def get_train_fig_dir(log_dir):
    return os.path.join(
        log_dir, 
        'train_figures'
    )

def get_checkpoint_basefilename(log_dir):
    return os.path.join(
        log_dir,
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

def get_eval_config_file_path(workspace_path, eval_cfg='eval.cfg'):
    return os.path.join(
        workspace_path, 
        'cfg_eval',
        eval_cfg
    )

def get_eval_dir(log_dir, evaluation_name, data):
    return os.path.join(
        log_dir,
        'eval',
        evaluation_name,
        data
    )


