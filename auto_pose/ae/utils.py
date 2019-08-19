
import os
import numpy as np
import functools
import cv2

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
    for i in range(end):
        a = i*batch_size
        e = i*batch_size+batch_size
        e = e if e <= N else N
        yield (a, e)

def get_dataset_path(workspace_path):
    return os.path.join(
        workspace_path, 
        'tmp_datasets',
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

def get_train_config_exp_file_path(log_dir, experiment_name):
    return os.path.join(
        log_dir,
        '{}.cfg'.format(experiment_name)
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


def tiles(batch, rows, cols, spacing_x=0, spacing_y=0, scale=1.0):
    if batch.ndim == 4:
        N, H, W, C = batch.shape
    elif batch.ndim == 3:
        N, H, W = batch.shape
        C = 1
    else:
        raise ValueError('Invalid batch shape: {}'.format(batch.shape))

    H = int(H*scale)
    W = int(W*scale)
    img = np.ones((rows*H+(rows-1)*spacing_y, cols*W+(cols-1)*spacing_x, C))
    i = 0
    for row in range(rows):
        for col in range(cols):
            start_y = row*(H+spacing_y)
            end_y = start_y + H
            start_x = col*(W+spacing_x)
            end_x = start_x + W
            if i < N:
                if C > 1:
                    img[start_y:end_y,start_x:end_x,:] = cv2.resize(batch[i], (W,H))
                else:
                    img[start_y:end_y,start_x:end_x,0] = cv2.resize(batch[i], (W,H))
            i += 1
    return img
