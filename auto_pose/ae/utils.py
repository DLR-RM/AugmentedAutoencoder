
import os
import pickle
import numpy as np
import functools
import cv2
import tensorflow as tf

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
    log_dir = os.path.join(
        workspace_path, 
        'experiments',
        experiment_group,
        experiment_name
    )
    if not os.path.exists(log_dir):
        log_dir = os.path.join(
            workspace_path, 
            experiment_group,
            experiment_name
        )
    return log_dir

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

def get_checkpoint_basefilename(log_dir, model_path=False, latest=False, joint=False):
    import glob

    file_name = os.path.join(
        log_dir,
        'checkpoints',
        'chkpt'
    )
    if joint:
        file_name += '-joint'
    if model_path:
        file_name += '-' + os.path.basename(model_path).split('.')[0]
    if latest:
        file_name = file_name + '-' + str(latest)

    return file_name


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

def create_summaries(multi_queue, decoders, ae):
    tf.summary.histogram('mean_loss', ae._encoder.z)
    tf.summary.scalar('total_loss', ae.loss)
    for j,d in enumerate(decoders):
        tf.summary.scalar('reconst_loss_%s' % j, d.reconstr_loss)
    rand_idcs = tf.random_shuffle(tf.range(multi_queue._batch_size * multi_queue._num_objects), seed=0)
    print(len(decoders), rand_idcs.shape[0], rand_idcs)
    tf.summary.image('input', tf.gather(tf.concat([el[0] for el in multi_queue.next_element],0),rand_idcs), max_outputs=4)
    tf.summary.image('reconstruction_target', tf.gather(tf.concat([el[2] for el in multi_queue.next_element],0),rand_idcs), max_outputs=4)
    tf.summary.image('reconstruction', tf.gather(tf.concat([decoder.x for decoder in decoders], axis=0), rand_idcs), max_outputs=4)
    return

def save_pickled_data(data, file_path):
    """Saves data to file_path. Tries to be python2/3 compatible
    
    Args:
        data (any): Data to be saved
        file_path (String): Output filepath
    
    Return:
        See pickle.dump return
    """
    with open(file_path, "wb") as f:
        try:
            # Python3
            return pickle.dump(data,
                               f,
                               protocol=2,
                               fix_imports=True)
        except TypeError:
            # Python2
            return pickle.dump(data,
                               f,
                               protocol=2)

def load_pickled_data(file_path):
    """Loads a pickled file. Tries to be python2/3 compatible
    
    Args:
        file_path (string): Input filepath
        
    Return:
        loaded_obj (any): Data which was loaded from file
    """
    with open(file_path, "rb") as f:
        try:
            # Python3
            loaded_obj = pickle.load(f,
                        fix_imports=True,
                        encoding="bytes")
        except TypeError:
            # Python2
            loaded_obj = pickle.load(f)
    return loaded_obj
            

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

def load_depth2(path):
    import scipy.misc
    d = scipy.misc.imread(path)
    d = d.astype(np.float32)
    return d