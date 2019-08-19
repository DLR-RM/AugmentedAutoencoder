# -*- coding: utf-8 -*-


from .dataset import Dataset
from .queue import Queue
from .ae import AE
from .encoder import Encoder
from .decoder import Decoder
from .codebook import Codebook

def build_dataset(dataset_path, args):
    dataset_args = { k:v for k,v in
        args.items('Dataset') +
        args.items('Paths') +
        args.items('Augmentation')+
        args.items('Queue') +
        args.items('Embedding')}
    dataset = Dataset(dataset_path, **dataset_args)
    return dataset

def build_queue(dataset, args):
    NUM_THREADS = args.getint('Queue', 'NUM_THREADS')
    QUEUE_SIZE = args.getint('Queue', 'QUEUE_SIZE')
    BATCH_SIZE = args.getint('Training', 'BATCH_SIZE')
    queue = Queue(
        dataset,
        NUM_THREADS,
        QUEUE_SIZE,
        BATCH_SIZE
    )
    return queue

def build_encoder(x, args, is_training=False):
    LATENT_SPACE_SIZE = args.getint('Network', 'LATENT_SPACE_SIZE')
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_ENCODER = args.getint('Network', 'KERNEL_SIZE_ENCODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')
    encoder = Encoder(
        x,
        LATENT_SPACE_SIZE,
        NUM_FILTER,
        KERNEL_SIZE_ENCODER,
        STRIDES,
        BATCH_NORM,
        is_training=is_training
    )
    return encoder

def build_decoder(reconstruction_target, encoder, args, is_training=False):
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_DECODER = args.getint('Network', 'KERNEL_SIZE_DECODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    LOSS = args.get('Network', 'LOSS')
    BOOTSTRAP_RATIO = args.getint('Network', 'BOOTSTRAP_RATIO')
    VARIATIONAL = args.getfloat('Network', 'VARIATIONAL') if is_training else False
    AUXILIARY_MASK = args.getboolean('Network', 'AUXILIARY_MASK')
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')
    decoder = Decoder(
        reconstruction_target,
        encoder.sampled_z if VARIATIONAL else encoder.z,
        list( reversed(NUM_FILTER) ),
        KERNEL_SIZE_DECODER,
        list( reversed(STRIDES) ),
        LOSS,
        BOOTSTRAP_RATIO,
        AUXILIARY_MASK,
        BATCH_NORM,
        is_training=is_training
    )
    return decoder

def build_ae(encoder, decoder, args):
    NORM_REGULARIZE = args.getfloat('Network', 'NORM_REGULARIZE')
    VARIATIONAL = args.getfloat('Network', 'VARIATIONAL')
    ae = AE(encoder, decoder, NORM_REGULARIZE, VARIATIONAL)
    return ae

def build_train_op(ae, args):
    LEARNING_RATE = args.getfloat('Training', 'LEARNING_RATE')
    OPTIMIZER_NAME = args.get('Training', 'OPTIMIZER')
    import tensorflow
    optimizer = eval('tensorflow.train.{}Optimizer'.format(OPTIMIZER_NAME))
    optim = optimizer(LEARNING_RATE)

    train_op = tensorflow.contrib.training.create_train_op(ae.loss, optim, global_step=ae.global_step)

    return train_op

def build_codebook(encoder, dataset, args):
    embed_bb = args.getboolean('Embedding', 'EMBED_BB')
    codebook = Codebook(encoder, dataset, embed_bb)
    return codebook

def build_codebook_from_name(experiment_name, experiment_group='', return_dataset=False, return_decoder = False):
    import os
    import configparser
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    from . import utils as u
    import tensorflow as tf

    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)
    checkpoint_file = u.get_checkpoint_basefilename(log_dir)
    cfg_file_path = u.get_train_config_exp_file_path(log_dir, experiment_name)
    dataset_path = u.get_dataset_path(workspace_path)

    if os.path.exists(cfg_file_path):
        args = configparser.ConfigParser()
        args.read(cfg_file_path)
    else:
        print('ERROR: Config File not found: ', cfg_file_path)
        exit()

    with tf.variable_scope(experiment_name):
        dataset = build_dataset(dataset_path, args)
        x = tf.placeholder(tf.float32, [None,] + list(dataset.shape))
        encoder = build_encoder(x, args)
        codebook = build_codebook(encoder, dataset, args)
        if return_decoder:
            reconst_target = tf.placeholder(tf.float32, [None,] + list(dataset.shape))
            decoder = build_decoder(reconst_target, encoder, args)

    if return_dataset:
        if return_decoder:
            return codebook, dataset, decoder
        else:
            return codebook, dataset
    else:
        return codebook


def restore_checkpoint(session, saver, ckpt_dir, at_step=None):

    import tensorflow as tf
    import os

    chkpt = tf.train.get_checkpoint_state(ckpt_dir)

    if chkpt and chkpt.model_checkpoint_path:
        if at_step is None:
            saver.restore(session, chkpt.model_checkpoint_path)
        else:
            for ckpt_path in chkpt.all_model_checkpoint_paths:

                if str(at_step) in str(ckpt_path):
                    saver.restore(session, ckpt_path)
                    print('restoring' , os.path.basename(ckpt_path))
    else:
        print('No checkpoint found. Expected one in:\n')
        print('{}\n'.format(ckpt_dir))
        exit(-1)
