# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils

from .dataset import Dataset
from .queue import Queue
from .multi_queue import MultiQueue
from .ae import AE
from .encoder import Encoder
from .decoder import Decoder
from .codebook_multi import Codebook


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

def build_multi_queue(dataset, args):
    BATCH_SIZE = args.getint('Training', 'BATCH_SIZE')
    SHAPE = (args.getint('Dataset', 'W'), args.getint('Dataset', 'H'), args.getint('Dataset', 'C'))
    NOOF_TRAINING_IMGS = args.getint('Dataset', 'NOOF_TRAINING_IMGS')
    MODEL_PATHS = eval(args.get('Paths', 'MODEL_PATH'))
    AUG_ARGS = { k:v for k,v in args.items('Augmentation')}
    queue = MultiQueue(
        dataset, 
        BATCH_SIZE,
        NOOF_TRAINING_IMGS,
        MODEL_PATHS,
        SHAPE,
        AUG_ARGS
    )
    return queue

def build_encoder(x, args, target=None, is_training=False):
    LATENT_SPACE_SIZE = args.getint('Network', 'LATENT_SPACE_SIZE')
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_ENCODER = args.getint('Network', 'KERNEL_SIZE_ENCODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')
    RESNET50 = args.getboolean('Network', 'RESNET50')
    RESNET101 = args.getboolean('Network', 'RESNET101')
    ASPP = eval(args.get('Network', 'ASPP'))
    PRE_TRAINED_MODEL = args.get('Training', 'PRE_TRAINED_MODEL')
    EMB_INVARIANCE_LOSS = args.getfloat('Network', 'EMB_INVARIANCE_LOSS')

    if target is not None and EMB_INVARIANCE_LOSS > 0:
        x = tf.concat((x, target), axis=0)

    encoder = Encoder(
        x,
        LATENT_SPACE_SIZE, 
        NUM_FILTER, 
        KERNEL_SIZE_ENCODER, 
        STRIDES,
        BATCH_NORM,
        RESNET50,
        RESNET101,
        ASPP,
        PRE_TRAINED_MODEL,
        EMB_INVARIANCE_LOSS,
        is_training=is_training
    )
    return encoder

def build_decoder(reconstruction_target, encoder_z_split, args, is_training=False,idx=0):
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
        encoder_z_split,
        list( reversed(NUM_FILTER) ),
        KERNEL_SIZE_DECODER,
        list( reversed(STRIDES) ),
        LOSS,
        BOOTSTRAP_RATIO,
        AUXILIARY_MASK,
        BATCH_NORM,
        is_training=is_training,
        idx=idx
    )
    return decoder

def build_ae(encoder, decoder, args):
    NORM_REGULARIZE = args.getfloat('Network', 'NORM_REGULARIZE')
    VARIATIONAL = args.getfloat('Network', 'VARIATIONAL')
    EMB_INVARIANCE_LOSS = args.getfloat('Network', 'EMB_INVARIANCE_LOSS')
    ae = AE(encoder, decoder, NORM_REGULARIZE, VARIATIONAL, EMB_INVARIANCE_LOSS)
    return ae

def build_train_op(ae, args):
    import tensorflow as tf

    LEARNING_RATE = args.getfloat('Training', 'LEARNING_RATE')
    LEARNING_RATE_SCHEDULE = args.get('Training','LEARNING_RATE_SCHEDULE')
    LAYERS_TO_FREEZE = eval(args.get('Training', 'LAYERS_TO_FREEZE'))

    if LEARNING_RATE_SCHEDULE=='poly':
        FINAL_LEARNING_RATE = args.getfloat('Training','FINAL_LEARNING_RATE')
        NUM_ITER = args.getfloat('Training','NUM_ITER')
        print('using poly learning rate schedule')
        LEARNING_RATE = tf.train.polynomial_decay(LEARNING_RATE, ae._encoder.global_step,
                                                NUM_ITER, FINAL_LEARNING_RATE, power=0.9)
    

    OPTIMIZER_NAME = args.get('Training', 'OPTIMIZER')

    optimizer = eval('tf.train.{}Optimizer'.format(OPTIMIZER_NAME))
    optim = optimizer(LEARNING_RATE)
    if len(LAYERS_TO_FREEZE)>0:
        freeze_vars = []
        all_vars = set([var for var in tf.trainable_variables()])
        for layer_to_freeze in LAYERS_TO_FREEZE:
            freeze_vars += [v for v in all_vars if layer_to_freeze in v.name]
        train_vars = list(all_vars.symmetric_difference(freeze_vars))
        train_op = tf.contrib.training.create_train_op(ae.loss, 
                                                        optim, 
                                                        global_step=ae._encoder.global_step, 
                                                        variables_to_train=train_vars,
                                                        colocate_gradients_with_ops=True)
    else:
        train_op = tf.contrib.training.create_train_op(ae.loss, 
                                                        optim, 
                                                        global_step=ae._encoder.global_step, 
                                                        colocate_gradients_with_ops=True)

    return train_op

def build_codebook(encoder, dataset, args):
    embed_bb = args.getboolean('Embedding', 'EMBED_BB')
    from .codebook import Codebook
    codebook = Codebook(encoder, dataset, embed_bb)
    return codebook

def build_codebook_multi(encoder, dataset, args, checkpoint_file_basename=None):
    embed_bb = args.getboolean('Embedding', 'EMBED_BB')

    existing_embs = []
    if checkpoint_file_basename is not None:
        var_list = checkpoint_utils.list_variables(checkpoint_file_basename)
        for v in var_list:
            if 'embedding_normalized_' in v[0]:
                print(v)
                existing_embs.append(v[0].split('/embedding_normalized_')[-1].split('.')[0])

    print(existing_embs)
    codebook = Codebook(encoder, dataset, embed_bb, existing_embs)
    return codebook

def build_codebook_from_name(experiment_name, experiment_group='', return_dataset=False, return_decoder = False, joint=False):
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
    cfg_file_path = u.get_train_config_exp_file_path(log_dir, experiment_name)
    dataset_path = u.get_dataset_path(workspace_path)

    if os.path.exists(cfg_file_path):
        args = configparser.ConfigParser(inline_comment_prefixes="#")
        args.read(cfg_file_path)
    else:
        print(('ERROR: Config File not found: ', cfg_file_path))
        exit()

    if joint:
        checkpoint_file = u.get_checkpoint_basefilename(log_dir, joint=joint, latest=args.getint('Training', 'NUM_ITER'))
    else:
        checkpoint_file = u.get_checkpoint_basefilename(log_dir, joint=joint)

    with tf.variable_scope(experiment_name):
        dataset = build_dataset(dataset_path, args)
        x = tf.placeholder(tf.float32, [None,] + list(dataset.shape))
        encoder = build_encoder(x, args)
        if joint:
            codebook = build_codebook_multi(encoder, dataset, args, checkpoint_file)
        else:
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
                    print(('restoring' , os.path.basename(ckpt_path)))
    else:
        print('No checkpoint found. Expected one in:\n')
        print(('{}\n'.format(ckpt_dir)))
        exit(-1)

