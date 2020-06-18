 # -*- coding: utf-8 -*-
from imgaug.augmenters import *
import os
import configparser
import argparse
import numpy as np
import signal
import shutil
import cv2
import glob


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import progressbar
import tensorflow as tf

from auto_pose.ae import ae_factory as factory
from auto_pose.ae import utils as u
from auto_pose.eval import eval_plots,eval_utils,latent_utils

import matplotlib.pyplot as plt
from sixd_toolkit.pysixd import transform,pose_error,view_sampler
import time

def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    gentle_stop = np.array((1,), dtype=np.bool)
    gentle_stop[0] = False

    def on_ctrl_c(signal, frame):
        gentle_stop[0] = True

    signal.signal(signal.SIGINT, on_ctrl_c)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument('--model_path', default=False)
    parser.add_argument('--config_path', default='eval_latent_template.cfg', required=True)
    parser.add_argument("-d", action='store_true', default=False)
    parser.add_argument("-gen", action='store_true', default=False)
    parser.add_argument("-vis_emb", action='store_true', default=False)
    parser.add_argument('--at_step', default=None,  type=int, required=False)


    arguments = parser.parse_args()

    full_name = arguments.experiment_name.split('/')
    model_path = arguments.model_path
    joint = False if model_path else True
    
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    
    at_step = arguments.at_step

    cfg_file_path = u.get_config_file_path(workspace_path, experiment_name, experiment_group)
    latent_cfg_file_path = u.get_eval_config_file_path(workspace_path, arguments.config_path)
    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)

    if not os.path.exists(cfg_file_path):
        print('Could not find config file:\n')
        print('{}\n'.format(cfg_file_path))
        exit(-1)

    args = configparser.ConfigParser(inline_comment_prefixes="#")
    args.read(cfg_file_path)

    args_latent = configparser.ConfigParser(inline_comment_prefixes="#")
    args_latent.read(latent_cfg_file_path)

    if at_step is None:
        checkpoint_file = u.get_checkpoint_basefilename(log_dir, model_path, latest=args.getint('Training', 'NUM_ITER'), joint=joint)
    else:
        checkpoint_file = u.get_checkpoint_basefilename(log_dir, model_path, latest=at_step, joint=joint)

    model_type = args.get('Dataset', 'MODEL')

    codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset = True, joint=joint)
    encoder = codebook._encoder
    
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)

    ######
    dataset_exp = args_latent.get('Data', 'dataset')
    base_path = args_latent.get('Data', 'base_path')
    split = args_latent.get('Data', 'split')
    num_obj = args_latent.getint('Data', 'num_obj')
    num_views = args_latent.getint('Data', 'num_views')
    test_class = args_latent.get('Data', 'test_class')

    # for test_class in test_classes:
    models = sorted(glob.glob(os.path.join(base_path, test_class, split, '*_normalized.off')))

    if split == 'test': or split=='train':
        if os.path.exists(os.path.dirname(models[0])):
            dataset._kw['model_path'] = models[0:num_obj]
        else:
            print((models[0], ' does not exist'))
    else:
        print('split must be train or test')
        exit()
    print(dataset._kw['model_path'])

    if args_latent.getboolean('Experiment', 'emb_invariance'):
        latent_utils.compute_plot_emb_invariance(args_latent, codebook)
    if args_latent.getboolean('Experiment', 'refinement_pert_category_agnostic'):
        res_preds = latent_utils.relative_pose_refinement(sess, args_latent, dataset, codebook)
        np.save(os.path.join(log_dir, 'preds_%s.npy' % test_class), res_preds)
    if args_latent.getboolean('Experiment', 'compute_pose_errors'):
        res_preds = np.load(os.path.join(log_dir, 'preds_%s.npy' % test_class)).item()
        res_errors = latent_utils.compute_pose_errors(res_preds, args_latent, dataset)
        np.save(os.path.join(log_dir, 'pose_errors_%s.npy' % test_class), res_errors)
    if args_latent.getboolean('Visualization', 'pca_embedding_azelin'):
        latent_utils.plot_latent_revolutions(num_obj, codebook)



if __name__ == '__main__':
    main()
    
