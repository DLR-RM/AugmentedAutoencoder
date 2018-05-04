import cv2
import tensorflow as tf
import numpy as np
import glob
import imageio
import os
import ConfigParser

from ae import factory
from ae import utils as u
from eval import eval_utils

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("-f", "--file_str", required=True, help='folder or filename to image(s)')
# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

file_str = arguments.file_str
# gt_bb = arguments.gt_bb

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
train_cfg_file_path = u.get_config_file_path(workspace_path, experiment_name, experiment_group)
eval_cfg_file_path = u.get_eval_config_file_path(workspace_path)
train_args = ConfigParser.ConfigParser()
eval_args = ConfigParser.ConfigParser()
train_args.read(train_cfg_file_path)
eval_args.read(eval_cfg_file_path)

codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)


log_dir = u.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = u.get_checkpoint_dir(log_dir)

with tf.Session() as sess:

    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)

    if os.path.isdir(file_str):
        files = glob.glob(os.path.join(str(file_str),'*.png'))+glob.glob(os.path.join(str(file_str),'*.jpg'))
    else:
        files = [file_str]

    for file in files:

        im = cv2.imread(file)
        im = cv2.resize(im,(128,128))
        if train_args.getint('Dataset','C')==1:
            im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)[:,:,None]

        R = codebook.nearest_rotation(sess, im/255.)

        pred_view = dataset.render_rot( R ,downSample = 1)
        
        
        cv2.imshow('resized img', cv2.resize(im/255.,(256,256)))
        cv2.imshow('pred_view', cv2.resize(pred_view,(256,256)))
        print R
        cv2.waitKey(0)


