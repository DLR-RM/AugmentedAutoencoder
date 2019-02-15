import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import configparser
import time

from auto_pose.ae import factory
from auto_pose.ae import utils as u
from auto_pose.eval import eval_utils

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
log_dir = u.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = u.get_checkpoint_dir(log_dir)

train_cfg_file_path = u.get_train_config_exp_file_path(log_dir, experiment_name)
eval_cfg_file_path = u.get_eval_config_file_path(workspace_path)
train_args = configparser.ConfigParser()
eval_args = configparser.ConfigParser()
train_args.read(train_cfg_file_path)
eval_args.read(eval_cfg_file_path)

codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)


with tf.Session() as sess:

    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)

    if os.path.isdir(file_str):
        files = glob.glob(os.path.join(str(file_str),'*.png'))+glob.glob(os.path.join(str(file_str),'*.jpg'))+glob.glob(os.path.join(str(file_str),'*.pgm'))
    else:
        files = [file_str]
    print files
    for file in files*10:

        im = cv2.imread(file)
        
        im = cv2.copyMakeBorder(im,50,50,50,50,cv2.BORDER_CONSTANT,value=[0,0,0])
        h,w = im.shape[:2]
        size = int(np.minimum(h, w)*1.2)
        
        left = np.maximum(w/2-size/2, 0)
        right = np.minimum(w/2+size/2,w)
        top = np.maximum(h/2-size/2, 0)
        bottom = np.minimum(h/2+size/2,h)

        im = im[top:bottom, left:right, :]
        im = cv2.resize(im,(128,128))

        if train_args.getint('Dataset','C')==1:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[:,:,None]
        st = time.time()
        R = codebook.nearest_rotation(sess, im)
        print time.time()-st
        pred_view = dataset.render_rot( R ,downSample = 1)
        
        
        cv2.imshow('resized img', cv2.resize(im/255.,(256,256)))
        cv2.imshow('pred_view', cv2.resize(pred_view,(256,256)))
        print R
        cv2.waitKey(0)


