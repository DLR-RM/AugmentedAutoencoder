import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import configparser
from meshrenderer import meshrenderer_phong
from sixd_toolkit.pysixd import misc

from auto_pose.ae import factory
from auto_pose.ae import utils as u
from auto_pose.eval import eval_utils

import argparse


def render_rot(renderer, obj_i, train_args, R):
    h, w = train_args.getint('Dataset','H'),train_args.getint('Dataset','W')
    radius = float(train_args.getfloat('Dataset','RADIUS'))
    render_dims = eval(train_args.get('Dataset','RENDER_DIMS'))

    K = eval(train_args.get('Dataset','K'))
    K = np.array(K).reshape(3,3)

    clip_near = float(train_args.getfloat('Dataset','CLIP_NEAR'))
    clip_far = float(train_args.getfloat('Dataset','CLIP_FAR'))
    pad_factor = float(train_args.getfloat('Dataset','PAD_FACTOR'))

    t = np.array([0, 0, radius])

    bgr_y, depth_y = renderer.render( 
        obj_id=obj_i,
        W=render_dims[0], 
        H=render_dims[1],
        K=K.copy(), 
        R=R, 
        t=t,
        near=clip_near,
        far=clip_far,
        random_light=False
    )

    ys, xs = np.nonzero(depth_y > 0)
    obj_bb = misc.calc_2d_bbox(xs, ys, render_dims)
    x, y, w, h = obj_bb

    size = int(np.maximum(h, w) * pad_factor)
    left = x+w/2-size/2
    right = x+w/2+size/2
    top = y+h/2-size/2
    bottom = y+h/2+size/2

    bgr_y = bgr_y[top:bottom, left:right]
    return cv2.resize(bgr_y, (w,h))


parser = argparse.ArgumentParser()
parser.add_argument("experiment_names", nargs='+',type=str)
parser.add_argument("-f", "--file_str", required=True, help='folder or filename to image(s)')
# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()


file_str = arguments.file_str
if os.path.isdir(file_str):
    files = glob.glob(os.path.join(str(file_str),'*.png'))+glob.glob(os.path.join(str(file_str),'*.jpg'))
else:
    files = [file_str]



all_sessions = []
all_codebooks = []
all_train_args = []

model_paths = []

for i,experiment_name in enumerate(arguments.experiment_names):

    full_name = experiment_name.split('/')
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''

    
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
    model_paths.append(train_args.get('Paths','MODEL_PATH'))
    all_train_args.append(train_args)


    with tf.Graph().as_default():

        # Sessions created in this scope will run operations from `g_1`.
        all_codebooks.append(factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=False))
        all_sessions.append(tf.Session())

        factory.restore_checkpoint(all_sessions[-1], tf.train.Saver(), ckpt_dir)


renderer = meshrenderer_phong.Renderer(
    model_paths, 
    1
)


for file in files*10:
    for i,(sess,codebook,train_args) in enumerate(zip(all_sessions,all_codebooks,all_train_args)):
        h, w = train_args.getint('Dataset','H'),train_args.getint('Dataset','W')
        
        im = cv2.imread(file)
        im = cv2.resize(im,(w,h))
        if train_args.getint('Dataset','C')==1:
            im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)[:,:,None]

        R = codebook.nearest_rotation(sess, im/255.)

        pred_view = render_rot(renderer, i, train_args, R)
        
        
        cv2.imshow('resized img', cv2.resize(im/255.,(256,256)))
        cv2.imshow('pred_view', cv2.resize(pred_view/255.,(256,256)))
        print R
        cv2.waitKey(0)


