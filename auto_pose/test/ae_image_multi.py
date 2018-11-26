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


def render_rot(renderer, obj_i, train_args, R, downSample=1):
    h, w = train_args.getint('Dataset','H'),train_args.getint('Dataset','W')
    radius = float(train_args.getfloat('Dataset','RADIUS'))
    render_dims = eval(train_args.get('Dataset','RENDER_DIMS'))

    K = eval(train_args.get('Dataset','K'))
    K = np.array(K).reshape(3,3)
    K[:2,:] = K[:2,:] / downSample

    
    clip_near = float(train_args.getfloat('Dataset','CLIP_NEAR'))
    clip_far = float(train_args.getfloat('Dataset','CLIP_FAR'))
    pad_factor = float(train_args.getfloat('Dataset','PAD_FACTOR'))

    t = np.array([0, 0, radius])

    bgr_y, depth_y = renderer.render( 
        obj_id=obj_i,
        W=render_dims[0]/ downSample, 
        H=render_dims[1]/ downSample,
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



all_codebooks = []
all_train_args = []
model_paths = []

sess = tf.Session()
for i,experiment_name in enumerate(arguments.experiment_names):

    full_name = experiment_name.split('/')
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''

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

    cb,dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)
    all_codebooks.append(cb)
    factory.restore_checkpoint(sess, tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=experiment_name)), ckpt_dir)
    # factory.restore_checkpoint(all_sessions[-1], tf.train.Saver(), ckpt_dir)


renderer = meshrenderer_phong.Renderer(
    model_paths, 
    1,
    vertex_tmp_store_folder=u.get_dataset_path(workspace_path)
)


for file in files*10:
    for i,(codebook,train_args) in enumerate(zip(all_codebooks,all_train_args)):
                
        # detect some objects here

        #######################
        ## if you have a trained object detector, use something like this:
        #######################

        # det_bb = np.array([x,y,w,h])
        # img_crop = dataset.extract_square_patch(im,det_bb,train_args.getfloat('Dataset','PAD_FACTOR'))
        # R,t,_ = codebook.auto_pose6d(sess, img_crop, det_bb,K_test,1,train_args)
        # print R,t

        # R = R.squeeze()
        # t = t.squeeze()
        # img_crop
        # Rs = []
        # ts = []
        # for k,bb,img_crop in zip(det_aae_objects_k,det_aae_bbs,img_crops):
        #     R, t = all_codebooks[k].auto_pose6d(sess, img_crop, bb, K_test, 1, all_train_args[k], upright=False)
        #     Rs.append(R.squeeze())
        #     ts.append(t.squeeze())

        # Rs = np.array(Rs)
        # ts = np.array(ts)                                    

        # bgr_y,_,_ = renderer.render_many( 
        #     obj_ids=np.array(det_aae_objects_k).astype(np.int32),
        #     W=width/arguments.down,
        #     H=height/arguments.down,
        #     K=K_down, 
        #     Rs=Rs, 
        #     ts=ts,
        #     near=1.,
        #     far=10000.,
        #     random_light=False,
        #     # calc_bbs=False,
        #     # depth=False
        # )

        # bgr_y = cv2.resize(bgr_y,(width,height))
        # g_y = np.zeros_like(bgr_y)
        # g_y[:,:,1]= bgr_y[:,:,1]    
        # im_bg = cv2.bitwise_and(image,image,mask=(g_y[:,:,1]==0).astype(np.uint8))                 
        # image = cv2.addWeighted(im_bg,1,g_y,1,0)
