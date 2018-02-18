import cv2
import tensorflow as tf
import numpy as np
import glob
import imageio
import os

from ae import factory, utils
from eval import eval_utils

import argparse
import rmcssd.bin.detector as detector


parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("ssd_name")
parser.add_argument("-f", "--folder_str", required=True)
# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

folder_str = arguments.folder_str
ssd_name = arguments.ssd_name
ssd = detector.Detector(os.path.join('/home_local/sund_ma/ssd_ws/checkpoints', ssd_name))

start_var_list =set([var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])


codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)

all_var_list = set([var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
ae_var_list = all_var_list.symmetric_difference(start_var_list)
saver = tf.train.Saver(ae_var_list)

workspace_path = os.environ.get('AE_WORKSPACE_PATH')

if workspace_path == None:
    print 'Please define a workspace path:\n'
    print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
    exit(-1)
log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = utils.get_checkpoint_dir(log_dir)
factory.restore_checkpoint(ssd._isess, saver, ckpt_dir)


files = glob.glob(os.path.join(str(folder_str),'*.png'))
for file in files:

    img = cv2.imread(file)
    rclasses, rscores, rbboxes = ssd.process(img)

    H, W = img.shape[:2]
    for i in xrange(len(rscores)):
        score = rscores[i]
        ymin = int(rbboxes[i, 0] * H)
        xmin = int(rbboxes[i, 1] * W)
        ymax = int(rbboxes[i, 2] * H)
        xmax = int(rbboxes[i, 3] * W)
        cv2.rectangle(img, (xmin,ymin),(xmax,ymax), (255,0,0), 2)
        cv2.putText(img, '%1.3f' % score, (xmin, ymax+20), cv2.FONT_ITALIC, .6, (255,0,0), 2)
    cv2.imshow()
    continue

    ssd_boxes = [ (int(rbboxes[i][0]*H), int(rbboxes[i][1]*W), int(rbboxes[i][2]*H), int(rbboxes[i][3]*W)) for i in xrange(len(rbboxes)) if rclasses[i] == 1 ]
    ssd_imgs = np.empty((len(rbboxes),) + dataset.shape)

    vis_img = 0.3 * np.ones((np.max([len(rbboxes),3])*dataset.shape[0],2*dataset.shape[1],dataset.shape[2]))
    #print vis_img.shape

    for j,ssd_box in enumerate(ssd_boxes):
        ymin, xmin, ymax, xmax = ssd_box

        ssd_img = img[ymin:ymax,xmin:xmax]
        h, w = ssd_img.shape[:2]
        size = int(np.maximum(h, w) * 1.1)
        cx = xmin + (xmax - xmin)/2
        cy = ymin + (ymax - ymin)/2

        left = np.maximum(cx-size/2, 0)
        top = np.maximum(cy-size/2, 0)

        ssd_img = img[top:cy+size/2,left:cx+size/2]
        ssd_img = cv2.resize(ssd_img, dataset.shape[:2])
        ssd_img = ssd_img / 255.
        ssd_imgs[j,:,:,:] = ssd_img

    if len(rbboxes) > 0:

        Rs = codebook.nearest_rotation(ssd._isess, ssd_imgs)
        ssd_rot_imgs = 0.3*np.ones_like(ssd_imgs)

        for j,R in enumerate(Rs):
            ssd_rot_imgs[j,:,:,:] = dataset.render_rot( R ,downSample = 1)/255.

        ssd_imgs = ssd_imgs.reshape(-1,*ssd_imgs.shape[2:])
        ssd_rot_imgs = ssd_rot_imgs.reshape(-1,*ssd_rot_imgs.shape[2:])
        vis_img[:ssd_imgs.shape[0],:dataset.shape[1],:] = ssd_imgs
        vis_img[:ssd_rot_imgs.shape[0],dataset.shape[1]:,:] = ssd_rot_imgs




    cv2.imshow('preds', vis_img)
    cv2.imshow('img', img)
    cv2.waitKey(0)
        




    # print rclasses, rscores, rbboxes


    # im = cv2.resize(im,(128,128))

    # R = codebook.nearest_rotation(ssd._isess, im/255.)

    # pred_view = dataset.render_rot( R ,downSample = 1)
    
    # cv2.imshow('resized img', cv2.resize(im/255.,(256,256)))
    # cv2.imshow('pred_view', cv2.resize(pred_view,(256,256)))
    # print R
    # cv2.waitKey(0)




