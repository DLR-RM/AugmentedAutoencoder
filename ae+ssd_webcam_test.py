# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import thread
# import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import time
import argparse

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import pygame
import pygame.camera

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ae import factory
# import pysixd

thresh = 0.24

import time

tab20 = (
    (0.12156862745098039, 0.4666666666666667,  0.7058823529411765  ),  # 1f77b4
    (0.6823529411764706,  0.7803921568627451,  0.9098039215686274  ),  # aec7e8
    (1.0,                 0.4980392156862745,  0.054901960784313725),  # ff7f0e
    (1.0,                 0.7333333333333333,  0.47058823529411764 ),  # ffbb78
    (0.17254901960784313, 0.6274509803921569,  0.17254901960784313 ),  # 2ca02c
    (0.596078431372549,   0.8745098039215686,  0.5411764705882353  ),  # 98df8a
    (0.8392156862745098,  0.15294117647058825, 0.1568627450980392  ),  # d62728
    (1.0,                 0.596078431372549,   0.5882352941176471  ),  # ff9896
    (0.5803921568627451,  0.403921568627451,   0.7411764705882353  ),  # 9467bd
    (0.7725490196078432,  0.6901960784313725,  0.8352941176470589  ),  # c5b0d5
    (0.5490196078431373,  0.33725490196078434, 0.29411764705882354 ),  # 8c564b
    (0.7686274509803922,  0.611764705882353,   0.5803921568627451  ),  # c49c94
    (0.8901960784313725,  0.4666666666666667,  0.7607843137254902  ),  # e377c2
    (0.9686274509803922,  0.7137254901960784,  0.8235294117647058  ),  # f7b6d2
    (0.4980392156862745,  0.4980392156862745,  0.4980392156862745  ),  # 7f7f7f
    (0.7803921568627451,  0.7803921568627451,  0.7803921568627451  ),  # c7c7c7
    (0.7372549019607844,  0.7411764705882353,  0.13333333333333333 ),  # bcbd22
    (0.8588235294117647,  0.8588235294117647,  0.5529411764705883  ),  # dbdb8d
    (0.09019607843137255, 0.7450980392156863,  0.8117647058823529  ),  # 17becf
    (0.6196078431372549,  0.8549019607843137,  0.8980392156862745),    # 9edae5
    (0.12156862745098039, 0.4666666666666667,  0.7058823529411765  ),  # 1f77b4
    (0.6823529411764706,  0.7803921568627451,  0.9098039215686274  ),  # aec7e8
    (1.0,                 0.4980392156862745,  0.054901960784313725),  # ff7f0e
    (1.0,                 0.7333333333333333,  0.47058823529411764 ),  # ffbb78
    (0.17254901960784313, 0.6274509803921569,  0.17254901960784313 ),  # 2ca02c
    (0.596078431372549,   0.8745098039215686,  0.5411764705882353  ),  # 98df8a
    (0.8392156862745098,  0.15294117647058825, 0.1568627450980392  ),  # d62728
    (1.0,                 0.596078431372549,   0.5882352941176471  ),  # ff9896
    (0.5803921568627451,  0.403921568627451,   0.7411764705882353  ),  # 9467bd
    (0.7725490196078432,  0.6901960784313725,  0.8352941176470589  ),  # c5b0d5
    (0.12156862745098039, 0.4666666666666667,  0.7058823529411765  ),  # 1f77b4
    (0.6823529411764706,  0.7803921568627451,  0.9098039215686274  ),  # aec7e8
    (1.0,                 0.4980392156862745,  0.054901960784313725),  # ff7f0e
    (1.0,                 0.7333333333333333,  0.47058823529411764 ),  # ffbb78
    (0.17254901960784313, 0.6274509803921569,  0.17254901960784313 ),  # 2ca02c
    (0.596078431372549,   0.8745098039215686,  0.5411764705882353  ),  # 98df8a
)


# bridge = CvBridge()
# webcam_image_depth = np.zeros((480,640), dtype=np.float32)
# detection_image = np.zeros((480,640,3), dtype=np.uint8)

# def callback(data):
#     global new_image
#     webcam_image[:] = bridge.imgmsg_to_cv2(data, "passthrough")
#     new_image = True

# def callback_depth(data):
#     webcam_image_depth[:] = bridge.imgmsg_to_cv2(data, "passthrough")

# rospy.init_node('listener', anonymous=True, disable_signals=True)
# rospy.Subscriber("/rgb/image", Image, callback, queue_size=1)
# rospy.Subscriber("/depth/image", Image, callback_depth, queue_size=1)
# thread.start_new_thread( rospy.spin, ())


import sys
sys.path.append('/home_local/sund_ma/src/SSD_Tensorflow')

from nets import ssd_vgg_300, ssd_common, np_methods, nets_factory
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NCHW'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None

ssd_class = nets_factory.get_network('ssd_300_vgg')
# ssd_net = ssd_class(ssd_params)
ssd_params = ssd_class.default_params._replace(num_classes=2, no_annotation_label=2)
ssd_net = ssd_vgg_300.SSDNet(ssd_params)
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)


ckpt_filename = '/home_local/sund_ma/src/SSD_Tensorflow/mug'
# ckpt_filename = 'logs_scene11_1000/model.ckpt-8572'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, tf.train.latest_checkpoint(ckpt_filename))

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)



parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
arguments = parser.parse_args()
experiment_name = arguments.experiment_name
# experiment_name = 'bigger_network'


start_var_list =set([var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
codebook, dataset = factory.build_codebook_from_name(experiment_name, True)

all_var_list = set([var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
ae_var_list = all_var_list.symmetric_difference(start_var_list)
saver = tf.train.Saver(ae_var_list)
factory.restore_checkpoint(isess, saver, experiment_name)

# Main image processing routine.
def process_image(img, select_threshold=0.6, nms_threshold=.05, net_shape=(300, 300)):
    # Run SSD network.

    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    #print rbbox_img

    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

#print "start"

scene_list = []

time_buf = np.zeros((20,))
time_buf_i = 0
flag = False
m = 0


def initializeWebcam(width, height):
    #initialise pygame   
    pygame.init()
    pygame.camera.init()
    cam = pygame.camera.Camera("/dev/video0",(width,height))
    cam.start()

    #setup window
    windowSurfaceObj = pygame.display.set_mode((width,height),1,16)
    pygame.display.set_caption('Camera')

    return cam


width = 720/2#480
height = 1280/2#640
webcam_image = np.zeros((height,width,3), dtype=np.uint8)
# img = cv2.cvtColor(webcam_image, cv2.COLOR_RGB2BGR)
cam = initializeWebcam(width, height)


while True:

    start_time = time.time()
    image = cam.get_image()
    arr = pygame.surfarray.array3d(image)

    webcam_image = np.swapaxes(arr,0,1)
    img = cv2.cvtColor(webcam_image, cv2.COLOR_RGB2BGR)
    # img = arr[:,(width-height)/2:width-(width-height)/2,:]/255.0
    # img = cv2.resize(img,(128,128))
    cam_read_time = time.time()
    print 'cam_read_time, ', cam_read_time - start_time

    t0 = time.time()
    rclasses, rscores, rbboxes =  process_image(img)
    time_buf[time_buf_i] = time.time() - t0

    ssd_time = time.time()
    print 'ssd_time, ', ssd_time - cam_read_time

    vis_img = webcam_image.copy()
    H, W = vis_img.shape[:2]

    mug_boxes = [ (int(rbboxes[i][0]*H), int(rbboxes[i][1]*W), int(rbboxes[i][2]*H), int(rbboxes[i][3]*W)) for i in xrange(len(rbboxes)) if rclasses[i] == 1 ]
    mug_imgs = []

    for mug_box in mug_boxes:
        ymin, xmin, ymax, xmax = mug_box

        mug_img = img[ymin:ymax,xmin:xmax]
        h, w = mug_img.shape[:2]
        size = int(np.maximum(h, w) * 1.2)
        cx = xmin + (xmax - xmin)/2
        cy = ymin + (ymax - ymin)/2

        left = np.maximum(cx-size/2, 0)
        top = np.maximum(cy-size/2, 0)

        mug_img = img[top:cy+size/2,left:cx+size/2]
        mug_img = cv2.resize(mug_img, (128, 128))
        mug_img = mug_img / 255.
        mug_imgs.append(mug_img)

    if len(mug_imgs) > 0:
        mug_rot_imgs = []
        mug_imgs = np.array(mug_imgs)


        t0 = time.time()
        Rs = codebook.nearest_rotation(isess, mug_imgs)
        time_buf[time_buf_i] += time.time() - t0

        lookup_time = time.time()
        print 'lookup_time, ', lookup_time - ssd_time

        for R in Rs:
            mug_rot_imgs.append(dataset.render_rot( R ,downSample = 4))

        render_time = time.time()
        print 'render_time, ', render_time - lookup_time

        mug_imgs = np.vstack(mug_imgs)
        
        padding = 0.3 * np.ones( (H-mug_imgs.shape[0], mug_imgs.shape[1], 3) )
        
        mug_imgs = np.vstack( (mug_imgs, padding) )
        mug_imgs = (mug_imgs * 255.).astype(np.uint8)

        # mug_imgs = cv2.cvtColor(mug_imgs, cv2.COLOR_BGR2RGB)

        mug_rot_imgs = np.vstack( mug_rot_imgs )
        mug_rot_imgs = np.vstack( (mug_rot_imgs, padding) )
        mug_rot_imgs = (mug_rot_imgs * 255.).astype(np.uint8)
        # mug_rot_imgs = cv2.cvtColor(mug_rot_imgs, cv2.COLOR_BGR2RGB)
        vis_img = np.hstack( (img, mug_imgs, mug_rot_imgs) )
    else:
        mug_rot_imgs = 0.3 * np.ones( (H, 2*128, 3) )
        mug_rot_imgs = (mug_rot_imgs * 255.).astype(np.uint8)
        # mug_rot_imgs = cv2.cvtColor(mug_rot_imgs, cv2.COLOR_BGR2RGB)
        vis_img = np.hstack( (img, mug_rot_imgs) )

    if flag or (time_buf_i == len(time_buf) -1):
        _H, _W = vis_img.shape[:2]
        vis_img[_H-40:,_W-125:] = 255.
        cv2.putText(vis_img, '{:03d} Hz'.format( int(1./time_buf.mean()) ), (_W-120, _H-10), cv2.FONT_ITALIC, 1., [20, 20, 20], 2)
        flag = True


    for i in xrange(len(rscores)):
        score = rscores[i]
        ymin = int(rbboxes[i, 0] * H)
        xmin = int(rbboxes[i, 1] * W)
        ymax = int(rbboxes[i, 2] * H)
        xmax = int(rbboxes[i, 3] * W)
        cv2.rectangle(vis_img, (xmin,ymin),(xmax,ymax), tuple([255*x for x in tab20[i]]), 2)
        cv2.putText(vis_img, '%1.3f' % score, (xmin, ymax+20), cv2.FONT_ITALIC, .6, tuple([255*x for x in tab20[i]]), 2)

    cv2.imshow('img', vis_img)
    cv2.waitKey(1)
        

    m += 1


    time_buf_i = (time_buf_i + 1) % len(time_buf)

    display_time = time.time()
    try:
        print 'display_time, ', display_time - render_time
    except:
        pass