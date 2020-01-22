import cv2
import argparse
import tensorflow as tf
import numpy as np
import os
import configparser

from auto_pose.ae import factory
from auto_pose.ae import utils as u
from webcam_video_stream import WebcamVideoStream

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
arguments = parser.parse_args()

full_name = arguments.experiment_name.split('/')

experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

codebook,dataset = factory.build_codebook_from_name(experiment_name,experiment_group,return_dataset=True)

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
log_dir = u.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = u.get_checkpoint_dir(log_dir)

train_cfg_file_path = u.get_train_config_exp_file_path(log_dir, experiment_name)
train_args = configparser.ConfigParser()
train_args.read(train_cfg_file_path)  

width = 960
height = 720
videoStream = WebcamVideoStream(0,width,height).start()

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.9)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)

    while videoStream.isActive():
        image = videoStream.read()
        
        # try your detector here:
        # bb_xywh = detector.detect(image)
        # image_crop = dataset.extract_square_patch(image, bb_xywh, train_args.getfloat('Dataset','PAD_FACTOR'))
        # Rs, ts = codebook.auto_pose6d(sess, image_crop, bb_xywh, K_test, 1, train_args)

        img = cv2.resize(image,(128,128))

        R = codebook.nearest_rotation(sess, img)
        pred_view = dataset.render_rot(R,downSample = 1)
        print(R)
        cv2.imshow('resized webcam input', img)
        cv2.imshow('pred view rendered', pred_view)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
