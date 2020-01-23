import cv2
import argparse
import tensorflow as tf
import numpy as np
import os
import configparser

from auto_pose.ae import factory
from auto_pose.ae import utils as u
from webcam_video_stream import WebcamVideoStream

import keras
from keras_retinanet.models import load_model, backbone


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

width = 640
height = 480
videoStream = WebcamVideoStream(0,width,height).start()
object_id = 246 # Mug in the OID dataset

with tf.Session() as sess:
    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)
    # RetinaNet ResNet101 trained on OID dataset, pre-trained model from:
    # https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018
    model = load_model('/shared-folder/autoencoder_ws/retinanet.h5')

    while videoStream.isActive():
        image = videoStream.read()

        img_array = np.expand_dims(image, axis=0)
        boxes, scores, labels = model.predict(img_array)

        valid_dets = np.where((labels[0] == object_id) & (scores[0] > 0.3))

        scores = scores[0][valid_dets]
        boxes = boxes[0][valid_dets]
        labels = labels[0][valid_dets]

        # Loop through the detections
        for i in np.arange(len(boxes)):
            # Convert from x1,y1,x2,y2 to xywh bounding box format
            boxes[i][2] = boxes[i][2]-boxes[i][0]
            boxes[i][3] = boxes[i][3]-boxes[i][1]

            # Extract ROI for detection
            detection = dataset.extract_square_patch(image, boxes[i], train_args.getfloat('Dataset','PAD_FACTOR'))
            # Rs, ts = codebook.auto_pose6d(sess, image_crop, bb_xywh, K_test, 1, train_args)

            # Find nearest neighbors in codebook
            n = 10
            R = codebook.nearest_rotation(sess, detection, top_n=n)
            pred_view = None
            for k in np.arange(n):
                curr_view = dataset.render_rot(R[k],downSample = 1)
                if(pred_view is None):
                    pred_view = curr_view
                else:
                    pred_view = np.concatenate((pred_view, curr_view), axis=1)
                #print(R[k])
            cv2.imshow('prediction{0}'.format(i), pred_view)

        # Draw bounding boxes
        id = 0
        for label,box,score in zip(labels,boxes,scores):
            box = box.astype(np.int32)
            xmin,ymin,xmax,ymax = box[0],box[1],box[0]+box[2],box[1]+box[3]
            cv2.putText(image, "ID: {0}".format(id), (xmin, ymin-20), cv2.FONT_ITALIC, .5, (255,0,0), 2)
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),2)
            id = id+1

        # Show camera image
        cv2.imshow('camera', image)
        k = cv2.waitKey(1)
        if k == 27:
            break
    print("Closing....")
    videoStream.stop()
    sess.close()
