# import pygame
# import pygame.camera
import cv2
import argparse
import tensorflow as tf
import numpy as np
import os

from auto_pose.ae import factory
from auto_pose.ae import utils as u
from webcam_video_stream import WebcamVideoStream

# def initializeWebcam(width, height):
#     #initialise pygame   
#     pygame.init()
#     pygame.camera.init()
#     cam = pygame.camera.Camera("/dev/video0",(width,height))
#     cam.start()

#     #setup window
#     windowSurfaceObj = pygame.display.set_mode((width,height),1,16)
#     pygame.display.set_caption('Camera')

#     return cam



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


width = 960
height = 720
videoStream = WebcamVideoStream(0,width,height).start()


with tf.Session() as sess:
    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)


    # width = 1280/2
    # height = 720/2

    # cam = initializeWebcam(width, height)

    while videoStream.isActive():
        image = videoStream.read()
        # image = cam.get_image()
        # arr = pygame.surfarray.array3d(image)

        # arr = np.swapaxes(arr,0,1)
        # arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        img = image[:,(width-height)/2:width-(width-height)/2,:]/255.0
        # img = cv2.resize(arr,(128,128))
        try:
            img = cv2.resize(img,(128,128))
            R = codebook.nearest_rotation(sess, img)
            pred_view = dataset.render_rot(R,downSample = 1)
        except:
            print 'empty img'
        print R
        cv2.imshow('webcam input', image)
        cv2.imshow('pred view rendered', pred_view)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
