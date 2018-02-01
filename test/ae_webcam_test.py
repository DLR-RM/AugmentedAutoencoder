import pygame
import pygame.camera
import cv2
import tensorflow as tf
import numpy as np

from ae import factory
import argparse


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



parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
arguments = parser.parse_args()

full_name = arguments.experiment_name.split('/')

experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

codebook = factory.build_codebook_from_name(experiment_name)

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
log_dir = u.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = u.get_checkpoint_dir(log_dir)

with tf.Session() as sess:
    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)


    width = 1280/2
    height = 720/2

    cam = initializeWebcam(width, height)

    while True:
        image = cam.get_image()
        arr = pygame.surfarray.array3d(image)

        arr = np.swapaxes(arr,0,1)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        arr = arr[:,(width-height)/2:width-(width-height)/2,:]/255.0
        img = cv2.resize(arr,(128,128))

        cv2.imshow('resized webcam input', arr)
        cv2.waitKey(1)

        R = codebook.nearest_rotation(sess, img)

        print R

if __name__ == '__main__':
    main()
