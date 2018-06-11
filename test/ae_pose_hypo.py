import cv2
import tensorflow as tf
import numpy as np
import glob
import imageio
import os
import ConfigParser

from ae import factory, utils
from eval import eval_utils

# from mvis_pose_estimator import mvis_pose_estimator
import m3vision

class AePoseEstimator(PoseEstInterface):
    """ """

    # Takes a configPath only!
    def __init__(self, test_configpath):

        test_args = ConfigParser.ConfigParser()
        test_args.read(test_configpath) 

        self._needs = ['rgb', 'camK', 'bboxes']
        self._upright = test_args.getboolean('MODEL','upright')
        self._topk = test_args.getint('MODEL','topk')
        self._image_format = {'color_format':test_args.get('MODEL','image_format'), 
                              'type': eval(test_args.get('MODEL','image_type')) }

        if test_args.getboolean('MODEL','icp'):
            self._needs.append('depth')
        if test_args.getboolean('MODEL','camPose'):
            self._needs.append('camPose')

        workspace_path = test_args.get('MODEL','workspace_path')
        experiment_group = test_args.get('MODEL','experiment_group')
        experiment_name = test_args.get('MODEL','experiment_name')

        train_cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
        
        self.train_args = ConfigParser.ConfigParser()
        self.train_args.read(train_cfg_file_path)  

        log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
        ckpt_dir = utils.get_checkpoint_dir(log_dir)

        self.codebook, self.dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)
        saver = tf.train.Saver()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.3))
        self.sess = tf.InteractiveSession(config=config)
        factory.restore_checkpoint(self.sess, saver, ckpt_dir)


    def set_parameter(self, string_name, string_val):
        pass

    # ABS
    def query_needs():
        return self._needs

    def query_image_format():
        return self._image_format
#    def get_native_coord_frame():
#        """With respect to x -> right; y -> down; z away from the camera.
#        E.g.: OGL system would be:
#        [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]"""
#        pass
    
    # ABS
    def process(self, bboxes=[], color_img=None, depth_img=None, camK=None, camPose=None, rois3ds=[]):
        """ roi3ds is a list of roi3d"""

        H, W = color_img.shape[:2]

        det_imgs = np.empty((len(bboxes),) + dataset.shape)
        
        #print vis_img.shape
        all_Rs, all_ts = [],[]
        for j,box in enumerate(bboxes):
            h_box, w_box = (box.ymax-box.ymin)*H, (box.xmax-box.xmin)*W
            cy, cx = int(box.ymin*H + h_box/2), int(box.xmin*W + w_box/2)
            size = int(np.maximum(h_box, w_box) * 1.25)

            left = np.maximum(cx-size/2, 0)
            top = np.maximum(cy-size/2, 0)

            det_img = color_img[top:cy+size/2,left:cx+size/2]
            det_img = cv2.resize(det_img, self.dataset.shape[:2])

            # if dataset.shape[2]  == 1:
            #     det_img = cv2.cvtColor(det_img,cv2.COLOR_BGR2GRAY)[:,:,None]

            box_xywh = [int(box.xmin*W),int(box.ymin*H),w_box,h_box]
            R, t,_,_= codebook.nearest_rotation_with_bb_depth(self.sess, det_img, box_xywh, camK, self._topk, self.train_args, upright=self._upright)

            all_Rs.append(R)
            all_ts.append(t)
        #TODO camPose
        return all_Rs, all_ts