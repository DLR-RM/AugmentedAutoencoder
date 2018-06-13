import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import configparser

from auto_pose.ae import factory, utils

from m3vision.interfaces.pose_estimator import PoseEstInterface,PoseEstimate,Roi3D

class AePoseEstimator(PoseEstInterface):
    """ """

    # Takes a configPath only!
    def __init__(self, test_configpath):

        test_args = configparser.ConfigParser()
        test_args.read(test_configpath) 

        workspace_path = test_args.get('MODEL','workspace_path')
        experiment_group = test_args.get('MODEL','experiment_group')
        experiment_name = test_args.get('MODEL','experiment_name')

        train_cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
        
        self.train_args = configparser.ConfigParser()
        self.train_args.read(train_cfg_file_path)  

        log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
        ckpt_dir = utils.get_checkpoint_dir(log_dir)


        self._process_requirements = ['color_img', 'camK', 'bboxes']
        self._upright = test_args.getboolean('MODEL','upright')
        self._topk = test_args.getint('MODEL','topk')
        self._image_format = {'color_format':test_args.get('MODEL','color_format'), 
                              'color_data_type': eval(test_args.get('MODEL','color_data_type')),
                              'depth_data_type': eval(test_args.get('MODEL','depth_data_type')) }

        if test_args.getboolean('MODEL','icp'):
            from auto_pose.icp import icp
            self._process_requirements.append('depth_img')
            self.icp_handle = icp.ICP(train_args)
        if test_args.getboolean('MODEL','camPose'):
            self._process_requirements.append('camPose')


        self.codebook, self.dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)
        saver = tf.train.Saver()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.3))
        self.sess = tf.InteractiveSession(config=config)
        factory.restore_checkpoint(self.sess, saver, ckpt_dir)


    def set_parameter(self, string_name, string_val):
        pass

    # ABS
    def query_process_requirements():
        return self._process_requirements

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
        
        #print vis_img.shape
        all_Rs, all_ts = [],[]
        H_est = np.eye(4)

        for j,box in enumerate(bboxes):
            h_box, w_box = (box.ymax-box.ymin)*H, (box.xmax-box.xmin)*W
            cy, cx = int(box.ymin*H + h_box/2), int(box.xmin*W + w_box/2)
            size = int(np.maximum(h_box, w_box) * self.train_args.getfloat('Dataset','PAD_FACTOR'))

            left = np.maximum(cx-size/2, 0)
            top = np.maximum(cy-size/2, 0)

            det_img = color_img[top:cy+size/2,left:cx+size/2]
            det_img = cv2.resize(det_img, self.dataset.shape[:2])

            # if dataset.shape[2]  == 1:
            #     det_img = cv2.cvtColor(det_img,cv2.COLOR_BGR2GRAY)[:,:,None]

            box_xywh = [int(box.xmin*W),int(box.ymin*H),w_box,h_box]
            Rs_est, ts_est = self.codebook.nearest_rotation_with_bb_depth(self.sess, det_img, box_xywh, camK, self._topk, self.train_args, upright=self._upright)

            if self._topk == 1:
                R_est = Rs_est.squeeze()
                t_est = ts_est.squeeze()
            else:
                print 'ERROR: Not topk > 1 not implemented yet'
                exit()

            if 'depth_img' in self.query_process_requirements():
                depth_crop = depth_img[top:cy+size/2,left:cx+size/2]
                R_est, t_est = self.icp_handle.icp_refinement(depth_crop, R_est, t_est, camK, (W,H))

            H_est[:3,:3] = R_est
            H_est[:3,3] = t_est
            
            top1_pose = PoseEstimate(name=box.classes[max(box.classes)],trafo=H_est)
            all_pose_estimates.append(top1_pose)
        #TODO camPose
        return all_pose_estimates



