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

        workspace_path = os.environ.get('AE_WORKSPACE_PATH')

        if workspace_path == None:
            print 'Please define a workspace path:\n'
            print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
            exit(-1)

        self._process_requirements = ['color_img', 'camK', 'bboxes']
        if test_args.getboolean('MODEL','camPose'):
            self._process_requirements.append('camPose')
        self._camPose = test_args.getboolean('MODEL','camPose')
        self._upright = test_args.getboolean('MODEL','upright')
        self._topk = test_args.getint('MODEL','topk')
        if self._topk > 1:
            print 'ERROR: topk > 1 not implemented yet'
            exit()

        self._image_format = {'color_format':test_args.get('MODEL','color_format'), 
                              'color_data_type': eval(test_args.get('MODEL','color_data_type')),
                              'depth_data_type': eval(test_args.get('MODEL','depth_data_type')) }

        self.all_experiments = eval(test_args.get('MODEL','experiments'))
        self.class_names = eval(test_args.get('MODEL','class_names'))
        self.all_codebooks = []
        self.all_train_args = []

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        config.gpu_options.per_process_gpu_memory_fraction = test_args.getfloat('MODEL','gpu_memory_fraction')

        self.sess = tf.Session(config=config)

        for i,experiment in enumerate(self.all_experiments):
            full_name = experiment.split('/')
            experiment_name = full_name.pop()
            experiment_group = full_name.pop() if len(full_name) > 0 else ''
            log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
            ckpt_dir = utils.get_checkpoint_dir(log_dir)
            train_cfg_file_path = utils.get_train_config_exp_file_path(log_dir, experiment_name)
            print train_cfg_file_path
            # train_cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
            train_args = configparser.ConfigParser()
            train_args.read(train_cfg_file_path)
            self.all_train_args.append(train_args)

            self.all_codebooks.append(factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=False))
            saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=experiment_name))
            factory.restore_checkpoint(self.sess, saver, ckpt_dir)


            if test_args.getboolean('MODEL','icp'):
                assert len(self.all_experiments) == 1, 'icp currently only works for one object'
                # currently works only for one object
                from auto_pose.icp import icp
                self._process_requirements.append('depth_img')
                self.icp_handle = icp.ICP(train_args)

            
            
            # self.codebook = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)
            # saver = tf.train.Saver()
            # config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.3))
            # self.sess = tf.InteractiveSession(config=config)
            # factory.restore_checkpoint(self.sess, saver, ckpt_dir)


    def set_parameter(self, string_name, string_val):
        pass

    # ABS
    def query_process_requirements(self):
        return self._process_requirements

    def query_image_format(self):
        return self._image_format
#    def get_native_coord_frame():
#        """With respect to x -> right; y -> down; z away from the camera.
#        E.g.: OGL system would be:
#        [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]"""
#        pass
    
    # ABS
    def process(self, bboxes, color_img, camK, depth_img=None, camPose=None, rois3ds=[]):
        """ roi3ds is a list of roi3d"""

        H, W = color_img.shape[:2]
        
        #print vis_img.shape
         
        all_Rs, all_ts = [],[]
        H_est = np.eye(4)
        all_pose_estimates = []

        for j,box in enumerate(bboxes):

            pred_clas = max(box.classes)

            try:
                clas_idx = self.class_names.index(pred_clas)
            except:
                print('%s not contained in config class_names %s', (pred_clas, self.class_names))
                continue

            h_box, w_box = (box.ymax-box.ymin)*H, (box.xmax-box.xmin)*W
            cy, cx = int(box.ymin*H + h_box/2), int(box.xmin*W + w_box/2)
            size = int(np.maximum(h_box, w_box) * self.all_train_args[clas_idx].getfloat('Dataset','PAD_FACTOR'))

            left = np.maximum(cx-size/2, 0)
            top = np.maximum(cy-size/2, 0)

            det_img = color_img[top:cy+size/2,left:cx+size/2]
            det_img = cv2.resize(det_img, (self.all_train_args[clas_idx].getint('Dataset','W'),self.all_train_args[clas_idx].getint('Dataset','H')))

            # cv2.imshow('',det_img)
            # cv2.waitKey(0)
            # if dataset.shape[2]  == 1:
            #     det_img = cv2.cvtColor(det_img,cv2.COLOR_BGR2GRAY)[:,:,None]

            box_xywh = [int(box.xmin*W),int(box.ymin*H),w_box,h_box]
            Rs_est, ts_est, _ = self.all_codebooks[clas_idx].auto_pose6d(self.sess, 
                                                                        det_img, 
                                                                        box_xywh, 
                                                                        camK,#*2 REMOVE, and input correct calibration with 1280x960 instead 640x480 
                                                                        self._topk, 
                                                                        self.all_train_args[clas_idx], 
                                                                        upright=self._upright)

            R_est = Rs_est.squeeze()
            t_est = ts_est.squeeze()

            if 'depth_img' in self.query_process_requirements():
                depth_crop = depth_img[top:cy+size/2,left:cx+size/2]
                R_est, t_est = self.icp_handle.icp_refinement(depth_crop, R_est, t_est, camK, (W,H))


            H_est[:3,:3] = R_est
            H_est[:3,3] = t_est / 1000. #mm in m
            print 'translation from camera: ',  H_est[:3,3]

            if self._camPose:
                H_est = np.dot(camPose, H_est)           

            top1_pose = PoseEstimate(name=pred_clas,trafo=H_est)
            all_pose_estimates.append(top1_pose)
        #TODO camPose

        return all_pose_estimates


