import cv2
try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
except:
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
    def __init__(self, test_config_path):
        
        test_args = self.get_params(test_config_path)

        workspace_path = os.environ.get('AE_WORKSPACE_PATH')

        if workspace_path == None:
            print('Please define a workspace path:\n')
            print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
            exit(-1)

        self._process_requirements = ['color_img', 'camK', 'bboxes']
        if test_args.getboolean('auto_pose','camPose'):
            self._process_requirements.append('camPose')
        self._camPose = test_args.getboolean('auto_pose','camPose')
        self._upright = test_args.getboolean('auto_pose','upright')
        self._topk = test_args.getint('auto_pose','topk')
        if self._topk > 1:
            print('ERROR: topk > 1 not implemented yet')
            exit()

        self._image_format = {'color_format':test_args.get('auto_pose','color_format'), 
                              'color_data_type': eval(test_args.get('auto_pose','color_data_type')),
                              'depth_data_type': eval(test_args.get('auto_pose','depth_data_type')) }

        # self.vis = test_args.getboolean('auto_pose','pose_visualization')

        # self.all_experiments = eval(test_args.get('auto_pose','experiments'))
        self.class_2_encoder = eval(test_args.get('auto_pose','class_2_encoder'))

        self.all_codebooks = {}
        self.all_train_args = {}
        self.pad_factors = {}
        self.patch_sizes = {}

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        config.gpu_options.per_process_gpu_memory_fraction = test_args.getfloat('auto_pose','gpu_memory_fraction')

        self.sess = tf.Session(config=config)

        for clas_name,experiment in self.class_2_encoder.items():
            full_name = experiment.split('/')
            experiment_name = full_name.pop()
            experiment_group = full_name.pop() if len(full_name) > 0 else ''
            log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
            ckpt_dir = utils.get_checkpoint_dir(log_dir)
            train_cfg_file_path = utils.get_train_config_exp_file_path(log_dir, experiment_name)
            print(train_cfg_file_path)
            # train_cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
            train_args = configparser.ConfigParser(inline_comment_prefixes="#")
            train_args.read(train_cfg_file_path)
            self.all_train_args[clas_name] = train_args
            self.pad_factors[clas_name] = train_args.getfloat('Dataset','PAD_FACTOR')
            self.patch_sizes[clas_name] = (train_args.getint('Dataset','W'), train_args.getint('Dataset','H'))

            self.all_codebooks[clas_name] = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=False)
            saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=experiment_name))
            factory.restore_checkpoint(self.sess, saver, ckpt_dir)


    def set_parameter(self, string_name, string_val):
        pass

    # ABS
    def query_process_requirements(self):
        return self._process_requirements

    def query_image_format(self):
        return self._image_format


    # def extract_square_patch(self, scene_img, bb_xywh, pad_factor,resize=(128,128),interpolation=cv2.INTER_NEAREST):

    #     x, y, w, h = np.array(bb_xywh).astype(np.int32)
    #     size = int(np.maximum(h, w) * pad_factor)
        
    #     left = np.maximum(x+w//2-size//2, 0)
    #     right = x+w//2+size/2
    #     top = np.maximum(y+h//2-size//2, 0)
    #     bottom = y+h//2+size//2

    #     scene_crop = scene_img[top:bottom, left:right]
    #     scene_crop = cv2.resize(scene_crop, resize, interpolation = interpolation)
    #     return scene_crop

    def extract_square_patch(self, scene_img, bb_xywh, pad_factor,resize=(128,128),interpolation=cv2.INTER_NEAREST,black_borders=False):

        x, y, w, h = np.array(bb_xywh).astype(np.int32)
        size = int(np.maximum(h, w) * pad_factor)


        scene_crop = np.zeros((size, size, 3),dtype=np.uint8)
        if black_borders:
            scene_crop[(size-h)//2:(size-h)//2 + h,
                    (size-w)//2:(size-w)//2 + w] = scene_img[y:y+h, x:x+w].copy()
        else:

            left_trunc = np.maximum(x+w/2-size/2, 0)
            right_trunc = np.minimum(x+w/2+size/2, scene_img.shape[1])
            top_trunc = np.maximum(y+h/2-size/2, 0)
            bottom_trunc = np.minimum(y+h/2+size/2, scene_img.shape[0])

            size_h = (bottom_trunc - top_trunc)
            size_w = (right_trunc - left_trunc)

            scene_crop[(size-size_h)//2:(size-size_h)//2 + size_h,
                       (size-size_w)//2:(size-size_w)//2 + size_w] = scene_img[top_trunc:bottom_trunc, left_trunc:right_trunc].copy()

        scene_crop = cv2.resize(scene_crop, resize, interpolation=interpolation)

        return scene_crop

    def process(self, bboxes, color_img, camK, depth_img=None, camPose=None, rois3ds=[], mm=False):

        H, W = color_img.shape[:2]

        all_Rs, all_ts = [],[]
        all_pose_estimates = []
        # if self.vis:
        #     img_show = color_img.copy()
        #     depth_img_show = np.dstack((depth_img.copy(),depth_img.copy(),depth_img.copy()))

        for j,box in enumerate(bboxes):
            H_est = np.eye(4)
            pred_clas = max(box.classes, key=box.classes.get)

            if not pred_clas in self.class_2_encoder:
                print(('%s not contained in config class_names %s' % (pred_clas, self.class_2_encoder.keys())))
                continue

            box_xywh = [box.xmin * W, box.ymin * H, (box.xmax - box.xmin) * W, (box.ymax - box.ymin) * H]
            if np.any(np.array(box_xywh) < 0):
                print(('invalid bb', box_xywh))
                continue

            print(box_xywh)
            det_img = self.extract_square_patch(color_img, 
                                                box_xywh, 
                                                self.pad_factors[pred_clas],
                                                resize=self.patch_sizes[pred_clas], 
                                                interpolation=cv2.INTER_LINEAR,
                                                black_borders=True)

            Rs_est, ts_est = self.all_codebooks[pred_clas].auto_pose6d(self.sess, 
                                                                        det_img, 
                                                                        box_xywh, 
                                                                        camK,
                                                                        self._topk, 
                                                                        self.all_train_args[pred_clas], 
                                                                        upright=self._upright)

            R_est = Rs_est.squeeze()
            t_est = ts_est.squeeze()

            # if 'depth_img' in self.query_process_requirements():
            #     print 'depth im shape:', depth_img.shape
            #     print 'color im shape:', color_img.shape
            #     assert H == depth_img.shape[0]
            #     depth_crop = depth_img
            #     depth_crop = self.extract_square_patch(depth_img, 
            #                                         box_xywh,
            #                                         self.pad_factors[pred_clas],
            #                                         resize=self.patch_sizes[pred_clas], 
            #                                         interpolation=cv2.INTER_NEAREST) * 1000.
            #     R_est_auto = R_est.copy()
            #     t_est_auto = t_est.copy()

            #     R_est, t_est = self.icp_handle.icp_refinement(depth_crop, R_est, t_est, camK, (W,H), pred_clas=pred_clas, depth_only=True)
            #     _, ts_est, _ = self.all_codebooks[pred_clas].auto_pose6d(self.sess, 
            #                                                                 det_img, 
            #                                                                 box_xywh, 
            #                                                                 camK,
            #                                                                 self._topk, 
            #                                                                 self.all_train_args[pred_clas], 
            #                                                                 upright=self._upright,
            #                                                                 depth_pred=t_est[2])
            #     t_est = ts_est.squeeze()
            #     R_est, _ = self.icp_handle.icp_refinement(depth_crop, R_est, ts_est.squeeze(), camK, (W,H), pred_clas=pred_clas, no_depth=True)

            #     if self.vis:
            #         bgr, depth = self.icp_handle.syn_renderer.render_trafo(camK, R_est, t_est, (W,H), pred_clas=pred_clas)
            #         bgr_auto, depth_auto = self.icp_handle.syn_renderer.render_trafo(camK, R_est_auto, t_est_auto, (W,H), pred_clas=pred_clas)
            #         g_y = np.zeros_like(bgr)
            #         g_y[:,:,1]= bgr[:,:,1]
            #         g_y = g_y/255.   
            #         r_y = np.zeros_like(bgr_auto)
            #         r_y[:,:,0]= bgr_auto[:,:,0]
            #         r_y = r_y/255.   
            #         img_show[depth > 0] = g_y[depth > 0]*2./3. + img_show[depth > 0]*1./3.
            #         img_show[depth_auto > 0] = r_y[depth_auto > 0]*2./3. + img_show[depth_auto > 0]*1./3.

            #         depth_img_show[depth > 0] = g_y[depth > 0]*2./3. + depth_img_show[depth > 0]*1./3.
            #         depth_img_show[depth_auto > 0] = r_y[depth_auto > 0]*2./3. + depth_img_show[depth_auto > 0]*1./3.
            #         cv2.imshow('pose est',img_show)
            #         cv2.imshow('pose est depth',depth_img_show)  
                    
           
            H_est[:3,:3] = R_est
            if mm:
                H_est[:3,3] = t_est
            else:
                H_est[:3,3] = t_est / 1000.
            # print 'translation from camera: ',  H_est[:3,3]

            if self._camPose:
                H_est = np.dot(camPose, H_est)           

            top1_pose = PoseEstimate(name=pred_clas,trafo=H_est)
            all_pose_estimates.append(top1_pose)


        return all_pose_estimates

