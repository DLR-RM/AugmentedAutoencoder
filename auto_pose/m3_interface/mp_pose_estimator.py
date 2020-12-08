import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import configparser

from auto_pose.ae import ae_factory, utils

from auto_pose.m3_interface.m3_interfaces import PoseEstInterface, PoseEstimate, Roi3D

class MPPoseEstimator(PoseEstInterface):
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
        self._full_model_name = test_args.get('mp_encoder', 'full_model_name')
        if test_args.getboolean('mp_encoder','camPose'):
            self._process_requirements.append('camPose')
        self._camPose = test_args.getboolean('mp_encoder','camPose')
        self._upright = test_args.getboolean('mp_encoder','upright')
        self._topk = test_args.getint('mp_encoder','topk')
        if self._topk > 1:
            print('ERROR: topk > 1 not implemented yet')
            exit()

        self._image_format = {'color_format':test_args.get('mp_encoder','color_format'), 
                              'color_data_type': eval(test_args.get('mp_encoder','color_data_type')),
                              'depth_data_type': eval(test_args.get('mp_encoder','depth_data_type')) }

        # self.vis = test_args.getboolean('mp_encoder','pose_visualization')

        # self.all_experiments = eval(test_args.get('mp_encoder','experiments'))
        # self.class_2_codebook = eval(test_args.get('mp_encoder','class_2_codebook'))

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        config.gpu_options.per_process_gpu_memory_fraction = test_args.getfloat('mp_encoder','gpu_memory_fraction')

        full_name = self._full_model_name.split('/')
        experiment_name = full_name.pop()
        experiment_group = full_name.pop() if len(full_name) > 0 else ''
        log_dir = utils.get_log_dir(workspace_path, experiment_name, experiment_group)

        train_cfg_file_path = utils.get_train_config_exp_file_path(log_dir, experiment_name)
        print(train_cfg_file_path)
        # train_cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
        self.train_args = configparser.ConfigParser(inline_comment_prefixes="#")
        self.train_args.read(train_cfg_file_path)

        checkpoint_file = utils.get_checkpoint_basefilename(log_dir, False, latest=self.train_args.getint('Training', 'NUM_ITER'), joint=True)

        self.codebook_multi, self.dataset = ae_factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True, joint=True)
        encoder = self.codebook_multi._encoder
        
        try:
            base_path = test_args.get('mp_encoder','base_path')
            class_2_objs = eval(test_args.get('mp_encoder','class_2_objs'))
            self.class_2_objpath, self.class_2_codebook = {}, {}
            for class_name, obj_path in class_2_objs.items():
                self.class_2_objpath[class_name] = os.path.join(base_path, obj_path)
                self.class_2_codebook[class_name] = self.codebook_multi._get_codebook_name(self.class_2_objpath[class_name])
        except:
            self.class_2_codebook = eval(test_args.get('mp_encoder','class_2_codebook'))
        self.sess = tf.Session(config=config)
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_file)

        self.pad_factor = self.train_args.getfloat('Dataset','PAD_FACTOR')
        self.patch_size = (self.train_args.getint('Dataset','W'), self.train_args.getint('Dataset','H'))

    def set_parameter(self, string_name, string_val):
        pass

    # ABS
    def query_process_requirements(self):
        return self._process_requirements

    def query_image_format(self):
        return self._image_format


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

        for j,box in enumerate(bboxes):
            H_est = np.eye(4)
            pred_clas = max(box.classes)

            if not pred_clas in self.class_2_codebook:
                print(('%s not contained in config class_names %s', (pred_clas, self.class_2_codebook)))
                continue

            box_xywh = [box.xmin*W, box.ymin*H, (box.xmax-box.xmin)*W, (box.ymax-box.ymin)*H]

            det_img = self.extract_square_patch(color_img, 
                                                box_xywh, 
                                                self.pad_factor,
                                                resize=self.patch_size, 
                                                black_borders=True,
                                                interpolation=cv2.INTER_NEAREST)

            Rs_est, ts_est, _ = self.codebook_multi.auto_pose6d(self.sess,
                                                        det_img, 
                                                        box_xywh, 
                                                        camK,
                                                        self._topk, 
                                                        self.train_args, 
                                                        self.class_2_codebook[pred_clas],
                                                        upright=self._upright)

            R_est = Rs_est.squeeze()
            t_est = ts_est.squeeze()
           
            H_est[:3,:3] = R_est
            H_est[:3,3] = t_est
            print('translation from camera: ',  H_est[:3,3])

            if self._camPose:
                H_est = np.dot(camPose, H_est)           

            top1_pose = PoseEstimate(name=pred_clas,trafo=H_est)
            all_pose_estimates.append(top1_pose)


        return all_pose_estimates

