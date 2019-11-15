 # -*- coding: utf-8 -*-
from imgaug.augmenters import *
import os
import configparser
import argparse
import numpy as np
import signal
import shutil
import cv2
import glob


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import progressbar
import tensorflow as tf

from auto_pose.ae import ae_factory as factory
from auto_pose.ae import utils as u
from auto_pose.eval import eval_plots,eval_utils,latent_utils

import matplotlib.pyplot as plt
from sixd_toolkit.pysixd import transform,pose_error,view_sampler
import time


def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print 'Please define a workspace path:\n'
        print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
        exit(-1)

    gentle_stop = np.array((1,), dtype=np.bool)
    gentle_stop[0] = False

    def on_ctrl_c(signal, frame):
        gentle_stop[0] = True

    signal.signal(signal.SIGINT, on_ctrl_c)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument('--model_path', default=None, required=True)
    parser.add_argument('--config_path', default='eval_latent_template.cfg', required=True)
    parser.add_argument("-d", action='store_true', default=False)
    parser.add_argument("-gen", action='store_true', default=False)
    parser.add_argument("-vis_emb", action='store_true', default=False)
    parser.add_argument('--at_step', default=None,  type=int, required=False)


    arguments = parser.parse_args()

    full_name = arguments.experiment_name.split('/')
    model_path = arguments.model_path
    
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    
    at_step = arguments.at_step

    cfg_file_path = u.get_config_file_path(workspace_path, experiment_name, experiment_group)
    latent_cfg_file_path = u.get_eval_config_file_path(workspace_path, arguments.config_path)
    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)

    if not os.path.exists(cfg_file_path):
        print 'Could not find config file:\n'
        print '{}\n'.format(cfg_file_path)
        exit(-1)

    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    args_latent = configparser.ConfigParser()
    args_latent.read(latent_cfg_file_path)

    if at_step is None:
        checkpoint_file = u.get_checkpoint_basefilename(log_dir, model_path, latest=args.getint('Training', 'NUM_ITER'))
    else:
        checkpoint_file = u.get_checkpoint_basefilename(log_dir, model_path, latest=at_step)

    model_type = args.get('Dataset', 'MODEL')

    codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset = True)
    encoder = codebook._encoder
    
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)

    ######
    dataset_exp = args_latent.get('Data', 'dataset')
    base_path = args_latent.get('Data', 'base_path')
    test_class = args_latent.get('Data', 'test_class')
    split = args_latent.get('Data', 'split')
    num_obj = args_latent.getint('Data', 'num_obj')
    num_views = args_latent.getint('Data', 'num_views')
    print(base_path, test_class)
    models_train = sorted(glob.glob(os.path.join(base_path, test_class, 'train', '*_normalized.off')))
    models_test = sorted(glob.glob(os.path.join(base_path, test_class, 'test', '*_normalized.off')))
    # models_test = [m for m in models_test if not '_normalized' in m]
    # models_train = [m for m in models_train if not '_normalized' in m]
    # print models_train
    # print models_test[:10]
    print(models_test)
    if split == 'train':
        dataset._kw['model_path'] = models_train[0:num_obj]
    elif split == 'test':
        dataset._kw['model_path'] = models_test[0:num_obj]
    else:
        print('split must be train or test')
        exit()

    print dataset._kw['model_path']

    if args_latent.getboolean('Experiment', 'emb_invariance'):
        latent_utils.compute_plot_emb_invariance(args_latent)
    if args_latent.getboolean('Experiment', 'refinement_pert_category_agnostic'):
        res_preds = latent_utils.relative_pose_refinement(sess, args_latent, dataset, codebook)
        np.save(os.path.join(log_dir, 'preds_%s.npy' % test_class), res_preds)
    if args_latent.getboolean('Experiment', 'compute_pose_errors'):
        res_preds = np.load(os.path.join(log_dir, 'preds_%s.npy' % test_class)).item()
        res_errors = latent_utils.compute_pose_errors(res_preds, args_latent, dataset)
        np.save(os.path.join(log_dir, 'pose_errors_%s.npy' % test_class), res_errors)
    if args_latent.getboolean('Visualization', 'pca_embedding_azelin'):
        latent_utils.plot_latent_revolutions(num_obj)


# ## rot error histogram CB with 3D model
# ############################
#     pose_errs = []
#     for i in range(1,num_o+1):
#         for j in range(3):
#             random_R = transform.random_rotation_matrix()[:3,:3]
#             # DeepIM


#             while True:
#                 rand_direction = transform.make_rand_vector(3)
#             #     rand_angle_x = np.random.normal(0,(15/180.*np.pi)**2)
#             #     rand_angle_y = np.random.normal(0,(15/180.*np.pi)**2)
#             #     rand_angle_z = np.random.normal(0,(15/180.*np.pi)**2)

#             #     R_off = transform.euler_matrix(rand_angle_x,rand_angle_y,rand_angle_z)
#             #     angle_off,_,_ = transform.rotation_from_matrix(R_off)
#             #     print angle_off*180/np.pi

#                 rand_angle = np.random.normal(0,45/180.*np.pi)
#                 R_off = transform.rotation_matrix(rand_angle,rand_direction)[:3,:3]
#                 random_R_pert = np.dot(R_off,random_R)
#                 print rand_angle
#                 if abs(rand_angle) < 45/180.*np.pi and abs(rand_angle) > 5/180.*np.pi:
#                     break

            
#             ###
#             random_t_pert = np.array([0,0,700])# + np.array([np.random.normal(0,10),np.random.normal(0,10),np.random.normal(0,50)])
#             print random_t_pert
#             # random_R = dataset.viewsphere_for_embedding[np.random.randint(0,92000)]
#             rand_test_view_crop, bb = dataset.render_rot(random_R, obj_id=i, return_bb=True)

#             # _, _, rand_test_view_whole_target = dataset.render_rot(random_R_pert, obj_id=i, t=random_t_pert, return_bb=True, return_orig=True)
#             # rand_test_view_crop = dataset.extract_square_patch(rand_test_view_whole_target, bb, float(dataset._kw['pad_factor']))
#             # rand_test_view_whole_target = rand_test_view_whole_target/255.
#             # rand_test_view_crop = rand_test_view_crop/255.
            
            
#             # K = eval(dataset._kw['k'])
#             # K = np.array(K).reshape(3,3)
#             # bgr_y,_ = dataset.renderer.render( 
#             #     obj_id=i,
#             #     W=720, 
#             #     H=540,
#             #     K=K.copy(), 
#             #     R=random_R, 
#             #     t=np.array([0.,0,650]),
#             #     near=10,
#             #     far=10000,
#             #     random_light=False
#             # )


#             # cv2.imshow('in',rand_test_view_crop)
#             # cv2.imshow('translated and rotated', rand_test_view_crop)
#             # cv2.waitKey(0)

#             # Rs_est = codebook.nearest_rotation(sess, rand_test_view, top_n=1)
#             st = time.time()
#             # session, x, top_n, budget=10, epochs=3, high=6./180*np.pi, obj_id=0, top_n_refine=1
#             R_refined,_ = codebook.refined_nearest_rotation(sess, rand_test_view_crop, 1, R_init=random_R_pert, budget=40, epochs=4, high=45./180*np.pi, obj_id=i, top_n_refine=1)
#             # R = codebook.nearest_rotation(sess, rand_test_view, 1)
#             # R = codebook.nearest_rotation(sess, rand_test_view, 1)
#             # R_refined = R_refined[np.newaxis,:]
#             print time.time() - st

#             pose_errs.append(pose_error.re(random_R,R_refined[0]))



#             # _, _, rand_test_view_whole = dataset.render_rot(R_refined[0], obj_id=i, return_bb=True,return_orig=True)
#             # z_est = eval_utils.align_images(rand_test_view_whole_target, rand_test_view_whole/255., random_t_pert[2], warp_mode = cv2.MOTION_AFFINE)
#             # print z_est



#             # pose_errs[-1] = np.minimum(pose_errs[-1],np.abs(pose_errs[-1]-180))
#             # import cv2
#             # cv2.imshow('inserted_view',rand_test_view)
#             # cv2.imshow('pert_view',rand_init_view)
#             # cv2.imshow('est_view', dataset.render_rot(R_refined[0],obj_id=i)/255.)
#             # cv2.waitKey(0)

#             if pose_errs[-1]>170:
#                 cv2.imshow('inserted_view',rand_test_view_crop)
#                 cv2.imshow('est_view', dataset.render_rot(R_refined[0],obj_id=i)/255.)
#                 cv2.waitKey(1)



if __name__ == '__main__':
    main()
    
