import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import time
import argparse
import configparser 

from auto_pose.ae import factory,utils



parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("-f", "--file_str", required=True, help='folder or filename to image(s)')
# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

file_str = arguments.file_str
if os.path.isdir(file_str):
    files = sorted(glob.glob(os.path.join(str(file_str),'*.png'))+glob.glob(os.path.join(str(file_str),'*.jpg')))
else:
    files = [file_str]

workspace_path = os.environ.get('AE_WORKSPACE_PATH')

if workspace_path == None:
    print 'Please define a workspace path:\n'
    print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
    exit(-1)

log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = utils.get_checkpoint_dir(log_dir)
   
train_cfg_file_path = utils.get_train_config_exp_file_path(log_dir, experiment_name)
train_args = configparser.ConfigParser()
train_args.read(train_cfg_file_path)  

codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)


import yaml
# yaml_files = sorted(glob.glob('/net/rmc-lx0318/home_local/public/dataset/201810_bosch/data_recorded/bboxes_manually/scene1/*.yaml'))
yaml_files = sorted(glob.glob('/volume/USERSTORE/proj_bosch_pose-estimation/bosch_zivid_data/RGB_only_for_detection/*.yaml'))
bb_dicts = {}
for y in yaml_files:
    with open(y) as file:
        bb_dicts[os.path.basename(y).split('yaml')[0]] = yaml.load(file)
print bb_dicts
# [[2730.3754266211604,0.0,960.0],[0.0,2730.3754266211604,600.0],[0.0,0.0,1.0]]
K_test = np.array([[2730.3754266211604,0.0,800.0],[0.0,2730.3754266211604,600.0],[0.0,0.0,1.0]])

with tf.Session() as sess:

    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)

    for file in files:
        orig_im = cv2.imread(file)
        
        if bb_dicts.has_key(os.path.basename(file).split('png')[0]):
            bb_dict = bb_dicts[os.path.basename(file).split('png')[0]]

            for bb in bb_dict['labels']:
                if bb['class'] == '4':
                    print orig_im.shape
                    H,W = orig_im.shape[:2]
                    x = int(W * bb['bbox']['minx']) -160
                    y = int(H * bb['bbox']['miny'])
                    w = int(W * bb['bbox']['maxx'] - 160 - x )
                    h = int(H * bb['bbox']['maxy'] - y)
                    pixel_bb = np.array([x,y,w,h])
                    cropped_im = orig_im[:,160:orig_im.shape[1]-160,:]
                    H,W =  cropped_im.shape[:2]
                    img_crop = dataset.extract_square_patch(cropped_im,pixel_bb,train_args.getfloat('Dataset','PAD_FACTOR'),interpolation=cv2.INTER_LINEAR)

                    R,t,_ = codebook.auto_pose6d(sess, img_crop, pixel_bb,K_test,1,train_args)
                    print R,t

                    R = R.squeeze()
                    t = t.squeeze()

                    rendered_pose_est,_ = dataset.renderer.render(0, W, H, K_test, R, t, 10, 10000) 

                    g_y = np.zeros_like(rendered_pose_est)
                    g_y[:,:,1]= rendered_pose_est[:,:,1]
                    cropped_im[rendered_pose_est > 0] = g_y[rendered_pose_est > 0]*2./3. + cropped_im[rendered_pose_est > 0]*1./3.
                    cv2.rectangle(cropped_im, (x,y),(x+w,y+h), (255,0,0), 3)
                
                    cv2.imshow('img crop', cv2.resize(img_crop,(512,512)))
                    cv2.imshow('orig img', cropped_im)
                    # cv2.imshow('pred_pose', cv2.resize(pred_view,(512,512)))

                    # cv2.imshow('orig_R', rendered_pose_est2)
                    # cv2.imshow('corrected_R', rendered_pose_est)
                    # key = cv2.waitKey(0)
                    # if key == ord('a'):
                    # 	t[0]-=10
                    # if key == ord('d'):
                    # 	t[0]+=10
                    # if key == ord('s'):
                    # 	t[1]-=10
                    # if key == ord('w'):
                    # 	t[1]+=10
                    	
                    
                    if cv2.waitKey(0) == ord('k'):
                    	cv2.imwrite('/home_local/sund_ma/autoencoder_ws/bosch/scene1_pressure_pump_pose_ests_aae/img_crop_%s.jpg' % os.path.basename(file).split('png')[0],cv2.resize(img_crop,(512,512)))
                    	cv2.imwrite('/home_local/sund_ma/autoencoder_ws/bosch/scene1_pressure_pump_pose_ests_aae/pred_pose_%s.jpg' % os.path.basename(file).split('png')[0],orig_im)
                    	cv2.imwrite('/home_local/sund_ma/autoencoder_ws/bosch/scene1_pressure_pump_pose_ests_aae/pred_rot%s.jpg' % os.path.basename(file).split('png')[0],cv2.resize(pred_view,(512,512)))


