import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import time
import argparse
import configparser 

from auto_pose.ae import factory,utils
from meshrenderer import meshrenderer


def extract_square_patch(scene_img, bb_xywh, pad_factor,resize=(128,128),interpolation=cv2.INTER_NEAREST):

    x, y, w, h = np.array(bb_xywh).astype(np.int32)
    size = int(np.maximum(h, w) * pad_factor)
    
    left = np.maximum(x+w//2-size//2, 0)
    right = x+w//2+size/2
    top = np.maximum(y+h//2-size//2, 0)
    bottom = y+h//2+size//2

    scene_crop = scene_img[top:bottom, left:right]
    scene_crop = cv2.resize(scene_crop, resize, interpolation = interpolation)
    return scene_crop

parser = argparse.ArgumentParser()
parser.add_argument("experiment_names", nargs='+', type=str)
parser.add_argument("-f", "--file_str", required=True, help='folder or filename to image(s)')
parser.add_argument("--out", "--img_out_folder", required=False, type=str, help='folder')
# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()

# full_name = arguments.experiment_name.split('/')
# experiment_name = full_name.pop()
# experiment_group = full_name.pop() if len(full_name) > 0 else ''

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


all_codebooks = []
all_train_args = []

model_paths = []
sess = tf.Session()


for i,exp_name in enumerate(arguments.experiment_names):

    full_name = exp_name.split('/')
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''

    log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
    ckpt_dir = utils.get_checkpoint_dir(log_dir)
    
    train_cfg_file_path = utils.get_train_config_exp_file_path(log_dir, experiment_name)
    train_args = configparser.ConfigParser()
    train_args.read(train_cfg_file_path)  
    h_train, w_train, c = train_args.getint('Dataset','H'),train_args.getint('Dataset','W'), train_args.getint('Dataset','C')
    model_paths.append(train_args.get('Paths','MODEL_PATH'))

    all_train_args.append(train_args)
    cb,dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)
    all_codebooks.append(cb)


    factory.restore_checkpoint(sess, tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=experiment_name)), ckpt_dir)


renderer = meshrenderer.Renderer(
    model_paths, 
    1,
    vertex_tmp_store_folder=utils.get_dataset_path(workspace_path)
)




# log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
# ckpt_dir = utils.get_checkpoint_dir(log_dir)
   
# train_cfg_file_path = utils.get_train_config_exp_file_path(log_dir, experiment_name)
# train_args = configparser.ConfigParser()
# train_args.read(train_cfg_file_path)  

# codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)


import yaml
# yaml_files = sorted(glob.glob('/net/rmc-lx0318/home_local/public/dataset/201810_bosch/data_recorded/bboxes_manually/scene1/*.yaml'))
# yaml_files = sorted(glob.glob('/volume/USERSTORE/proj_bosch_pose-estimation/bosch_zivid_data/RGB_only_for_detection/*.yaml'))
yaml_files = sorted(glob.glob('/volume/USERSTORE/proj_bosch_pose-estimation/bosch_zivid_data/scene5/*.yaml'))
bb_dicts = {}
for y in yaml_files:
    with open(y) as file:
        bb_dicts[os.path.basename(y).split('yaml')[0]] = yaml.load(file)
print bb_dicts
# [[2730.3754266211604,0.0,960.0],[0.0,2730.3754266211604,600.0],[0.0,0.0,1.0]]
K_test = np.array([[2730.3754266211604,0.0,960.0],[0.0,2730.3754266211604,600.0],[0.0,0.0,1.0]])

# with tf.Session() as sess:


for file in files:
    orig_im = cv2.imread(file)
    
    if bb_dicts.has_key(os.path.basename(file).split('png')[0]):
        bb_dict = bb_dicts[os.path.basename(file).split('png')[0]]

        Rs = []
        ts = []
        det_objects_k = []
        pixel_bbs = []

        for bb in bb_dict['labels']:
            

            k = 0 if bb['class'] == 9 else 1

            if bb['class'] == 10 or bb['class'] == 9:
                
                im_show =orig_im.copy()
                print orig_im.shape
                H,W = orig_im.shape[:2]
                x = int(W * bb['bbox']['minx'])
                y = int(H * bb['bbox']['miny'])
                w = int(W * bb['bbox']['maxx'] - x)
                h = int(H * bb['bbox']['maxy'] - y)
                pixel_bb = np.array([x,y,w,h])
                pixel_bbs.append(pixel_bb)

                # cropped_im = orig_im[:,:orig_im.shape[1],:]
                H,W =  orig_im.shape[:2]
                img_crop = extract_square_patch(orig_im,pixel_bb,train_args.getfloat('Dataset','PAD_FACTOR'),interpolation=cv2.INTER_LINEAR)

                R, t, _ = all_codebooks[k].auto_pose6d(sess, img_crop, pixel_bb, K_test, 1, all_train_args[k], upright=False)
                Rs.append(R.squeeze())
                ts.append(t.squeeze())


                det_objects_k.append(k)

                # R,t,_ = codebook.auto_pose6d(sess, img_crop, pixel_bb,K_test,1,train_args)
                # print R,t

                # R = R.squeeze()
                # t = t.squeeze()

        
        rendered_pose_ests = {}
        for i,k in enumerate(det_objects_k):
            rendered_pose_est,_ = renderer.render( 
                obj_id=np.array(det_objects_k).astype(np.int32)[i],
                W=W,
                H=H,
                K=K_test, 
                R=np.array(Rs)[i], 
                t=np.array(ts)[i],
                near=1.,
                far=10000.,
                random_light=False
            )
            rendered_pose_ests[k] = rendered_pose_est
        

        for (key,rendered_pose_est),pixel_bb in zip(rendered_pose_ests.items(),pixel_bbs):
            x,y,w,h = pixel_bb
            cv2.rectangle(im_show, (x,y),(x+w,y+h),(255,0,0), 3)
            g_y = np.zeros_like(rendered_pose_est)
            g_y[:,:,key+1]= rendered_pose_est[:,:,key+1]
            im_show[rendered_pose_est > 0] = g_y[rendered_pose_est > 0]*2./3. + orig_im[rendered_pose_est > 0]*1./3.
            # im_show[rendered_pose_est > 0] = rendered_pose_est[rendered_pose_est > 0]
            
    
        cv2.imshow('img crop', cv2.resize(img_crop,(512,512)))
        cv2.imshow('orig img', im_show)


        if arguments.out is not None:
            if not os.path.exists(arguments.out):
                os.makedirs(arguments.out)
            cv2.imwrite(os.path.join(arguments.out, os.path.basename(file)),im_show)



        cv2.waitKey(0)
        # # cv2.imshow('pred_pose', cv2.resize(pred_view,(512,512)))

        # # cv2.imshow('orig_R', rendered_pose_est2)
        # # cv2.imshow('corrected_R', rendered_pose_est)
        # key = cv2.waitKey(0)


        # if key == ord('a'):
        # 	t[0]-=10
        # if key == ord('d'):
        # 	t[0]+=10
        # if key == ord('s'):
        # 	t[1]-=10
        # if key == ord('w'):
        # 	t[1]+=10
        # if key == ord('e'):
        #     t[2]-=10
        # if key == ord('r'):
        #     t[2]+=10
        	
        
        # if cv2.waitKey(0) == ord('k'):
        # 	cv2.imwrite('/home_local/sund_ma/autoencoder_ws/bosch/scene1_pressure_pump_pose_ests_aae/img_crop_%s.jpg' % os.path.basename(file).split('png')[0],cv2.resize(img_crop,(512,512)))
        # 	cv2.imwrite('/home_local/sund_ma/autoencoder_ws/bosch/scene1_pressure_pump_pose_ests_aae/pred_pose_%s.jpg' % os.path.basename(file).split('png')[0],orig_im)
        # 	cv2.imwrite('/home_local/sund_ma/autoencoder_ws/bosch/scene1_pressure_pump_pose_ests_aae/pred_rot%s.jpg' % os.path.basename(file).split('png')[0],cv2.resize(pred_view,(512,512)))