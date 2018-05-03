import cv2
import tensorflow as tf
import numpy as np
import glob
import imageio
import os
import ConfigParser

from ae import factory, utils
from eval import eval_utils

import argparse
import rmcssd.bin.detector as detector
from sixd_toolkit.pysixd import inout


parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("ssd_name")
parser.add_argument("-f", "--folder_str", required=True)
parser.add_argument("-eval", action='store_true',default=False)
# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''



folder_str = arguments.folder_str
ssd_name = arguments.ssd_name
ssd = detector.Detector(os.path.join('/home_local/sund_ma/ssd_ws/checkpoints', ssd_name))

start_var_list =set([var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])


codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)

all_var_list = set([var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
ae_var_list = all_var_list.symmetric_difference(start_var_list)
saver = tf.train.Saver(ae_var_list)

workspace_path = os.environ.get('AE_WORKSPACE_PATH')

if workspace_path == None:
    print 'Please define a workspace path:\n'
    print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
    exit(-1)

train_cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
train_args = ConfigParser.ConfigParser()
train_args.read(train_cfg_file_path)  
  
log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = utils.get_checkpoint_dir(log_dir)
factory.restore_checkpoint(ssd.isess, saver, ckpt_dir)

K_test = np.array([5.08912781e+002,0.,2.60586884e+002,0.,5.08912781e+002,2.52437561e+002,0.,0.,1.]).reshape(3,3)

result_dict = {}

files = glob.glob(os.path.join(str(folder_str),'**/image.png'))
for file in files:
    

    img = cv2.imread(file)
    H, W = img.shape[:2]
    img_show = img.copy()

    rclasses, rscores, rbboxes = ssd.process(img,select_threshold=0.5)

    ssd_boxes = [ (int(rbboxes[i][0]*H), int(rbboxes[i][1]*W), int(rbboxes[i][2]*H), int(rbboxes[i][3]*W)) for i in xrange(len(rbboxes)) if rclasses[i] == 1 ]
    ssd_imgs = np.empty((len(rbboxes),) + dataset.shape)

    vis_img = 0.3 * np.ones((np.max([len(rbboxes),3])*dataset.shape[0],2*dataset.shape[1],dataset.shape[2]))
    #print vis_img.shape

    for j,ssd_box in enumerate(ssd_boxes):
        ymin, xmin, ymax, xmax = ssd_box

        ssd_img = img[ymin:ymax,xmin:xmax]
        h, w = ssd_img.shape[:2]
        size = int(np.maximum(h, w) * 1.25)
        cx = xmin + (xmax - xmin)/2
        cy = ymin + (ymax - ymin)/2

        left = np.maximum(cx-size/2, 0)
        top = np.maximum(cy-size/2, 0)

        ssd_img = img[top:cy+size/2,left:cx+size/2]
        ssd_img = cv2.resize(ssd_img, dataset.shape[:2])
        if dataset.shape[2]  == 1:
            ssd_img = cv2.cvtColor(ssd_img,cv2.COLOR_BGR2GRAY)[:,:,None]
        ssd_img = ssd_img / 255.
        ssd_imgs[j,:,:,:] = ssd_img

    if len(rbboxes) > 0:
        cv2.imshow('ae_input',ssd_imgs[0])
        Rs = []
        ts = []

        for j,ssd_img in enumerate(ssd_imgs):
            predicted_bb = [ssd_boxes[j][1],ssd_boxes[j][0],ssd_boxes[j][3]-ssd_boxes[j][1],ssd_boxes[j][2]-ssd_boxes[j][0]]
            R, t,_,_= codebook.nearest_rotation_with_bb_depth(ssd.isess, ssd_img, predicted_bb, K_test, 1, train_args, upright=False)
            Rs.append(R.squeeze())
            ts.append(t.squeeze())
        # Rs = codebook.nearest_rotation(ssd.isess, ssd_imgs)
        ssd_rot_imgs = 0.3*np.ones_like(ssd_imgs)
        print ts

        # idcs = np.argsort(rscores)

        for j,R in enumerate(Rs):
            rendered_R_est = dataset.render_rot( R ,downSample = 1)
            if dataset.shape[2]  == 1:
                rendered_R_est = cv2.cvtColor(rendered_R_est,cv2.COLOR_BGR2GRAY)[:,:,None]
            ssd_rot_imgs[j,:,:,:] = rendered_R_est/255.



        ssd_imgs = ssd_imgs.reshape(-1,*ssd_imgs.shape[2:])
        ssd_rot_imgs = ssd_rot_imgs.reshape(-1,*ssd_rot_imgs.shape[2:])
        vis_img[:ssd_imgs.shape[0],:dataset.shape[1],:] = ssd_imgs
        vis_img[:ssd_rot_imgs.shape[0],dataset.shape[1]:,:] = ssd_rot_imgs


    if arguments.eval:
        dict_key = ''.join(os.path.basename(os.path.dirname(file)).split('_H')[0])
        result_dict[dict_key] = {}
        Rs_flat = np.zeros((len(Rs),9))
        ts_flat = np.zeros((len(Rs),3))
        for i in xrange(len(Rs)):
            Rs_flat[i]=Rs[i].flatten()
            ts_flat[i]=ts[i].flatten()

        result_dict[dict_key]['cam_R_m2c'] = Rs_flat.tolist()
        result_dict[dict_key]['cam_t_m2c'] = ts_flat.tolist()
        print result_dict[dict_key]['cam_R_m2c']
        print ssd_boxes
        result_dict[dict_key]['ssd_bboxes'] = np.array(ssd_boxes).tolist()
        result_dict[dict_key]['ssd_scores'] = rscores.tolist()



    z_sort = np.argsort(ts_flat[:,2])
    print z_sort
    for t,R in zip(ts_flat[z_sort[::-1]],Rs_flat[z_sort[::-1]]):
        bgr_y, depth_y  = dataset.renderer.render( 
            obj_id=0,
            W=640, 
            H=512,
            K=K_test, 
            R=np.array(R).reshape(3,3), 
            t=t,
            near=10,
            far=10000,
            random_light=False
        )

        img_show[bgr_y > 0] = bgr_y[bgr_y > 0]
        # cv2.imshow('render6D',img_show)
    for i in xrange(len(rscores)):
        score = rscores[i]
        ymin = int(rbboxes[i, 0] * H)
        xmin = int(rbboxes[i, 1] * W)
        ymax = int(rbboxes[i, 2] * H)
        xmax = int(rbboxes[i, 3] * W)
        cv2.putText(img_show, '%1.3f' % score, (xmin, ymax+10), cv2.FONT_ITALIC, .5, (0,255,0), 1)
        cv2.rectangle(img_show, (xmin,ymin),(xmax,ymax), (0,255,0), 1)

    cv2.imshow('preds', vis_img)
    cv2.imshow('img', img_show)
    cv2.waitKey(0)
        



    # print rclasses, rscores, rbboxes


    # im = cv2.resize(im,(128,128))

    # R = codebook.nearest_rotation(ssd.isess, im/255.)

    # pred_view = dataset.render_rot( R ,downSample = 1)
    
    # cv2.imshow('resized img', cv2.resize(im/255.,(256,256)))
    # cv2.imshow('pred_view', cv2.resize(pred_view,(256,256)))
    # print R
    # cv2.waitKey(0)
if arguments.eval:
    # print result_dict
    inout.save_yaml('/home_local2/sund_ma/data/kuka_results/kuka_results.yaml',result_dict)


