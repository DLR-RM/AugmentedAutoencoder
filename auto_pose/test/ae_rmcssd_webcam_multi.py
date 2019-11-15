import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import configparser
import re
import time

from auto_pose.meshrenderer import meshrenderer_phong
from sixd_toolkit.pysixd import misc

from auto_pose.ae import factory, utils
from auto_pose.eval import eval_utils

import argparse
import rmcssd.bin.detector as detector
from webcam_video_stream import WebcamVideoStream


parser = argparse.ArgumentParser()
parser.add_argument("experiment_names", nargs='+',type=str)
parser.add_argument("--ssd_frozen_ckpt_path", type=str)
parser.add_argument("--width",type=int,default=960)
parser.add_argument("--height",type=int,default=720)
parser.add_argument("--K_test",type=str,default='[810.4968405 , 0. ,487.55096072,  0., 810.61326022 ,354.6674888,  0. ,   0.,  1.]')
parser.add_argument('--down', default=1, type=int)
parser.add_argument("--s", action='store_true', default=False)

# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()

width = arguments.width
height = arguments.height
K_test =  np.array(eval(arguments.K_test)).reshape(3,3)

K_down = K_test.copy()
K_down[:2,:] = K_down[:2,:] / arguments.down



videoStream = WebcamVideoStream(0,width,height).start()

if arguments.s:
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))


ssd = detector.Detector(arguments.ssd_frozen_ckpt_path)

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print 'Please define a workspace path:\n'
    print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
    exit(-1)


all_codebooks = []
all_train_args = []
class_i_mapping = {}

model_paths = []
sess = tf.Session()
for i,experiment_name in enumerate(arguments.experiment_names):

    full_name = experiment_name.split('/')
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''

    log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
    ckpt_dir = utils.get_checkpoint_dir(log_dir)
    
    try:
        obj_id = int(re.findall(r"(\d+)", experiment_name)[-1])
        class_i_mapping[obj_id] = i
    except:
        print 'no obj_id in name, needed to get the mapping for the detector'
        class_i_mapping[i] = i

    train_cfg_file_path = utils.get_train_config_exp_file_path(log_dir, experiment_name)
    train_args = configparser.ConfigParser()
    train_args.read(train_cfg_file_path)  
    h_train, w_train, c = train_args.getint('Dataset','H'),train_args.getint('Dataset','W'), train_args.getint('Dataset','C')
    model_paths.append(train_args.get('Paths','MODEL_PATH'))

    all_train_args.append(train_args)
    all_codebooks.append(factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=False))

    factory.restore_checkpoint(sess, tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=experiment_name)), ckpt_dir)


renderer = meshrenderer_phong.Renderer(
    model_paths, 
    1,
    vertex_tmp_store_folder=utils.get_dataset_path(workspace_path)
)

# sequential vs. concurrent executeion
# 8 objects: sequential:  0.0308971405029 concurrent:  0.0260059833527
# test = np.random.rand(1,128,128,3)
# for j in range(100):
#     st=time.time()
#     for i,op in enumerate(all_codebooks):
#         cosine_similarity = sess.run(op.cos_similarity, {all_codebooks[i]._encoder.x: test})
#     print 'sequential: ', time.time()-st

#     st=time.time()
#     cosine_similarity = sess.run([op.cos_similarity for op in all_codebooks], feed_dict={all_codebooks[0]._encoder.x:test,all_codebooks[1]._encoder.x:test,all_codebooks[2]._encoder.x:test,all_codebooks[3]._encoder.x:test,all_codebooks[4]._encoder.x:test,all_codebooks[5]._encoder.x:test,all_codebooks[6]._encoder.x:test,all_codebooks[7]._encoder.x:test})
#     print 'concurrent: ', time.time()-st
# exit()

print class_i_mapping
while videoStream.isActive():

    img = videoStream.read()

    H, W = img.shape[:2]
    img_show = img.copy()
    # img_show = cv2.resize(img_show, (width/arguments.down,height/arguments.down))

    rclasses, rscores, rbboxes = ssd.process(img,select_threshold=0.4,nms_threshold=.5)
    print rclasses

    ssd_boxes = [ (int(rbboxes[i][0]*H), int(rbboxes[i][1]*W), int(rbboxes[i][2]*H), int(rbboxes[i][3]*W)) for i in xrange(len(rbboxes))]
    ssd_imgs = np.empty((len(rbboxes),) + (h_train,w_train,c))

    #print vis_img.shape

    for j,ssd_box in enumerate(ssd_boxes):
        ymin, xmin, ymax, xmax = ssd_box

        h, w = (ymax-ymin,xmax-xmin)
        size = int(np.maximum(h, w) * train_args.getfloat('Dataset','PAD_FACTOR'))
        cx = xmin + (xmax - xmin)/2
        cy = ymin + (ymax - ymin)/2

        left = np.maximum(cx-size/2, 0)
        top = np.maximum(cy-size/2, 0)

        ssd_img = img[top:cy+size/2,left:cx+size/2]
        ssd_img = cv2.resize(ssd_img, (h_train,w_train))

        if c == 1:
            ssd_img = cv2.cvtColor(ssd_img,cv2.COLOR_BGR2GRAY)[:,:,None]

        ssd_img = ssd_img / 255.
        ssd_imgs[j,:,:,:] = ssd_img


    if len(rbboxes) > 0:
        # cv2.imshow('ae_input',ssd_imgs[0])
        Rs = []
        ts = []
        det_objects_k = []

        for j,ssd_img in enumerate(ssd_imgs):
            predicted_bb = [ssd_boxes[j][1],ssd_boxes[j][0],ssd_boxes[j][3]-ssd_boxes[j][1],ssd_boxes[j][2]-ssd_boxes[j][0]]
            obj = rclasses[j]
            if not obj in class_i_mapping:
                continue
            k = class_i_mapping[obj]

            R, t, _ = all_codebooks[k].auto_pose6d(sess, ssd_img, predicted_bb, K_test, 1, all_train_args[k], upright=False)
            Rs.append(R.squeeze())
            ts.append(t.squeeze())
            det_objects_k.append(k)

        if len(det_objects_k) == 0:
            continue

        bgr_y,_,_ = renderer.render_many( 
            obj_ids=np.array(det_objects_k).astype(np.int32),
            W=width/arguments.down,
            H=height/arguments.down,
            K=K_down, 
            Rs=np.array(Rs), 
            ts=np.array(ts),
            near=1.,
            far=10000.,
            random_light=False
        )
        
    
        bgr_y = cv2.resize(bgr_y,(width,height))
        
        g_y = np.zeros_like(bgr_y)
        g_y[:,:,1]= bgr_y[:,:,1]    
        im_bg = cv2.bitwise_and(img_show,img_show,mask=(g_y[:,:,1]==0).astype(np.uint8))                 
        img_show = cv2.addWeighted(im_bg,1,g_y,1,0)




        for i in xrange(len(rscores)):
            if not rclasses[i] in class_i_mapping:
                continue
            score = rscores[i]
            ymin = int(rbboxes[i, 0] * H)
            xmin = int(rbboxes[i, 1] * W)
            ymax = int(rbboxes[i, 2] * H)
            xmax = int(rbboxes[i, 3] * W)
            cv2.putText(img_show, '%1.3f' % score, (xmin, ymax+10), cv2.FONT_ITALIC, .5, (0,255,0), 1)
            cv2.rectangle(img_show, (xmin,ymin),(xmax,ymax), (0,255,0), 1)
        #cv2.putText(img_show, '1x', (0, 0), cv2.FONT_ITALIC, 2., (0,255,0), 1)
    # cv2.imshow('preds', vis_img)
    try:
        # img_show = cv2.resize(img_show, (width,height))
        if arguments.s:
            out.write(img_show)

        cv2.imshow('img', img_show)
        cv2.waitKey(10)
    except:
        print 'no frame'
if arguments.s:
    out.release()




