import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import configparser

from auto_pose.ae import factory, utils
from auto_pose.eval import eval_utils

import argparse
import rmcssd.bin.detector as detector
from webcam_video_stream import WebcamVideoStream


parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("ssd_name")
parser.add_argument("-s", action='store_true', default=False)

# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

width = 960
height = 720


videoStream = WebcamVideoStream(0,width,height).start()

print 'here'

if arguments.s:
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))

print 'here'
ssd_name = arguments.ssd_name
ssd = detector.Detector(os.path.join('/home_local/sund_ma/ssd_ws/checkpoints', ssd_name))
print 'here'

start_var_list =set([var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
print 'here'

codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)
print 'here'

all_var_list = set([var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
ae_var_list = all_var_list.symmetric_difference(start_var_list)
saver = tf.train.Saver(ae_var_list)
print 'here'

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
print 'here'


if workspace_path == None:
    print 'Please define a workspace path:\n'
    print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
    exit(-1)
log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = utils.get_checkpoint_dir(log_dir)

train_cfg_file_path = utils.get_train_config_exp_file_path(log_dir, experiment_name)
train_args = configparser.ConfigParser()
train_args.read(train_cfg_file_path)  
  
factory.restore_checkpoint(ssd.isess, saver, ckpt_dir)

# K_test = np.array([(width+height)/2.,0.,width/2.,0.,(width+height)/2.,height/2.,0.,0.,1.]).reshape(3,3)
# K_test =  np.array([[797.81194184 ,  0.      ,   496.49721123],[  0.    ,     799.72051992 ,350.63848832],[  0.     ,      0.         ,  1.        ]])
K_test =  np.array([[810.4968405 ,   0.         ,487.55096072],
 [  0.        , 810.61326022 ,354.6674888 ],
 [  0.        ,   0.         ,  1.        ]])

result_dict = {}


while videoStream.isActive():
    
    # if cam.query_image():
    #     image = cam.get_image()
    #     arr = pygame.surfarray.array3d(image)
    #     img = np.swapaxes(arr,0,1)        
    # else:
    #     continue

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = videoStream.read()

    H, W = img.shape[:2]
    img_show = img.copy()

    rclasses, rscores, rbboxes = ssd.process(img,select_threshold=0.5)

    ssd_boxes = [ (int(rbboxes[i][0]*H), int(rbboxes[i][1]*W), int(rbboxes[i][2]*H), int(rbboxes[i][3]*W)) for i in xrange(len(rbboxes)) if rclasses[i] == 1 ]
    ssd_imgs = np.empty((len(rbboxes),) + dataset.shape)

    vis_img = 0.3 * np.ones((np.max([len(rbboxes),3])*dataset.shape[0],2*dataset.shape[1],dataset.shape[2]))
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
        ssd_img = cv2.resize(ssd_img, dataset.shape[:2])
        if dataset.shape[2]  == 1:
            ssd_img = cv2.cvtColor(ssd_img,cv2.COLOR_BGR2GRAY)[:,:,None]
        ssd_img = ssd_img / 255.
        ssd_imgs[j,:,:,:] = ssd_img




    if len(rbboxes) > 0:
        # cv2.imshow('ae_input',ssd_imgs[0])
        Rs = []
        ts = []

        for j,ssd_img in enumerate(ssd_imgs):
            predicted_bb = [ssd_boxes[j][1],ssd_boxes[j][0],ssd_boxes[j][3]-ssd_boxes[j][1],ssd_boxes[j][2]-ssd_boxes[j][0]]
            R, t, _ = codebook.auto_pose6d(ssd.isess, ssd_img, predicted_bb, K_test, 1, train_args, upright=False)
            Rs.append(R.squeeze())
            ts.append(t.squeeze())
        # Rs = codebook.nearest_rotation(ssd.isess, ssd_imgs)
        ssd_rot_imgs = 0.3*np.ones_like(ssd_imgs)
        print ts[0][2]


        # for j,R in enumerate(Rs):
        #     rendered_R_est = dataset.render_rot( R ,downSample = 1)
        #     if dataset.shape[2]  == 1:
        #         rendered_R_est = cv2.cvtColor(rendered_R_est,cv2.COLOR_BGR2GRAY)[:,:,None]
        #     ssd_rot_imgs[j,:,:,:] = rendered_R_est/255.


        # ssd_imgs = ssd_imgs.reshape(-1,*ssd_imgs.shape[2:])
        # ssd_rot_imgs = ssd_rot_imgs.reshape(-1,*ssd_rot_imgs.shape[2:])
        # vis_img[:ssd_imgs.shape[0],:dataset.shape[1],:] = ssd_imgs
        # vis_img[:ssd_rot_imgs.shape[0],dataset.shape[1]:,:] = ssd_rot_imgs

        Rs_flat = np.zeros((len(Rs),9))
        ts_flat = np.zeros((len(Rs),3))
        for i in xrange(len(Rs)):
            Rs_flat[i]=Rs[i].flatten()
            ts_flat[i]=ts[i].flatten()

        z_sort = np.argsort(ts_flat[:,2])
        print z_sort
        for t,R in zip(ts_flat[z_sort[::-1]],Rs_flat[z_sort[::-1]]):
            bgr_y, depth_y  = dataset.renderer.render( 
                obj_id=0,
                W=width, 
                H=height,
                K=K_test, 
                R=np.array(R).reshape(3,3), 
                t=t,
                near=10,
                far=10000,
                random_light=False
            )

            g_y = np.zeros_like(bgr_y)
            g_y[:,:,1]= bgr_y[:,:,1]
            img_show[bgr_y > 0] = g_y[bgr_y > 0]*2./3. + img_show[bgr_y > 0]*1./3.
            # cv2.imshow('render6D',img_show)
        for i in xrange(len(rscores)):
            score = rscores[i]
            ymin = int(rbboxes[i, 0] * H)
            xmin = int(rbboxes[i, 1] * W)
            ymax = int(rbboxes[i, 2] * H)
            xmax = int(rbboxes[i, 3] * W)
            cv2.putText(img_show, '%1.3f' % score, (xmin, ymax+10), cv2.FONT_ITALIC, .5, (0,255,0), 1)
            cv2.rectangle(img_show, (xmin,ymin),(xmax,ymax), (0,255,0), 1)
        cv2.putText(img_show, '1x', (0, 0), cv2.FONT_ITALIC, 2., (0,255,0), 1)
    # cv2.imshow('preds', vis_img)
    try:
        if arguments.s:
            out.write(img_show)
        cv2.imshow('img', img_show)
        cv2.waitKey(1)
    except:
        print 'no frame'
if arguments.s:
    out.release()




