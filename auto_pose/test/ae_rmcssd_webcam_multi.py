import cv2
import tensorflow as tf
import numpy as np
import glob
import imageio
import os
import configparser
import re
import time

from meshrenderer import meshrenderer_phong
from sixd_toolkit.pysixd import misc

from auto_pose.ae import factory, utils
from auto_pose.eval import eval_utils

import argparse
import rmcssd.bin.detector as detector
from webcam_video_stream import WebcamVideoStream


parser = argparse.ArgumentParser()
parser.add_argument("experiment_names", nargs='+',type=str)
parser.add_argument("ssd_name")
parser.add_argument('-down', default=1, type=int)
parser.add_argument("-s", action='store_true', default=False)

# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()


width = 960
height = 720

# width = 1920
# height = 1080

# def initializeWebcam(width, height):
#     #initialise pygame   
#     pygame.init()
#     pygame.camera.init()
#     cam = pygame.camera.Camera("/dev/video0",(width,height))
#     cam.start()

#     #setup window
#     windowSurfaceObj = pygame.display.set_mode((width,height),1,16)
#     pygame.display.set_caption('Camera')

#     return cam

# cam = initializeWebcam(width, height)


videoStream = WebcamVideoStream(0,width,height).start()


if arguments.s:
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))

ssd_name = arguments.ssd_name

ssd = detector.Detector(os.path.join('/home_local/sund_ma/ssd_ws/checkpoints', ssd_name))


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
        exit()

    train_cfg_file_path = u.get_train_config_exp_file_path(log_dir, experiment_name)
    train_args = configparser.ConfigParser()
    train_args.read(train_cfg_file_path)  
    h_train, w_train, c = train_args.getint('Dataset','H'),train_args.getint('Dataset','W'), train_args.getint('Dataset','C')
    model_paths.append(train_args.get('Paths','MODEL_PATH'))
    all_train_args.append(train_args)
      

    # codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)
    # with tf.variable_scope('model1'):
    # with tf.Graph().as_default():

    all_codebooks.append(factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=False))
    # all_sessions.append(sess)
    factory.restore_checkpoint(sess, tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='obj%s' % obj_id)), ckpt_dir)

        # factory.restore_checkpoint(ssd.isess, saver, ckpt_dir)

# K_test = np.array([(width+height)/2.,0.,width/2.,0.,(width+height)/2.,height/2.,0.,0.,1.]).reshape(3,3)
# K_test =  np.array([[797.81194184 ,  0.      ,   496.49721123],[  0.    ,     799.72051992 ,350.63848832],[  0.     ,      0.         ,  1.        ]])
K_test =  np.array([[810.4968405 ,   0.         ,487.55096072],
 [  0.        , 810.61326022 ,354.6674888 ],
 [  0.        ,   0.         ,  1.        ]])

K_down = K_test.copy()

if arguments.down > 1:
    K_down[:2,:] = K_down[:2,:] / arguments.down

result_dict = {}


renderer = meshrenderer_phong.Renderer(
    model_paths, 
    1
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
    # img_show = cv2.resize(img_show, (width/arguments.down,height/arguments.down))

    rclasses, rscores, rbboxes = ssd.process(img,select_threshold=0.4)
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
        print ssd_img.shape
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

            R, t, _ = all_codebooks[k].nearest_rotation_with_bb_depth(sess, ssd_img, predicted_bb, K_test, 1, all_train_args[k], upright=False)
            Rs.append(R.squeeze())
            ts.append(t.squeeze())
            det_objects_k.append(k)
        # Rs = codebook.nearest_rotation(ssd.isess, ssd_imgs)
        if len(det_objects_k) == 0:
            continue


        # for j,R in enumerate(Rs):
        #     rendered_R_est = dataset.render_rot( R ,downSample = 1)
        #     if dataset.shape[2]  == 1:
        #         rendered_R_est = cv2.cvtColor(rendered_R_est,cv2.COLOR_BGR2GRAY)[:,:,None]
        #     ssd_rot_imgs[j,:,:,:] = rendered_R_est/255.


        # ssd_imgs = ssd_imgs.reshape(-1,*ssd_imgs.shape[2:])
        # ssd_rot_imgs = ssd_rot_imgs.reshape(-1,*ssd_rot_imgs.shape[2:])
        # vis_img[:ssd_imgs.shape[0],:dataset.shape[1],:] = ssd_imgs
        # vis_img[:ssd_rot_imgs.shape[0],dataset.shape[1]:,:] = ssd_rot_imgs
        det_objects_k = np.array(det_objects_k)
        Rs_flat = np.zeros((len(Rs),9))
        ts_flat = np.zeros((len(Rs),3))
        for i in xrange(len(Rs)):
            Rs_flat[i]=Rs[i].flatten()
            ts_flat[i]=ts[i].flatten()

        z_sort = np.argsort(ts_flat[:,2])
        print z_sort
        for t,R,k in zip(ts_flat[z_sort[::-1]],Rs_flat[z_sort[::-1]],det_objects_k[z_sort[::-1]]):
            st = time.time()
            #use render_many
            bgr_y, depth_y  = renderer.render( 
                obj_id=k,
                W=width/arguments.down, 
                H=height/arguments.down,
                K=K_down, 
                R=np.array(R).reshape(3,3), 
                t=t,
                near=10,
                far=10000,
                random_light=False
            )
            # print 'rendering time', time.time() - st
            g_y = np.zeros_like(bgr_y)
            g_y[:,:,1]= bgr_y[:,:,1]
            st = time.time()
            g_y = cv2.resize(g_y, (width,height))
            depth_y = cv2.resize(depth_y, (width,height))
            # print 'resize time', time.time() - st
            img_show[depth_y > 0] = g_y[depth_y > 0]*2./3. + img_show[depth_y > 0]*1./3.

            # cv2.imshow('render6D',img_show)


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




