import cv2
import numpy as np
import os
import argparse
import configparser

from webcam_video_stream import WebcamVideoStream
from auto_pose.ae.utils import get_dataset_path
from aae_retina_pose_estimator import AePoseEstimator


parser = argparse.ArgumentParser()
parser.add_argument("-test_config", type=str, required=False, default='test_config_webcam.cfg')
parser.add_argument("-vis", action='store_true', default=False)
args = parser.parse_args()


workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print 'Please define a workspace path:\n'
    print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
    exit(-1)


test_configpath = os.path.join(workspace_path,'cfg_eval',args.test_config)
test_args = configparser.ConfigParser()
test_args.read(test_configpath)

ae_pose_est = AePoseEstimator(test_configpath)

videoStream = WebcamVideoStream(0, ae_pose_est._width, ae_pose_est._height).start()

if args.vis:
    from auto_pose.meshrenderer import meshrenderer

    ply_model_paths = [str(train_args.get('Paths','MODEL_PATH')) for train_args in ae_pose_est.all_train_args]
    cad_reconst = [str(train_args.get('Dataset','MODEL')) for train_args in ae_pose_est.all_train_args]
    
    renderer = meshrenderer.Renderer(ply_model_paths, 
                    samples=1, 
                    vertex_tmp_store_folder=get_dataset_path(workspace_path),
                    vertex_scale=float(1)) # float(1) for some models

color_dict = [(0,255,0),(0,0,255),(255,0,0),(255,255,0)] * 10

while videoStream.isActive():

    image = videoStream.read()

    boxes, scores, labels = ae_pose_est.process_detection(image)

    all_pose_estimates, all_class_idcs = ae_pose_est.process_pose(boxes, labels, image)

    if args.vis:
        bgr, depth,_ = renderer.render_many(obj_ids = [clas_idx for clas_idx in all_class_idcs],
                    W = ae_pose_est._width,
                    H = ae_pose_est._height,
                    K = ae_pose_est._camK, 
                    # R = transform.random_rotation_matrix()[:3,:3],
                    Rs = [pose_est[:3,:3] for pose_est in all_pose_estimates],
                    ts = [pose_est[:3,3] for pose_est in all_pose_estimates],
                    near = 10,
                    far = 10000,
                    random_light=False,
                    phong={'ambient':0.4,'diffuse':0.8, 'specular':0.3})

        bgr = cv2.resize(bgr,(ae_pose_est._width,ae_pose_est._height))
        
        g_y = np.zeros_like(bgr)
        g_y[:,:,1]= bgr[:,:,1]    
        im_bg = cv2.bitwise_and(image,image,mask=(g_y[:,:,1]==0).astype(np.uint8))                 
        image_show = cv2.addWeighted(im_bg,1,g_y,1,0)

        #cv2.imshow('pred view rendered', pred_view)
        for label,box,score in zip(labels,boxes,scores):
            box = box.astype(np.int32)
            xmin,ymin,xmax,ymax = box[0],box[1],box[0]+box[2],box[1]+box[3]
            print label
            cv2.putText(image_show, '%s : %1.3f' % (label,score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, color_dict[int(label)], 2)
            cv2.rectangle(image_show,(xmin,ymin),(xmax,ymax),(255,0,0),2)

        #cv2.imshow('', bgr)
        cv2.imshow('real', image_show)
        cv2.waitKey(1)
