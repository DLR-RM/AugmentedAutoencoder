import cv2
import numpy as np
import os
import argparse

from m3vision.interfaces.detector_bb import BoundingBox
from auto_pose.m3_interface.ae_pose_estimator import AePoseEstimator
from auto_pose.ae.utils import get_dataset_path


def load_depth2(path):
    import scipy.misc
    d = scipy.misc.imread(path)
    d = d.astype(np.float32)
    return d


parser = argparse.ArgumentParser()
parser.add_argument("-vis", action='store_true', default=False)
args = parser.parse_args()

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print 'Please define a workspace path:\n'
    print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
    exit(-1)


this_dir = os.path.dirname(os.path.abspath(__file__))
# example input, replace with camera stream
# img = cv2.imread(os.path.join(this_dir,'sample_data','cup.png'))

# img = cv2.imread('/home_local/sund_ma/data/t-less/t-less_v2/test_primesense/01/rgb/0205.png')
# depth_img = load_depth2('/home_local/sund_ma/data/t-less/t-less_v2/test_primesense/01/depth/0205.png')

img = cv2.imread('/volume/USERSTORE/project_indu/sixd/test_sr300/01/rgb/100.png')
depth_img = load_depth2('/volume/USERSTORE/project_indu/sixd/test_sr300/01/depth/100.png')

H,W,_ = img.shape
# replace with a detector
# bb = BoundingBox(xmin=0.517,xmax=0.918,ymin=0.086,ymax=0.592,classes={'benchviseblue':1.0}) 
bb2 = BoundingBox(xmin=0.517,xmax=0.918,ymin=0.086,ymax=0.592,classes={'1':1.0}) 
# test camera matrix
camK = np.array([[1075.65,0,W//2],[0,1073.90,H//2],[0,0,1]]) 


ae_pose_est = AePoseEstimator(os.path.join(workspace_path,'cfg_m3vision/test_config.cfg'))
pose_ests = ae_pose_est.process([bb2],img,camK,depth_img=depth_img)

try:
    print pose_ests[0].trafo
except:
    print 'nothing detected'
if args.vis:
    ply_model_paths = [str(train_args.get('Paths','MODEL_PATH')) for train_args in ae_pose_est.all_train_args]
    cad_reconst = [str(train_args.get('Dataset','MODEL')) for train_args in ae_pose_est.all_train_args]
    print cad_reconst
    print ply_model_paths
    from meshrenderer import meshrenderer, meshrenderer_phong
    if all([model == 'cad' for model in cad_reconst]):   
        renderer = meshrenderer.Renderer(ply_model_paths, 
                        samples=1, 
                        vertex_tmp_store_folder=get_dataset_path(workspace_path),
                        vertex_scale=float(1000)) # float(1) for some models
    else:
        # for textured ply
        renderer = meshrenderer_phong.Renderer(ply_model_paths, 
                        samples=1, 
                        vertex_tmp_store_folder=get_dataset_path(workspace_path)
                        ) 

    bgr, depth,_ = renderer.render_many(obj_ids = [ae_pose_est.class_names.index(pose_est.name) for pose_est in pose_ests],
                W = W,
                H = H,
                K = camK, 
                # R = transform.random_rotation_matrix()[:3,:3],
                Rs = [pose_est.trafo[:3,:3] for pose_est in pose_ests],
                ts = [pose_est.trafo[:3,3] for pose_est in pose_ests],
                near = 10,
                far = 10000,
                random_light=False,
                phong={'ambient':0.4,'diffuse':0.8, 'specular':0.3})
    cv2.imshow('', bgr)
    cv2.imshow('real', img)
cv2.waitKey(0)