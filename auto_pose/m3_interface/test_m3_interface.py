import cv2
import numpy as np
from meshrenderer import meshrenderer
from m3vision.interfaces.detector_bb import BoundingBox
from auto_pose.m3_interface.ae_pose_estimator import AePoseEstimator

visualize = True

img = cv2.imread('./cup.png')
H,W,_ = img.shape
bb = BoundingBox(xmin=0.517,xmax=0.918,ymin=0.086,ymax=0.592,classes={'ikea_cup':1.0})
camK = np.array([[1075.65,0,W//2],[0,1073.90,H//2],[0,0,1]])

ae_pose_est = AePoseEstimator('../ae/cfg_m3vision/test_config.cfg')
pose_est = ae_pose_est.process([bb],img,camK)
print pose_est[0].trafo


if visualize:
    from meshrenderer import meshrenderer
    renderer = meshrenderer.Renderer(['/home_local/sund_ma/data/ikea_mugs/ikea_cup_model/ikea_mug_reduced.ply'], 
                    samples=1, 
                    vertex_tmp_store_folder='.',
                    vertex_scale=float(1000)) 


    bgr, depth = renderer.render(obj_id = 0,
                W = W,
                H = H,
                K = camK, 
                # R = transform.random_rotation_matrix()[:3,:3],
                R = pose_est[0].trafo[:3,:3],
                t = pose_est[0].trafo[:3,3],
                near = 10,
                far = 10000,
                random_light=False,
                phong={'ambient':0.4,'diffuse':0.8, 'specular':0.3})
    cv2.imshow('', bgr)
    cv2.imshow('real', img)
    cv2.waitKey(0)
