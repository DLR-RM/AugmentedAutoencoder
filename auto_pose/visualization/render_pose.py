
import cv2
import numpy as np
import os
import glob

from auto_pose.meshrenderer import meshrenderer
from auto_pose.ae.utils import lazy_property

class PoseVisualizer:

    def __init__(self, ae_pose_est, downsample=1):

        self.downsample = downsample
        self.classes = list(ae_pose_est.class_2_encoder.keys())
        self.vertex_scale = list(set([ae_pose_est.all_train_args[c].getint('Dataset','VERTEX_SCALE') for c in self.classes]))
        self.ply_model_paths = [str(ae_pose_est.all_train_args[c].get('Paths','MODEL_PATH')) for c in self.classes]
        print(('renderer', 'Model paths: ', self.ply_model_paths))

    @lazy_property
    def renderer(self):
       return meshrenderer.Renderer(self.ply_model_paths, 
                        samples=1, 
                        vertex_tmp_store_folder='.',
                        vertex_scale=float(self.vertex_scale[0])) #1000 for models in meters

    def render_poses(self, image, camK, pose_ests, dets, vis_bbs=True, vis_mask=False, all_pose_estimates_rgb=None, depth_image=None, waitKey=True):
        W_d = image.shape[1] // self.downsample
        H_d = image.shape[0] // self.downsample
        print( [self.classes.index(pose_est.name) for pose_est in pose_ests])
        bgr, depth,_ = self.renderer.render_many(obj_ids = [self.classes.index(pose_est.name) for pose_est in pose_ests],
                    W = W_d,
                    H = H_d,
                    K = camK.copy(), 
                    Rs = [pose_est.trafo[:3,:3] for pose_est in pose_ests],
                    ts = [pose_est.trafo[:3,3] for pose_est in pose_ests],
                    near = 10,
                    far = 10000)

        image_show = cv2.resize(image,(W_d,H_d))
        if all_pose_estimates_rgb is not None:
            image_show_rgb = image_show.copy()

        g_y = np.zeros_like(bgr)
        g_y[:,:,1]= bgr[:,:,1]
        image_show[bgr > 0] = g_y[bgr > 0]*2./3. + image_show[bgr > 0]*1./3.

        if all_pose_estimates_rgb is not None:
            bgr, depth,_ = self.renderer.render_many(obj_ids = [clas_idx for clas_idx in all_class_idcs],
                W = W_d,
                H = H_d,
                K = camK.copy(), 
                Rs = [pose_est.trafo[:3,:3] for pose_est in pose_ests],
                ts = [pose_est.trafo[:3,3] for pose_est in pose_ests],
                near = 10,
                far = 10000)

            bgr = cv2.resize(bgr,(W_d,H_d))

            b_y = np.zeros_like(bgr)
            b_y[:,:,0]= bgr[:,:,0]
            image_show_rgb[bgr > 0] = b_y[bgr > 0]*2./3. + image_show_rgb[bgr > 0]*1./3.
        if np.any(depth_image):
            depth_show = depth_image.copy()
            depth_show = np.dstack((depth_show,depth_show,depth_show))
            depth_show[bgr[:,:,0] > 0] = g_y[bgr[:,:,0] > 0]*2./3. + depth_show[bgr[:,:,0] > 0]*1./3.
            cv2.imshow('depth_refined_pose', depth_show)

        if vis_bbs:
            
            # for label,box,score in zip(labels,boxes,scores):
            for det in dets:
                # box = box.astype(np.int32) / self.downsample
                # xmin, ymin, xmax, ymax = box[0], box[1], box[0] + box[2], box[1] + box[3]
                xmin, ymin, xmax, ymax = int(det.xmin * W_d), int(det.ymin * H_d), int(det.xmax * W_d), int(det.ymax * H_d)
                label, score = list(det.classes.items())[0]
                try:
                    cv2.putText(image_show, '%s : %1.3f' % (label,score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, (0,0,255), 2)
                    cv2.rectangle(image_show,(xmin,ymin),(xmax,ymax),(255,0,0),2)
                    if all_pose_estimates_rgb is not None:
                        cv2.putText(image_show_rgb, '%s : %1.3f' % (label,score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, (0,0,255), 2)
                        cv2.rectangle(image_show_rgb,(xmin,ymin),(xmax,ymax),(255,0,0),2)
                except:
                    print('failed to plot boxes')

        if all_pose_estimates_rgb is not None:
            cv2.imshow('rgb_pose', image_show_rgb)
        cv2.imshow('refined_pose', image_show)
        if waitKey:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
        return (image_show)


        
