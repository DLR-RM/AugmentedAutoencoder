
import cv2
import numpy as np
import os

from auto_pose.meshrenderer import meshrenderer
from auto_pose.ae.utils import lazy_property

class PoseVisualizer:

    def __init__(self, mp_pose_estimator, downsample=1, vertex_scale=False):

        self.downsample = downsample
        self.vertex_scale = [mp_pose_estimator.train_args.getint('Dataset', 'VERTEX_SCALE')] if not vertex_scale else [1.]
        if hasattr(mp_pose_estimator, 'class_2_objpath'):
            self.classes, self.ply_model_paths = zip(*mp_pose_estimator.class_2_objpath.items())
        else:
            # For BOP evaluation (sry!):
            self.classes = mp_pose_estimator.class_2_codebook.keys()
            all_model_paths = eval(mp_pose_estimator.train_args.get('Paths', 'MODEL_PATH'))
            base_path = '/'.join(all_model_paths[0].split('/')[:-3])
            itodd_paths = [os.path.join(base_path, 'itodd/models/obj_0000{: 02d}.ply'.format(i)) for i in range(29)]
            all_model_paths = all_model_paths + itodd_paths
            all_model_paths = [model_p.replace('YCB_VideoDataset/original2sixd/bop_models/', 'bop/original/ycbv/models_eval/') for model_p in all_model_paths]
            self.ply_model_paths = []
            for cb_name in mp_pose_estimator.class_2_codebook.values():
                for model_path in all_model_paths:
                    bop_dataset = cb_name.split('_')[0]
                    bop_dataset = 'ycbv' if bop_dataset == 'original2sixd' else bop_dataset
                    model_type, obj, obj_id = cb_name.split('_')[-3:]
                    model_name = obj + '_' + obj_id
                    if bop_dataset in model_path and model_name in model_path:
                        self.ply_model_paths.append(model_path)

        print(('renderer', 'Model paths: ', self.ply_model_paths))

    @lazy_property
    def renderer(self):
       return meshrenderer.Renderer(self.ply_model_paths, 
                        samples=1, 
                        vertex_tmp_store_folder='.',
                        vertex_scale=float(self.vertex_scale[0]))  # 1000 for models in meters

    def render_poses(self, image, camK, pose_ests, dets, vis_bbs=True, vis_mask=False, all_pose_estimates_rgb=None, depth_image=None, waitKey=True):
        W_d = image.shape[1] / self.downsample
        H_d = image.shape[0] / self.downsample
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


        
