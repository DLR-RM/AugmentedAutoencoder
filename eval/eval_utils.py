import os
import numpy as np
import cv2
from collections import defaultdict
import hashlib

from sixd_toolkit.params import dataset_params
from sixd_toolkit.pysixd import inout

from ae import utils as u


def get_gt_scene_crops(scene_id, eval_args, train_args):
    

    dataset_name = eval_args.get('DATA','DATASET')
    cam_type = eval_args.get('DATA','CAM_TYPE')
    icp = eval_args.get('EVALUATION','ICP')

    delta = eval_args.get('METRIC', 'VSD_DELTA')

    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    dataset_path = u.get_dataset_path(workspace_path)

    H = train_args.getint('Dataset','H')

    cfg_string = str([scene_id] + eval_args.items('DATA') + eval_args.items('BBOXES') + [H])
    current_config_hash = hashlib.md5(cfg_string).hexdigest()

    current_file_name = os.path.join(dataset_path, current_config_hash + '.npz')
    

    if os.path.exists(current_file_name):
        data = np.load(current_file_name)
        test_img_crops = data['test_img_crops'].item()
        try:
            test_img_depth_crops = data['test_img_depth_crops'].item()
        except:
            test_img_depth_crops = {}

        bb_scores = data['bb_scores'].item()
        bb_vis = data['visib_gt'].item()
        bbs = data['bbs'].item()
        print 'loaded previously generated ground truth crops!'
    else:
        test_imgs = load_scenes(scene_id, eval_args)
        test_imgs_depth = load_scenes(scene_id, eval_args, depth=True) if icp else None

        data_params = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)

        # only available for primesense, sixdtoolkit can generate
        visib_gt = inout.load_yaml(data_params['scene_gt_stats_mpath'].format(scene_id, delta))
        
        bb_gt = inout.load_gt(data_params['scene_gt_mpath'].format(scene_id))

        test_img_crops, test_img_depth_crops, bbs, bb_scores, bb_vis = generate_scene_crops(test_imgs, test_imgs_depth, bb_gt, eval_args, 
                                                                                            train_args, visib_gt=visib_gt)

        np.savez(current_file_name, test_img_crops=test_img_crops, test_img_depth_crops=test_img_depth_crops, bbs = bbs, bb_scores=bb_scores, visib_gt=bb_vis)
        
        current_cfg_file_name = os.path.join(dataset_path, current_config_hash + '.cfg')
        with open(current_cfg_file_name, 'w') as f:
            f.write(cfg_string)
        print 'created new ground truth crops!'


    return (test_img_crops, test_img_depth_crops, bbs, bb_scores, bb_vis)


def generate_scene_crops(test_imgs, test_depth_imgs, bboxes, eval_args, train_args, visib_gt = None):


    scenes = eval(eval_args.get('DATA','SCENES'))
    objects = eval(eval_args.get('DATA','OBJECTS'))
    estimate_bbs = eval_args.getboolean('BBOXES', 'ESTIMATE_BBS')
    pad_factor = eval_args.getfloat('BBOXES','PAD_FACTOR')
    icp = eval_args.getboolean('EVALUATION','ICP')

    W_AE = train_args.getint('Dataset','W')
    H_AE = train_args.getint('Dataset','H')


    test_img_crops, test_img_depth_crops, bb_scores, bb_vis, bbs = {}, {}, {}, {}, {}

    H,W = test_imgs.shape[1:3]
    for view,img in enumerate(test_imgs):
        if icp:
            depth = test_depth_imgs[view]
            test_img_depth_crops[view] = {}

        test_img_crops[view], bb_scores[view], bb_vis[view], bbs[view] = {}, {}, {}, {}
        if len(bboxes[view]) > 0:
            for bbox_idx,bbox in enumerate(bboxes[view]):
                if bbox['obj_id'] in objects:
                    bb = np.array(bbox['obj_bb'])
                    obj_id = bbox['obj_id']
                    bb_score = bbox['score'] if estimate_bbs else 1.0
                    vis_frac = None if estimate_bbs else visib_gt[view][bbox_idx]['visib_fract']

                    x, y, w, h = bb
                    size = int(np.maximum(h,w) * pad_factor)
                    left = np.max([x+w/2-size/2, 0])
                    right = np.min([x+w/2+size/2, W])
                    top = np.max([y+h/2-size/2, 0])
                    bottom = np.min([y+h/2+size/2, H])

                    crop = img[top:bottom, left:right]
                    # print 'Original Crop Size: ', crop.shape
                    resized_crop = cv2.resize(crop, (H_AE,W_AE))

                    if icp:
                        depth_crop = depth[top:bottom, left:right]
                        test_img_depth_crops[view].setdefault(obj_id,[]).append(depth_crop)
                    test_img_crops[view].setdefault(obj_id,[]).append(resized_crop)
                    bb_scores[view].setdefault(obj_id,[]).append(bb_score)
                    bb_vis[view].setdefault(obj_id,[]).append(vis_frac)
                    bbs[view].setdefault(obj_id,[]).append(bb)

    return (test_img_crops, test_img_depth_crops, bbs, bb_scores, bb_vis)

def generate_depth_scene_crops(test_depth_imgs, scene_id, eval_args, train_args):

    dataset_name = eval_args.get('DATA','DATASET')
    cam_type = eval_args.get('DATA','CAM_TYPE')
    objects = eval(eval_args.get('DATA','OBJECTS'))[0]
    estimate_bbs = eval_args.getboolean('BBOXES', 'ESTIMATE_BBS')

    p = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)
    noof_imgs = len(os.listdir(os.path.join(p['base_path'], p['test_dir'], '{:02d}', 'depth').format(scene_id)))
    bb_gt = inout.load_gt(p['scene_gt_mpath'].format(scene_id))    
    all_real_depth_pts = []
    R_gts = []
    t_gts = []
    for view,depth in enumerate(test_depth_imgs):
        H,W = depth.shape
        for bbox_idx,bbox in enumerate(bb_gt[view]):
            if bbox['obj_id'] == obj_id:
                bb = np.array(bbox['obj_bb'])
                R_gts.append(np.array(bbox['cam_R_m2c']))
                t_gts.append(np.array(bbox['cam_t_m2c']))
                x, y, w, h = bb
                size = int(np.maximum(h,w) * 1.1)
                left = np.max([x+w/2-size/2, 0])
                right = np.min([x+w/2+size/2, W])
                top = np.max([y+h/2-size/2, 0])
                bottom = np.min([y+h/2+size/2, H])

                depth_crop = depth[top:bottom, left:right]
                #print 'Original Crop Size: ', depth_crop.shape
                real_depth_pts = misc.rgbd_to_point_cloud(np.array([1075.65, 0, depth_crop.shape[0]/2, 0, 1073.90, depth_crop.shape[0]/2, 0, 0, 1]).reshape(3,3),depth_crop)[0]
                # real_depth_pts = misc.rgbd_to_point_cloud(np.array([1075.65091572, 0.00000000, 372.06888344, 0.00000000, 1073.90347929, 300.72159802, 0.00000000, 0.00000000, 1.00000000]).reshape(3,3),depth)[0]
                all_real_depth_pts.append(real_depth_pts)

    return (all_real_depth_pts, R_gts, t_gts)

def noof_scene_views(scene_id, eval_args):
    dataset_name = eval_args.get('DATA','DATASET')
    cam_type = eval_args.get('DATA','CAM_TYPE')

    p = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)
    noof_imgs = len(os.listdir(os.path.join(p['base_path'], p['test_dir'], '{:02d}', 'rgb').format(scene_id)))

    return noof_imgs


def load_scenes(scene_id, eval_args, depth=False):

    dataset_name = eval_args.get('DATA','DATASET')
    cam_type = eval_args.get('DATA','CAM_TYPE')

    p = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)
    noof_imgs = noof_scene_views(scene_id, eval_args)
    if depth:
        imgs = np.empty((noof_imgs,) + p['test_im_size'][::-1], dtype=np.float32)
        for view_id in xrange(noof_imgs):
            depth_path = p['test_depth_mpath'].format(scene_id, view_id)
            imgs[view_id,...] = inout.load_depth(depth_path)/10.
    else:    
        print (noof_imgs,) + p['test_im_size'][::-1] + (3,)
        imgs = np.empty((noof_imgs,) + p['test_im_size'][::-1] + (3,), dtype=np.uint8)
        for view_id in xrange(noof_imgs):
            img_path = p['test_rgb_mpath'].format(scene_id, view_id)
            imgs[view_id,...] = cv2.imread(img_path)

    return imgs



def select_img_crops(crop_candidates, bbs, bb_scores, visibs, eval_args):

    estimate_bbs = eval_args.getboolean('BBOXES', 'ESTIMATE_BBS')
    single_instance = eval_args.getboolean('BBOXES', 'SINGLE_INSTANCE')

    if single_instance and estimate_bbs:
        idcs = np.array([np.argmax(bb_scores)])
    elif single_instance and not estimate_bbs:
        idcs = np.array([np.argmax(visibs)])
    elif not single_instance and estimate_bbs:
        idcs = np.argsort(-bb_scores)
    else:
        idcs = np.argsort(-visibs)
    
    return (np.array(crop_candidates)[idcs], np.array(bbs)[idcs], np.array(bb_scores)[idcs], np.array(visibs)[idcs])