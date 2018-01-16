import os
import numpy as np
import cv2
from collections import defaultdict
import hashlib

from sixd_toolkit.params import dataset_params
from sixd_toolkit.pysixd import inout

from ae import utils as u


def get_gt_scene_crops(scene_id, eval_args, train_args):
    
    pad_factor = train_args.getfloat('Dataset','PAD_FACTOR')

    dataset_name = eval_args.get('DATA','DATASET')
    cam_type = eval_args.get('DATA','CAM_TYPE')
    estimate_bbs = eval_args.get('BBOXES', 'ESTIMATE_BBS')

    delta = eval_args.get('METRIC', 'VSD_DELTA')

    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    dataset_path = u.get_dataset_path(workspace_path)

    cfg_string = str([scene_id] + eval_args.items('BBOXES') + eval_args.items('DATA'))
    current_config_hash = hashlib.md5(cfg_string).hexdigest()

    current_file_name = os.path.join(dataset_path, current_config_hash + '.npz')
    

    if os.path.exists(current_file_name):
        data = np.load(current_file_name)
        test_img_crops = data['test_img_crops'].item()
        bb_scores = data['bb_scores'].item()
        bb_vis = data['visib_gt'].item()
        bbs = data['bbs'].item()
        print 'loaded previously generated ground truth crops!'
    else:
        test_imgs = load_scenes(scene_id, eval_args)
        data_params = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)

        # only available for primesense, sixdtoolkit can generate
        visib_gt = inout.load_yaml(data_params['scene_gt_stats_mpath'].format(scene_id, delta))
        
        bb_gt = inout.load_gt(data_params['scene_gt_mpath'].format(scene_id))

        test_img_crops, bbs, bb_scores, bb_vis = generate_scene_crops(test_imgs, bb_gt, eval_args, train_args, visib_gt=visib_gt)
        np.savez(current_file_name, test_img_crops=test_img_crops, bbs = bbs, bb_scores=bb_scores, visib_gt=bb_vis)
        
        current_cfg_file_name = os.path.join(dataset_path, current_config_hash + '.cfg')
        with open(current_cfg_file_name, 'w') as f:
            f.write(cfg_string)
        print 'created new ground truth crops!'


    return (test_img_crops, bbs, bb_scores, bb_vis)


def generate_scene_crops(test_imgs, bboxes, eval_args, train_args, visib_gt = None):


    scenes = eval(eval_args.get('DATA','SCENES'))
    objects = eval(eval_args.get('DATA','OBJECTS'))
    estimate_bbs = eval_args.getboolean('BBOXES', 'ESTIMATE_BBS')

    pad_factor = train_args.getfloat('Dataset','PAD_FACTOR')
    W_AE = train_args.getint('Dataset','W')
    H_AE = train_args.getint('Dataset','H')

    test_img_crops, bb_scores, bb_vis, bbs = {}, {}, {}, {}

    H,W = test_imgs.shape[1:3]
    for view,img in enumerate(test_imgs):
        test_img_crops[view], bb_scores[view], bb_vis[view], bbs[view] = {}, {}, {}, {}
        if len(bboxes[view]) > 0:
            for bbox_idx,bbox in enumerate(bboxes[view]):
                if bbox['obj_id'] in objects:
                    bb = np.array(bbox['obj_bb'])
                    obj_id = bbox['obj_id']
                    bb_score = bbox['score'] if estimate_bbs else 1.0
                    vis_frac = visib_gt[view][bbox_idx]['visib_fract'] if not estimate_bbs else None

                    x, y, w, h = bb
                    size = int(np.maximum(h,w) * pad_factor)
                    left = np.max([x+w/2-size/2, 0])
                    right = np.min([x+w/2+size/2, W])
                    top = np.max([y+h/2-size/2, 0])
                    bottom = np.min([y+h/2+size/2, H])

                    crop = img[top:bottom, left:right]
                    resized_crop = cv2.resize(crop, (H_AE,W_AE))

                    test_img_crops[view].setdefault(obj_id,[]).append(resized_crop)
                    bb_scores[view].setdefault(obj_id,[]).append(bb_score)
                    bb_vis[view].setdefault(obj_id,[]).append(vis_frac)
                    bbs[view].setdefault(obj_id,[]).append(bb)

    return (test_img_crops, bbs, bb_scores, bb_vis)


def noof_scene_views(scene_id, eval_args):
    dataset_name = eval_args.get('DATA','DATASET')
    cam_type = eval_args.get('DATA','CAM_TYPE')

    p = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)
    noof_imgs = len(os.listdir(os.path.join(p['base_path'], p['test_dir'], '{:02d}', 'rgb').format(scene_id)))

    return noof_imgs


def load_scenes(scene_id, eval_args):

    dataset_name = eval_args.get('DATA','DATASET')
    cam_type = eval_args.get('DATA','CAM_TYPE')

    p = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)
    noof_imgs = noof_scene_views(scene_id, eval_args)
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
    
    return (np.array(crop_candidates)[idcs], np.array(bbs)[idcs], np.array(bb_scores)[idcs])