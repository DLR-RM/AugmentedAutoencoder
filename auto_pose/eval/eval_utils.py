import os
import numpy as np
import cv2
from collections import defaultdict
import hashlib

from sixd_toolkit.params import dataset_params
from sixd_toolkit.pysixd import inout

from auto_pose.ae import utils as u


def get_gt_scene_crops(scene_id, eval_args, train_args):
    

    dataset_name = eval_args.get('DATA','DATASET')
    cam_type = eval_args.get('DATA','CAM_TYPE')
    icp = eval_args.getboolean('EVALUATION','ICP')

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
        test_img_depth_crops = data['test_img_depth_crops'].item()
        bb_scores = data['bb_scores'].item()
        bb_vis = data['visib_gt'].item()
        bbs = data['bbs'].item()
    if not os.path.exists(current_file_name) or len(test_img_crops) == 0 or len(test_img_depth_crops) == 0:
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
    else:
        print 'loaded previously generated ground truth crops!'
        print len(test_img_crops), len(test_img_depth_crops)



    return (test_img_crops, test_img_depth_crops, bbs, bb_scores, bb_vis)


def generate_scene_crops(test_imgs, test_depth_imgs, bboxes, eval_args, train_args, visib_gt = None):


    scenes = eval(eval_args.get('DATA','SCENES'))
    obj_id = eval_args.getint('DATA','OBJ_ID')
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
                if bbox['obj_id'] == obj_id:
                    bb = np.array(bbox['obj_bb'])
                    obj_id = bbox['obj_id']
                    bb_score = bbox['score'] if estimate_bbs else 1.0
                    vis_frac = None if estimate_bbs else visib_gt[view][bbox_idx]['visib_fract']
                    #print bb
                    ## uebler hack: remove!
                    # xmin, ymin, xmax, ymax = bb
                    # x, y, w, h = xmin, ymin, xmax-xmin, ymax-ymin 
                    # bb = np.array([x, y, w, h])
                    ##
                    x,y,w,h = bb
                    
                    size = int(np.maximum(h,w) * pad_factor)
                    left = int(np.max([x+w/2-size/2, 0]))
                    right = int(np.min([x+w/2+size/2, W]))
                    top = int(np.max([y+h/2-size/2, 0]))
                    bottom = int(np.min([y+h/2+size/2, H]))

                    crop = img[top:bottom, left:right].copy()
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


def noof_scene_views(scene_id, eval_args):
    dataset_name = eval_args.get('DATA','DATASET')
    cam_type = eval_args.get('DATA','CAM_TYPE')

    p = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)

    noof_imgs = len(os.listdir(os.path.dirname(p['test_rgb_mpath']).format(scene_id)))
    return noof_imgs


def load_scenes(scene_id, eval_args, depth=False):

    dataset_name = eval_args.get('DATA','DATASET')
    cam_type = eval_args.get('DATA','CAM_TYPE')

    p = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)
    cam_p = inout.load_cam_params(p['cam_params_path'])
    noof_imgs = noof_scene_views(scene_id, eval_args)
    if depth:
        imgs = np.empty((noof_imgs,) + p['test_im_size'][::-1], dtype=np.float32)
        for view_id in range(noof_imgs):
            depth_path = p['test_depth_mpath'].format(scene_id, view_id)
            try:
                imgs[view_id,...] = inout.load_depth2(depth_path) * cam_p['depth_scale']
            except:
                print depth_path,' not found'
    
    else:    
        print (noof_imgs,) + p['test_im_size'][::-1] + (3,)
        imgs = np.empty((noof_imgs,) + p['test_im_size'][::-1] + (3,), dtype=np.uint8)
        print noof_imgs
        for view_id in range(noof_imgs):
            img_path = p['test_rgb_mpath'].format(scene_id, view_id)
            try:
                imgs[view_id,...] = cv2.imread(img_path)
            except:
                print img_path,' not found'

    return imgs

def get_all_scenes_for_obj(eval_args):
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    dataset_path = u.get_dataset_path(workspace_path)

    dataset_name = eval_args.get('DATA','DATASET')
    cam_type = eval_args.get('DATA','CAM_TYPE')
    try:
        obj_id = eval_args.getint('DATA', 'OBJ_ID')
    except:
        obj_id = eval(eval_args.get('DATA', 'OBJECTS'))[0]
    

    cfg_string = str(dataset_name)
    current_config_hash = hashlib.md5(cfg_string).hexdigest()
    current_file_name = os.path.join(dataset_path, current_config_hash + '.npy')

    if os.path.exists(current_file_name):
        obj_scene_dict = np.load(current_file_name).item()
    else:    
        p = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)
        
        obj_scene_dict = {}
        scene_gts = []
        for scene_id in range(1,p['scene_count']+1):
            print scene_id
            scene_gts.append(inout.load_yaml(p['scene_gt_mpath'].format(scene_id)))

        for obj in range(1,p['obj_count']+1):
            eval_scenes = set()
            for scene_i,scene_gt in enumerate(scene_gts):
                for view_gt in scene_gt[0]:
                    if view_gt['obj_id'] == obj:
                        eval_scenes.add(scene_i+1)
            obj_scene_dict[obj] = list(eval_scenes)
        np.save(current_file_name,obj_scene_dict)
    print obj_scene_dict

    eval_scenes = obj_scene_dict[obj_id]

    return eval_scenes


def select_img_crops(crop_candidates, test_crops_depth, bbs, bb_scores, visibs, eval_args):

    estimate_bbs = eval_args.getboolean('BBOXES', 'ESTIMATE_BBS')
    single_instance = eval_args.getboolean('BBOXES', 'SINGLE_INSTANCE')
    icp = eval_args.getboolean('EVALUATION', 'ICP')

    if single_instance and estimate_bbs:
        idcs = np.array([np.argmax(bb_scores)])
    elif single_instance and not estimate_bbs:
        idcs = np.array([np.argmax(visibs)])
    elif not single_instance and estimate_bbs:
        idcs = np.argsort(-np.array(bb_scores))
    else:
        idcs = np.argsort(-np.array(visibs))
    
    if icp:
        return (np.array(crop_candidates)[idcs], np.array(test_crops_depth)[idcs], np.array(bbs)[idcs], np.array(bb_scores)[idcs], np.array(visibs)[idcs])
    else:
        return (np.array(crop_candidates)[idcs], None, np.array(bbs)[idcs], np.array(bb_scores)[idcs], np.array(visibs)[idcs])
