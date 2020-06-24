import os
import numpy as np
import cv2
from collections import defaultdict
import hashlib
import glob
import configparser

from sixd_toolkit.params import dataset_params
from sixd_toolkit.pysixd import inout

from auto_pose.ae.pysixd_stuff import view_sampler
from auto_pose.ae import utils as u

import glob

def get_gt_scene_crops(scene_id, eval_args, train_args, load_gt_masks=False):
    
    dataset_name = eval_args.get('DATA','DATASET')
    cam_type = eval_args.get('DATA','CAM_TYPE')
    icp = eval_args.getboolean('EVALUATION','ICP')

    delta = eval_args.get('METRIC', 'VSD_DELTA')

    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    dataset_path = u.get_dataset_path(workspace_path)

    H_AE = train_args.getint('Dataset','H')
    W_AE = train_args.getint('Dataset','W')

    cfg_string = str([scene_id] + eval_args.items('DATA') + eval_args.items('BBOXES') + [H_AE])
    cfg_string = cfg_string.encode('utf-8')
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
        
        gt = inout.load_gt(data_params['scene_gt_mpath'].format(scene_id))
        
        gt_inst_masks = None
        if load_gt_masks:
            mask_paths = glob.glob(os.path.join(load_gt_masks, '{:02d}/masks/*.npy'.format(scene_id)))
            gt_inst_masks = [np.load(mp) for mp in mask_paths] 
        

        test_img_crops, test_img_depth_crops, bbs, bb_scores, bb_vis = generate_scene_crops(test_imgs, test_imgs_depth, gt, eval_args, 
                                                                                            (H_AE, W_AE), visib_gt=visib_gt,inst_masks=gt_inst_masks)


        np.savez(current_file_name, test_img_crops=test_img_crops, test_img_depth_crops=test_img_depth_crops, bbs = bbs, bb_scores=bb_scores, visib_gt=bb_vis)
        
        current_cfg_file_name = os.path.join(dataset_path, current_config_hash + '.cfg')
        with open(current_cfg_file_name, 'w') as f:
            f.write(cfg_string)
        print('created new ground truth crops!')
    else:
        print('loaded previously generated ground truth crops!')
        print((len(test_img_crops), len(test_img_depth_crops)))



    return (test_img_crops, test_img_depth_crops, bbs, bb_scores, bb_vis)

def get_sixd_gt_train_crops(obj_id, hw_ae, pad_factor=1.2, dataset='tless', cam_type='primesense'):
    
    data_params = dataset_params.get_dataset_params(dataset, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)

    eval_args = configparser.ConfigParser()
    eval_args.add_section("DATA")
    eval_args.add_section("BBOXES")
    eval_args.add_section("EVALUATION")

    eval_args.set('BBOXES', 'ESTIMATE_BBS', "False")
    eval_args.set('EVALUATION','ICP', "False")
    eval_args.set('BBOXES','PAD_FACTOR', str(pad_factor))
    eval_args.set('BBOXES','ESTIMATE_MASKS', "False")
    eval_args.set('DATA','OBJ_ID', str(obj_id))

    gt = inout.load_gt(data_params['obj_gt_mpath'].format(obj_id))
    imgs = []
    for im_id in range(504):
        imgs.append(cv2.imread(data_params['train_rgb_mpath'].format(obj_id, im_id)))

    test_img_crops, _, _, _, _ = generate_scene_crops(np.array(imgs), None, gt, eval_args, hw_ae)

    return test_img_crops

def generate_scene_crops(test_imgs, test_depth_imgs, gt, eval_args, hw_ae, visib_gt = None, inst_masks=None):

    obj_id = eval_args.getint('DATA','OBJ_ID')
    estimate_bbs = eval_args.getboolean('BBOXES', 'ESTIMATE_BBS')
    pad_factor = eval_args.getfloat('BBOXES','PAD_FACTOR')
    icp = eval_args.getboolean('EVALUATION','ICP')

    estimate_masks = eval_args.getboolean('BBOXES','ESTIMATE_MASKS')
    print(hw_ae)
    H_AE, W_AE = hw_ae


    test_img_crops, test_img_depth_crops, bb_scores, bb_vis, bbs = {}, {}, {}, {}, {}

    H,W = test_imgs.shape[1:3]
    for view,img in enumerate(test_imgs):
        if icp:
            depth = test_depth_imgs[view]
            test_img_depth_crops[view] = {}

        test_img_crops[view], bb_scores[view], bb_vis[view], bbs[view] = {}, {}, {}, {}
        if len(gt[view]) > 0:
            for bbox_idx,bbox in enumerate(gt[view]):
                if bbox['obj_id'] == obj_id:
                    if 'score' in bbox:
                        if bbox['score']==-1:
                            continue
                    bb = np.array(bbox['obj_bb'])
                    obj_id = bbox['obj_id']
                    bb_score = bbox['score'] if estimate_bbs else 1.0
                    if estimate_bbs and visib_gt is not None:
                        vis_frac = visib_gt[view][bbox_idx]['visib_fract']
                    else:
                        vis_frac = None

                    x,y,w,h = bb
                    
                    size = int(np.maximum(h,w) * pad_factor)
                    left = int(np.max([x+w/2-size/2, 0]))
                    right = int(np.min([x+w/2+size/2, W]))
                    top = int(np.max([y+h/2-size/2, 0]))
                    bottom = int(np.min([y+h/2+size/2, H]))
                    if inst_masks is None:

                        crop = img[top:bottom, left:right].copy()
                        if icp:
                            depth_crop = depth[top:bottom, left:right]
                    else:
                        if not estimate_masks:
                            mask = inst_masks[view]
                            img_copy = np.zeros_like(img)
                            img_copy[mask == (bbox_idx+1)] = img[mask == (bbox_idx+1)]
                            crop = img_copy[top:bottom, left:right].copy()
                            if icp:
                                depth_copy = np.zeros_like(depth)
                                depth_copy[mask == (bbox_idx+1)] = depth[mask == (bbox_idx+1)]
                                depth_crop = depth_copy[top:bottom, left:right]
                        else:
                            # chan = int(bbox['np_channel_id'])
                            chan = bbox_idx
                            mask = inst_masks[view][:,:,chan]
                            # kernel = np.ones((2,2), np.uint8) 
                            # mask_eroded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) 
                            # cv2.imshow('mask_erod',mask_eroded.astype(np.float32))
                            # cv2.imshow('mask',mask.astype(np.float32))
                            # cv2.waitKey(0)
                            
                            # mask = mask_eroded.astype(np.bool)
                            img_copy = np.zeros_like(img)
                            img_copy[mask] = img[mask]
                            crop = img_copy[top:bottom, left:right].copy()
                            if icp:
                                depth_copy = np.zeros_like(depth)
                                depth_copy[mask] = depth[mask]
                                depth_crop = depth_copy[top:bottom, left:right]

                    #print bb
                    ## uebler hack: remove!
                    # xmin, ymin, xmax, ymax = bb
                    # x, y, w, h = xmin, ymin, xmax-xmin, ymax-ymin 
                    # bb = np.array([x, y, w, h])
                    ##



                    resized_crop = cv2.resize(crop, (H_AE,W_AE))
                    if icp:
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
                print((depth_path,' not found'))
    
    else:    
        print(((noof_imgs,) + p['test_im_size'][::-1] + (3,)))
        imgs = np.empty((noof_imgs,) + p['test_im_size'][::-1] + (3,), dtype=np.uint8)
        print(noof_imgs)
        for view_id in range(noof_imgs):
            img_path = p['test_rgb_mpath'].format(scene_id, view_id)
            try:
                imgs[view_id,...] = cv2.imread(img_path)
            except:
                print((img_path,' not found'))

    return imgs

# def generate_masks(scene_id, eval_args):
#     dataset_name = eval_args.get('DATA','DATASET')
#     cam_type = eval_args.get('DATA','CAM_TYPE')

#     p = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)
    
#     for scene_id in range(1,21):
#         noof_imgs = noof_scene_views(scene_id, eval_args)
#         gts = inout.load_gt(dataset_params['scene_gt_mpath'].format(scene_id))

#         for view_gt in gts:
#             for gt in view_gt:





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
            print(scene_id)

            scene_gts.append(inout.load_yaml(p['scene_gt_mpath'].format(scene_id)))

        for obj in range(1,p['obj_count']+1):
            eval_scenes = set()
            for scene_i,scene_gt in enumerate(scene_gts):
                for view_gt in scene_gt[0]:
                    if view_gt['obj_id'] == obj:
                        eval_scenes.add(scene_i+1)
            obj_scene_dict[obj] = list(eval_scenes)
        np.save(current_file_name,obj_scene_dict)

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

