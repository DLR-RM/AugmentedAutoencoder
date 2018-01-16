import cv2
import tensorflow as tf
import numpy as np
import glob
import imageio
import os
import ConfigParser

from ae import factory, utils
from eval import eval_utils

import argparse
from sixd_toolkit.params import dataset_params 
from sixd_toolkit.pysixd import inout


# def generate_tless_crops(test_img, bboxes, scenes, objects, visib_gt = None):


#     pad_factor = 1.2
#     estimate_bbs = False

#     test_img_crops = {}#
#     bb_scores = {}
#     H,W = test_img.shape[1:3]

#     test_img_crops = {}
#     bb_scores = {}
#     visibilities = {}

#     for bbox_idx,bbox in enumerate(bboxes):
#         if bbox['obj_id'] in objects:
#             bb = bbox['obj_bb']
#             obj_id = bbox['obj_id']
#             bb_score = bbox['score'] if estimate_bbs else 1.0
            
                   
#             #from ymin,xmin,ymax,xmax [0,1] to xmin,xmax, W, H [px]
#             if estimate_bbs:
#                 x, y, h, w = int(bb[1]*W),int(bb[0]*H),int((bb[3]-bb[1])*W),int((bb[2]-bb[0])*H)
#             else:
#                 x, y, h, w = bb
#             size = int(np.maximum(h,w) * pad_factor)
#             left = np.max([x+w/2-size/2, 0])
#             right = np.min([x+w/2+size/2, W])
#             top = np.max([y+h/2-size/2, 0])
#             bottom = np.min([y+h/2+size/2, H])
#             print top, bottom, left, right
#             # if single_instance:
#             #     if bb_scores[0][obj_id] < bb_score
#             #         test_img_crops[0][obj_id] = [img[top:bottom, left:right]]
#             #         bb_scores[0][obj_id] = [bb_score]
#             # else:
#             test_img_crops.setdefault(obj_id,[]).append(test_img[top:bottom, left:right])
#             bb_scores.setdefault(obj_id,[]).append(bb_score)

#     return (test_img_crops, bb_scores)

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("-f", "--file_str", required=True)
parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

file_str = arguments.file_str
gt_bb = arguments.gt_bb

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
train_cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
eval_cfg_file_path = utils.get_eval_config_file_path(workspace_path)
train_args = ConfigParser.ConfigParser()
eval_args = ConfigParser.ConfigParser()
train_args.read(train_cfg_file_path)
eval_args.read(eval_cfg_file_path)

codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)

with tf.Session() as sess:
    factory.restore_checkpoint(sess, tf.train.Saver(), experiment_name, experiment_group)

    files = glob.glob(os.path.join(str(file_str),'*'))

    p = dataset_params.get_dataset_params('tless')
    bb_gt = inout.load_gt(p['scene_gt_mpath'].format(2))


    test_img_crops, bbs, bb_scores, visibilities = eval_utils.get_gt_scene_crops(2, eval_args, train_args)


    # for i,img_path in enumerate(files):
    noof_scene_views = eval_utils.noof_scene_views(2, eval_args)
    obj_id = eval(eval_args.get('DATA','OBJECTS'))[0]

    for view in xrange(noof_scene_views):
        print '#'*100
        test_crops, test_bbs, test_scores = eval_utils.select_img_crops(test_img_crops[view][obj_id], bbs[view][obj_id],
                                                                        bb_scores[view][obj_id], visibilities[view][obj_id], eval_args)


        for test_crop, test_bb, test_score in zip(test_crops, test_bbs, test_scores):
            # img = cv2.imread(img_path)
            # if gt_bb:
            #     img = cv2.resize(img[5][0],(128,128))
            # else:
            #     h,w = img.shape[:2]
            #     short = np.min([h,w])
            #     img = cv2.resize(img[int(h/2-short/2):int(h/2+short/2),int(w/2-short/2):int(w/2+short/2)],(128,128))

            R = codebook.nearest_rotation(sess, test_crop)
            R2 = codebook.nearest_rotation(sess, test_crop/255.)

            pred_view = dataset.render_rot( R ,downSample = 1)
            pred_view2 = dataset.render_rot( R2 ,downSample = 1)
            
            
            cv2.imshow('resized img', test_crop/255.)
            cv2.imshow('pred_view', pred_view)
            cv2.imshow('pred_view2', pred_view2)
            print R
            cv2.waitKey(1)

if __name__ == '__main__':
    main()
