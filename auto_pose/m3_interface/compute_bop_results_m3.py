import cv2
import numpy as np
import os
import argparse
import configparser
import glob
import yaml
import time
# from tqdm import tqdm
import png
import imageio
import json

from bop_toolkit_lib import dataset_params, inout, pycoco_utils
from auto_pose.m3_interface.m3_interfaces import BoundingBox
from auto_pose.visualization.render_pose import PoseVisualizer
from auto_pose.m3_interface.ae_pose_estimator import AePoseEstimator


def load_depth(path, mul=1., shift=(-4, -1)):
    r = png.Reader(filename=path)
    im = np.vstack(map(np.uint16, r.asDirect()[2]))
    im = (im * mul).astype(np.float32)
    im = np.roll(im, shift, (0, 1))
    return im

def load_depth_tif(path):
  """Loads a depth image from a file.

  :param path: Path to the depth image file to load.
  :return: ndarray with the loaded depth image.
  """
  d = imageio.imread(path)
  return d.astype(np.float32)

    
def convert_rmc2bop(pose_est, det, scene_id, im_id):
    b_res = {}
    b_res['time'] = -1
    b_res['scene_id'] = scene_id
    b_res['im_id'] = im_id
    b_res['obj_id'] = max(det.classes)
    b_res['score'] = max(det.classes, key=det.classes.get)
    b_res['R'] = pose_est.trafo[:3,:3]
    b_res['t'] = pose_est.trafo[:3,3]
    return b_res

def add_inference_time(res, img_proc_time):
    for r in res:
        r['time'] = img_proc_time
    return res
    

parser = argparse.ArgumentParser()
parser.add_argument('m3_config_path', type=str)
parser.add_argument("-vis", action='store_true', default=False)
parser.add_argument('--eval_name', type=str)

parser.add_argument('--result_folder', type=str, required=True)
parser.add_argument('--datasets_path', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--split_type', type=str, default=None)
args = parser.parse_args()

datasets_path = args.datasets_path
result_folder = args.result_folder
dataset_name = args.dataset_name
split = args.split
split_type = args.split_type
m3_config_path = args.m3_config_path

m3_args = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes="#")
m3_args.read(m3_config_path)

detector_method = m3_args.get('methods','object_detector')
pose_est_method = m3_args.get('methods', 'object_pose_estimator')

mask_inf_time = m3_args.getfloat(detector_method, 'inference_time')
mask_base = m3_args.get(detector_method, 'path_to_masks')
shift_depth = (-1,-4) if dataset_name == 'tless' else (0,0)

dp_split = dataset_params.get_split_params(datasets_path, dataset_name, split, split_type=split_type)
targets = inout.load_json(os.path.join(dp_split['base_path'], 'test_targets_bop19.json'))

if pose_est_method:
    ae_pose_est = AePoseEstimator(m3_config_path)

if args.vis:
    pose_visualizer = PoseVisualizer(ae_pose_est)

last_scene_id = -1
last_im_id = -1
bop_results = []
img_pose_ests = []
aae_time = 0.0

for target_idx, target in enumerate(targets):
    scene_id = target['scene_id']
    obj_id = target['obj_id']
    im_id = target['im_id']

    if scene_id != last_scene_id:
        scene_camera = inout.load_scene_camera(dp_split['scene_camera_tpath'].format(scene_id=scene_id))
        if detector_method == 'mask_rcnn':
            with open(os.path.join(mask_base, '{:02d}/mask_rcnn_predict.yml'.format(scene_id)), 'r') as f:
                mask_annot = yaml.load(f)                
            maskrcnn_scene_masks = {}
            for k in sorted(mask_annot.keys()):
                mp_path = os.path.join(mask_base, '{:02d}/masks/{}.npy'.format(scene_id,k))
                mask = np.unpackbits(np.load(mp_path)).reshape(dp_split['im_size'][1], dp_split['im_size'][0], -1)
                maskrcnn_scene_masks[k] = mask
        else:
            mask_paths = os.path.join(dp_split['base_path'], 'test', '{:06d}/mask_visib'.format(scene_id))
            scene_gt_info = inout.load_json(dp_split['scene_gt_info_tpath'].format(scene_id=scene_id))
        last_scene_id = scene_id
    
    if im_id != last_im_id:

        if args.vis and img_pose_ests:
            pose_visualizer.render_poses(img, cam_K, img_pose_ests, img_dets)

        if aae_time > 0:
            img_proc_time = mask_inf_time + aae_time + icp_time
            img_bop_res = add_inference_time(img_bop_res, img_proc_time)
            bop_results += img_bop_res
            
        img_bop_res = []
        img_pose_ests = []
        img_dets = []
        last_im_id = im_id
        aae_time = 0.0
        icp_time = 0.0

        im_p = dp_split['rgb_tpath'].format(scene_id=scene_id, im_id=im_id)
        if dataset_name == 'itodd':
            im_p = im_p.replace('.png','.tif')
            im_p = im_p.replace('rgb','gray')
        img = cv2.imread(im_p)            
        cam_K = scene_camera[im_id]['cam_K'].copy()

    inst_count = target['inst_count']
    # inst_count = 1

    for inst in range(inst_count):
        if detector_method == 'mask_rcnn':
            print((mask_annot[im_id]))
            print(im_id)
            score = mask_annot[im_id][obj_id][inst]['score']
            if score < 0:
                continue
            obj_bb_est = mask_annot[im_id][obj_id][inst]['obj_bb']
            obj_id_est = mask_annot[im_id][obj_id][inst]['obj_id']
            chan_id = mask_annot[im_id][obj_id][inst]['np_channel_id']
            mask = maskrcnn_scene_masks[im_id]
            inst_mask = mask[:,:, chan_id]
        else:
            obj_bb_est = scene_gt_info[str(im_id)][inst]['bbox_obj']
            obj_id_est = obj_id
            score = 1.0
            mask_p = os.path.join(mask_paths,'{:06d}_{:06d}.png'.format(im_id, inst))
            inst_mask = inout.load_depth(mask_p)/255.

        img_masked = img * inst_mask[..., None].astype(np.uint8)

        x, y, w, h = obj_bb_est
        xmin = float(x) / dp_split['im_size'][0]
        ymin = float(y) / dp_split['im_size'][1]
        xmax = float((x+w)) / dp_split['im_size'][0]
        ymax = float((y+h)) / dp_split['im_size'][1]
        det = [BoundingBox(xmin, ymin, xmax, ymax, classes={obj_id_est:score})]

        aae_start = time.time()
        pose_est = ae_pose_est.process(det, img_masked, cam_K, mm=True)
        aae_time += (time.time() - aae_start)

        if pose_est:
            img_bop_res.append(convert_rmc2bop(pose_est[0], det[0], scene_id, im_id))
        
        img_pose_ests += pose_est
        img_dets += det

img_proc_time = mask_inf_time + aae_time + icp_time
img_bop_res = add_inference_time(img_bop_res, img_proc_time)
bop_results += img_bop_res

res_path = os.path.join(result_folder, 'sundermeyer-{}_{}-{}.csv'.format(args.eval_name, dataset_name, split))
inout.save_bop_results(res_path, bop_results)
           