import cv2
import numpy as np
import os
import argparse
import configparser
import glob
import yaml
import time
from tqdm import tqdm
import png

from bop_toolkit_lib import dataset_params, inout
from auto_pose.visualization.render_pose import PoseVisualizer
from auto_pose.m3_interface.m3_interfaces import BoundingBox
from auto_pose.m3_interface.mp_pose_estimator import MPPoseEstimator
from auto_pose.ae.utils import get_dataset_path


def load_depth(path, mul=1., shift=(-4, -1)):
    r = png.Reader(filename=path)
    im = np.vstack(list(map(np.uint16, r.asDirect()[2])))
    im = (im * mul).astype(np.float32)
    im = np.roll(im, shift, (0, 1))
    return im

def load_depth_tif(path):
  """Loads a depth image from a file.

  :param path: Path to the depth image file to load.
  :return: ndarray with the loaded depth image.
  """
  import imageio
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

parser.add_argument('--result_folder', type=str, default='/net/rmc-lx0314/home_local/sund_ma/src/foreign_packages/bop/bop_results/bop_challenge_2019')
parser.add_argument('--datasets_path', type=str, default='/net/rmc-lx0314/home_local/sund_ma/src/foreign_packages/bop/datasets/bop')
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
pose_refiner_method = m3_args.get('methods', 'object_pose_refiner')

mask_inf_time = m3_args.getfloat('mask_rcnn', 'inference_time')
mask_base = m3_args.get('mask_rcnn', 'path_to_masks')
shift_depth = (-1,-4) if dataset_name == 'tless' else (0,0)
# if detector_method:
    # from retinanet_detector import RetinaNetDetector
    # detector = m3vision.get_detector(detector_method, m3_config_path)
    # detector = RetinaNetDetector('/net/rmc-lx0314/home_local/sund_ma/tmp/resnet50_csv_27_frozen2.h5', m3_config_path)

dp_split = dataset_params.get_split_params(datasets_path, dataset_name, split, split_type=split_type)
targets = inout.load_json(os.path.join(dp_split['base_path'], 'test_targets_bop19.json'))

if pose_est_method:
    # mp_pose_estimator = AePoseEstimator(os.path.join(workspace_path,'cfg_m3vision/test_config.cfg'))
    mp_pose_estimator = MPPoseEstimator(m3_config_path)
    
    if pose_refiner_method:
        pose_refiner = m3vision.get_pose_refiner(pose_refiner_method, m3_config_path)

        if dataset_name == 'ycbv':
            for m in list(pose_refiner.models.values()):
                for mp in m['pts']:
                    mp[0] *= 1000
                    mp[1] *= 1000
                    mp[2] *= 1000

if args.vis:
    pose_visualizer = PoseVisualizer(mp_pose_estimator, vertex_scale=1000. if dataset_name=='ycbv' else False)

last_scene_id = -1
last_im_id = -1
bop_results = []
img_pose_ests = []


for target_idx, target in tqdm(enumerate(targets)):

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

            
            # mask_paths = glob.glob(os.path.join(mask_base, '{:02d}/masks/*.npy'.format(scene_id)))
            # print(mask_paths)
            # maskrcnn_scene_masks = [np.unpackbits(np.load(mp)).reshape(dp_split['im_size'][1], dp_split['im_size'][0], -1) for mp in mask_paths]
            # maskrcnn_scene_masks = {k:m for m,k in zip(maskrcnn_scene_masks,sorted(mask_annot.keys()))}
        else:
            mask_paths = os.path.join(dp_split['base_path'], 'test', '{:06d}/mask_visib'.format(scene_id))
            scene_gt_info = inout.load_json(dp_split['scene_gt_info_tpath'].format(scene_id=scene_id))
        last_scene_id = scene_id
    
    if im_id != last_im_id:

        if args.vis and img_pose_ests:
            if pose_refiner_method:
                pose_visualizer.render_poses(img, cam_K, img_pose_ests, img_dets, depth_image=depth_img/depth_img.max())
            else:
                pose_visualizer.render_poses(img, cam_K, img_pose_ests, img_dets)

        if target_idx > 0:
            img_proc_time = mask_inf_time + aae_time + icp_time
            img_bop_res = add_inference_time(img_bop_res, img_proc_time)
            bop_results += img_bop_res
            
        img_bop_res = []
        img_pose_ests = []
        img_dets = []
        last_target_idx = target_idx
        last_im_id = im_id
        aae_time = 0.0
        icp_time = 0.0

        im_p = dp_split['rgb_tpath'].format(scene_id=scene_id, im_id=im_id)
        if dataset_name == 'itodd':
            im_p = im_p.replace('.png','.tif')
            im_p = im_p.replace('rgb','gray')
        img = cv2.imread(im_p)
        if pose_refiner_method:
            # depth_img = inout.load_depth(dp_split['depth_tpath'].format(scene_id=scene_id, im_id=im_id))
            d_p = dp_split['depth_tpath'].format(scene_id=scene_id, im_id=im_id)
            if dataset_name == 'itodd':
                d_p = d_p.replace('.png','.tif')
                print(d_p)
                depth_img = load_depth_tif(d_p)
            else:
                depth_img = load_depth(d_p, shift=shift_depth)
            # depth_img = cv2.imread(d_p)
            depth_img *= scene_camera[im_id]['depth_scale']  # Convert to [mm].
            print((np.histogram(depth_img)))
            
        cam_K = scene_camera[im_id]['cam_K'].copy()

    inst_count = target['inst_count']

    for inst in range(inst_count):
        # try:
        if detector_method == 'mask_rcnn':
            print(im_id)
            print((mask_annot[im_id]))
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

        print((img.shape))
        img_masked = img * inst_mask[..., None].astype(np.uint8)

        x, y, w, h = obj_bb_est
        xmin = float(x) / dp_split['im_size'][0]
        ymin = float(y) / dp_split['im_size'][1]
        xmax = float((x+w)) / dp_split['im_size'][0]
        ymax = float((y+h)) / dp_split['im_size'][1]
        det = [BoundingBox(xmin, ymin, xmax, ymax, classes={obj_id_est:score})]
        

        if pose_refiner_method:
            depth_masked = depth_img * inst_mask

        aae_start = time.time()
        pose_est = mp_pose_estimator.process(det, img_masked, cam_K, mm=True)
        aae_time += (time.time() - aae_start)
        
        #pose refinement
        if pose_refiner_method:
            # cv2.imshow('depth_mask', depth_masked / depth_img.max())
            # cv2.imshow('depth_img', depth_img / depth_img.max())
            # cam_K[0,2] = depth_img.shape[1] / 2
            # cam_K[1,2] = depth_img.shape[0] / 2
            icp_start = time.time()
            pose_est = pose_refiner.process(pose_est, depth_img=depth_img, camK=cam_K, masks=inst_mask)
            icp_time += (time.time() - icp_start)

        if pose_est:
            img_bop_res.append(convert_rmc2bop(pose_est[0], det[0], scene_id, im_id))
        
        img_pose_ests += pose_est
        img_dets += det
        # except:
        #     print((im_id,'not found'))

res_path = os.path.join(result_folder, 'sundermeyer-{}_{}-{}.csv'.format(args.eval_name, dataset_name, split))
inout.save_bop_results(res_path, bop_results)
           
