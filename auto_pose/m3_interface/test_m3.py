import cv2
import numpy as np
import os
import argparse

from auto_pose.m3_interface.m3_interfaces import BoundingBox
from auto_pose.m3_interface.mp_pose_estimator import MPPoseEstimator
from auto_pose.ae.utils import get_dataset_path, load_depth2

dir_name = os.path.dirname(os.path.abspath(__file__))
default_cfg = os.path.join(dir_name, '../ae/cfg_m3vision/m3_template.cfg')

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, required=True)
parser.add_argument("--m3_config_path", type=str, default=default_cfg)
parser.add_argument("-vis", action='store_true', default=False)

args = parser.parse_args()

if os.environ.get('AE_WORKSPACE_PATH') == None:
    print('Please define a workspace path:\n')
    print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
    exit(-1)

img = cv2.imread(args.img_path)
H,W,_ = img.shape

# test camera matrix
camK = np.array([479.99998569488565, 0., 320., 0., 479.99998569488565, 240., 0., 0., 1.]).reshape(3, 3)

# gt boxes and classes (replace with your favorite detector)
classes =  [1, 2, 2, 3, 3, 6, 6, 7, 7]
bboxes = [[276, 143, 131, 73], [195, 233, 32, 26], [550, 321, 41, 23], [181, 182, 61, 54],
        [175, 306, 74, 71],[327, 216, 48, 44],[244, 258, 48, 43],[118, 285, 37, 32],[168, 226, 33, 35]] 

bbs = []
h,w = float(H), float(W)
for b,c in zip(bboxes, classes):
    bbs.append(BoundingBox(xmin=b[0]/w, xmax=(b[0]+b[2])/w , ymin=b[1]/h, ymax=(b[1]+b[3])/h, classes={str(c):1.0}))

# MultiPath Encoder Initialization
mp_pose_estimator = MPPoseEstimator(args.m3_config_path)

# Predict 6-DoF poses
pose_ests = mp_pose_estimator.process(bbs,img,camK)
print(np.array([{p.name:p.trafo} for p in pose_ests]))

# Visualize
if args.vis:
    from auto_pose.visualization.render_pose import PoseVisualizer

    pose_visualizer = PoseVisualizer(mp_pose_estimator)
    pose_visualizer.render_poses(img, camK, pose_ests, bbs)
