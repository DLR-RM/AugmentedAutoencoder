import cv2
import numpy as np
import argparse
import configparser

import m3vision

parser = argparse.ArgumentParser()
parser.add_argument('m3_config_path', type=str)
args = parser.parse_args()

m3_args = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes="#")
m3_args.read(args.m3_config_path)

####### this is our new m3vision api:
# detector_method = m3_args.get('methods','object_detector')
# detector = m3vision.get_detector(detector_method, args.m3_config_path)
#######

####### this is manually hacked to load a frozen model without non-maxima suppression, change the path to your checkpoint: 
from retinanet_detector import RetinaNetDetector
detector = RetinaNetDetector('/net/rmc-lx0314/home_local/sund_ma/tmp/resnet50_csv_27_frozen2.h5', args.m3_config_path)
#######

dummy_img = np.ones((480,640,3), dtype=np.float32) #rgb change to your images
proc_img, _ = detector.preprocess_image(dummy_img, 'rgb', dummy_img.dtype)

#detection happens here:
dets = detector.process(proc_img)
print(dets)
