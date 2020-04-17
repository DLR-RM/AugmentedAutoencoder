import os
import shutil
import torch
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import configparser
import json
import argparse
import glob
import cv2

from utils.utils import *

from Model import Model
from BatchRender import BatchRender
from losses import Loss

from scipy.spatial.transform import Rotation as R

def eqv_dist_points(n):
    pi = np.pi
    sin = np.sin
    cos = np.cos
    n_count = 0
    a = 4*pi/n # 4 pi r^2 / N for r = 1
    d = np.sqrt(a)
    m_theta = int(np.floor(pi/d))
    d_theta = pi/m_theta
    d_phi = a/d_theta

    points = []
    for i in range(m_theta):
        theta = pi*(i + 0.5)/m_theta
        m_phi = int(np.floor(2*pi*sin(theta)/d_phi))
        for j in range(m_phi):
            point = {}
            phi = 2*pi*j/m_phi
            point['spherical'] = [theta, phi]
            point['cartesian'] = [sin(theta)*cos(phi), \
                                  sin(theta)*sin(phi), \
                                  cos(theta)]
            n_count += 1
            points.append(point)
    return points

def plot_points(points, name):
    cart = [point['cartesian'] for point in points]
    print(cart[0:1])
    x, y, z = zip(*cart)
    fig = plt.figure()
    plt.scatter(x, y, z)
    fig.savefig(name, dpi=fig.dpi)
    #plt.show()

def render_point(point, ts, br):
    r = R.from_euler('yz', point['spherical']) # select point wanted for comparison here
    # print(point['spherical'])
    # print(point['cartesian'])
    # print(r.as_matrix())
    # print(r.apply(vec))
    Rs = []
    Rs.append(r.as_matrix())

    images = br.renderBatch(Rs, ts)

    return images

def main():
    global learning_rate, optimizer, views, epoch
    # Read configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name')
    arguments = parser.parse_args()

    cfg_file_path = os.path.join('./experiments', arguments.experiment_name)
    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    # Prepare rotation matrices for multi view loss function
    #eulerViews = json.loads(args.get('Rendering', 'VIEWS'))
    #views = prepareViews(eulerViews)

    # Set the cuda device
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    # Set up batch renderer
    br = BatchRender(args.get('Dataset', 'CAD_PATH'),
                     device,
                     batch_size=args.getint('Training', 'BATCH_SIZE'),
                     render_method=args.get('Rendering', 'SHADER'),
                     image_size=args.getint('Rendering', 'IMAGE_SIZE'))

    output_path = args.get('Training', 'OUTPUT_PATH')
    batch_img_dir = os.path.join(output_path, 'images')
    prepareDir(batch_img_dir)

    # collect points to use
    points = eqv_dist_points(int(args.get('Sampling', 'NUM_SAMPLES')))
    plot_points(points, os.path.join(batch_img_dir, 'test2.png'))
    np.save(os.path.join(output_path, 'points.npy'), points)

    # Testing using existing rotaions, to be replaced
    # data = pickle.load(open(args.get('Dataset', 'TRAIN_DATA_PATH'),'rb'), encoding='latin1')

    t=json.loads(args.get('Rendering', 'T'))
    T = np.array(t, dtype=np.float32)
    ts = []
    #for b in curr_batch:
    #    Rs.append(data['Rs'][b])
    #    ts.append(T.copy())
    # Rs.append(data['Rs'][0])
    # print(Rs)
    ts.append(T.copy())

    ref_image = render_point(points[0], ts, br)

    i = 0
    losses = []
    for point in points:
        image = render_point(point, ts, br)

        prepareDir(output_path)
        shutil.copy(cfg_file_path, os.path.join(output_path, cfg_file_path.split('/')[-1]))

        gt_img = (image[0]).detach().cpu().numpy()

        im = np.array(gt_img * 255, dtype = np.uint8)
        threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        if args.getboolean('Training', 'SAVE_IMAGES'):
            cv2.imwrite(os.path.join(batch_img_dir, '{}.png'.format(i)), im)

        loss, batch_loss, gt_images, predicted_images = Loss(image, ref_image, br, ts, 0, 1, loss_method='loss_landscape')
        loss = (loss).detach().cpu().numpy()
        losses.append(loss)
        i += 1

    np.save(os.path.join(output_path, 'losses.npy'), losses)

if __name__ == '__main__':
    main()
