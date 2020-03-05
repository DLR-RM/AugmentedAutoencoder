import torch
import numpy as np
import math
import cv2
import csv
import matplotlib.pyplot as plt
import os

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def list2file(input_list, file_name):
    with open(file_name, 'w') as f:
        wr = csv.writer(f, delimiter='\n')
        wr.writerow(input_list)

def prepareDir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def plotLoss(loss, file_name):
    fig = plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.plot(loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    fig.tight_layout()
    fig.savefig(file_name, dpi=fig.dpi)
    plt.close()

def calcMeanVar(br, data, device, t):
    num_samples = len(data["codes"])
    data_indeces = np.arange(num_samples)
    np.random.shuffle(data_indeces)
    batch_size = br.batch_size

    all_data = []
    for i,curr_batch in enumerate(batch(data_indeces, batch_size)):
        # Render the ground truth images
        T = np.array(t, dtype=np.float32)
        Rs = []
        ts = []
        for b in curr_batch:
            Rs.append(data["Rs"][b])
            ts.append(T.copy())
        gt_images = br.renderBatch(Rs, ts)
        all_data.append(gt_images)
        print("Step: {0}/{1}".format(i,round(num_samples/batch_size)))
    result = torch.cat(all_data)
    print(torch.mean(result))
    print(torch.std(result))
    return torch.mean(result), torch.std(result)

def plotView(currView, numViews, vmin, vmax, groundtruth, predicted, predicted_pose, loss, batch_size):
    plt.subplot(3, numViews, currView+1)
    plt.imshow(groundtruth[currView*batch_size].detach().cpu().numpy(),
               vmin=vmin, vmax=vmax)
    plt.title("View {0} - GT".format(currView))
    
    plt.subplot(3, numViews, currView+(1+numViews))
    plt.imshow(predicted[currView*batch_size].detach().cpu().numpy(),
               vmin=vmin, vmax=vmax)
    if(currView == 0):
        plt.title("Predicted: " + np.array2string((predicted_pose[currView*batch_size]).detach().cpu().numpy(),precision=2))
    else:
        plt.title("Predicted")
    
    loss_contrib = np.abs((groundtruth[currView*batch_size]).detach().cpu().numpy() - (predicted[currView*batch_size]).detach().cpu().numpy())
    plt.subplot(3, numViews, currView+(1+2*numViews))
    plt.imshow(loss_contrib) #, vmin=vmin, vmax=vmax)
    plt.title("Loss: {0}".format((loss[currView*batch_size]).detach().cpu().numpy()))

# Convert quaternion to rotation matrix
# from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    #norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    #norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    #w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    norm_quat = quat/quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]
    
    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

# -*- coding: utf-8 -*-
# transform.py

# Copyright (c) 2006-2015, Christoph Gohlke
# Copyright (c) 2006-2015, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

def random_quaternion(rand=None):
    """Return uniform random unit quaternion.

    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.

    >>> q = random_quaternion()
    >>> numpy.allclose(1, vector_norm(q))
    True
    >>> q = random_quaternion(numpy.random.random(3))
    >>> len(q.shape), q.shape[0]==4
    (1, True)

    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array([np.cos(t2)*r2, np.sin(t1)*r1,
                        np.cos(t1)*r1, np.sin(t2)*r2])



def q2m(quat):
    _EPS = np.finfo(float).eps * 4.0
    q = torch.tensor(quat, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]]) 
    
def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def random_rotation_matrix(rand=None):
    """Return uniform random rotation matrix.

    rand: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quaternion.

    >>> R = random_rotation_matrix()
    >>> np.allclose(np.dot(R.T, R), np.identity(4))
    True

    """
    return quaternion_matrix(random_quaternion(rand))


def calc_2d_bbox(xs, ys, im_size):
    bbTL = (max(xs.min() - 1, 0),
            max(ys.min() - 1, 0))
    bbBR = (min(xs.max() + 1, im_size[0] - 1),
            min(ys.max() + 1, im_size[1] - 1))
    return [bbTL[0], bbTL[1], bbBR[0] - bbTL[0], bbBR[1] - bbTL[1]]


def extract_square_patch(scene_img, bb_xywh, pad_factor=1.2,resize=(128,128),
                         interpolation=cv2.INTER_NEAREST,black_borders=False):

        x, y, w, h = np.array(bb_xywh).astype(np.int32)
        size = int(np.maximum(h, w) * pad_factor)

        left = int(np.maximum(x+w/2-size/2, 0))
        right = int(np.minimum(x+w/2+size/2, scene_img.shape[1]))
        top = int(np.maximum(y+h/2-size/2, 0))
        bottom = int(np.minimum(y+h/2+size/2, scene_img.shape[0]))

        scene_crop = scene_img[top:bottom, left:right].copy()

        if black_borders:
            scene_crop[:(y-top),:] = 0
            scene_crop[(y+h-top):,:] = 0
            scene_crop[:,:(x-left)] = 0
            scene_crop[:,(x+w-left):] = 0

        scene_crop = cv2.resize(scene_crop, resize) #, interpolation = interpolation)
        return scene_crop
