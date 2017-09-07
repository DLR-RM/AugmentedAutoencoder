# -*- coding: utf-8 -*-

import multiprocessing
import numpy as np
import hashlib
import glob
import os

import progressbar
import pysixd
import cv2

from renderer import meshrenderer
from utils import lazy_property

class BackgroundRendering(multiprocessing.Process):

    def __init__(self, queue, dataset_path, **kwargs):
        multiprocessing.Process.__init__(self)
        self.exit = multiprocessing.Event()
        self._queue = queue
        self._kwargs = kwargs
        self._dataset_path = dataset_path

    def run(self):
        kw = self._kwargs
        H, W = int(kw['h']), int(kw['w'])
        azimuth_range = (0, 2 * np.pi)
        elev_range = (-0.5 * np.pi, 0.5 * np.pi)
        render_dims = eval(kw['render_dims'])
        K = eval(kw['k'])
        K = np.array(K).reshape(3,3)
        clip_near = float(kw['clip_near'])
        clip_far = float(kw['clip_far'])
        crop_factor = float(kw['crop_factor'])
        crop_offset_sigma = float(kw['crop_offset_sigma'])
        t = np.array([0, 0, float(kw['radius'])])

        renderer = meshrenderer.Renderer(
            [kw['model_path']], 
            int(kw['antialiasing']), 
            self._dataset_path, 
            float(kw['vertex_scale'])
        )
        while not self.exit.is_set():
            R = pysixd.transform.random_rotation_matrix()[:3,:3]
            rgb_x, depth_x = renderer.render( 
                obj_id=0,
                W=render_dims[0], 
                H=render_dims[1],
                K=K.copy(), 
                R=R, 
                t=t,
                near=clip_near,
                far=clip_far,
                randomLight=True
            )
            rgb_y, depth_y = renderer.render( 
                obj_id=0,
                W=render_dims[0], 
                H=render_dims[1],
                K=K.copy(), 
                R=R, 
                t=t,
                near=clip_near,
                far=clip_far,
                randomLight=False
            )
            ys, xs = np.nonzero(depth_x > 0)
            try:
                obj_bb = pysixd.misc.calc_2d_bbox(xs, ys, render_dims)
            except ValueError as e:
                print 'Object in Rendering not visible. Have you scaled the vertices to mm?'
                break
            x, y, w, h = obj_bb

            size = int(np.maximum(h, w) * crop_factor)
            left = int(x+w/2-size/2 + np.random.uniform(0, crop_offset_sigma))
            right = int(x+w/2+size/2 + np.random.uniform(0, crop_offset_sigma))
            top = int(y+h/2-size/2 + np.random.uniform(0, crop_offset_sigma))
            bottom = int(y+h/2+size/2 + np.random.uniform(0, crop_offset_sigma))

            rgb_x = rgb_x[top:bottom, left:right]
            depth_x = depth_x[top:bottom, left:right]
            rgb_x = cv2.resize(rgb_x, (W, H))
            depth_x = cv2.resize(depth_x, (W, H))
            mask_x = depth_x < 1.

            ys, xs = np.nonzero(depth_y > 0)
            obj_bb = pysixd.misc.calc_2d_bbox(xs, ys, render_dims)
            x, y, w, h = obj_bb

            size = int(np.maximum(h, w) * crop_factor)
            left = x+w/2-size/2
            right = x+w/2+size/2
            top = y+h/2-size/2
            bottom = y+h/2+size/2

            rgb_y = rgb_y[top:bottom, left:right]
            rgb_y = cv2.resize(rgb_y, (W, H))
            self._queue.put((rgb_x, mask_x, rgb_y))

        renderer.close()

from imgaug.augmenters import *

class Dataset(object):

    def __init__(self, train_mode, dataset_path, **kw):
        h, w = int(kw['h']), int(kw['w'])
        self.shape = (h, w, 3)
        self.background_imgs = glob.glob( kw['background_images_glob'] )

        self._aug = eval(kw['code'])
        if train_mode:
            self._queue = multiprocessing.Queue( int(kw['opengl_render_queue_size']) )
            self.bg_rendering = BackgroundRendering(self._queue, dataset_path, **kw)
            self.bg_rendering.daemon = True
        else:
            self.renderer = meshrenderer.Renderer(
                [kw['model_path']], 
                int(kw['antialiasing']), 
                dataset_path, 
                float(kw['vertex_scale'])
            )
        self._kw = kw

    @lazy_property
    def viewsphere_for_embedding(self):
        kw = self._kw
        num_cyclo = int(kw['num_cyclo'])
        azimuth_range = (0, 2 * np.pi)
        elev_range = (-0.5 * np.pi, 0.5 * np.pi)
        views, _ = pysixd.view_sampler.sample_views(
            int(kw['min_n_views']), 
            float(kw['radius']), 
            azimuth_range, 
            elev_range
        )
        Rs = np.empty( (len(views)*num_cyclo, 3, 3) )
        i = 0
        for view in views:
            for cyclo in np.linspace(0, 2.*np.pi, num_cyclo):
                rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
                Rs[i,:,:] = rot_z.dot(view['R'])
                i += 1
        return Rs

    def render_embedding_images(self, start, end):
        kw = self._kw
        h, w = self.shape[:2]
        azimuth_range = (0, 2 * np.pi)
        elev_range = (-0.5 * np.pi, 0.5 * np.pi)
        radius = float(kw['radius'])
        render_dims = eval(kw['render_dims'])
        K = eval(kw['k'])
        K = np.array(K).reshape(3,3)

        clip_near = float(kw['clip_near'])
        clip_far = float(kw['clip_far'])
        crop_factor = float(kw['crop_factor'])

        t = np.array([0, 0, float(kw['radius'])])
        batch = np.empty( (end-start,)+ self.shape)
        for i, R in enumerate(self.viewsphere_for_embedding[start:end]):
            rgb_y, depth_y = self.renderer.render( 
                obj_id=0,
                W=render_dims[0], 
                H=render_dims[1],
                K=K.copy(), 
                R=R, 
                t=t,
                near=clip_near,
                far=clip_far,
                randomLight=False
            )

            ys, xs = np.nonzero(depth_y > 0)
            obj_bb = pysixd.misc.calc_2d_bbox(xs, ys, render_dims)
            x, y, w, h = obj_bb

            size = int(np.maximum(h, w) * crop_factor)
            left = x+w/2-size/2
            right = x+w/2+size/2
            top = y+h/2-size/2
            bottom = y+h/2+size/2

            rgb_y = rgb_y[top:bottom, left:right]
            batch[i] = cv2.resize(rgb_y, self.shape[:2]) / 255.

        return batch

    @property
    def embedding_size(self):
        return len(self.viewsphere_for_embedding)

    def start(self):
        self.bg_rendering.start()

    def stop(self):
        self.bg_rendering.exit.set()

    def batch(self, batch_size):
        batch_x = np.empty( (batch_size,) + self.shape, dtype=np.float32 )
        batch_y = np.empty( (batch_size,) + self.shape, dtype=np.float32 )
        for i in xrange(batch_size):
            rgb_x, mask, rgb_y = self._queue.get()

            rand_voc = cv2.imread( self.background_imgs[np.random.randint( len(self.background_imgs) )] )
            rand_voc = cv2.resize(rand_voc, self.shape[:2])

            rand_voc = rand_voc
            rgb_x[mask] = rand_voc[mask]

            batch_x[i] = rgb_x
            batch_y[i] = rgb_y

        batch_x = self._aug.augment_images(batch_x)

        batch_x = batch_x / 255.
        batch_y = batch_y / 255.

        return (batch_x, batch_y)