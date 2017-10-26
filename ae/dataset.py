# -*- coding: utf-8 -*-

import multiprocessing
import numpy as np
import hashlib
import glob
import os

import progressbar
from pysixd_stuff import transform
from pysixd_stuff import view_sampler

import cv2


from renderer import meshrenderer
from utils import lazy_property

from imgaug.augmenters import *

class Dataset(object):

    def __init__(self, train_mode, dataset_path, **kw):
        
        h, w = int(kw['h']), int(kw['w'])
        self.shape = (h, w, 3)
        self.noof_training_imgs = int(kw['noof_training_imgs'])

        self.bg_img_paths = glob.glob( kw['background_images_glob'] )
        self.noof_bg_imgs = min(int(kw['noof_bg_imgs']), len(self.bg_img_paths))
        
        self._aug = eval(kw['code'])
        self._kw = kw

        self.renderer = meshrenderer.Renderer(
           [kw['model_path']], 
           int(kw['antialiasing']), 
           dataset_path, 
           float(kw['vertex_scale'])
        )

        self.train_x = np.empty( (self.noof_training_imgs,) + self.shape, dtype=np.uint8 )
        self.mask_x = np.empty( (self.noof_training_imgs,) + self.shape[:2], dtype= bool)
        self.train_y = np.empty( (self.noof_training_imgs,) + self.shape, dtype=np.uint8 )
        self.bg_imgs = np.empty( (self.noof_bg_imgs,) + self.shape, dtype=np.uint8 )


    @lazy_property
    def viewsphere_for_embedding(self):
        kw = self._kw
        num_cyclo = int(kw['num_cyclo'])
        azimuth_range = (0, 2 * np.pi)
        elev_range = (-0.5 * np.pi, 0.5 * np.pi)
        views, _ = view_sampler.sample_views(
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

    def get_training_images(self, dataset_path, args):

        current_config_hash = hashlib.md5(str(args.items('Dataset')+args.items('Paths'))).hexdigest()
        current_file_name = os.path.join(dataset_path, current_config_hash + '.npz')
        print current_file_name
        if os.path.exists(current_file_name):
            training_data = np.load(current_file_name)
            self.train_x = training_data['train_x'].astype(np.uint8)
            self.mask_x = training_data['mask_x']
            print np.max(self.mask_x[3])

            self.train_y = training_data['train_y'].astype(np.uint8)
        else:
            self.render_training_images()
            np.savez(current_file_name, train_x = self.train_x, mask_x = self.mask_x, train_y = self.train_y)

    def load_bkgd_images(self, dataset_path):
        current_config_hash = hashlib.md5(str(self.shape) + str(self.bg_img_paths)).hexdigest()
        current_file_name = os.path.join(dataset_path, current_config_hash + '.npy')
        if os.path.exists(current_file_name):
            self.bg_imgs = np.load(current_file_name)
        else:
            file_list = self.bg_img_paths[:self.noof_bg_imgs]

            for j,fname in enumerate(file_list):
                print 'loading bg img %s/%s' % (j,self.noof_bg_imgs)
                rgb = cv2.imread(fname)
                rgb = cv2.resize(rgb, self.shape[:2])

                self.bg_imgs[j] = rgb
            np.save(current_file_name,self.bg_imgs)
        print 'loaded %s bg images' % self.noof_bg_imgs


    def render_training_images(self):
        kw = self._kw
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

        for i in np.arange(self.noof_training_imgs):

            print '%s/%s' % (i,self.noof_training_imgs)
            import time
            start_time = time.time()
            R = transform.random_rotation_matrix()[:3,:3]
            rgb_x, depth_x = self.renderer.render( 
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
            render_time = time.time() - start_time

            ys, xs = np.nonzero(depth_x > 0)
            try:
                obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)
            except ValueError as e:
                print 'Object in Rendering not visible. Have you scaled the vertices to mm?'
                break
            x, y, w, h = obj_bb

            size = int(np.maximum(h, w) * crop_factor)
            left = int(x+w/2-size/2 + np.random.uniform(-crop_offset_sigma, crop_offset_sigma))
            right = int(x+w/2+size/2 + np.random.uniform(-crop_offset_sigma, crop_offset_sigma))
            top = int(y+h/2-size/2 + np.random.uniform(-crop_offset_sigma, crop_offset_sigma))
            bottom = int(y+h/2+size/2 + np.random.uniform(-crop_offset_sigma, crop_offset_sigma))

            rgb_x = rgb_x[top:bottom, left:right]
            depth_x = depth_x[top:bottom, left:right]
            rgb_x = cv2.resize(rgb_x, (W, H))
            depth_x = cv2.resize(depth_x, (W, H))

            mask_x = depth_x < 1.

            ys, xs = np.nonzero(depth_y > 0)
            obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)
            x, y, w, h = obj_bb

            size = int(np.maximum(h, w) * crop_factor)
            left = x+w/2-size/2
            right = x+w/2+size/2
            top = y+h/2-size/2
            bottom = y+h/2+size/2

            rgb_y = rgb_y[top:bottom, left:right]
            rgb_y = cv2.resize(rgb_y, (W, H))

            self.train_x[i] = rgb_x.astype(np.uint8)
            self.mask_x[i] = mask_x
            self.train_y[i] = rgb_y.astype(np.uint8)

            print 'rendertime ', render_time, 'processing ', time.time() - start_time



    def render_rot(self, R, downSample = 1):
        kw = self._kw
        h, w = self.shape[:2]
        azimuth_range = (0, 2 * np.pi)
        elev_range = (-0.5 * np.pi, 0.5 * np.pi)
        radius = float(kw['radius'])
        render_dims = eval(kw['render_dims'])
        K = eval(kw['k'])
        K = np.array(K).reshape(3,3)
        K[:2,:] = K[:2,:] / downSample

        clip_near = float(kw['clip_near'])
        clip_far = float(kw['clip_far'])
        crop_factor = float(kw['crop_factor'])

        t = np.array([0, 0, float(kw['radius'])])
        rgb_y, depth_y = self.renderer.render( 
            obj_id=0,
            W=render_dims[0]/downSample, 
            H=render_dims[1]/downSample,
            K=K.copy(), 
            R=R, 
            t=t,
            near=clip_near,
            far=clip_far,
            randomLight=False
        )

        ys, xs = np.nonzero(depth_y > 0)
        obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)
        x, y, w, h = obj_bb

        size = int(np.maximum(h, w) * crop_factor)
        left = x+w/2-size/2
        right = x+w/2+size/2
        top = y+h/2-size/2
        bottom = y+h/2+size/2

        rgb_y = rgb_y[top:bottom, left:right]
        return cv2.resize(rgb_y, self.shape[:2]) / 255.


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
            obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)
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

    def batch(self, batch_size):
        # import time
        # start_time = time.time()

        batch_x = np.empty( (batch_size,) + self.shape, dtype=np.uint8 )
        batch_y = np.empty( (batch_size,) + self.shape, dtype=np.uint8 )
        
        rand_idcs = np.random.choice(self.noof_training_imgs, batch_size, replace=False)
        rand_idcs_bg = np.random.choice(self.noof_bg_imgs, batch_size, replace=False)
        

        for i in xrange(batch_size):

            rgb_x, mask, rgb_y = self.train_x[rand_idcs[i]], self.mask_x[rand_idcs[i]], self.train_y[rand_idcs[i]]
            rand_voc = self.bg_imgs[rand_idcs_bg[i]]

            rgb_x[mask] = rand_voc[mask]

            batch_x[i] = rgb_x
            batch_y[i] = rgb_y


        # augm_time = time.time()
        # print 'rendering: ', time.time()-start_time

        batch_x = self._aug.augment_images(batch_x)

        batch_x = batch_x / 255.
        batch_y = batch_y / 255.
        
        # print 'augmentation:', time.time()-augm_time 

        return (batch_x, batch_y)
