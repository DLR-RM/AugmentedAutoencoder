# -*- coding: utf-8 -*-

import numpy as np
import time
import hashlib
import glob
import os
import progressbar
import cv2
import xml.etree.ElementTree as ET

from .pysixd_stuff import transform
from .pysixd_stuff import view_sampler
from .utils import lazy_property


class Dataset(object):

    def __init__(self, dataset_path, **kw):
        
        self.shape = (int(kw['h']), int(kw['w']), int(kw['c']))
        self.noof_training_imgs = int(kw['noof_training_imgs'])
        self.dataset_path = dataset_path

        self.bg_img_paths = glob.glob(kw['background_images_glob'])
        self.noof_bg_imgs = min(int(kw['noof_bg_imgs']), len(self.bg_img_paths))
        
        self._kw = kw
        # self._aug = eval(self._kw['code'])

        self.train_x = np.empty( (self.noof_training_imgs,) + self.shape, dtype=np.uint8 )
        self.mask_x = np.empty( (self.noof_training_imgs,) + self.shape[:2], dtype= bool)
        self.noof_obj_pixels = np.empty( (self.noof_training_imgs,), dtype= bool)
        self.train_y = np.empty( (self.noof_training_imgs,) + self.shape, dtype=np.uint8 )
        self.bg_imgs = np.empty( (self.noof_bg_imgs,) + self.shape, dtype=np.float32 )

        if 'realistic_occlusion' in self._kw and np.float(eval(self._kw['realistic_occlusion'])):
            self.random_syn_masks


    @lazy_property
    def viewsphere_for_embedding(self):
        kw = self._kw
        num_cyclo = int(kw['num_cyclo'])
        azimuth_range = eval(kw['azimuth_range']) if 'azimuth_range' in kw else (0, 2 * np.pi)
        elev_range = eval(kw['elev_range']) if 'elev_range' in kw else (-0.5 * np.pi, 0.5 * np.pi)
        views, _ = view_sampler.sample_views(
            int(kw['min_n_views']), 
            float(kw['radius']), 
            azimuth_range, 
            elev_range
        )
        Rs = np.empty( (len(views)*num_cyclo, 3, 3) )
        i = 0
        
        for view in views:
            for cyclo in np.linspace(0, 2.*np.pi, num_cyclo, endpoint=False):
                rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
                Rs[i,:,:] = rot_z.dot(view['R'])
                i += 1
        return Rs

    @lazy_property
    def renderer(self):
        from auto_pose.meshrenderer import meshrenderer, meshrenderer_phong

        if self._kw['model'] == 'cad':
            renderer = meshrenderer.Renderer(
               eval(str(self._kw['model_path'])), 
               int(self._kw['antialiasing']), 
               self.dataset_path, 
               float(self._kw['vertex_scale'])
            )
        elif self._kw['model'] == 'reconst':
            # print(meshrenderer)
            renderer = meshrenderer_phong.Renderer(
               eval(str(self._kw['model_path'])), 
               int(self._kw['antialiasing']), 
               vertex_tmp_store_folder = self.dataset_path,
               vertex_scale = float(self._kw['vertex_scale'])
            )
        else:
            print('Error: neither cad nor reconst in model path!')
            exit()
        return renderer

    def get_training_images(self, dataset_path, args):
        md5_string = str(args.items('Dataset')+args.items('Paths'))
        md5_string = md5_string.encode('utf-8')
        current_config_hash = hashlib.md5(md5_string).hexdigest()
        current_file_name = os.path.join(dataset_path, current_config_hash + '.npz')

        if os.path.exists(current_file_name):
            training_data = np.load(current_file_name)
            self.train_x = training_data['train_x'].astype(np.uint8)
            self.mask_x = training_data['mask_x']
            self.train_y = training_data['train_y'].astype(np.uint8)
        else:
            self.render_training_images()
            np.savez(current_file_name, train_x = self.train_x, mask_x = self.mask_x, train_y = self.train_y)
        self.noof_obj_pixels = np.count_nonzero(self.mask_x==0,axis=(1,2))

        print(('loaded %s training images' % len(self.train_x)))

    def get_sprite_training_images(self, train_args):
        
        dataset_path= train_args.get('Paths','MODEL_PATH')
        dataset_zip = np.load(dataset_path)

        # print('Keys in the dataset:', dataset_zip.keys())
        imgs = dataset_zip['imgs']
        latents_values = dataset_zip['latents_values']
        latents_classes = dataset_zip['latents_classes']
        metadata = dataset_zip['metadata'][()]

        latents_sizes = metadata['latents_sizes']
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1,])))

        latents_classes_heart = latents_classes[:245760]
        latents_classes_heart_rot = latents_classes_heart.copy()

        latents_classes_heart_rot[:, 0] = 0
        latents_classes_heart_rot[:, 1] = 0
        latents_classes_heart_rot[:, 2] = 5
        latents_classes_heart_rot[:, 4] = 16
        latents_classes_heart_rot[:, 5] = 16

        def latent_to_index(latents):
          return np.dot(latents, latents_bases).astype(int)

        indices_sampled = latent_to_index(latents_classes_heart_rot)
        imgs_sampled_rot = imgs[indices_sampled]
        indices_sampled = latent_to_index(latents_classes_heart)
        imgs_sampled_all = imgs[indices_sampled]

        self.train_x = np.expand_dims(imgs_sampled_all, 3)*255
        self.train_y = np.expand_dims(imgs_sampled_rot, 3)*255


    # def get_embedding_images(self, dataset_path, args):

    #     current_config_hash = hashlib.md5(str(args.items('Embedding') + args.items('Dataset')+args.items('Paths'))).hexdigest()
    #     current_file_name = os.path.join(dataset_path, current_config_hash + '.npz')

    #     if os.path.exists(current_file_name):
    #         embedding_data = np.load(current_file_name)
    #         self.embedding_data = embedding_data.astype(np.uint8)
    #     else:
    #         self.render_embedding_images()
    #         np.savez(current_file_name, train_x = self.train_x, mask_x = self.mask_x, train_y = self.train_y)
    #     print 'loaded %s training images' % len(self.train_x)

    @property
    def embedding_size(self):
        return len(self.viewsphere_for_embedding)

    def load_bg_images(self, dataset_path):
        md5_string = str(str(self.shape) + str(self.noof_bg_imgs) + str(self._kw['background_images_glob']))
        md5_string = md5_string.encode('utf-8')
        current_config_hash = hashlib.md5(md5_string).hexdigest()
        current_file_name = os.path.join(dataset_path, current_config_hash +'.npy')
    

        if os.path.exists(current_file_name):
            self.bg_imgs = np.load(current_file_name)
        else:
            
            file_list = self.bg_img_paths[:self.noof_bg_imgs]
            
            from random import shuffle
            shuffle(file_list)


            for j,fname in enumerate(file_list):
                print(('loading bg img %s/%s' % (j,self.noof_bg_imgs)))
                bgr = cv2.imread(fname)
                H,W = bgr.shape[:2]
                y_anchor = int(np.random.rand() * (H-self.shape[0]))
                x_anchor = int(np.random.rand() * (W-self.shape[1]))
                # bgr = cv2.resize(bgr, self.shape[:2])                    
                bgr = bgr[y_anchor:y_anchor+self.shape[0],x_anchor:x_anchor+self.shape[1],:]
                if bgr.shape[0]!=self.shape[0] or bgr.shape[1]!=self.shape[1]:
                    continue
                if self.shape[2] == 1:
                    bgr = cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
                self.bg_imgs[j] = bgr
            np.save(current_file_name,self.bg_imgs)


        import tensorflow as tf
        self.bg_imgs = self.bg_imgs/255.
        print(('loaded %s bg images' % self.noof_bg_imgs))


    def filter_voc_paths(self, bg_paths):
        model_set = set()
        m_paths = eval(str(self._kw['model_path']))
        for path in m_paths:
            if 'ModelNet' in path:
                model_set.add(os.path.basename(path).split('_')[0])
        if len(model_set)==0:
            return bg_paths

        filtered_bg_paths = []
        xml_path_glob = os.path.join(os.path.dirname(os.path.dirname(bg_paths[0])),'Annotations','*.xml')
        xml_glob = glob.glob(xml_path_glob)
        dic = {}
        for xml,bg_p in zip(xml_glob,bg_paths):
            root = ET.parse(xml).getroot()
            if root.find('object').find('name').text in model_set:
                filtered_bg_paths.append(bg_p)
        self.noof_bg_imgs = len(filtered_bg_paths)
        return filtered_bg_paths

    def render_rot(self, R, K=None, downSample = 1, obj_id = 0, t=None, return_bb = False,return_orig = False):
        kw = self._kw
        h, w = self.shape[:2]
        radius = float(kw['radius'])
        render_dims = eval(kw['render_dims'])
        if K is None:
            K = eval(kw['k'])
            K = np.array(K).reshape(3,3)


        K[:2,2] = K[:2,2] / downSample


        clip_near = float(kw['clip_near'])
        clip_far = float(kw['clip_far'])
        pad_factor = float(kw['pad_factor'])

        if t is None:
            t = np.array([0, 0, float(kw['radius'])])

        bgr_y, depth_y = self.renderer.render( 
            obj_id=obj_id,
            W=render_dims[0]//downSample, 
            H=render_dims[1]//downSample,
            K=K.copy(), 
            R=R, 
            t=t,
            near=clip_near,
            far=clip_far,
            random_light=False
        )

        ys, xs = np.nonzero(depth_y > 0)
        obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)
        bgr_y_cropped = self.extract_square_patch(bgr_y, obj_bb, pad_factor)
        
        if downSample > 1:
            obj_bb[0] = obj_bb[0]*downSample+obj_bb[2]/downSample
            obj_bb[1] = obj_bb[1]*downSample+obj_bb[3]/downSample

        
        if return_bb:
            if return_orig:
                return bgr_y_cropped, obj_bb,bgr_y
            return bgr_y_cropped, obj_bb
        else:
            return bgr_y_cropped



    def render_training_images(self, serialize_func = None, obj_id=0, tfrec_writer=None):
        kw = self._kw
        H, W = int(kw['h']), int(kw['w'])
        render_dims = eval(kw['render_dims'])
        K = eval(kw['k'])
        K = np.array(K).reshape(3,3)
        clip_near = float(kw['clip_near'])
        clip_far = float(kw['clip_far'])
        pad_factor = float(kw['pad_factor'])
        max_rel_offset = float(kw['max_rel_offset'])
        t = np.array([0, 0, float(kw['radius'])])
        lighting = eval(kw['lighting']) if 'lighting' in kw else None

        widgets = ['Rendering Training Data: ', progressbar.Percentage(),

             ' ', progressbar.Bar(),
             ' ', progressbar.Counter(), ' / %s' % self.noof_training_imgs,
             ' ', progressbar.ETA(), ' ']
        bar = progressbar.ProgressBar(maxval=self.noof_training_imgs,widgets=widgets)
        bar.start()

        for i in np.arange(self.noof_training_imgs):
            bar.update(i)

            # print '%s/%s' % (i,self.noof_training_imgs)
            # start_time = time.time()
            R = transform.random_rotation_matrix()[:3,:3]

            if lighting is None:
                bgr_x, depth_x = self.renderer.render( 
                    obj_id=obj_id,
                    W=render_dims[0], 
                    H=render_dims[1],
                    K=K.copy(), 
                    R=R, 
                    t=t,
                    near=clip_near,
                    far=clip_far,
                    random_light=True,
                )
                bgr_y, depth_y = self.renderer.render( 
                    obj_id=obj_id,
                    W=render_dims[0], 
                    H=render_dims[1],
                    K=K.copy(), 
                    R=R, 
                    t=t,
                    near=clip_near,
                    far=clip_far,
                    random_light=False
                )
            else:
                bgr_x, depth_x = self.renderer.render( 
                    obj_id=obj_id,
                    W=render_dims[0], 
                    H=render_dims[1],
                    K=K.copy(), 
                    R=R, 
                    t=t,
                    near=clip_near,
                    far=clip_far,
                    random_light=True,
                    phong = lighting
                )
                bgr_y, depth_y = self.renderer.render( 
                    obj_id=obj_id,
                    W=render_dims[0], 
                    H=render_dims[1],
                    K=K.copy(), 
                    R=R, 
                    t=t,
                    near=clip_near,
                    far=clip_far,
                    random_light=False,
                    phong = lighting
                )




            # cv2.imshow('bgr_y',bgr_y)
            # cv2.waitKey(0)
            ys, xs = np.nonzero(depth_x > 0)
            try:
                obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)
            except ValueError as e:
                print('Object in Rendering not visible. Have you scaled the vertices to mm?')
                break

            x, y, w, h = obj_bb

            rand_trans_x = np.random.uniform(-max_rel_offset, max_rel_offset) * w
            rand_trans_y = np.random.uniform(-max_rel_offset, max_rel_offset) * h

            obj_bb_off = obj_bb + np.array([rand_trans_x,rand_trans_y,0,0])

            bgr_x = self.extract_square_patch(bgr_x, obj_bb_off, pad_factor,resize=(W,H),interpolation = cv2.INTER_NEAREST)
            depth_x = self.extract_square_patch(depth_x, obj_bb_off, pad_factor,resize=(W,H),interpolation = cv2.INTER_NEAREST)
            mask_x = depth_x == 0.

            ys, xs = np.nonzero(depth_y > 0)
            obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)

            bgr_y = self.extract_square_patch(bgr_y, obj_bb, pad_factor,resize=(W,H),interpolation = cv2.INTER_NEAREST)

            if self.shape[2] == 1:
                bgr_x = cv2.cvtColor(np.uint8(bgr_x), cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
                bgr_y = cv2.cvtColor(np.uint8(bgr_y), cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]

            if 'target_bg_color' in kw:
                depth_y = self.extract_square_patch(depth_y, obj_bb, pad_factor, resize=(W, H), interpolation=cv2.INTER_NEAREST)
                mask_y = depth_y == 0.
                bgr_y[mask_y] = eval(kw['target_bg_color'])
                
            train_x = bgr_x.astype(np.uint8)
            mask_x = mask_x
            train_y = bgr_y.astype(np.uint8)

            serialize_func(train_x, mask_x, train_y, writer=tfrec_writer)

            #print 'rendertime ', render_time, 'processing ', time.time() - start_time
        bar.finish()
        return (self.train_x,self.mask_x,self.train_y)

    def render_embedding_image_batch(self, start, end):
        kw = self._kw
        h, w = self.shape[:2]
        azimuth_range = (0, 2 * np.pi)
        elev_range = (-0.5 * np.pi, 0.5 * np.pi)
        radius = float(kw['radius'])
        render_dims = eval(kw['render_dims'])
        K = eval(kw['k'])
        K = np.array(K).reshape(3,3)
        lighting = eval(kw['lighting']) if 'lighting' in kw else None

        clip_near = float(kw['clip_near'])
        clip_far = float(kw['clip_far'])
        pad_factor = float(kw['pad_factor'])

        t = np.array([0, 0, float(kw['radius'])])
        batch = np.empty( (end-start,)+ self.shape)
        obj_bbs = np.empty( (end-start,)+ (4,))

        for i, R in enumerate(self.viewsphere_for_embedding[start:end]):
            bgr_y, depth_y = self.renderer.render( 
                obj_id=0,
                W=render_dims[0], 
                H=render_dims[1],
                K=K.copy(), 
                R=R, 
                t=t,
                near=clip_near,
                far=clip_far,
                random_light=False
                # phong=lighting
            )
            # cv2.imshow('depth',depth_y)
            # cv2.imshow('bgr',bgr_y)
            # print depth_y.max()
            # cv2.waitKey(0)
            ys, xs = np.nonzero(depth_y > 0)
            obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)

            obj_bbs[i] = obj_bb

            resized_bgr_y = self.extract_square_patch(bgr_y, obj_bb, pad_factor,resize=self.shape[:2],interpolation = cv2.INTER_NEAREST)

            if self.shape[2] == 1:
                resized_bgr_y = cv2.cvtColor(resized_bgr_y, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
            batch[i] = resized_bgr_y / 255.
        return (batch, obj_bbs)

    def extract_square_patch(self, scene_img, bb_xywh, pad_factor,resize=(128,128),interpolation=cv2.INTER_NEAREST, black_borders=False):

        x, y, w, h = np.array(bb_xywh).astype(np.int32)
        size = int(np.maximum(h, w) * pad_factor)
        
        left = np.maximum(x+w//2-size//2, 0)
        right = x+w//2+size//2
        top = np.maximum(y+h//2-size//2, 0)
        bottom = y+h//2+size//2

        scene_crop = scene_img[top:bottom, left:right].copy()

        if black_borders:
            scene_crop[:(y-top),:] = 0
            scene_crop[(y+h-top):,:] = 0
            scene_crop[:,:(x-left)] = 0
            scene_crop[:,(x+w-left):] = 0

        scene_crop = cv2.resize(scene_crop, resize, interpolation = interpolation)
        return scene_crop


    @lazy_property
    def _aug(self):
        from imgaug.augmenters import Sequential,SomeOf,OneOf,Sometimes,WithColorspace,WithChannels, \
            Noop,Lambda,AssertLambda,AssertShape,Scale,CropAndPad, \
            Pad,Crop,Fliplr,Flipud,Superpixels,ChangeColorspace, PerspectiveTransform, \
            Grayscale,GaussianBlur,AverageBlur,MedianBlur,Convolve, \
            Sharpen,Emboss,EdgeDetect,DirectedEdgeDetect,Add,AddElementwise, \
            AdditiveGaussianNoise,Multiply,MultiplyElementwise,Dropout, \
            CoarseDropout,Invert,ContrastNormalization,Affine,PiecewiseAffine, ElasticTransformation
        return eval(self._kw['code'])

    @lazy_property
    def _aug_occl(self):
        from imgaug.augmenters import Sequential,SomeOf,OneOf,Sometimes,WithColorspace,WithChannels, \
            Noop,Lambda,AssertLambda,AssertShape,Scale,CropAndPad, \
            Pad,Crop,Fliplr,Flipud,Superpixels,ChangeColorspace, PerspectiveTransform, \
            Grayscale,GaussianBlur,AverageBlur,MedianBlur,Convolve, \
            Sharpen,Emboss,EdgeDetect,DirectedEdgeDetect,Add,AddElementwise, \
            AdditiveGaussianNoise,Multiply,MultiplyElementwise,Dropout, \
            CoarseDropout,Invert,ContrastNormalization,Affine,PiecewiseAffine, \
            ElasticTransformation
        return Sequential([Sometimes(0.7, CoarseDropout( p=0.4, size_percent=0.01) )])

    
    @lazy_property
    def random_syn_masks(self):
        import bitarray
        workspace_path = os.environ.get('AE_WORKSPACE_PATH')

        random_syn_masks = bitarray.bitarray()
        with open(os.path.join(workspace_path,'random_tless_masks/arbitrary_syn_masks_1000.bin'), 'r') as fh:
            random_syn_masks.fromfile(fh)
        occlusion_masks = np.fromstring(random_syn_masks.unpack(), dtype=np.bool)
        occlusion_masks = occlusion_masks.reshape(-1,224,224,1).astype(np.float32)
        print((occlusion_masks.shape))

        occlusion_masks = np.array([cv2.resize(mask,(self.shape[0],self.shape[1]), interpolation = cv2.INTER_NEAREST) for mask in occlusion_masks])           
        return occlusion_masks


    def augment_occlusion_mask(self, masks, verbose=False, min_trans = 0.2, max_trans=0.7, max_occl = 0.25,min_occl = 0.0):

        new_masks = np.zeros_like(masks,dtype=np.bool)
        occl_masks_batch = self.random_syn_masks[np.random.choice(len(self.random_syn_masks),len(masks))]
        for idx,mask in enumerate(masks):
            occl_mask = occl_masks_batch[idx]
            while True:
                trans_x = int(np.random.choice([-1,1])*(np.random.rand()*(max_trans-min_trans) + min_trans)*occl_mask.shape[0])
                trans_y = int(np.random.choice([-1,1])*(np.random.rand()*(max_trans-min_trans) + min_trans)*occl_mask.shape[1])
                M = np.float32([[1,0,trans_x],[0,1,trans_y]])

                transl_occl_mask = cv2.warpAffine(occl_mask,M,(occl_mask.shape[0],occl_mask.shape[1]))

                overlap_matrix = np.invert(mask.astype(np.bool)) * transl_occl_mask.astype(np.bool)
                overlap = len(overlap_matrix[overlap_matrix==True])/float(len(mask[mask==0]))

                if overlap < max_occl and overlap > min_occl:
                    new_masks[idx,...] = np.logical_xor(mask.astype(np.bool), overlap_matrix)
                    if verbose:
                        print(('overlap is ', overlap))    
                    break

        return new_masks

    def augment_squares(self,masks,rand_idcs,max_occl=0.25):
        new_masks = np.invert(masks)

        idcs = np.arange(len(masks))
        while len(idcs) > 0:
            new_masks[idcs] = self._aug_occl.augment_images(np.invert(masks[idcs]))
            new_noof_obj_pixels = np.count_nonzero(new_masks,axis=(1,2))
            idcs = np.where(new_noof_obj_pixels/self.noof_obj_pixels[rand_idcs].astype(np.float32) < 1-max_occl)[0]
            # print idcs
        return np.invert(new_masks)


    def preprocess_aae(self, batch_x, masks_x, batch_y):
        rand_idcs_bg = np.random.choice(self.noof_bg_imgs, len(batch_x), replace=False)
        rand_vocs = self.bg_imgs[rand_idcs_bg]
        print(rand_vocs.shape)
        print(masks_x.shape)
        batch_x[masks_x] = rand_vocs[masks_x]
        batch_x = self._aug.augment_images(batch_x)

        return (batch_x,masks_x,batch_y)

    def batch(self, batch_size):

        # batch_x = np.empty( (batch_size,) + self.shape, dtype=np.uint8 )
        # batch_y = np.empty( (batch_size,) + self.shape, dtype=np.uint8 )
        
        rand_idcs = np.random.choice(self.noof_training_imgs, batch_size, replace=False)
        
        assert self.noof_bg_imgs > 0

        rand_idcs_bg = np.random.choice(self.noof_bg_imgs, batch_size, replace=False)
        
        batch_x, masks, batch_y = self.train_x[rand_idcs], self.mask_x[rand_idcs], self.train_y[rand_idcs]
        rand_vocs = self.bg_imgs[rand_idcs_bg]

        if 'realistic_occlusion' in self._kw and eval(self._kw['realistic_occlusion']):
            masks = self.augment_occlusion_mask(masks.copy(),max_occl=np.float(self._kw['realistic_occlusion']))
        
        if 'square_occlusion' in self._kw and eval(self._kw['square_occlusion']):
            masks = self.augment_squares(masks.copy(),rand_idcs,max_occl=np.float(self._kw['square_occlusion']))

        batch_x[masks] = rand_vocs[masks]


        # random in-plane rotation, not necessary
        # for i in xrange(batch_size):
        #   rot_angle= np.random.rand()*360
        #   cent = int(self.shape[0]//2)
        #   M = cv2.getRotationMatrix2D((cent,cent),rot_angle,1)
        #   batch_x[i] = cv2.warpAffine(batch_x[i],M,self.shape[:2])[:,:,np.newaxis]
        #   batch_y[i] = cv2.warpAffine(batch_y[i],M,self.shape[:2])[:,:,np.newaxis]


        #needs uint8
        batch_x = self._aug.augment_images(batch_x)

        #slow...
        batch_x = batch_x / 255.
        batch_y = batch_y / 255.
        

        return (batch_x, batch_y)
