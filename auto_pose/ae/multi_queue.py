# -*- coding: utf-8 -*-

import tensorflow as tf


from .utils import lazy_property
from .image_augmentation_functions import *
import time
import hashlib
import os
import glob

class MultiQueue(object):

    def __init__(self, dataset, batch_size, noof_training_imgs, model_paths, shape, aug_args):

        self._dataset = dataset
        self._batch_size = batch_size
        self._noof_training_imgs = noof_training_imgs

        self._model_paths = model_paths
        self._shape = shape

        self._num_objects = len(self._model_paths)
        self.next_element = None

        self.zoom_range = eval(aug_args['zoom_pad'])
        self.g_noise = eval(aug_args['gaussian_noise'])
        self.contrast_norm_range = eval(aug_args['contrast_norm'])
        self.mult_brightness = eval(aug_args['mult_brightness'])
        self.max_off_brightness = eval(aug_args['max_off_brightness'])
        self.gaussian_blur = eval(aug_args['gaussian_blur'])
        self.invert = eval(aug_args['invert'])
        self.invert_whole = eval(aug_args['invert_whole'])
        self._random_bg = eval(aug_args['random_bg'])
        self.occl = eval(aug_args['transparent_shape_occlusion'])


        print(('zoom_range: ', self.zoom_range))
        print(('g_noise: ', self.g_noise)) 
        print(('contrast_norm_range: ', self.contrast_norm_range))
        print(('mult_brightness: ', self.mult_brightness))
        print(('max_off_brightness: ', self.max_off_brightness))
        print(('gaussian_blur: ', self.gaussian_blur))
        print(('invert: ', self.invert))
        print(('occl: ', self.occl))
    
        self.bg_img_init = None
        self.next_bg_element = None
        

    def serialize_tfrecord(self, train_x, mask_x, train_y, writer, output_file='train.tfrecord'):
        train_x_bytes = train_x.tostring()
        mask_x_bytes = mask_x.tostring()
        train_y_bytes = train_y.tostring()

        feature = {}
        feature['train_x'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_x_bytes]))
        feature['train_y'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_y_bytes]))
        feature['mask'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_x_bytes]))


        example = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized = example.SerializeToString()
        writer.write(serialized)
    
    def deserialize_tfrecord(self, example_proto):
        keys_to_features = {'train_x':tf.FixedLenFeature([], tf.string),
                            'train_y':tf.FixedLenFeature([], tf.string),
                            'mask':tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        train_x = tf.reshape(tf.decode_raw(parsed_features['train_x'], tf.uint8), self._shape)
        mask_x = tf.reshape(tf.decode_raw(parsed_features['mask'], tf.uint8), self._shape[:2])
        train_y = tf.reshape(tf.decode_raw(parsed_features['train_y'], tf.uint8), self._shape)

        return (train_x, mask_x, train_y)

    def create_tfrecord_training_images(self, dataset_path, args):

        for m,model in enumerate(self._model_paths):
            md5_string = str(str(args.items('Dataset')) + model)
            md5_string = md5_string.encode('utf-8')
            current_config_hash = hashlib.md5(md5_string).hexdigest()

            current_file_name = os.path.join(dataset_path, current_config_hash + '.tfrecord')
            
            if not os.path.exists(current_file_name):
                writer = tf.python_io.TFRecordWriter(current_file_name)
                self._dataset.render_training_images(serialize_func = self.serialize_tfrecord, obj_id = m, tfrec_writer = writer)
                writer.close()
                print(('generated tfrecord for training images', model))
            else:
                print(('tfrecord exists for', model))

    def _tf_augmentations(self, train_x, mask_x, train_y, bg):
        # train_x = add_black_patches(train_x)
        train_x = zoom_image_object(train_x,np.linspace(self.zoom_range[0], self.zoom_range[1], 50).astype(np.float32))
        train_x = add_black_patches(train_x, max_area_cov = self.occl) if self.occl > 0 else train_x
        train_x = add_background(train_x, bg) if self._random_bg else train_x
        train_x = gaussian_noise(train_x) if self.g_noise else train_x
        # train_x = gaussian_blur(train_x) if self.gaussian_blur else train_x
        train_x = random_brightness(train_x, self.max_off_brightness)
        train_x = invert_color(train_x) if self.invert else train_x
        train_x = invert_color_all(train_x) if self.invert_whole else train_x
        train_x = multiply_brightness(train_x, self.mult_brightness)
        train_x = contrast_normalization(train_x, self.contrast_norm_range)
        # train_x = gaussian_blur(train_x)
        return (train_x, mask_x, train_y)

    def load_bg_imgs(self, in_path):

        return tf.image.resize_images(tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(in_path)),tf.float32),
                    [self._shape[0],self._shape[1]])



    def create_background_image_iterator(self, bg_paths):
        # change to load from numpy array
        background_imgs_dataset = tf.data.Dataset.from_tensor_slices(bg_paths)
        background_imgs_dataset = background_imgs_dataset.map(map_func=self.load_bg_imgs, num_parallel_calls = 4)
        # background_imgs_dataset = background_imgs_dataset.cache()
        background_imgs_dataset = background_imgs_dataset.shuffle(1000)
        background_imgs_dataset = background_imgs_dataset.repeat()
        background_imgs_dataset = background_imgs_dataset.prefetch(1)

        self.bg_img_init = background_imgs_dataset.make_initializable_iterator()

        self.next_bg_element = self.bg_img_init.get_next()

    def _float_cast(self, train_x, mask_x, train_y):

        train_x = tf.image.convert_image_dtype(train_x,tf.float32)
        train_y = tf.image.convert_image_dtype(train_y,tf.float32)
        return (train_x, mask_x, train_y)

    def _recover_shapes(self, train_x, mask_x, train_y):
        train_x.set_shape((None,) + self._shape)
        train_y.set_shape((None,) + self._shape)
        mask_x.set_shape((None,) + self._shape[:2])
        return (train_x, mask_x, train_y)

    def preprocess_pipeline(self, dataset):
        dataset = dataset.map(self.deserialize_tfrecord)  
        dataset = dataset.shuffle(buffer_size=1000)#self._noof_training_imgs//self._num_objects)
        dataset = dataset.map(lambda train_x, mask_x, train_y : self._float_cast(train_x, mask_x, train_y))
        dataset = dataset.repeat()
        dataset = dataset.map(lambda train_x, mask_x, train_y : self._tf_augmentations(train_x, mask_x, train_y, self.bg_img_init.get_next()))
        dataset = dataset.batch(self._batch_size)
        # dataset = dataset.map(lambda train_x, mask_x, train_y : tuple(tf.py_func(self._dataset.preprocess_aae, 
        #                                                                         [train_x, mask_x, train_y], 
        #                                                                         [tf.uint8, tf.uint8, tf.uint8])))
        # dataset = dataset.map(lambda train_x, mask_x, train_y : self._recover_shapes(train_x, mask_x, train_y))
        dataset = dataset.prefetch(1)
        return dataset

    def create_iterator(self, dataset_path, args):
        background_img_paths = glob.glob(args.get('Paths','BACKGROUND_IMAGES_GLOB'))

        self.create_background_image_iterator(background_img_paths)
        dsets = []
        for m,model in enumerate(self._model_paths):
            print(model)
            current_config_hash = hashlib.md5((str(args.items('Dataset')) + model).encode('utf-8')).hexdigest()
            current_file_name = os.path.join(dataset_path, current_config_hash + '.tfrecord')
            tfrecord_dataset = tf.data.TFRecordDataset(current_file_name)
            dsets.append(self.preprocess_pipeline(tfrecord_dataset))

        joint_dataset = tf.data.Dataset.zip(tuple(dsets))
        iterator = joint_dataset.make_initializable_iterator()
        self.next_element = iterator.get_next()
        return iterator


