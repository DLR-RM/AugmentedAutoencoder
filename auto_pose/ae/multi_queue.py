# -*- coding: utf-8 -*-

import threading

import tensorflow as tf

from utils import lazy_property
from image_augmentation_functions import *
import time
import hashlib
import os
import glob

class MultiQueue(object):

    def __init__(self, dataset, batch_size, noof_training_imgs, model_paths, shape):

        self._dataset = dataset
        self._batch_size = batch_size
        self._noof_training_imgs = noof_training_imgs
        self._model_paths = model_paths
        self._shape = shape

        self._num_objects = len(self._model_paths)
        self.next_element = None
    
        self.bg_img_init = None
        self.next_bg_element = None

    def serialize_tfrecord(self, train_x, mask_x, train_y, writer, output_file='train.tfrecord'):
        train_x_bytes = train_x.tostring()
        mask_x_bytes = mask_x.tostring()
        train_y_bytes = train_y.tostring()

        feature = {}
        feature['train_x'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_x_bytes]))
        feature['train_y'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_y_bytes]))
        feature['mask'] =  tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_x_bytes]))

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

            current_config_hash = hashlib.md5(str(args.items('Dataset')) + model).hexdigest()
            current_file_name = os.path.join(dataset_path, current_config_hash + '.tfrecord')
            
            if not os.path.exists(current_file_name):
                writer = tf.python_io.TFRecordWriter(current_file_name)
                self._dataset.render_training_images(serialize_func = self.serialize_tfrecord, obj_id = m, tfrec_writer = writer)
                writer.close()
                print 'generated tfrecord for training images', model
            else:
                print 'tfrecord exists for', model

            

    def _tf_augmentations(self, train_x, mask_x, train_y, bg):
        # train_x = add_black_patches(train_x)
        train_x = add_background(train_x,bg)
        train_x = gaussian_noise(train_x)
        train_x = random_brightness(train_x,0.1)
        train_x = contrast_normalization(train_x)
        train_x = multiply_brightness(train_x)
        # train_x = gaussian_blur(train_x)
        return (train_x, mask_x, train_y)

    def load_bg_imgs(self, in_path):

        return tf.image.resize_images(tf.to_float(tf.image.decode_jpeg(tf.read_file(in_path))),
                    [self._shape[0],self._shape[1]],
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    def create_background_image_iterator(self, bg_paths):
        # change to load from numpy array
        background_imgs_dataset = tf.data.Dataset.from_tensor_slices(bg_paths)
        
        background_imgs_dataset = background_imgs_dataset.shuffle(self._dataset.noof_bg_imgs)
        background_imgs_dataset = background_imgs_dataset.apply(tf.contrib.data.map_and_batch(
            map_func=self.load_bg_imgs, 
            batch_size=self._batch_size, 
            num_parallel_calls = 2,
            drop_remainder = True))
        background_imgs_dataset = background_imgs_dataset.repeat()
        background_imgs_dataset = background_imgs_dataset.prefetch(buffer_size = 1)

        # iterator_bg = tf.data.Iterator.from_structure(background_imgs_dataset.output_types, 
        #                                         background_imgs_dataset.output_shapes)

        # iterator_bg = background_imgs_dataset.make_initializable_iterator()


        self.bg_img_init = background_imgs_dataset.make_initializable_iterator()

        self.next_bg_element = self.bg_img_init.get_next()

    def _float_cast(self, train_x, mask_x, train_y):
        print 'here', 100*'g'
        train_x = tf.image.convert_image_dtype(train_x,tf.float32)
        train_y = tf.image.convert_image_dtype(train_y,tf.float32)
        return (train_x, mask_x, train_y)

    def _recover_shapes(self, train_x, mask_x, train_y):
        print 'here'
        train_x.set_shape((None,) + self._shape)
        train_y.set_shape((None,) + self._shape)
        mask_x.set_shape((None,) + self._shape[:2])
        return (train_x, mask_x, train_y)

    def preprocess_pipeline(self, dataset):
        dataset = dataset.map(self.deserialize_tfrecord)  
        dataset = dataset.shuffle(buffer_size=self._noof_training_imgs//10)
        dataset = dataset.repeat()
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.map(lambda train_x, mask_x, train_y : self._float_cast(train_x, mask_x, train_y))
        dataset = dataset.map(lambda train_x, mask_x, train_y : self._tf_augmentations(train_x, mask_x, train_y,  self.bg_img_init.get_next()/255.))
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
            current_config_hash = hashlib.md5(str(args.items('Dataset')) + model).hexdigest()
            current_file_name = os.path.join(dataset_path, current_config_hash + '.tfrecord')
            tfrecord_dataset = tf.data.TFRecordDataset(current_file_name)
            dsets.append(self.preprocess_pipeline(tfrecord_dataset))

        joint_dataset = tf.data.Dataset.zip(tuple(dsets))
        iterator = joint_dataset.make_initializable_iterator()
        self.next_element = iterator.get_next()
        return iterator


