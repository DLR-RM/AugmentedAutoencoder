# -*- coding: utf-8 -*-

import threading

import tensorflow as tf

from utils import lazy_property
import time
import hashlib
import os

class MultiQueue(object):

    def __init__(self, dataset, batch_size, noof_training_imgs, model_paths, shape):

        self._dataset = dataset
        self._batch_size = batch_size
        self._noof_training_imgs = noof_training_imgs
        self._model_paths = model_paths
        self._shape = shape

        self._num_objects = len(self._model_paths)
        self.next_element = None



    
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
                            'mask_x':tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        train_x = tf.reshape(tf.decode_raw(parsed_features['train_x'], tf.uint8), self._shape)
        mask_x = tf.reshape(tf.decode_raw(parsed_features['mask_x'], tf.uint8), self._shape[:2])
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

            

    def _float_cast(self, train_x, mask_x, train_y):
        print 'here', 100*'g'
        train_x = tf.image.convert_image_dtype(train_x,tf.float32)
        train_y = tf.image.convert_image_dtype(train_y,tf.float32)
        return (train_x, mask_x, train_y)

    def preprocess_pipeline(self, dataset):
        dataset = dataset.map(self.deserialize_tfrecord)  
        dataset = dataset.shuffle(buffer_size=self._noof_training_imgs)
        dataset = dataset.repeat()
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.map(lambda train_x, mask_x, train_y : tuple(tf.py_func(self._dataset.preprocess_aae, 
                                                                                [train_x, mask_x, train_y], 
                                                                                [tf.uint8, tf.uint8, tf.uint8])))
        dataset = dataset.map(lambda train_x, mask_x, train_y : self._float_cast(train_x, mask_x, train_y))
        dataset = dataset.prefetch(1)
        return dataset

    def create_iterator(self, dataset_path, args):
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


