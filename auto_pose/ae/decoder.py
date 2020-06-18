# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf

from .utils import lazy_property

class Decoder(object):

    def __init__(self, reconstruction_target, latent_code, num_filters, 
                kernel_size, strides, loss, bootstrap_ratio, 
                auxiliary_mask, batch_norm, is_training=False,idx=0):
	
        self._reconstruction_target = reconstruction_target[2]
        self._latent_code = latent_code
        self._auxiliary_mask = auxiliary_mask
        if self._auxiliary_mask:
            self._xmask = None
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._loss = loss
        self._bootstrap_ratio = bootstrap_ratio
        self._batch_normalization = batch_norm
        self._is_training = is_training
        with tf.variable_scope('decoder_' + str(idx)):
            self.reconstr_loss

    @property
    def reconstruction_target(self):
        return self._reconstruction_target

    @lazy_property
    def x(self):
        z = self._latent_code

        h, w, c = self._reconstruction_target.get_shape().as_list()[1:]
        print((h,w,c))
        layer_dimensions = [ [h//np.prod(self._strides[i:]), w//np.prod(self._strides[i:])]  for i in range(len(self._strides))]
        print(layer_dimensions)
        x = tf.layers.dense(
            inputs=self._latent_code,
            units= layer_dimensions[0][0]*layer_dimensions[0][1]*self._num_filters[0],
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        if self._batch_normalization:
            x = tf.layers.batch_normalization(x, training=self._is_training)
        x = tf.reshape( x, [-1, layer_dimensions[0][0], layer_dimensions[0][1], self._num_filters[0] ] )

        for filters, layer_size in zip(self._num_filters[1:], layer_dimensions[1:]):
            # shoudl have     align_corners=True
            x = tf.image.resize_nearest_neighbor(x, layer_size)

            x = tf.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=self._kernel_size,
                padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=tf.nn.relu
            )
            if self._batch_normalization:
                x = tf.layers.batch_normalization(x, training=self._is_training)
        
        x = tf.image.resize_nearest_neighbor( x, [h, w] )

        if self._auxiliary_mask:
            self._xmask = tf.layers.conv2d(
                    inputs=x,
                    filters=1,
                    kernel_size=self._kernel_size,
                    padding='same',
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    activation=tf.nn.sigmoid
                )

        x = tf.layers.conv2d(
                inputs=x,
                filters=c,
                kernel_size=self._kernel_size,
                padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=tf.nn.sigmoid
            )
        return x



    @lazy_property
    def reconstr_loss(self):
        print((self.x.shape))
        print((self._reconstruction_target.shape))
        if self._loss == 'L2':
            if self._bootstrap_ratio > 1:

                x_flat = tf.contrib.layers.flatten(self.x)
                reconstruction_target_flat = tf.contrib.layers.flatten(self._reconstruction_target)
                l2 = tf.losses.mean_squared_error (
                    reconstruction_target_flat,
                    x_flat,
                    reduction=tf.losses.Reduction.NONE
                )
                l2_val,_ = tf.nn.top_k(l2,k=l2.shape[1]//self._bootstrap_ratio)
                loss = tf.reduce_mean(l2_val)
            else:
                loss = tf.losses.mean_squared_error (
                    self._reconstruction_target,
                    self.x,
                    reduction=tf.losses.Reduction.MEAN
                )
        elif self._loss == 'L1':
            if self._bootstrap_ratio > 1:

                x_flat = tf.contrib.layers.flatten(self.x)
                reconstruction_target_flat = tf.contrib.layers.flatten(self._reconstruction_target)
                l1 = tf.losses.absolute_difference(
                    reconstruction_target_flat,
                    x_flat,
                    reduction=tf.losses.Reduction.NONE
                )
                print((l1.shape))
                l1_val,_ = tf.nn.top_k(l1,k=l1.shape[1]/self._bootstrap_ratio)
                loss = tf.reduce_mean(l1_val)
            else:
                x_flat = tf.contrib.layers.flatten(self.x)
                reconstruction_target_flat = tf.contrib.layers.flatten(self._reconstruction_target)
                l1 = tf.losses.absolute_difference(
                    reconstruction_target_flat,
                    x_flat,
                    reduction=tf.losses.Reduction.MEAN
                )
        else:
            print(('ERROR: UNKNOWN LOSS ', self._loss))
            exit()
        
        #tf.summary.scalar('reconst_loss', loss)
        if self._auxiliary_mask:
            mask_loss = tf.losses.mean_squared_error (
                tf.cast(tf.greater(tf.reduce_sum(self._reconstruction_target,axis=3,keepdims=True),0.0001),tf.float32),
                self._xmask,
                reduction=tf.losses.Reduction.MEAN
            )
            loss += mask_loss

            tf.summary.scalar('mask_loss', mask_loss)

        return loss
