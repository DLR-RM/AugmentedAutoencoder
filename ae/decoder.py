# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf

from utils import lazy_property

class Decoder(object):

    def __init__(self, reconstruction_target, latent_code, num_filters, kernel_size, strides, loss, bootstrap_ratio):
        self._reconstruction_target = reconstruction_target
        self._latent_code = latent_code
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._loss = loss
        self._bootstrap_ratio = bootstrap_ratio
        self.reconstr_loss

    @property
    def reconstruction_target(self):
        return self._reconstruction_target


    @lazy_property
    def x(self):
        z = self._latent_code

        h, w, c = self._reconstruction_target.get_shape().as_list()[1:]
        print h,w,c
        layer_dimensions = [ [h/np.prod(self._strides[i:]), w/np.prod(self._strides[i:])]  for i in xrange(len(self._strides))]
        print layer_dimensions
        x = tf.layers.dense(
            inputs=self._latent_code,
            units= layer_dimensions[0][0]*layer_dimensions[0][1]*self._num_filters[0],
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        x = tf.layers.batch_normalization(x)
        x = tf.reshape( x, [-1, layer_dimensions[0][0], layer_dimensions[0][1], self._num_filters[0] ] )

        for filters, layer_size in zip(self._num_filters[1:], layer_dimensions[1:]):
            x = tf.image.resize_nearest_neighbor(x, layer_size)

            x = tf.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=self._kernel_size,
                padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=tf.nn.relu
            )
            x = tf.layers.batch_normalization(x)
        
        x = tf.image.resize_nearest_neighbor( x, [h, w] )
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
        print self.x.shape
        print self._reconstruction_target.shape
        if self._loss == 'L2':
            return tf.losses.mean_squared_error (
                self._reconstruction_target,
                self.x,
                reduction=tf.losses.Reduction.MEAN
            )
        elif self._loss == 'L2_bootstrapped':

            x_flat = tf.contrib.layers.flatten(self.x)
            reconstruction_target_flat = tf.contrib.layers.flatten(self._reconstruction_target)

            l2 = tf.losses.mean_squared_error (
                reconstruction_target_flat,
                x_flat,
                reduction=tf.losses.Reduction.NONE
            )

            l2_val,_ = tf.nn.top_k(l2,k=l2.shape[1]/self._bootstrap_ratio)

            l2_bootstrapped = tf.reduce_mean(l2_val)

            return l2_bootstrapped
        else:
            print 'ERROR: UNKNOWN LOSS ', self._loss
            exit()

