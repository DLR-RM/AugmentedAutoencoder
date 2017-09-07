# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf

from utils import lazy_property

class Decoder(object):

    def __init__(self, reconstruction_target, latent_code, num_filters, kernel_size, strides):
        self._reconstruction_target = reconstruction_target
        self._latent_code = latent_code
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self.reconstr_loss

    @property
    def reconstruction_target(self):
        return self._reconstruction_target

    @lazy_property
    def x(self):
        z = self._latent_code

        h, w, c = self._reconstruction_target.get_shape().as_list()[1:]
        layer_dimensions = [ [h/np.prod(self._strides[i:]), w/np.prod(self._strides[i:])]  for i in xrange(len(self._strides))]

        x = tf.layers.dense(
            inputs=self._latent_code,
            units= layer_dimensions[0][0]*layer_dimensions[0][1]*self._num_filters[0],
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )
        x = tf.layers.batch_normalization(x)
        x = tf.reshape( x, [-1, layer_dimensions[0][0], layer_dimensions[0][1], self._num_filters[0] ] )

        for filters, layer_size in zip(self._num_filters[1:], layer_dimensions[1:]):
            x = tf.image.resize_nearest_neighbor( x, layer_size )

            x = tf.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=self._kernel_size,
                padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=tf.nn.relu
            )
            x = tf.layers.batch_normalization(x)

        x = tf.layers.conv2d(
                inputs=x,
                filters=c,
                kernel_size=self._kernel_size,
                padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=None
            )
        return x

    @lazy_property
    def reconstr_loss(self):
        return tf.losses.mean_squared_error (
            self._reconstruction_target,
            self.x,
            reduction=tf.losses.Reduction.MEAN
        )