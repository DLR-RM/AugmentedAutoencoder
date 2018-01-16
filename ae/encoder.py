# -*- coding: utf-8 -*-

import tensorflow as tf

from utils import lazy_property

class Encoder(object):

    def __init__(self, input, latent_space_size, num_filters, kernel_size, strides):
        self._input = input
        self._latent_space_size = latent_space_size
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self.z

    @property
    def x(self):
        return self._input

    @property
    def latent_space_size(self):
        return self._latent_space_size

    @lazy_property
    def z(self):
        x = self._input

        for filters, stride in zip(self._num_filters, self._strides):
            padding = 'same'
            x = tf.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=self._kernel_size,
                strides=stride,
                padding=padding,
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=tf.nn.relu
            )
            x = tf.layers.batch_normalization(x)

        x = tf.contrib.layers.flatten(x)
        z = tf.layers.dense(
            x,
            self._latent_space_size,   
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

        return z