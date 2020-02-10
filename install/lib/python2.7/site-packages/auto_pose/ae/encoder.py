# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from .utils import lazy_property

class Encoder(object):

    def __init__(self, input, latent_space_size, num_filters, kernel_size, strides, batch_norm, is_training=False):
        self._input = input
        self._latent_space_size = latent_space_size
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._batch_normalization = batch_norm
        self._is_training = is_training
        self.encoder_out
        self.z
        # self.q_sigma
        # self.sampled_z
        # self.reg_loss
        # self.kl_div_loss

    @property
    def x(self):
        return self._input

    @property
    def latent_space_size(self):
        return self._latent_space_size

    @lazy_property
    def encoder_out(self):
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
            if self._batch_normalization:
                x = tf.layers.batch_normalization(x, training=self._is_training)

        encoder_out = tf.contrib.layers.flatten(x)
        
        return encoder_out

    @lazy_property
    def z(self):
        x = self.encoder_out

        z = tf.layers.dense(
            x,
            self._latent_space_size,   
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

        return z
    
    @lazy_property
    def q_sigma(self):
        x = self.encoder_out

        q_sigma = 1e-8 + tf.layers.dense(inputs=x,
                        units=self._latent_space_size,
                        activation=tf.nn.softplus,
                        kernel_initializer=tf.zeros_initializer())

        return q_sigma

    @lazy_property
    def sampled_z(self):
        epsilon = tf.random_normal(tf.shape(self._latent_space_size), 0., 1.)
        # epsilon = tf.contrib.distributions.Normal(
        #             np.zeros(self._latent_space_size, dtype=np.float32), 
        #             np.ones(self._latent_space_size, dtype=np.float32))
        return self.z + self.q_sigma * epsilon


    @lazy_property
    def kl_div_loss(self):
        p_z = tf.contrib.distributions.Normal(
            np.zeros(self._latent_space_size, dtype=np.float32), 
            np.ones(self._latent_space_size, dtype=np.float32))
        q_z = tf.contrib.distributions.Normal(self.z, self.q_sigma)

        return tf.reduce_mean(tf.distributions.kl_divergence(q_z,p_z))


    @lazy_property
    def reg_loss(self):
        reg_loss = tf.reduce_mean(tf.abs(tf.norm(self.z,axis=1) - tf.constant(1.)))
        return reg_loss