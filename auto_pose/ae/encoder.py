# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os

from utils import lazy_property

class Encoder(object):

    def __init__(self, input, latent_space_size, num_filters, kernel_size, strides, batch_norm, resnet50, resnet101, aspp, pre_trained_model, is_training=False):
        
        self._input = input #tf.concat([inp[0] for inp in input],0)
        print self._input.shape
        self._latent_space_size = latent_space_size
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._batch_normalization = batch_norm
        self._resnet50 = resnet50
        self._resnet101 = resnet101
        self._aspp = aspp
        self._pre_trained_model = pre_trained_model
        self._is_training = is_training
        self.encoder_out
        self.z
        self.global_step

        if self._pre_trained_model != 'False':
            exclude = ['resnet_v2_101' + '/logits', 'resnet_v2_50' + '/logits', 'global_step']
            self.variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
        else:
            self.variables_to_restore = None

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
        
        if self._resnet50 or self._resnet101:
            from auto_pose.ae import deeplab_v3_encoder
            base_architecture = 'resnet_v2_50' if self._resnet50 else 'resnet_v2_101'
            pre_trained_model = self._pre_trained_model if os.path.exists(self._pre_trained_model) else None
            params = {'output_stride':16, 'base_architecture':base_architecture, 'pre_trained_model':pre_trained_model, 'batch_norm_decay':None}
            x = deeplab_v3_encoder.deeplab_v3_encoder(x, 
                params,
                is_training=self._is_training, 
                depth=self._num_filters[-1], 
                atrous_rates=self._aspp)
        else:
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

    @lazy_property
    def global_step(self):
        return tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

    