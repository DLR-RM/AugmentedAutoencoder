# -*- coding: utf-8 -*-

import tensorflow as tf

from .utils import lazy_property

class AE(object):

    def __init__(self, encoder, decoder, norm_regularize, variational):
        self._encoder = encoder
        self._decoder = decoder
        self._norm_regularize = norm_regularize
        self._variational = variational
        self.loss
        tf.summary.scalar('total_loss', self.loss)
        self.global_step

    @property
    def x(self):
        return self._encoder.x

    @property
    def z(self):
        return self._encoder.z

    @property
    def reconstruction(self):
        return self._decoder.x

    @property
    def reconstruction_target(self):
        return self._decoder.reconstruction_target

    @lazy_property
    def global_step(self):
        return tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

    @lazy_property
    def loss(self):
        loss = self._decoder.reconstr_loss
        if self._norm_regularize > 0:
            loss += self._encoder.reg_loss * tf.constant(self._norm_regularize,dtype=tf.float32)
            tf.summary.scalar('reg_loss', self._encoder.reg_loss)
        if self._variational:
            loss +=  self._encoder.kl_div_loss * tf.constant(self._variational, dtype=tf.float32)
            tf.summary.scalar('KL_loss', self._encoder.kl_div_loss)
            tf.summary.histogram('Variance', self._encoder.q_sigma)
        tf.summary.histogram('Mean', self._encoder.z)
        return loss



