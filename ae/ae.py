# -*- coding: utf-8 -*-

import tensorflow as tf

from utils import lazy_property

class AE(object):

    def __init__(self, encoder, decoder):
        self._encoder = encoder
        self._decoder = decoder
        self.loss
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

    @property
    def loss(self):
        return self._decoder.reconstr_loss

