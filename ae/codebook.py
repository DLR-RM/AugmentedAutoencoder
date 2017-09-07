# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
import progressbar

import utils as u

class Codebook(object):

    def __init__(self, encoder, dataset):
        self._encoder = encoder
        self._dataset = dataset

        normalized_embedding_query = tf.nn.l2_normalize(self._encoder.z, 1)
        J = encoder.latent_space_size
        embedding_size = self._dataset.embedding_size

        self.embedding_normalized = tf.Variable(
            np.zeros((J, embedding_size)),
            dtype=tf.float32,
            trainable=False,
            name='embedding_normalized'
        )
        self.embedding = tf.placeholder(tf.float32, shape=[J, embedding_size])
        self.embedding_assign_op = tf.assign(self.embedding_normalized, self.embedding)
        
        self.cos_similarity= tf.matmul(normalized_embedding_query, self.embedding_normalized)
        self.nearest_neighbor_idx = tf.argmax(self.cos_similarity, axis=1)


    def nearest_rotation(self, session, x):
        if x.ndim == 3:
            x = np.expand_dims(x, 0)
            cosine_similarity = session.run(self.cos_similarity, {self._encoder.x: x})
            idcs = np.argmax(cosine_similarity, axis=1)
            return self._dataset.viewsphere_for_embedding[idcs][0]      
        else:
            cosine_similarity = session.run(self.cos_similarity, {self._encoder.x: x})
            idcs = np.argmax(cosine_similarity, axis=1)
            return self._dataset.viewsphere_for_embedding[idcs]


    def nearest_rotation_batch(self, session, x):
        idcs = session.run(self.nearest_neighbor_idx, {self._encoder.x: x})
        return self._dataset.viewsphere_for_embedding[idcs]


    def update_embedding(self, session, batch_size):
        embedding_size = self._dataset.embedding_size
        J = self._encoder.latent_space_size
        embedding_z = np.empty( (embedding_size, J) )
        print 'Creating embedding ..'
        bar = progressbar.ProgressBar(
            maxval=embedding_size, 
            widgets=[' [', progressbar.Timer(), ' | ', progressbar.Counter('%0{}d / {}'.format(len(str(embedding_size)), embedding_size)), ' ] ', 
            progressbar.Bar(), 
            ' (', progressbar.ETA(), ') ']
        )
        bar.start()
        for a, e in u.batch_iteration_indices(embedding_size, batch_size):
            batch = self._dataset.render_embedding_images(a, e)
            embedding_z[a:e] = session.run(self._encoder.z, feed_dict={self._encoder.x: batch})
            bar.update(e)
        bar.finish()
        embedding_z = embedding_z.T
        normalized_embedding = embedding_z / np.linalg.norm( embedding_z, axis=0 )     
        session.run(self.embedding_assign_op, {self.embedding: normalized_embedding})