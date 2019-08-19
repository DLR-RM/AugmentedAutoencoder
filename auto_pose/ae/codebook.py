# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
import progressbar

from .utils import lazy_property
from . import utils as u


class Codebook(object):

    def __init__(self, encoder, dataset, embed_bb):
        
        self._encoder = encoder
        self._dataset = dataset
        self.embed_bb = embed_bb
        
        J = encoder.latent_space_size
        embedding_size = self._dataset.embedding_size

        self.normalized_embedding_query = tf.nn.l2_normalize(self._encoder.z, 1)
        self.embedding_normalized = tf.Variable(
            np.zeros((embedding_size, J)),
            dtype=tf.float32,
            trainable=False,
            name='embedding_normalized'
        )

        self.embedding = tf.placeholder(tf.float32, shape=[embedding_size, J])
        self.embedding_assign_op = tf.assign(self.embedding_normalized, self.embedding)
        

        if embed_bb:
            self.embed_obj_bbs_var = tf.Variable(
                np.zeros((embedding_size, 4)),
                dtype=tf.int32,
                trainable=False,
                name='embed_obj_bbs_var'
            )
            self.embed_obj_bbs = tf.placeholder(tf.int32, shape=[embedding_size, 4])
            self.embed_obj_bbs_assign_op = tf.assign(self.embed_obj_bbs_var, self.embed_obj_bbs)
            self.embed_obj_bbs_values = None
        
        self.cos_similarity = tf.matmul(self.normalized_embedding_query, self.embedding_normalized,transpose_b=True)
        self.nearest_neighbor_idx = tf.argmax(self.cos_similarity, axis=1)



    def nearest_rotation(self, session, x, top_n=1, upright=False, return_idcs=False):
        #R_model2cam

        if x.dtype == 'uint8':
            x = x/255.
        if x.ndim == 3:
            x = np.expand_dims(x, 0)
        
        cosine_similarity = session.run(self.cos_similarity, {self._encoder.x: x})
        if top_n == 1:
            if upright:
                idcs = np.argmax(cosine_similarity[:,::int(self._dataset._kw['num_cyclo'])], axis=1)*int(self._dataset._kw['num_cyclo'])
            else:
                idcs = np.argmax(cosine_similarity, axis=1)
        else:
            unsorted_max_idcs = np.argpartition(-cosine_similarity.squeeze(), top_n)[:top_n]
            idcs = unsorted_max_idcs[np.argsort(-cosine_similarity.squeeze()[unsorted_max_idcs])]
        if return_idcs:
            return idcs
        else:
            return self._dataset.viewsphere_for_embedding[idcs].squeeze()



    def auto_pose6d(self, session, x, predicted_bb, K_test, top_n, train_args, depth_pred=None, upright=False):
        
        idcs = self.nearest_rotation(session, x, top_n=top_n, upright=upright,return_idcs=True)
        Rs_est = self._dataset.viewsphere_for_embedding[idcs]



        # test_depth = f_test / f_train * render_radius * diag_bb_ratio
        K_train = np.array(eval(train_args.get('Dataset','K'))).reshape(3,3)
        render_radius = train_args.getfloat('Dataset','RADIUS')

        K00_ratio = K_test[0,0] / K_train[0,0]  
        K11_ratio = K_test[1,1] / K_train[1,1]  
        
        mean_K_ratio = np.mean([K00_ratio,K11_ratio])

        if self.embed_obj_bbs_values is None:
            self.embed_obj_bbs_values = session.run(self.embed_obj_bbs_var)

        ts_est = np.empty((top_n,3))
        for i,idx in enumerate(idcs):

            rendered_bb = self.embed_obj_bbs_values[idx].squeeze()
            if depth_pred is None:
                diag_bb_ratio = np.linalg.norm(np.float32(rendered_bb[2:])) / np.linalg.norm(np.float32(predicted_bb[2:]))
                z = diag_bb_ratio * mean_K_ratio * render_radius
            else:
                z = depth_pred


            # object center in image plane (bb center =/= object center)
            center_obj_x_train = rendered_bb[0] + rendered_bb[2]/2. - K_train[0,2]
            center_obj_y_train = rendered_bb[1] + rendered_bb[3]/2. - K_train[1,2]

            center_obj_x_test = predicted_bb[0] + predicted_bb[2]/2 - K_test[0,2]
            center_obj_y_test = predicted_bb[1] + predicted_bb[3]/2 - K_test[1,2]
            
            center_obj_mm_x = center_obj_x_test * z / K_test[0,0] - center_obj_x_train * render_radius / K_train[0,0]  
            center_obj_mm_y = center_obj_y_test * z / K_test[1,1] - center_obj_y_train * render_radius / K_train[1,1]  


            t_est = np.array([center_obj_mm_x, center_obj_mm_y, z])
            ts_est[i] = t_est

            # correcting the rotation matrix 
            # the codebook consists of centered object views, but the test image crop is not centered
            # we determine the rotation that preserves appearance when translating the object
            d_alpha_x = - np.arctan(t_est[0]/t_est[2])
            d_alpha_y = - np.arctan(t_est[1]/t_est[2])
            R_corr_x = np.array([[1,0,0],
                                [0,np.cos(d_alpha_y),-np.sin(d_alpha_y)],
                                [0,np.sin(d_alpha_y),np.cos(d_alpha_y)]]) 
            R_corr_y = np.array([[np.cos(d_alpha_x),0,-np.sin(d_alpha_x)],
                                [0,1,0],
                                [np.sin(d_alpha_x),0,np.cos(d_alpha_x)]]) 
            R_corrected = np.dot(R_corr_y,np.dot(R_corr_x,Rs_est[i]))
            Rs_est[i] = R_corrected
        return (Rs_est, ts_est)
        



    def nearest_rotation_batch(self, session, x):
        idcs = session.run(self.nearest_neighbor_idx, {self._encoder.x: x})
        return self._dataset.viewsphere_for_embedding[idcs]

    def test_embedding(self, sess, x, normalized=True):
        
        if x.dtype == 'uint8':
            x = x/255.

        if x.ndim == 3:
            x = np.expand_dims(x, 0)
        if normalized:
            return sess.run(self.normalized_embedding_query, {self._encoder.x: x}).squeeze()
        else:
            return sess.run(self._encoder.z, {self._encoder.x: x}).squeeze()
        



    # def knearest_rotation(self, session, x, k):
    #     if x.ndim == 3:
    #         x = np.expand_dims(x, 0)
    #         cosine_similarity = session.run(self.cos_similarity, {self._encoder.x: x})
            
    #         unsorted_idcs = np.argpartition(cosine_similarity, -k, axis=1)[-k:]
    #         idcs = unsorted_idcs[np.argsort(cosine_similarity[unsorted_idcs], axis=1)]
            
    #         return self._dataset.viewsphere_for_embedding[idcs][0]      
    #     else:
    #         cosine_similarity = session.run(self.cos_similarity, {self._encoder.x: x})
    #         idcs = np.argmax(cosine_similarity, axis=1)
    #         return self._dataset.viewsphere_for_embedding[idcs]

    def update_embedding_dsprites(self, session, train_args):
        batch_size = train_args.getint('Training', 'BATCH_SIZE')
        embedding_size = train_args.getint('Embedding', 'MIN_N_VIEWS')

        J = self._encoder.latent_space_size
        embedding_z = np.empty( (embedding_size, J) )

        print('Creating embedding ..')

        self._dataset.get_sprite_training_images(train_args)

        emb_imgs = self._dataset.train_y[::1024][40:80]


        embedding_z = session.run(self._encoder.z, feed_dict={self._encoder.x: emb_imgs})



        # embedding_z = embedding_z.T
        normalized_embedding = embedding_z / np.linalg.norm( embedding_z, axis=1, keepdims=True )

        session.run(self.embedding_assign_op, {self.embedding: normalized_embedding})




    def update_embedding(self, session, batch_size):
        embedding_size = self._dataset.embedding_size
        J = self._encoder.latent_space_size
        embedding_z = np.empty( (embedding_size, J) )
        obj_bbs = np.empty( (embedding_size, 4) )

        widgets = ['Creating embedding ..: ', progressbar.Percentage(),
         ' ', progressbar.Bar(),
         ' ', progressbar.Counter(), ' / %s' % embedding_size,
         ' ', progressbar.ETA(), ' ']
        bar = progressbar.ProgressBar(maxval=embedding_size,widgets=widgets)

        bar.start()
        for a, e in u.batch_iteration_indices(embedding_size, batch_size):

            batch, obj_bbs_batch = self._dataset.render_embedding_image_batch(a, e)
            embedding_z[a:e] = session.run(self._encoder.z, feed_dict={self._encoder.x: batch})

            if self.embed_bb:
                obj_bbs[a:e] = obj_bbs_batch

            bar.update(e)
        bar.finish()
        # embedding_z = embedding_z.T
        normalized_embedding = embedding_z / np.linalg.norm( embedding_z, axis=1, keepdims=True )

        session.run(self.embedding_assign_op, {self.embedding: normalized_embedding})

        if self.embed_bb:
            session.run(self.embed_obj_bbs_assign_op, {self.embed_obj_bbs: obj_bbs})
