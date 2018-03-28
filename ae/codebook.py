# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
import progressbar

import utils as u

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
        
        self.cos_similarity = tf.matmul(self.normalized_embedding_query, self.embedding_normalized,transpose_b=True)
        self.nearest_neighbor_idx = tf.argmax(self.cos_similarity, axis=1)


    def nearest_rotation(self, session, x):
        if x.dtype == 'uint8':
            x = x/255.

        if x.ndim == 3:
            x = np.expand_dims(x, 0)
            cosine_similarity = session.run(self.cos_similarity, {self._encoder.x: x})
            idcs = np.argmax(cosine_similarity, axis=1)
            return self._dataset.viewsphere_for_embedding[idcs][0]      
        else:
            cosine_similarity = session.run(self.cos_similarity, {self._encoder.x: x})
            idcs = np.argmax(cosine_similarity, axis=1)
            return self._dataset.viewsphere_for_embedding[idcs]

    def nearest_rotation_with_bb_depth(self, session, x, predicted_bb, K_test, top_n, train_args, depth_pred=None):
        
        if x.dtype == 'uint8':
            x = x/255.
            # print 'converted uint8 to float type'

        if x.ndim == 3:
            x = np.expand_dims(x, 0)
        # print np.max(x)
        cosine_similarity, normalized_test_code = session.run([self.cos_similarity,self.normalized_embedding_query], {self._encoder.x: x})


        if top_n > 1:
            unsorted_max_idcs = np.argpartition(-cosine_similarity.squeeze(), top_n)[:top_n]
            idcs = unsorted_max_idcs[np.argsort(-cosine_similarity.squeeze()[unsorted_max_idcs])]
        else:
            idcs = np.argmax(cosine_similarity, axis=1)
        
        nearest_train_codes = session.run(self.embedding_normalized)[idcs]
        Rs_est = self._dataset.viewsphere_for_embedding[idcs]

        
        # test_depth = f_test / f_train * render_radius * diag_bb_ratio
        K_train = np.array(eval(train_args.get('Dataset','K'))).reshape(3,3)
        render_radius = train_args.getfloat('Dataset','RADIUS')

        K00_ratio = K_test[0,0] / K_train[0,0]  
        K11_ratio = K_test[1,1] / K_train[1,1]  
        mean_K_ratio = np.mean([K00_ratio,K11_ratio])
        
        embed_obj_bbs = session.run(self.embed_obj_bbs_var) if depth_pred is None else None
        
        ts_est = np.empty((top_n,3))
        for i,idx in enumerate(idcs):
            if depth_pred is None:
                rendered_bb = embed_obj_bbs[idx].squeeze()
                diag_bb_ratio = np.linalg.norm(np.float32(rendered_bb[2:])) / np.linalg.norm(np.float32(predicted_bb[2:]))
                depth_pred = diag_bb_ratio * render_radius * mean_K_ratio

            center_bb_x = (predicted_bb[0] + predicted_bb[2]/2)
            center_bb_y = (predicted_bb[1] + predicted_bb[3]/2)

            # t = K_test_cam_inv * center_bb * depth_pred
            center_mm_tx = (center_bb_x - K_test[0,2]) * depth_pred / K_test[0,0]
            center_mm_ty = (center_bb_y - K_test[1,2]) * depth_pred / K_test[1,1]
            t_est = np.array([center_mm_tx, center_mm_ty, depth_pred])
            ts_est[i] = t_est
        return (Rs_est, ts_est, normalized_test_code.squeeze(),nearest_train_codes.squeeze())
        



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

        print 'Creating embedding ..'

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
        print 'Creating embedding ..'
        bar = progressbar.ProgressBar(
            maxval=embedding_size, 
            widgets=[' [', progressbar.Timer(), ' | ', progressbar.Counter('%0{}d / {}'.format(len(str(embedding_size)), embedding_size)), ' ] ', 
            progressbar.Bar(), 
            ' (', progressbar.ETA(), ') ']
        )
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
