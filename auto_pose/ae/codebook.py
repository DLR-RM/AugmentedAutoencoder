# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
import progressbar

from .utils import lazy_property
from . import utils as u

import time

class Codebook(object):

    def __init__(self, encoder, dataset, embed_bb):
        
        self._encoder = encoder
        self._dataset = dataset
        self.embed_bb = embed_bb

        latent_dims = encoder.latent_space_size
        embedding_size = self._dataset.embedding_size

        self.normalized_embedding_query = tf.nn.l2_normalize(self._encoder.z, 1)
        self.embedding_normalized = tf.Variable(
            np.zeros((embedding_size, latent_dims)),

            dtype=tf.float32,
            trainable=False,
            name='embedding_normalized'
        )


        self.embedding = tf.placeholder(tf.float32, shape=[embedding_size, latent_dims])
        self.embedding_assign_op = tf.assign(self.embedding_normalized, self.embedding)
        
        if embed_bb:
            self.embed_obj_bbs_var = tf.Variable(
                np.zeros((self.embedding_size, 4)),
                dtype=tf.int32,
                trainable=False,
                name='embed_obj_bbs_var'
            )
            self.embed_obj_bbs = tf.placeholder(tf.int32, shape=[self.embedding_size, 4])
            self.embed_obj_bbs_assign_op = tf.assign(self.embed_obj_bbs_var, self.embed_obj_bbs)
            self.embed_obj_bbs_values = None
        
        self.cos_similarity = tf.matmul(self.normalized_embedding_query, self.embedding_normalized,transpose_b=True)
        self._image_ph = tf.placeholder(tf.float32, [None,] + list(self._dataset.shape))
        self.image_ph_tofloat = self._image_ph/255.

    def refined_nearest_rotation(self, session, target_view, top_n, R_init=None, t_init=None, budget=10, epochs=3,
                                 high=6./180*np.pi, obj_id=0, top_n_refine=1, target_bb=None):

        from sixd_toolkit.pysixd import transform,pose_error
        from sklearn.metrics.pairwise import cosine_similarity

        if target_view.dtype == 'uint8':
            target_view = target_view/255.
        if target_view.ndim == 3:
            target_view = np.expand_dims(target_view, 0)

        cosine_similar, orig_in_emb = session.run([self.cos_similarity,self.normalized_embedding_query], {self._encoder.x: target_view})
        
        if top_n_refine==1:
            idcs = np.argmax(cosine_similar, axis=1)
            # orig_cosine_sim = cosine_similar[0,idcs]
        else:
            unsorted_max_idcs = np.argpartition(-cosine_similar.squeeze(), top_n_refine)[:top_n_refine]
            idcs = unsorted_max_idcs[np.argsort(-cosine_similar.squeeze()[unsorted_max_idcs])]
            # orig_cosine_sim = cosine_similar[0,idcs[0]]
        print('original cosine sim: ', cosine_similar[0,idcs])

        ### intitializing rotation estimates from existing codebook
        Rs = self._dataset.viewsphere_for_embedding[idcs].copy()
        Rs_new = [Rs[0]]
        for R in Rs:
            res = [pose_error.re(R_new,R) for R_new in Rs_new] 
            if np.min(res) > 80:
                Rs_new.append(R)

        if R_init is None:
            Rs = Rs_new[:]
        ######
        else:
            Rs = [R_init]


        top_n_new = len(Rs)
        max_cosine_sim = 0.0
        K = eval(self._dataset._kw['k'])
        K = np.array(K).reshape(3,3)
        render_dims = eval(self._dataset._kw['render_dims'])
        clip_near = float(self._dataset._kw['clip_near'])
        clip_far = float(self._dataset._kw['clip_far'])
        pad_factor = float(self._dataset._kw['pad_factor'])

        R_perts = []
        fine_views = []
        bbs = []

        for j in range(epochs):
            noof_perts = budget * top_n_new
            for i in range(noof_perts):
                if j>0 and i==0:
                    R_perts = Rs
                    print(noof_perts)
                    # fine_views = list(view_best) if not isinstance(view_best, list) else view_best
                    # bbs = list(bb_best) if not isinstance(bbs, list) else bbs
                    continue
                if i < top_n_new and j==0:
                    R_off = np.eye(3,3)
                else:
                    rand_direction = transform.make_rand_vector(3)
                    rand_angle = np.random.uniform(0,high/(j+1))
                    R_off = transform.rotation_matrix(rand_angle,rand_direction)[:3,:3]

                R_pert = np.dot(R_off,Rs[i%top_n_new])
                R_perts.append(R_pert)
             
                if target_bb is not None and t_init is not None:
                    bgr_full, _ = self._dataset.renderer.render(
                        obj_id=obj_id,
                        W=render_dims[0],
                        H=render_dims[1],
                        K=K.copy(),
                        R=R_pert,
                        t=t_init,
                        near=clip_near,
                        far=clip_far,
                        random_light=False
                    )
                    bgr = self._dataset.extract_square_patch(bgr_full, target_bb, pad_factor)
                    obj_bb = np.array([0,0,1,1])
                else:
                    bgr,obj_bb = self._dataset.render_rot(R_pert, downSample=1, obj_id=obj_id, return_bb=True)

                fine_views.append(bgr)
                bbs.append(obj_bb)


            float_imgs = session.run(self.image_ph_tofloat,{self._image_ph:np.array(fine_views)})
            normalized_embedding_query = session.run(self.normalized_embedding_query, {self._encoder.x: float_imgs})
            cosine_sim = cosine_similarity(orig_in_emb, normalized_embedding_query)
            idx = np.argmax(cosine_sim, axis=1)
            R_perts = np.array(R_perts)
            
            if cosine_sim[0,idx] >= max_cosine_sim:

                max_cosine_sim = cosine_sim[0, idx]

                fine_views = np.array(fine_views)
                bbs = np.array(bbs)
                
                unsorted_max_idcs = np.argpartition(-cosine_sim.squeeze(), top_n_refine)[:top_n_refine]
                idcs = unsorted_max_idcs[np.argsort(-cosine_sim.squeeze()[unsorted_max_idcs])]
                
                Rs = R_perts[idcs]
                view_best = fine_views[idcs[0]]
                bb_best = bbs[idcs[0]]
                # if top_n_new > 1:´

                

                # cv2.imshow('refined',view_best)
                # cv2.imshow('orig_rendered',fine_views[0])
                # cv2.imshow('orig_in',x[0])
                # cv2.waitKey(0)

                view_best_new = [view_best.squeeze()]
                bb_best_new = [bb_best.squeeze()]

                ##  if more than one neighbor, look for far apart alternatives 
                Rs_new = [Rs[0]]
                for r,R in enumerate(Rs):
                    res = [pose_error.re(R_new,R) for R_new in Rs_new] 
                    if np.min(res) > 80:
                        Rs_new.append(R)
                        view_best_new.append(fine_views[idcs[r]])
                        bb_best_new.append(bbs[idcs[r]])


                Rs = Rs_new[:]
                fine_views = view_best_new[:]
                bbs = bb_best_new[:]

                top_n_new = len(view_best_new)

                print('refined')
                # cv2.imshow('chosen', fine_views[idx])
                # cv2.waitKey(0)
                
                
                # Rs_best = list(Rs)
            else:
                print('not refined')
        print('final cosine sim: ', max_cosine_sim)
        # idx = np.argmax(cosine_sim)

        return np.array(Rs)[0:top_n], bbs[0:top_n]



    def nearest_rotation(self, session, x, top_n=1, upright=False, return_idcs=False):
        
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



    def auto_pose6d(self, session, x, predicted_bb, K_test, top_n, train_args, depth_pred=None, upright=False, refine=False):
        
        if refine:
            Rs_est,rendered_bbs = self.refined_nearest_rotation(session, x, top_n, budget=30, epochs=2, obj_id=0, top_n_refine=2)
            rendered_bb = rendered_bbs[0]
        else:
            idcs = self.nearest_rotation(session, x, top_n=top_n, upright=upright,return_idcs=True)
            Rs_est = self._dataset.viewsphere_for_embedding[idcs]


        # if test_codes:
        #     if x.ndim == 3:
        #         x = np.expand_dims(x, 0)
        #     normalized_test_code = session.run(self.normalized_embedding_query, {self._encoder.x: x})

        # test_depth = f_test / f_train * render_radius * bb_diag_ratio
        K_train = np.array(eval(train_args.get('Dataset','K'))).reshape(3,3)
        render_radius = train_args.getfloat('Dataset','RADIUS')

        K_diag_ratio = np.sqrt(K_test[0,0]**2 + K_test[1,1]**2) / np.sqrt(K_train[0,0]**2 + K_train[1,1]**2)  
        # mean_K_ratio = np.mean([K00_ratio,K11_ratio])

        if self.embed_obj_bbs_values is None:
            self.embed_obj_bbs_values = session.run(self.embed_obj_bbs_var)

        ts_est = np.empty((top_n,3))

        for i,R_est in enumerate(Rs_est):

            if not refine:
                rendered_bb = self.embed_obj_bbs_values[idcs[i]].squeeze()
                
            if depth_pred is None:
                bb_diag_ratio = np.linalg.norm(np.float32(rendered_bb[2:])) / np.linalg.norm(np.float32(predicted_bb[2:]))
                z = bb_diag_ratio * K_diag_ratio * render_radius
            else:
                z = depth_pred

            # object center in image plane (bb center =/= object center)

            # center_obj_x = predicted_bb[0] + predicted_bb[2]/2 - K_test[0,2] - (rendered_bb[0] + rendered_bb[2]/2. - K_train[0,2])
            # center_obj_y = predicted_bb[1] + predicted_bb[3]/2 - K_test[1,2] - (rendered_bb[1] + rendered_bb[3]/2. - K_train[1,2])

            # t = K_test_cam_inv * center_bb * depth_pred
            # center_mm_tx = center_obj_x * z / K_test[0,0]
            # center_mm_ty = center_obj_y * z / K_test[1,1]

            center_obj_x_train = rendered_bb[0] + rendered_bb[2]/2. - K_train[0,2]
            center_obj_y_train = rendered_bb[1] + rendered_bb[3]/2. - K_train[1,2]

            center_obj_x_test = predicted_bb[0] + predicted_bb[2]/2 - K_test[0,2]
            center_obj_y_test = predicted_bb[1] + predicted_bb[3]/2 - K_test[1,2]

            center_mm_tx = center_obj_x_test * z / K_test[0,0] - center_obj_x_train * render_radius / K_train[0,0]  
            center_mm_ty = center_obj_y_test * z / K_test[1,1] - center_obj_y_train * render_radius / K_train[1,1]  

            t_est = np.array([center_mm_tx, center_mm_ty, z])

            ts_est[i] = t_est
            
            # correcting the rotation matrix 
            # the codebook consists of centered object views, but the test image crop is not centered
            # we determine the rotation that preserves appearance when translating the object

            d_alpha_y = np.arctan(t_est[0]/np.sqrt(t_est[2]**2+t_est[1]**2))
            d_alpha_x = - np.arctan(t_est[1]/t_est[2])
            R_corr_x = np.array([[1,0,0],
                                [0,np.cos(d_alpha_x),-np.sin(d_alpha_x)],
                                [0,np.sin(d_alpha_x),np.cos(d_alpha_x)]]) 
            R_corr_y = np.array([[np.cos(d_alpha_y),0,np.sin(d_alpha_y)],
                                [0,1,0],
                                [-np.sin(d_alpha_y),0,np.cos(d_alpha_y)]]) 

            R_corrected = np.dot(R_corr_y,np.dot(R_corr_x,R_est))

            Rs_est[i] = R_corrected
        return (Rs_est, ts_est,None)
        

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

    @lazy_property
    def embedding_size(self):
        return len(self._dataset.viewsphere_for_embedding)

    def update_embedding(self, session, batch_size, model_path):

        self._dataset._kw['model_path'] = list([str(model_path)])
        self._dataset._kw['model'] = 'cad' if 'cad' in model_path else self._dataset._kw['model']
        self._dataset._kw['model'] = 'reconst' if 'reconst' in model_path else self._dataset._kw['model']

        embedding_size = self._dataset.embedding_size
        J = self._encoder.latent_space_size
        embedding_z = np.empty( (embedding_size, J) )
        obj_bbs = np.empty( (embedding_size, 4) )
        widgets = ['Creating Embedding: ', progressbar.Percentage(),
             ' ', progressbar.Bar(),
             ' ', progressbar.Counter(), ' / %s' % embedding_size,
             ' ', progressbar.ETA(), ' ']
        bar = progressbar.ProgressBar(maxval=embedding_size,widgets=widgets)
        bar.start()
        for a, e in u.batch_iteration_indices(self.embedding_size, batch_size):

            batch, obj_bbs_batch = self._dataset.render_embedding_image_batch(a, e)
            # import cv2
            # cv2.imshow('',u.tiles(batch,10,10))
            # cv2.waitKey(0)
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
