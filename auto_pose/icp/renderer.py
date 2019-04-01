import os
import numpy as np
import configparser

from auto_pose.meshrenderer import meshrenderer, meshrenderer_phong
from auto_pose.ae.utils import lazy_property
from auto_pose.ae.pysixd_stuff import misc

class SynRenderer(object):
    def __init__(self,test_args, all_train_args, has_vertex_color = False):
        self.model_paths = []

        for train_args in all_train_args:
            self.model_paths.append(train_args.get('Paths','model_path'))

            # self.has_vertex_color.append(True if train_args.get('Dataset','model')=='reconst' else False)

        self.has_vertex_color = has_vertex_color # try True 

        self.vertex_scale = all_train_args[0].getint('Dataset','vertex_scale')
        self.renderer



    @lazy_property
    def renderer(self):
        if self.has_vertex_color:
            # meshrenderer works also for models with vertex color
            return meshrenderer_phong.Renderer(self.model_paths,1,'.',vertex_scale=self.vertex_scale)
        else:
            return meshrenderer.Renderer(self.model_paths,1,'.',vertex_scale=self.vertex_scale)

    def generate_synthetic_depth(self,K_test, R_est, t_est, test_shape,clas_idx=0):

        # renderer = meshrenderer.Renderer(['/net/rmc-lx0050/home_local/sund_ma/data/SLC_precise_blue.ply'],1,'.',1)
        # R = transform.random_rotation_matrix()[:3,:3]
        W_test, H_test = test_shape[:2]

        _, depth_x = self.renderer.render( 
                        obj_id=clas_idx,
                        W=W_test, 
                        H=H_test,
                        K=K_test, 
                        R=R_est, 
                        t=np.array([0,0,t_est[2]]), #TODO use t_est because R is corrected now!!!
                        near=10,
                        far=10000,
                        random_light=False
                    )

        pts = misc.rgbd_to_point_cloud(K_test,depth_x)[0]

        return pts

    def render_trafo(self, K_test, R_est, t_est, test_shape, clas_idx=0):
        W_test, H_test = test_shape[:2]

        bgr, depth_x = self.renderer.render( 
                        obj_id=clas_idx,
                        W=W_test, 
                        H=H_test,
                        K=K_test, 
                        R=R_est, 
                        t=t_est,
                        near=10,
                        far=10000,
                        random_light=False
                    )
        return bgr,depth_x

