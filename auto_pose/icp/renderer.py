import os
import numpy as np
import configparser

from meshrenderer import meshrenderer, meshrenderer_phong
from auto_pose.ae.utils import lazy_property
from auto_pose.ae.pysixd_stuff import misc

class SynRenderer(object):
    def __init__(self,test_args):
        model_base_path = test_args.get('ICP','base_path')
        models = eval(test_args.get('ICP','models'))
        self.model_paths = [os.path.join(model_base_path, model) for model in models]
        self.has_vertex_color = test_args.getboolean('ICP','has_vertex_color')
        self.vertex_scale = test_args.getint('ICP','vertex_scale')
        self.renderer



    @lazy_property
    def renderer(self):
        if self.has_vertex_color:
            # meshrenderer works also for models with vertex color
            return meshrenderer_phong.Renderer(self.model_paths,1,'.')
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
                        t=np.array([0,0,t_est[2]]),
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

