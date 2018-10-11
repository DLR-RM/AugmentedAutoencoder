from meshrenderer import meshrenderer, meshrenderer_phong
from auto_pose.ae.utils import lazy_property
from auto_pose.ae.pysixd_stuff import misc

class SynRenderer(object):
    def __init__(self,train_args):
        self.model_path = train_args.get('Paths','MODEL_PATH')
        self.model = train_args.get('Dataset','MODEL')
        if os.path.exists(self.model_path.replace(self.model,'cad')):
            self.model_path = MODEL_PATH.replace(self.model,'cad')
        self.renderer

    @lazy_property
    def renderer(self):
        return meshrenderer.Renderer([self.model_path],1,'.',1)
        # if self.model == 'cad':
        # if self.model == 'reconst':
        #   return meshrenderer_phong.Renderer([self.model_path],1,'.',1)

    def generate_synthetic_depth(self,K_test, R_est, t_est, test_shape):

        # renderer = meshrenderer.Renderer(['/net/rmc-lx0050/home_local/sund_ma/data/SLC_precise_blue.ply'],1,'.',1)
        # R = transform.random_rotation_matrix()[:3,:3]
        W_test, H_test = test_shape[:2]

        _, depth_x = self.renderer.render( 
                        obj_id=0,
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

    def render_trafo(self, K_test, R_est, t_est, test_shape, downSample = 1):
        W_test, H_test = test_shape[:2]

        bgr, depth_x = self.renderer.render( 
                        obj_id=0,
                        W=W_test,#/downSample, 
                        H=H_test,#/downSample,
                        K=K_test, 
                        R=R_est, 
                        t=np.array(t_est),
                        near=10,
                        far=10000,
                        random_light=False
                    )
        return bgr

