import numpy as np
import cv2
import time

from auto_pose.meshrenderer.pysixd import transform, misc
from auto_pose.meshrenderer import meshrenderer_phong, meshrenderer_phong_normals, meshrenderer

# renderer = meshrenderer.Renderer(['/home_local/sund_ma/tmp/DB9_rgb_n.ply'], 
#                 samples=1, 
#                 vertex_tmp_store_folder='.')

# renderer = meshrenderer_phong.Renderer(['/home_local/sund_ma/tmp/DB9_rgb_n.ply'], 
#                 samples=1, 
#                 vertex_tmp_store_folder='.')#,
                #vertex_scale=float(1)) 

renderer = meshrenderer_phong.Renderer(['/volume/pekdat/datasets/public/YCB_VideoDataset/original2sixd/models/002_master_chef_can/textured.ply'], 
              samples=1,
              vertex_tmp_store_folder='.',
              vertex_scale=float(1000))

for i in range(100):
    st = time.time()
    rgb, depth = renderer.render(obj_id = 0,
                W = 1920,
                H = 1200,
                K = np.array([2745.772652, 0, 973.410518, 0, 2745.392277, 613.325858, 0, 0, 1]).reshape(3,3),
                R = transform.random_rotation_matrix()[:3,:3],
                t = np.array([0, 0, 1000.]),
                near = 10,
                far = 10000,
                random_light=False,
                phong={'ambient':0.4, 'diffuse':0.8+0.2*np.random.rand(), 'specular':0.3+0.2*np.random.rand()})
    print((time.time() - st))

    cv2.imshow('rgb',rgb)
    #### visualization ####

    ys, xs = np.nonzero(depth > 0)
    obj_bb = misc.calc_2d_bbox(xs, ys, (1000,1000))
    x, y, w, h = obj_bb
    size = np.maximum(h, w)
    left = x+w/2-size/2
    right = x+w/2+size/2
    top = y+h/2-size/2
    bottom = y+h/2+size/2

    rgb = rgb[top:bottom, left:right]
    depth_img = depth[top:bottom, left:right]
    # depth_img = depth.copy()

    # normal_img = normal[top:bottom, left:right]

    # plt.hist(depth_img.reshape(-1))
    # plt.show()

    depth_img[depth_img>0] = (depth_img[depth_img>0]-np.mean(depth_img[depth_img>0]))/(10*np.std(depth_img[depth_img>0]))
    depth_img[depth_img!=0] = 1-(depth_img[depth_img!=0]+0.5)

    depth_img = np.dstack((depth_img/np.max(depth_img),)*3)
    rgb = cv2.resize( rgb, (224,224) )
    depth_img = cv2.resize( depth_img, (224,224) )
    # normal_img = cv2.resize( normal_img, (224,224) )

    cv2.imshow('', np.hstack((rgb/255.,depth_img)))#,normal_img/255.)))
    cv2.waitKey(0)
cv2.destroyAllWindows()