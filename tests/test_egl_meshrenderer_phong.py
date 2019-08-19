import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, '..'))
import glob
import numpy as np
import random
from auto_pose.meshrenderer import meshrenderer_phong
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
random.seed(12)


def test_egl():
    # cad_path = '/home_local/sund_ma/data/linemod_dataset/models'
    # NOTE: in $ROOT, mkdir -p data; ln -sf /path/to/SIXD_DATASETS data/SIXD_DATASETS
    cad_path = osp.join(cur_dir, '../data/SIXD_DATASETS/hinterstoisser/models')
    assert osp.exists(cad_path), "cad_path {} does not exist. Check your dataset path!".format(cad_path)
    K = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])
    height = 480
    width = 640
    clip_near = 10
    clip_far = 5000

    models_cad_files = sorted(glob.glob(os.path.join(cad_path, '*.ply')))
    num_obj = len(models_cad_files)
    renderer = meshrenderer_phong.Renderer(
        models_cad_files,
        1
    )
    renderer = meshrenderer_phong.Renderer(models_cad_files,
        samples=1, vertex_tmp_store_folder='.', clamp=False, vertex_scale=1.0)
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    t = np.array([0, 0, 0.7], dtype=np.float32) * 1000
    start = time.time()
    num_runs = 3000
    for i in tqdm(range(num_runs)):
        rand_obj_id = random.randint(0, num_obj-1)
        # print(i, rand_obj_id)
        color, depth = renderer.render(
            rand_obj_id, width, height, K
            , R, t, clip_near, clip_far)
    dt = time.time() - start
    print('{}s, {}fps'.format(dt, num_runs/dt))
    plt.subplot(1,2,1)
    plt.imshow(color[:,:,[2,1,0]])
    plt.title('color')
    plt.subplot(1,2,2)
    plt.imshow(depth)
    plt.title('depth')
    plt.show()


if __name__=="__main__":
    test_egl()
