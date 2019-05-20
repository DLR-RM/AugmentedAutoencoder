import os
import numpy as np
import cv2
from collections import defaultdict
import hashlib
import glob

import matplotlib.pyplot as plt
from sixd_toolkit.pysixd import transform, pose_error, view_sampler, inout
from sixd_toolkit.params import dataset_params

from auto_pose.eval import eval_plots, eval_utils
from auto_pose.ae import utils as u


def compute_plot_emb_invariance(args_latent):
    Rs, lon_lat, pts = eval_plots.generate_view_points(noof=101)
    syn_crops = []
    z_train = np.zeros((len(Rs), encoder.latent_space_size))
    for R in Rs:
        syn_crops.append(dataset.render_rot(R, obj_id=1)/255.)
    for a, e in u.batch_iteration_indices(len(Rs), 200):
        print a
        z_train[a:e] = sess.run(encoder.z, feed_dict={
                                encoder._input: syn_crops[a:e]})

    aug = eval(args_latent.get('Emb_invariance', 'aug'))

    batch = []
    orig_img = (syn_crops[100]*255).astype(np.uint8)  # H, W, C,  C H W
    for i in xrange(200):
        print i
        img = aug.augment_image(orig_img.copy()).astype(np.float32) / 255.
        #img = img.transpose( (1, 2, 0) ) #C H, W 1, 2,
        batch.append(img)
    batch = np.array(batch)
    z_test = sess.run(encoder.z, feed_dict={encoder._input: batch})

    eval_plots.compute_pca_plot_embedding(
        '', z_train, z_test=z_test, lon_lat=None, save=False, inter_factor=1)
    from gl_utils import tiles
    import cv2
    mean_var = np.mean(np.var(z_test, axis=0))
    cv2.imshow('mean_var: %s' % mean_var, tiles(batch, 10, 20))
    cv2.waitKey(0)
    plt.show()
