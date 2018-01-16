
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from matplotlib2tikz import save as tikz_save
import tensorflow as tf
from gl_utils import tiles


def plot_reconstruction(sess, decoder, normed_test_code):
    reconst = sess.run(decoder.x, feed_dict={decoder._latent_code: normed_test_code})
    cv2.imshow('reconst',reconst)
    cv2.waitKey(1)



def show_nearest_rotation(pred_views, test_crop):
    nearest_views = tiles(pred_views,1,len(pred_views),10,10)
    cv2.imshow('nearest_views',nearest_views)
    cv2.imshow('test_crop',test_crop)
    cv2.waitKey(1)

def compute_pca_plot_embedding(eval_dir, z_train, z_test):
    pass
def plot_viewsphere_for_embedding(Rs_viewpoints, eval_dir):
    pass
def plot_t_err_hist(t_errors, eval_dir):
    pass
def plot_R_err_hist(top_n, eval_dir, scene_id):
    pass
def animate_embedding_path(z_test):
    pass
