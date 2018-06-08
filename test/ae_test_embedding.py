from eval import eval_plots
from ae import factory
from ae import utils as u
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

import argparse
import ConfigParser
import os
import tensorflow as tf
import numpy as np
import cv2

from gl_utils import tiles

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('experiment_name')
    arguments = parser.parse_args()

    full_name = arguments.experiment_name.split('/')
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    train_cfg_file_path = u.get_config_file_path(workspace_path, experiment_name, experiment_group)
    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)
    ckpt_dir = u.get_checkpoint_dir(log_dir)
    train_args = ConfigParser.ConfigParser()
    train_args.read(train_cfg_file_path)

    print train_args.items('Dataset')

    codebook, dataset, decoder = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset = True, return_decoder = True)

    model_type = train_args.get('Dataset','MODEL')
    if model_type=='dsprites':
        dataset.get_sprite_training_images(train_args)
    else:
        dataset_path = u.get_dataset_path(workspace_path)
        dataset.get_training_images(dataset_path, train_args)

    # print np.max(dataset.train_y[::1024][40:80])
    idcs_random_trans = []
    idcs_random = []
    idcs_random_scale = np.arange(1024*40+512+16,1024*80,1024)
    idcs_orig = np.arange(1024*200+512+16,1024*240,1024)
    for i in xrange(200,240):
        idcs_random_trans.append(np.random.randint(i*1024,(i+1)*1024))
    for i in xrange(40):
        s = np.random.randint(0,6)
        idcs_random.append(np.random.randint((s*40+i)*1024,(s*40+i+1)*1024))

    print idcs_orig

    # im = tiles(dataset.train_x[np.array(idcs_random_trans)[::3]],2,7,3,3)
    # im2 = tiles(dataset.train_x[np.array(idcs_random_scale)[::3]],2,7,3,3)
    # im3 = tiles(dataset.train_x[np.array(idcs_orig)[::3]],2,7,3,3)
    # im4 = tiles(dataset.train_x[np.array(idcs_random)[::3]],2,7,3,3)
    # cv2.imshow('trans',im)
    # cv2.imshow('scale',im2)
    # cv2.imshow('orig',im3)
    # cv2.imshow('orig',im4)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(os.path.join(log_dir,'train_figures','trans.png'),im)
    # cv2.imwrite(os.path.join(log_dir,'train_figures','scale.png'),im2)
    # cv2.imwrite(os.path.join(log_dir,'train_figures','y.png'),im3)
    # cv2.imwrite(os.path.join(log_dir,'train_figures','rand.png'),im4)

    with tf.Session() as sess:
        factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)
        if model_type=='dsprites':
            y_codes = sess.run(codebook._encoder.z, {codebook._encoder.x: dataset.train_y[idcs_orig]/255.})
            test_codes = sess.run(codebook._encoder.z, {codebook._encoder.x: dataset.train_x[idcs_random_scale]/255.})
            test_codes_trans = sess.run(codebook._encoder.z, {codebook._encoder.x: dataset.train_x[np.array(idcs_random_trans)]/255.})
            # y_codes_normed = sess.run(codebook.embedding_normalized, {codebook._encoder.x: dataset.train_y[::1024][:160]/255.})
        else:
            test_codes=sess.run(codebook._encoder.z, {codebook._encoder.x: dataset.train_y[:256]/255.})
            test_codes_normed=sess.run(codebook.embedding_normalized, {codebook._encoder.x: dataset.train_y[:256]/255.})
        # train_embedding = sess.run(codebook.embedding_normalized)
        print test_codes.shape, test_codes_trans.shape

    # plt.figure()
    norm = np.linalg.norm(y_codes,axis=1)
    matplotlib.rcParams.update({'font.size': 22})
    c = np.linspace(0,1,4)
    x = np.linspace(0,360,40)
    if model_type=='dsprites':
        norm_test = np.linalg.norm(test_codes,axis=1)
        norm_test_trans = np.linalg.norm(test_codes_trans,axis=1)
        plt.figure()
        plt.xticks(np.arange(0,361,90))
        plt.xlabel('rotation angle [deg]')
        plt.ylabel('z1')
        plt.grid()
        plt.plot(x,y_codes[:,0]/norm,color='green',linewidth=3,label='z1: y',alpha=0.7,markersize = 5)
        plt.plot(x,test_codes[:,0]/norm_test,linestyle='--',linewidth=3,color='red',label='z1: y @ scale = 0.6',dashes=(5, 5),alpha=0.7, markersize = 5)
        plt.plot(x,test_codes_trans[:,0]/norm_test_trans,linestyle='--',linewidth=3,color='blue',label='z1: y @ random translation',dashes=(5, 5),alpha=0.7, markersize = 5)
        # plt.scatter(y_codes[:,0]/norm,y_codes[:,1]/norm,c='red',marker=marker='-.',label=)
        # for i in xrange(4):
        plt.legend(fontsize='small',loc=4)
        plt.tight_layout()
        tikz_save(os.path.join(log_dir,'train_figures','z1.tikz'))
        plt.savefig(os.path.join(log_dir,'train_figures','z1.pdf'))

        plt.figure()
        plt.ylabel('z2')
        plt.xlabel('rotation angle [deg]')
        plt.xticks(np.arange(0,361,90))
        plt.grid()
        plt.plot(x,y_codes[:,1]/norm,color='green',linewidth=3,label='z2: y',alpha=0.7,markersize = 5)
        plt.plot(x,test_codes[:,1]/norm_test,linestyle='--',linewidth=3,color='red',label='z2: y @ scale = 0.6',dashes=(5, 5),alpha=0.7, markersize = 5)
        plt.plot(x,test_codes_trans[:,1]/norm_test_trans,linestyle='--',linewidth=3,color='blue',label='z2: y @ random translation',dashes=(5, 5),alpha=0.7, markersize = 5)

        # plt.scatter(y_codes[:,0]/norm,y_codes[:,1]/'--',norm,c='red',label=)
        plt.legend(fontsize='small', loc=4)
        plt.tight_layout()
        tikz_save(os.path.join(log_dir,'train_figures','z2.tikz'))
        plt.savefig(os.path.join(log_dir,'train_figures','z2.pdf'))
    else:
        eval_plots.compute_pca_plot_embedding('',test_codes,save=False)
        eval_plots.compute_pca_plot_embedding('',test_codes_normed[:256],save=False)
    # plt.figure()
    # plt.plot(train_embedding[:,0],train_embedding[:,1])


    plt.show()

if __name__ == '__main__':
    main()
