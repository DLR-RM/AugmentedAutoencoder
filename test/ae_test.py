from eval import eval_plots
from ae import factory
from ae import utils as u


import argparse
import ConfigParser
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

    with tf.Session() as sess:
        factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)
        if model_type=='dsprites':
            y_codes=sess.run(codebook._encoder.z, {codebook._encoder.x: dataset.train_y[::1024][40:80]/255.})
            test_codes=sess.run(codebook._encoder.z, {codebook._encoder.x: dataset.train_x[::1024][:160]/255.})
            y_codes_normed=sess.run(codebook.embedding_normalized, {codebook._encoder.x: dataset.train_y[::1024][:160]/255.})
        else:
            test_codes=sess.run(codebook._encoder.z, {codebook._encoder.x: dataset.train_y[:256]/255.})
            test_codes_normed=sess.run(codebook.embedding_normalized, {codebook._encoder.x: dataset.train_y[:256]/255.})
        # train_embedding = sess.run(codebook.embedding_normalized)
        print test_codes.shape
        # print train_embedding.shape

    # plt.figure()
    norm = np.linalg.norm(y_codes,axis=1)

    c = np.linspace(0,1,40)
    if model_type=='dsprites':
        plt.scatter(y_codes[:,0]/norm,y_codes[:,1]/norm,c=c,marker='o')
        for i in xrange(4):
            plt.scatter(y_codes_normed[i*40:(i+1)*40,0],y_codes_normed[i*40:(i+1)*40,1],c=c,marker='triangle_up')
    else:
        eval_plots.compute_pca_plot_embedding('',test_codes,save=False)
        eval_plots.compute_pca_plot_embedding('',test_codes_normed[:256],save=False)
    # plt.figure()
    # plt.plot(train_embedding[:,0],train_embedding[:,1])

    plt.show()

if __name__ == '__main__':
    main()
