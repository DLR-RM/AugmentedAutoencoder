# -*- coding: utf-8 -*-
import os
import ConfigParser
import argparse
import numpy as np
import signal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import progressbar

import ae_factory as factory
import utils as u

def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print 'Please define a workspace path:\n'
        print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
        exit(-1)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    arguments = parser.parse_args()
    experiment_name = arguments.experiment_name

    cfg_file_path = u.get_config_file_path(workspace_path, experiment_name)
    checkpoint_file = u.get_checkpoint_basefilename(workspace_path, experiment_name)
    log_dir = u.get_log_dir(workspace_path, experiment_name)
    dataset_path = u.get_dataset_path(workspace_path)

    if not os.path.exists(cfg_file_path):
        print 'Could not find config file:\n'
        print '{}\n'.format(cfg_file_path)
        exit(-1)

    args = ConfigParser.ConfigParser()
    args.read(cfg_file_path)

    with tf.variable_scope(experiment_name):
        dataset = factory.build_dataset(False, dataset_path, args)
        queue = factory.build_queue(dataset, args)
        encoder = factory.build_encoder(queue.x, args)
        decoder = factory.build_decoder(queue.y, encoder, args)
        ae = factory.build_ae(encoder, decoder)
        optimize = factory.build_optimizer(ae, args)
        codebook = factory.build_codebook(encoder, dataset)
        saver = tf.train.Saver()

    batch_size = args.getint('Training', 'BATCH_SIZE')

    with tf.Session() as sess:
        chkpt = tf.train.get_checkpoint_state(log_dir)
        if chkpt and chkpt.model_checkpoint_path:
            saver.restore(sess, chkpt.model_checkpoint_path)
        else:
            print 'No checkpoint found. Expected one in:\n'
            print '{}\n'.format(log_dir)
            exit(-1)
        
        codebook.update_embedding(sess, batch_size)
        print 'Saving new checkoint ..',
        saver.save(sess, checkpoint_file, global_step=ae.global_step)
        print 'done',

if __name__ == '__main__':
    main()