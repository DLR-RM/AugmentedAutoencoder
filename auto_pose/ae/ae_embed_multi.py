# -*- coding: utf-8 -*-
import os
import configparser
import argparse
import numpy as np
import signal
import progressbar
import tensorflow as tf

from auto_pose.ae import ae_factory as factory
from auto_pose.ae import utils as u

def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument('--at_step', default=None,  type=int, required=False)
    parser.add_argument('--model_path', type=str, required=True)
    arguments = parser.parse_args()
    full_name = arguments.experiment_name.split('/')

    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    at_step = arguments.at_step
    model_path = arguments.model_path

    cfg_file_path = u.get_config_file_path(workspace_path, experiment_name, experiment_group)
    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)

    ckpt_dir = u.get_checkpoint_dir(log_dir)
    dataset_path = u.get_dataset_path(workspace_path)

    if not os.path.exists(cfg_file_path):
        print('Could not find config file:\n')
        print('{}\n'.format(cfg_file_path))
        exit(-1)

    args = configparser.ConfigParser()
    args.read(cfg_file_path)
    iteration = args.getint('Training', 'NUM_ITER') if at_step is None else at_step
    
    checkpoint_file_basename = u.get_checkpoint_basefilename(log_dir, latest=iteration, joint=True)
    if not tf.train.checkpoint_exists(checkpoint_file_basename):
        checkpoint_file_basename = u.get_checkpoint_basefilename(log_dir, latest=iteration, joint=False)

    checkpoint_single_encoding = u.get_checkpoint_basefilename(log_dir, latest=iteration, model_path=model_path)
    target_checkpoint_file = u.get_checkpoint_basefilename(log_dir, joint=True)

    print(checkpoint_file_basename)
    print(target_checkpoint_file)
    print(ckpt_dir)
    print('#'*20)
    
    with tf.variable_scope(experiment_name):
        dataset = factory.build_dataset(dataset_path, args)
        queue = factory.build_queue(dataset, args)
        encoder = factory.build_encoder(queue.x, args)
        # decoder = factory.build_decoder(queue.y, encoder, args)
        # ae = factory.build_ae(encoder, decoder, args)
        # before_cb = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        codebook_multi = factory.build_codebook_multi(encoder, dataset, args, checkpoint_file_basename)
        restore_saver = tf.train.Saver(save_relative_paths=True, max_to_keep=100)

        codebook_multi.add_new_codebook_to_graph(model_path)
        # inters_vars = before_cb.intersection(set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        saver = tf.train.Saver(save_relative_paths=True, max_to_keep=100)

    batch_size = args.getint('Training', 'BATCH_SIZE')*len(eval(args.get('Paths', 'MODEL_PATH')))
    model = args.get('Dataset', 'MODEL')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        print(ckpt_dir)
        # print sess.run(encoder.global_step)
        print('#'*20)

        # factory.restore_checkpoint(sess, saver, ckpt_dir, at_step=at_step)
        sess.run(tf.global_variables_initializer())
        restore_saver.restore(sess, checkpoint_file_basename)

        print('#'*20)
        # chkpt = tf.train.get_checkpoint_state(ckpt_dir)
        # if chkpt and chkpt.model_checkpoint_path:
        #     print chkpt.model_checkpoint_path
        #     saver.restore(sess, chkpt.model_checkpoint_path)
        # else:
        #     print 'No checkpoint found. Expected one in:\n'
        #     print '{}\n'.format(ckpt_dir)
        #     exit(-1)

        try:
            loaded_emb = tf.train.load_variable(checkpoint_single_encoding, experiment_name + '/embedding_normalized')
            loaded_obj_bbs = tf.train.load_variable(checkpoint_single_encoding, experiment_name + '/embed_obj_bbs_var')
        except:
            loaded_emb = None
            loaded_obj_bbs = None

        if model=='dsprites':
            codebook_multi.update_embedding_dsprites(sess, args)
        else:
            codebook_multi.update_embedding(sess, batch_size, model_path, loaded_emb=loaded_emb, loaded_obj_bbs=loaded_obj_bbs)

        print('Saving new checkoint ..')
        
        saver.save(sess, target_checkpoint_file, global_step=iteration)

        print('done')

if __name__ == '__main__':
    main()