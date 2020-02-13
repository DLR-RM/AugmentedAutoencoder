# -*- coding: utf-8 -*-
import os
import configparser
import argparse
import numpy as np
import signal
import progressbar
import tensorflow as tf
import pickle

from . import ae_factory as factory
from . import utils as u

def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument('--at_step', default=None, required=False)
    arguments = parser.parse_args()
    full_name = arguments.experiment_name.split('/')
    
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    at_step = arguments.at_step

    cfg_file_path = u.get_config_file_path(workspace_path, experiment_name, experiment_group)
    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)
    checkpoint_file = u.get_checkpoint_basefilename(log_dir)
    ckpt_dir = u.get_checkpoint_dir(log_dir)
    dataset_path = u.get_dataset_path(workspace_path)

    print(checkpoint_file)
    print(ckpt_dir)
    print('#'*20)

    if not os.path.exists(cfg_file_path):
        print('Could not find config file:\n')
        print('{}\n'.format(cfg_file_path))
        exit(-1)

    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    with tf.variable_scope(experiment_name):
        dataset = factory.build_dataset(dataset_path, args)
        queue = factory.build_queue(dataset, args)
        encoder = factory.build_encoder(queue.x, args)
        decoder = factory.build_decoder(queue.y, encoder, args)
        ae = factory.build_ae(encoder, decoder, args)
        codebook = factory.build_codebook(encoder, dataset, args)
        saver = tf.train.Saver(save_relative_paths=True)

    batch_size = args.getint('Training', 'BATCH_SIZE')
    model = args.get('Dataset', 'MODEL')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options)

    data = pickle.load(open("/shared-folder/AugmentedAutoencoder/pytorch3d/training-data/training-images.p","rb"))
    codes = []

    with tf.Session(config=config) as sess:

        print(ckpt_dir)
        print('#'*20)

        factory.restore_checkpoint(sess, saver, ckpt_dir, at_step=at_step)

        for i,img in enumerate(data["images"]):
            code = sess.run(encoder.z, feed_dict={encoder.x: [img]})
            codes.append(np.array(code[0]))
            print(i)

        coded_data = {"images": data["images"], "Rs": data["Rs"], "ts": data["ts"], "codes": codes}
        pickle.dump(coded_data, open("/shared-folder/AugmentedAutoencoder/pytorch3d/training-data/training-codes.p", "wb")) 
            
if __name__ == '__main__':
    main()
