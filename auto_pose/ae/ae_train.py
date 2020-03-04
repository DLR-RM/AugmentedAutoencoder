 # -*- coding: utf-8 -*-
import os
import configparser
import argparse
import numpy as np
import signal
import shutil
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import progressbar
import tensorflow as tf

from auto_pose.ae import ae_factory as factory
from auto_pose.ae import utils as u

try:
    range = xrange
except NameError:
    # when running on Python3
    pass


def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path is None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    gentle_stop = np.array((1,), dtype=np.bool)
    gentle_stop[0] = False
    def on_ctrl_c(signal, frame):
        gentle_stop[0] = True
    signal.signal(signal.SIGINT, on_ctrl_c)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("-d", action='store_true', default=False)
    parser.add_argument("-gen", action='store_true', default=False)
    parser.add_argument('--at_step', default=None,  type=int, required=False)

    arguments = parser.parse_args()

    full_name = arguments.experiment_name.split('/')
    
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    
    debug_mode = arguments.d
    generate_data = arguments.gen
    at_step = arguments.at_step

    cfg_file_path = u.get_config_file_path(workspace_path, experiment_name, experiment_group)
    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)
    checkpoint_file = u.get_checkpoint_basefilename(log_dir)
    ckpt_dir = u.get_checkpoint_dir(log_dir)
    train_fig_dir = u.get_train_fig_dir(log_dir)
    dataset_path = u.get_dataset_path(workspace_path)
    
    if not os.path.exists(cfg_file_path):
        print('Could not find config file:\n')
        print(('{}\n'.format(cfg_file_path)))
        exit(-1)
        
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(train_fig_dir):
        os.makedirs(train_fig_dir)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    args = configparser.ConfigParser(inline_comment_prefixes="#")
    args.read(cfg_file_path)

    shutil.copy2(cfg_file_path, log_dir)

    num_iter = args.getint('Training', 'NUM_ITER') if not debug_mode else np.iinfo(np.int32).max
    save_interval = args.getint('Training', 'SAVE_INTERVAL')
    num_gpus = args.getint('Training', 'NUM_GPUS')

    with tf.device('/device:CPU:0'): 
        with tf.variable_scope(experiment_name, reuse=tf.AUTO_REUSE):
            
            dataset = factory.build_dataset(dataset_path, args)
            multi_queue = factory.build_multi_queue(dataset, args)
            if generate_data:
                # dataset.load_bg_images(dataset_path)
                multi_queue.create_tfrecord_training_images(dataset_path, args)
                print('finished generating training images')
                exit()

            dev_splits = np.array_split(np.arange(multi_queue._num_objects), num_gpus)

            iterator = multi_queue.create_iterator(dataset_path, args)

            all_x, all_y = list(zip(*[(inp[0], inp[2]) for inp in multi_queue.next_element]))
            all_x, all_y = tf.concat(all_x, axis=0), tf.concat(all_y, axis=0)
            print(all_x.shape)
            encoding_splits = []
            for dev in range(num_gpus):
                with tf.device('/device:GPU:%s' % dev):   
                    sta = dev_splits[dev][0] * multi_queue._batch_size
                    end = (dev_splits[dev][-1]+1) * multi_queue._batch_size
                    print(sta, end)
                    encoder = factory.build_encoder(all_x[sta:end], args, target=all_y[sta:end], is_training=True)
                    encoding_splits.append(tf.split(encoder.z, len(dev_splits[dev]),0))

        with tf.variable_scope(experiment_name):
            decoders = []
            for dev in range(num_gpus):     
                with tf.device('/device:GPU:%s' % dev):  
                    for j,i in enumerate(dev_splits[dev]):
                        print(len(encoding_splits))
                        decoders.append(factory.build_decoder(multi_queue.next_element[i], encoding_splits[dev][j], args, is_training=True, idx=i))
            
            ae = factory.build_ae(encoder, decoders, args)
            codebook = factory.build_codebook(encoder, dataset, args)
            train_op = factory.build_train_op(ae, args)
            saver = tf.train.Saver(save_relative_paths=True, max_to_keep=1)

        # dataset.get_training_images(dataset_path, args)
    # dataset.load_bg_images(dataset_path)
    multi_queue.create_tfrecord_training_images(dataset_path, args)

    if generate_data:
        print(('finished generating synthetic training data for ' + experiment_name))
        print('exiting...')
        exit()


    widgets = ['Training: ', progressbar.Percentage(),
         ' ', progressbar.Bar(),
         ' ', progressbar.Counter(), ' / %s' % num_iter,
         ' ', progressbar.ETA(), ' ']
    bar = progressbar.ProgressBar(maxval=num_iter,widgets=widgets)


    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.9)
    config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)

    with tf.Session(config=config) as sess:

        sess.run(multi_queue.bg_img_init.initializer)
        sess.run(iterator.initializer)
       
        u.create_summaries(multi_queue, decoders, ae)
        merged_loss_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(ckpt_dir, sess.graph)

        chkpt = tf.train.get_checkpoint_state(ckpt_dir)
        if chkpt and chkpt.model_checkpoint_path:
            if at_step is None:
                # checkpoint_file_basename = u.get_checkpoint_basefilename(log_dir,latest=args.getint('Training', 'NUM_ITER'))
                checkpoint_file_basename = chkpt.model_checkpoint_path
            else:
                checkpoint_file_basename = u.get_checkpoint_basefilename(log_dir,latest=at_step)
            print(('loading ', checkpoint_file_basename))
            saver.restore(sess, checkpoint_file_basename)
            # except:
            #     print 'loading ', chkpt.model_checkpoint_path
            #     saver.restore(sess, chkpt.model_checkpoint_path)
        else:            
            if encoder._pre_trained_model != 'False':
                encoder.saver.restore(sess, encoder._pre_trained_model)
                all_vars = set([var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
                var_list = all_vars.symmetric_difference([v[1] for v in list(encoder.fil_var_list.items())])
                sess.run(tf.variables_initializer(var_list))
                print(sess.run(tf.report_uninitialized_variables()))
            else:
                sess.run(tf.global_variables_initializer())

        if not debug_mode:
            print(('Training with %s model' % args.get('Dataset','MODEL'), os.path.basename(args.get('Paths','MODEL_PATH'))))
            bar.start()

        for i in range(encoder.global_step.eval(), num_iter):
            if not debug_mode:
                # print 'before optimize'
                sess.run([train_op,multi_queue.next_bg_element])
                # print 'after optimize'
                if (i+1) % 100 == 0:
                    merged_summaries = sess.run(merged_loss_summary)
                    summary_writer.add_summary(merged_summaries, i)

                bar.update(i)

                if (i+1) % save_interval == 0:
                    saver.save(sess, checkpoint_file, global_step=encoder.global_step)

                    # this_x, this_y = sess.run([queue.x, queue.y])
                    # reconstr_train = sess.run(decoder.x,feed_dict={queue.x:this_x})

                    this, reconstr_train = sess.run([multi_queue.next_element,[decoder.x for decoder in decoders]])
                    this_x = np.concatenate([el[0] for el in this])
                    this_y = np.concatenate([el[2] for el in this])
                    # reconstr_train = sess.run(,feed_dict={queue.x:this_x})
                    reconstr_train = np.concatenate(reconstr_train)
                    for imgs in [this_x,this_y,reconstr_train]:
                        np.random.seed(0)
                        np.random.shuffle(imgs)
                    train_imgs = np.hstack(( u.tiles(this_x, 4, 4), u.tiles(reconstr_train, 4,4),u.tiles(this_y, 4, 4)))
                    cv2.imwrite(os.path.join(train_fig_dir,'training_images_%s.png' % i), train_imgs*255)
            else:
                
                this,_,reconstr_train  = sess.run([multi_queue.next_element,multi_queue.next_bg_element,[decoder.x for decoder in decoders]])

                this_x = np.concatenate([el[0] for el in this])
                this_y = np.concatenate([el[2] for el in this])
                print(this_x.shape, reconstr_train[0].shape, len(reconstr_train))
                reconstr_train = np.concatenate(reconstr_train,axis=0)
                for imgs in [this_x,this_y,reconstr_train]:
                    np.random.seed(0)
                    np.random.shuffle(imgs)
                print(this_x.shape)
                cv2.imshow('sample batch', np.hstack(( u.tiles(this_x, 4, 6), u.tiles(reconstr_train, 4,6),u.tiles(this_y, 4, 6))) )
                k = cv2.waitKey(0)
                if k == 27:
                    break

            if gentle_stop[0]:
                break

        if not debug_mode:
            bar.finish()
        if not gentle_stop[0] and not debug_mode:
            print('To create the embedding run:\n')
            print(('ae_embed {}\n'.format(full_name)))

if __name__ == '__main__':
    main()
    
