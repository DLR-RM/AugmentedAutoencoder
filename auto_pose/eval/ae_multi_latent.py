 # -*- coding: utf-8 -*-
import os
import configparser
import argparse
import numpy as np
import signal
import shutil
import cv2
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import progressbar
import tensorflow as tf

from auto_pose.ae import ae_factory as factory
from auto_pose.ae import utils as u
from auto_pose.eval import eval_plots,eval_utils


def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print 'Please define a workspace path:\n'
        print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
        exit(-1)

    gentle_stop = np.array((1,), dtype=np.bool)
    gentle_stop[0] = False
    def on_ctrl_c(signal, frame):
        gentle_stop[0] = True
    signal.signal(signal.SIGINT, on_ctrl_c)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument('--model_path', default=None, required=True)
    parser.add_argument("-d", action='store_true', default=False)
    parser.add_argument("-gen", action='store_true', default=False)
    parser.add_argument("-vis_emb", action='store_true', default=False)
    parser.add_argument('--at_step', default=None,  type=int, required=False)


    arguments = parser.parse_args()

    full_name = arguments.experiment_name.split('/')
    model_path = arguments.model_path
    
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    
    debug_mode = arguments.d
    generate_data = arguments.gen
    at_step = arguments.at_step

    cfg_file_path = u.get_config_file_path(workspace_path, experiment_name, experiment_group)
    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)
    ckpt_dir = u.get_checkpoint_dir(log_dir)
    train_fig_dir = u.get_train_fig_dir(log_dir)
    dataset_path = u.get_dataset_path(workspace_path)
    
    if not os.path.exists(cfg_file_path):
        print 'Could not find config file:\n'
        print '{}\n'.format(cfg_file_path)
        exit(-1)


    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    if at_step is None:
        checkpoint_file = u.get_checkpoint_basefilename(log_dir, model_path, latest=args.getint('Training', 'NUM_ITER'))
    else:
        checkpoint_file = u.get_checkpoint_basefilename(log_dir, model_path, latest=at_step)

    num_iter = args.getint('Training', 'NUM_ITER') if not debug_mode else np.iinfo(np.int32).max
    save_interval = args.getint('Training', 'SAVE_INTERVAL')
    num_gpus = 1
    model_type = args.get('Dataset', 'MODEL')

    codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset = True)
    encoder = codebook._encoder
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)


    # with tf.variable_scope(experiment_name, reuse=tf.AUTO_REUSE):
    #     dataset = factory.build_dataset(dataset_path, args)
    #     multi_queue = factory.build_multi_queue(dataset, args)
    #     dev_splits = np.array_split(np.arange(24), num_gpus)

    #     iterator = multi_queue.create_iterator(dataset_path, args)
    #     all_object_views = tf.concat([inp[0] for inp in multi_queue.next_element],0)

    #     bs = multi_queue._batch_size
    #     encoding_splits = []
    #     for dev in xrange(num_gpus):
    #         with tf.device('/device:GPU:%s' % dev):   
    #             encoder = factory.build_encoder(all_object_views[dev_splits[dev][0]*bs:(dev_splits[dev][-1]+1)*bs], args, is_training=False)
    #             encoding_splits.append(tf.split(encoder.z, len(dev_splits[dev]),0))

    # with tf.variable_scope(experiment_name):
    #     decoders = []
    #     for dev in xrange(num_gpus):     
    #         with tf.device('/device:GPU:%s' % dev):  
    #             for j,i in enumerate(dev_splits[dev]):
    #                 decoders.append(factory.build_decoder(multi_queue.next_element[i], encoding_splits[dev][j], args, is_training=False, idx=i))

    #     ae = factory.build_ae(encoder, decoders, args)
    #     codebook = factory.build_codebook(encoder, dataset, args)
    #     train_op = factory.build_train_op(ae, args)
    #     saver = tf.train.Saver(save_relative_paths=True)

    # dataset.load_bg_images(dataset_path)
    # multi_queue.create_tfrecord_training_images(dataset_path, args)

    # widgets = ['Training: ', progressbar.Percentage(),
    #      ' ', progressbar.Bar(),
    #      ' ', progressbar.Counter(), ' / %s' % num_iter,
    #      ' ', progressbar.ETA(), ' ']
    # bar = progressbar.ProgressBar(maxval=num_iter,widgets=widgets)


    # gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.9)
    # config = tf.ConfigProto(gpu_options=gpu_options,log_device_placement=True,allow_soft_placement=True)

    # all_aligned_train = sorted(glob.glob('/net/rmc-lx0314/home_local/sund_ma/data/modelnet_aligned/modelnet40_aligned_normalized_cent/car/train/*_normalized.off'))
    # all_aligned_test = sorted(glob.glob('/net/rmc-lx0314/home_local/sund_ma/data/modelnet_aligned/modelnet40_aligned_normalized_cent/car/test/*_normalized.off'))
    all_aligned_train = sorted(glob.glob('/net/rmc-lx0314/home_local/sund_ma/data/modelnet_aligned/modelnet40_manually_aligned/car/train/*_normalized.off'))
    all_aligned_test = sorted(glob.glob('/net/rmc-lx0314/home_local/sund_ma/data/modelnet_aligned/modelnet40_manually_aligned/car/test/*_normalized.off'))
    all_aligned_sim_dif_cat = sorted(glob.glob('/net/rmc-lx0314/home_local/sund_ma/data/modelnet_aligned/modelnet40_manually_aligned/sofa/train/*_normalized.off'))
    all_aligned_dif_cat = sorted(glob.glob('/net/rmc-lx0314/home_local/sund_ma/data/modelnet_aligned/modelnet40_manually_aligned/toilet/test/*_normalized.off'))
    num_o = 3
    dataset._kw['model_path'] = all_aligned_train[3:3+num_o] + all_aligned_test[3:3+num_o] + all_aligned_sim_dif_cat[3:3+num_o] + all_aligned_dif_cat[3:3+num_o]
    # dataset._kw['model_path'] = all_aligned_train[:num_o] + all_aligned_test[:3]# + all_aligned_test[3:3+num_o] + all_aligned_sim_dif_cat[3:3+num_o] + all_aligned_dif_cat[3:3+num_o]
    print dataset._kw['model_path']



    import matplotlib.pyplot as plt
    from sixd_toolkit.pysixd import transform,pose_error,view_sampler
    import time


## rot error histogram CB with 3D model
############################
    # pose_errs = []
    # for i in range(1,num_o+1):
    #     for j in range(3):
    #         random_R = transform.random_rotation_matrix()[:3,:3]
    #         # DeepIM


    #         while True:
    #             rand_direction = transform.make_rand_vector(3)
    #         #     rand_angle_x = np.random.normal(0,(15/180.*np.pi)**2)
    #         #     rand_angle_y = np.random.normal(0,(15/180.*np.pi)**2)
    #         #     rand_angle_z = np.random.normal(0,(15/180.*np.pi)**2)

    #         #     R_off = transform.euler_matrix(rand_angle_x,rand_angle_y,rand_angle_z)
    #         #     angle_off,_,_ = transform.rotation_from_matrix(R_off)
    #         #     print angle_off*180/np.pi

    #             rand_angle = np.random.normal(0,45/180.*np.pi)
    #             R_off = transform.rotation_matrix(rand_angle,rand_direction)[:3,:3]
    #             random_R_pert = np.dot(R_off,random_R)
    #             print rand_angle
    #             if abs(rand_angle) < 45/180.*np.pi and abs(rand_angle) > 5/180.*np.pi:
    #                 break

            
    #         ###
    #         random_t_pert = np.array([0,0,700])# + np.array([np.random.normal(0,10),np.random.normal(0,10),np.random.normal(0,50)])
    #         print random_t_pert
    #         # random_R = dataset.viewsphere_for_embedding[np.random.randint(0,92000)]
    #         import cv2
    #         rand_test_view_crop, bb = dataset.render_rot(random_R, obj_id=i, return_bb=True)

    #         # _, _, rand_test_view_whole_target = dataset.render_rot(random_R_pert, obj_id=i, t=random_t_pert, return_bb=True, return_orig=True)
    #         # rand_test_view_crop = dataset.extract_square_patch(rand_test_view_whole_target, bb, float(dataset._kw['pad_factor']))
    #         # rand_test_view_whole_target = rand_test_view_whole_target/255.
    #         # rand_test_view_crop = rand_test_view_crop/255.
            
            
    #         # K = eval(dataset._kw['k'])
    #         # K = np.array(K).reshape(3,3)
    #         # bgr_y,_ = dataset.renderer.render( 
    #         #     obj_id=i,
    #         #     W=720, 
    #         #     H=540,
    #         #     K=K.copy(), 
    #         #     R=random_R, 
    #         #     t=np.array([0.,0,650]),
    #         #     near=10,
    #         #     far=10000,
    #         #     random_light=False
    #         # )


    #         # cv2.imshow('in',rand_test_view_crop)
    #         # cv2.imshow('translated and rotated', rand_test_view_crop)
    #         # cv2.waitKey(0)

    #         # Rs_est = codebook.nearest_rotation(sess, rand_test_view, top_n=1)
    #         st = time.time()
    #         # session, x, top_n, budget=10, epochs=3, high=6./180*np.pi, obj_id=0, top_n_refine=1
    #         R_refined,_ = codebook.refined_nearest_rotation(sess, rand_test_view_crop, 1, R_init=random_R_pert, budget=40, epochs=4, high=45./180*np.pi, obj_id=i, top_n_refine=1)
    #         # R = codebook.nearest_rotation(sess, rand_test_view, 1)
    #         # R = codebook.nearest_rotation(sess, rand_test_view, 1)
    #         # R_refined = R_refined[np.newaxis,:]
    #         print time.time() - st

    #         pose_errs.append(pose_error.re(random_R,R_refined[0]))



    #         # _, _, rand_test_view_whole = dataset.render_rot(R_refined[0], obj_id=i, return_bb=True,return_orig=True)
    #         # z_est = eval_utils.align_images(rand_test_view_whole_target, rand_test_view_whole/255., random_t_pert[2], warp_mode = cv2.MOTION_AFFINE)
    #         # print z_est



    #         # pose_errs[-1] = np.minimum(pose_errs[-1],np.abs(pose_errs[-1]-180))
    #         # import cv2
    #         # cv2.imshow('inserted_view',rand_test_view)
    #         # cv2.imshow('pert_view',rand_init_view)
    #         # cv2.imshow('est_view', dataset.render_rot(R_refined[0],obj_id=i)/255.)
    #         # cv2.waitKey(0)

    #         if pose_errs[-1]>170:
    #             cv2.imshow('inserted_view',rand_test_view)
    #             cv2.imshow('est_view', dataset.render_rot(R_refined[0],obj_id=i)/255.)
    #             cv2.waitKey(1)

    # plt.hist(pose_errs, bins=180)
    # pose_errs = np.array(pose_errs)
    # plt.title('median: ' + str(np.median(pose_errs)) + ', mean: ' + str(np.mean(pose_errs)) + ', <5deg: ' + str(len(pose_errs[pose_errs<5])/1.0/len(pose_errs)))
    # plt.show()



##############################################

# generate PCA directions from all objects

    Rs,lon_lat,_ = eval_plots.generate_view_points(noof=201,num_cyclo=5)
    all_ztrain=[]
    for i in range(0,num_o*2,4):
        syn_crops = []
        z_train = np.zeros((len(Rs),encoder.latent_space_size))
        for R in Rs:
            
            syn_crops.append(dataset.render_rot(R,obj_id=i)/255.)
        for a, e in u.batch_iteration_indices(len(Rs), 200):
            print e
            z_train[a:e] = sess.run(encoder.z,feed_dict={encoder._input:syn_crops[a:e]})
        all_ztrain.append(z_train)
    all_ztrain = np.array(all_ztrain).reshape(-1,128)
    pca_all = eval_plots.compute_pca_plot_embedding('',all_ztrain,lon_lat=list(lon_lat)*5,save=False)

##########################


    Rs,lon_lat,_ = eval_plots.generate_azim_elev_points(noof=36*8)

    fig = plt.figure( figsize=(3*num_o,3*4))
    fig.subplots_adjust(top=0.95, bottom=0.05)
    # plt.title('Embedding Principal Components')
    imgs = []
    axes = []
    for o in xrange(0,num_o*4):
        syn_crops = []
        for R in Rs:
            if o >= 2*num_o and o<3*num_o:
                R_rot = np.array([[0,1,0],[1,0,0],[0,0,1]])
                syn_crops.append(dataset.render_rot(np.dot(R,R_rot),obj_id=o)/255.)
            else:
                syn_crops.append(dataset.render_rot(R,obj_id=o)/255.)
        syn_crops=np.array(syn_crops)
        imgs.append(syn_crops[np.linspace(0,len(syn_crops),8,endpoint=False).astype(np.int32)])
        # im = u.tiles(np.array(syn_crops),12,18*4,scale=0.66)
        z_train = np.zeros((len(Rs),encoder.latent_space_size))
        # cv2.imshow('',im)
        # cv2.waitKey(1)

        for a, e in u.batch_iteration_indices(len(Rs), 200):
            print e
            z_train[a:e] = sess.run(encoder.z,feed_dict={encoder._input:syn_crops[a:e]})
        # eval_plots.compute_pca_plot_embedding('',z_train,lon_lat=lon_lat,save=False)


        ax = fig.add_subplot(4, num_o, o+1, projection='3d')
        # if o>=3*num_o:
        #     pca_all=None
        eval_plots.compute_pca_plot_azelin(36*8+1,z_train,pca=pca_all,save=False,inter_factor=1, normalize=False, fig=fig,ax=ax)
        axes.append(ax)
    
    axes[-1].legend()



        # for j in range(len(Rs)):
        #     Rs_est = codebook.nearest_rotation(sess, syn_crops[j], top_n=1)
        #     est_view = dataset.render_rot(Rs_est.squeeze(),obj_id=0)/255.
        #     cv2.imshow('inserted_view',syn_crops[j])
        #     cv2.imshow('est_view',est_view)
        #     cv2.waitKey(0)

    def on_move(event):
        ax_i = axes.index(event.inaxes)
        for ax_ in axes:
            # if ax_ is not axes[ax_i]:
            ax_.view_init(elev=axes[ax_i].elev, azim=axes[ax_i].azim)
            ax_.set_xlim3d(axes[ax_i].get_xlim3d())
            ax_.set_ylim3d(axes[ax_i].get_ylim3d())
            ax_.set_zlim3d(axes[ax_i].get_zlim3d())
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    
    im = u.tiles(np.array(imgs).reshape(-1,128,128,3),num_o*4,8,scale=1)
    cv2.imshow('',im)
    cv2.waitKey(1)
    plt.show()       


if __name__ == '__main__':
    main()
    
