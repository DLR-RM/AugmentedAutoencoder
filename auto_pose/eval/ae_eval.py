import tensorflow as tf
import numpy as np
import cv2
import argparse
import configparser
import shutil
import os
import sys
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from auto_pose.ae import factory
from auto_pose.ae import utils as u
from auto_pose.eval import eval_utils, icp_utils, eval_plots, latex_report
from sixd_toolkit.pysixd import inout, pose_error
from sixd_toolkit.params import dataset_params
from sixd_toolkit.tools import eval_calc_errors, eval_loc

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('experiment_name')
    parser.add_argument('evaluation_name')
    parser.add_argument('--eval_cfg', default='eval.cfg', required=False)
    parser.add_argument('--at_step', default=None, required=False)
    arguments = parser.parse_args()
    full_name = arguments.experiment_name.split('/')
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    evaluation_name = arguments.evaluation_name
    eval_cfg = arguments.eval_cfg
    at_step = arguments.at_step

    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    train_cfg_file_path = u.get_config_file_path(workspace_path, experiment_name, experiment_group)
    eval_cfg_file_path = u.get_eval_config_file_path(workspace_path, eval_cfg=eval_cfg)

    train_args = configparser.ConfigParser()
    eval_args = configparser.ConfigParser()
    train_args.read(train_cfg_file_path)
    eval_args.read(eval_cfg_file_path)
    
    #[DATA]
    dataset_name = eval_args.get('DATA','DATASET')
    obj_id = eval_args.getint('DATA','OBJ_ID')
    scenes = eval(eval_args.get('DATA','SCENES')) if len(eval(eval_args.get('DATA','SCENES'))) > 0 else eval_utils.get_all_scenes_for_obj(eval_args)
    cam_type = eval_args.get('DATA','cam_type')
    data_params = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)
    #[BBOXES]
    estimate_bbs = eval_args.getboolean('BBOXES', 'ESTIMATE_BBS')
    #[METRIC]
    top_nn = eval_args.getint('METRIC','TOP_N')
    #[EVALUATION]
    icp = eval_args.getboolean('EVALUATION','ICP')    

    evaluation_name = evaluation_name + '_icp' if icp else evaluation_name
    evaluation_name = evaluation_name + '_bbest' if estimate_bbs else evaluation_name

    data = dataset_name + '_' + cam_type if len(cam_type) > 0 else dataset_name

    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)
    ckpt_dir = u.get_checkpoint_dir(log_dir)
    eval_dir = u.get_eval_dir(log_dir, evaluation_name, data)

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    shutil.copy2(eval_cfg_file_path, eval_dir)

    codebook, dataset, decoder = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset = True, return_decoder = True)
    dataset.renderer
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.Session(config=config)
    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir, at_step=at_step)
    

    # if estimate_bbs:
    #     #Object Detection, seperate from main
    #     # sys.path.append('/net/rmc-lx0050/home_local/sund_ma/src/SSD_Tensorflow')
    #     # from ssd_detector import SSD_detector
    #     # #TODO: set num_classes, network etc.
    #     # ssd = SSD_detector(sess, num_classes=31, net_shape=(300,300))
    #     from rmcssd.bin import detector
    #     ssd = detector.Detector(eval_args.get('BBOXES','CKPT'))

    
    t_errors = []
    R_errors = []
    all_test_visibs = []

    # if eval_args.getboolean('EVALUATION','EVALUATE_ERRORS'):    
    #     eval_loc.match_and_eval_performance_scores(eval_args, eval_dir)
    #     exit()

    test_embeddings = []  
    for scene_id in scenes:

        test_imgs = eval_utils.load_scenes(scene_id, eval_args)
        test_imgs_depth = eval_utils.load_scenes(scene_id, eval_args, depth=True) if icp else None

        if estimate_bbs:
            print eval_args.get('BBOXES','EXTERNAL')
            if eval_args.get('BBOXES','EXTERNAL') == 'False':
                # bb_preds = {}
                # for i,img in enumerate(test_imgs):
                #     print img.shape
                #     bb_preds[i] = ssd.detectSceneBBs(img, min_score=.2, nms_threshold=.45)
                # # inout.save_yaml(os.path.join(scene_res_dir,'bb_preds.yml'), bb_preds)
                # print bb_preds
                print('only externally loaded BBOXES suppported. Precompute and save them as yaml files.')
                exit()
            else:
                bb_preds = inout.load_yaml(os.path.join(eval_args.get('BBOXES','EXTERNAL'),'{:02d}.yml'.format(scene_id)))


            test_img_crops, test_img_depth_crops, bbs, bb_scores, visibilities = eval_utils.generate_scene_crops(test_imgs, test_imgs_depth, bb_preds, eval_args, train_args)
        else:
            test_img_crops, test_img_depth_crops, bbs, bb_scores, visibilities = eval_utils.get_gt_scene_crops(scene_id, eval_args, train_args)

        if len(test_img_crops) == 0:
            print 'ERROR: object %s not in scene %s' % (obj_id,scene_id)
            exit()

        info = inout.load_info(data_params['scene_info_mpath'].format(scene_id))
        Ks_test = [np.array(v['cam_K']).reshape(3,3) for v in info.values()]

        ######remove
        gts = inout.load_gt(data_params['scene_gt_mpath'].format(scene_id))
        visib_gts = inout.load_yaml(data_params['scene_gt_stats_mpath'].format(scene_id, 15))
        #######
        W_test, H_test = data_params['test_im_size']

        icp_renderer = icp_utils.SynRenderer(train_args) if icp else None
        noof_scene_views = eval_utils.noof_scene_views(scene_id, eval_args)

        test_embeddings.append([])

        scene_res_dir = os.path.join(eval_dir, '{scene_id:02d}'.format(scene_id = scene_id))
        if not os.path.exists(scene_res_dir):
            os.makedirs(scene_res_dir)

        for view in range(noof_scene_views):
            try:
                test_crops, test_crops_depth, test_bbs, test_scores, test_visibs = eval_utils.select_img_crops(test_img_crops[view][obj_id], 
                                                                                                               test_img_depth_crops[view][obj_id] if icp else None,
                                                                                                               bbs[view][obj_id],
                                                                                                               bb_scores[view][obj_id], 
                                                                                                               visibilities[view][obj_id], 
                                                                                                               eval_args)
            except:
                print 'no detections'
                continue

            print view
            preds = {}
            pred_views = []
            all_test_visibs.append(test_visibs[0])
            t_errors_crop = []
            R_errors_crop = []
            
            for i, (test_crop, test_bb, test_score) in enumerate(zip(test_crops, test_bbs, test_scores)):    

                if train_args.getint('Dataset','C') == 1:
                    test_crop = cv2.cvtColor(test_crop,cv2.COLOR_BGR2GRAY)[:,:,None]
                start = time.time()
                Rs_est, ts_est = codebook.auto_pose6d(sess, 
                                                                    test_crop, 
                                                                    test_bb, 
                                                                    Ks_test[view].copy(), 
                                                                    top_nn, 
                                                                    train_args)
                ae_time = time.time() - start
                run_time = ae_time + bb_preds[view][0]['det_time'] if estimate_bbs else ae_time

                if eval_args.getboolean('PLOT','EMBEDDING_PCA'):
                    test_embeddings[-1].append(codebook.test_embedding(sess,test_crop,normalized=True))



                # icp = False if view<350 else True
                #TODO: 
                Rs_est_old, ts_est_old = Rs_est.copy(), ts_est.copy()
                for p in range(top_nn):
                    if icp:
                        start = time.time()
                        # icp only along tz
                        R_est_refined, t_est_refined = icp_utils.icp_refinement(test_crops_depth[i], icp_renderer,Rs_est[p],
                            ts_est[p], Ks_test[view].copy(), (W_test, H_test),depth_only=True, max_mean_dist_factor=5.0)
                        print ts_est[p]
                        print t_est_refined

                        # x,y update,does not change tz:
                        _, ts_est_refined, _ = codebook.auto_pose6d(sess, test_crop, test_bb, Ks_test[view].copy(), top_nn, train_args,depth_pred=t_est_refined[2])
                        t_est_refined = ts_est_refined[p]

                        # rotation icp, only accepted if below 20 deg change
                        R_est_refined, _ = icp_utils.icp_refinement(test_crops_depth[i], icp_renderer,R_est_refined,t_est_refined, Ks_test[view].copy(), (W_test, H_test), no_depth=True)
                        print Rs_est[p]
                        print R_est_refined


                        icp_time = time.time() - start
                        Rs_est[p], ts_est[p] = R_est_refined, t_est_refined
                    
                        
                    preds.setdefault('ests',[]).append({'score':test_score, 'R': Rs_est[p], 't':ts_est[p]})
                run_time = run_time + icp_time if icp else run_time

                min_t_err, min_R_err = eval_plots.print_trans_rot_errors(gts[view], obj_id, ts_est, ts_est_old, Rs_est, Rs_est_old)
                t_errors_crop.append(min_t_err)
                R_errors_crop.append(min_R_err)
                                       
                if eval_args.getboolean('PLOT','RECONSTRUCTION'):
                    eval_plots.plot_reconstruction_test(sess, codebook._encoder, decoder, test_crop)
                    # eval_plots.plot_reconstruction_train(sess, decoder, nearest_train_codes[0])
                if eval_args.getboolean('PLOT','NEAREST_NEIGHBORS') and not icp:
                    for R_est, t_est in zip(Rs_est,ts_est):
                        pred_views.append(dataset.render_rot( R_est ,downSample = 2))
                    eval_plots.show_nearest_rotation(pred_views, test_crop, view)
                if eval_args.getboolean('PLOT','SCENE_WITH_ESTIMATE'):
                    eval_plots.plot_scene_with_estimate(test_imgs[view].copy(),icp_renderer.renderer if icp else dataset.renderer,Ks_test[view].copy(), Rs_est_old[0], 
                                                        ts_est_old[0],Rs_est[0], ts_est[0],test_bb, test_score, obj_id, gts[view], bb_preds[view] if estimate_bbs else None)


                if cv2.waitKey(1) == 32:
                    cv2.waitKey(0)


            t_errors.append(t_errors_crop[np.argmin(np.linalg.norm(np.array(t_errors_crop),axis=1))])
            R_errors.append(R_errors_crop[np.argmin(np.linalg.norm(np.array(t_errors_crop),axis=1))])
                    

            # save predictions in sixd format
            res_path = os.path.join(scene_res_dir,'%04d_%02d.yml' % (view,obj_id))
            inout.save_results_sixd17(res_path, preds, run_time=run_time)
            
    if not os.path.exists(os.path.join(eval_dir,'latex')):
        os.makedirs(os.path.join(eval_dir,'latex'))
    if not os.path.exists(os.path.join(eval_dir,'figures')):
        os.makedirs(os.path.join(eval_dir,'figures'))

    if eval_args.getboolean('EVALUATION','COMPUTE_ERRORS'):
        eval_calc_errors.eval_calc_errors(eval_args, eval_dir)
    if eval_args.getboolean('EVALUATION','EVALUATE_ERRORS'):    
        eval_loc.match_and_eval_performance_scores(eval_args, eval_dir)

    cyclo = train_args.getint('Embedding','NUM_CYCLO')
    if eval_args.getboolean('PLOT','EMBEDDING_PCA'):
        embedding = sess.run(codebook.embedding_normalized)
        eval_plots.compute_pca_plot_embedding(eval_dir, embedding[::cyclo], np.array(test_embeddings[0]))
    if eval_args.getboolean('PLOT','VIEWSPHERE'):
        eval_plots.plot_viewsphere_for_embedding(dataset.viewsphere_for_embedding[::cyclo], eval_dir)
    if eval_args.getboolean('PLOT','CUM_T_ERROR_HIST'):
        eval_plots.plot_t_err_hist(np.array(t_errors), eval_dir)
        eval_plots.plot_t_err_hist2(np.array(t_errors), eval_dir)
    if eval_args.getboolean('PLOT','CUM_R_ERROR_HIST'):
        eval_plots.plot_R_err_hist(eval_args, eval_dir, scenes)
        eval_plots.plot_R_err_hist2(np.array(R_errors), eval_dir)
    if eval_args.getboolean('PLOT','CUM_VSD_ERROR_HIST'):
        eval_plots.plot_vsd_err_hist(eval_args, eval_dir, scenes)
    if eval_args.getboolean('PLOT','VSD_OCCLUSION'):
        eval_plots.plot_vsd_occlusion(eval_args, eval_dir, scenes, np.array(all_test_visibs))
    if eval_args.getboolean('PLOT','R_ERROR_OCCLUSION'):
        eval_plots.plot_re_rect_occlusion(eval_args, eval_dir, scenes, np.array(all_test_visibs))
    if eval_args.getboolean('PLOT','ANIMATE_EMBEDDING_PCA'):
        eval_plots.animate_embedding_path(test_embeddings[0])
    if eval_args.getboolean('PLOT','RECONSTRUCTION_TEST_BATCH'):
        eval_plots.plot_reconstruction_test_batch(sess, codebook, decoder, test_img_crops, noof_scene_views, obj_id, eval_dir=eval_dir)
        # plt.show()    

        # calculate 6D pose errors
        # print 'exiting ...'
        # eval_calc_errors.eval_calc_errors(eval_args, eval_dir)
        # calculate 6D pose errors


    report = latex_report.Report(eval_dir,log_dir)
    report.write_configuration(train_cfg_file_path,eval_cfg_file_path)
    report.merge_all_tex_files()
    report.include_all_figures()
    report.save(open_pdf=False)

if __name__ == '__main__':
    main()
