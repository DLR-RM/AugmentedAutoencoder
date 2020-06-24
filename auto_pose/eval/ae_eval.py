import tensorflow as tf
import numpy as np
import cv2
import argparse
import configparser
import shutil
import os
import sys
import time
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from auto_pose.ae import ae_factory as factory
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
    parser.add_argument('--at_step', default=None, type=str, required=False)
    parser.add_argument('--model_path', default=None, required=True)
    arguments = parser.parse_args()
    full_name = arguments.experiment_name.split('/')
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    evaluation_name = arguments.evaluation_name
    eval_cfg = arguments.eval_cfg
    at_step = arguments.at_step
    model_path = arguments.model_path

    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)
    train_cfg_file_path = u.get_train_config_exp_file_path(log_dir, experiment_name)
    eval_cfg_file_path = u.get_eval_config_file_path(workspace_path, eval_cfg=eval_cfg)

    train_args = configparser.ConfigParser(inline_comment_prefixes="#")
    eval_args = configparser.ConfigParser(inline_comment_prefixes="#")
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
    gt_masks = eval_args.getboolean('BBOXES','gt_masks')
    estimate_masks = eval_args.getboolean('BBOXES','estimate_masks')

    #[METRIC]
    top_nn = eval_args.getint('METRIC','TOP_N')
    #[EVALUATION]
    icp = eval_args.getboolean('EVALUATION','ICP')
    gt_trans = eval_args.getboolean('EVALUATION','gt_trans')
    iterative_code_refinement = eval_args.getboolean('EVALUATION','iterative_code_refinement')
    
    H_AE = train_args.getint('Dataset','H')
    W_AE = train_args.getint('Dataset','W')

    evaluation_name = evaluation_name + '_icp' if icp else evaluation_name
    evaluation_name = evaluation_name + '_bbest' if estimate_bbs else evaluation_name
    evaluation_name = evaluation_name + '_maskest' if estimate_masks else evaluation_name
    evaluation_name = evaluation_name + '_gttrans' if gt_trans else evaluation_name
    evaluation_name = evaluation_name + '_gtmasks' if gt_masks else evaluation_name
    evaluation_name = evaluation_name + '_refined' if iterative_code_refinement else evaluation_name


    data = dataset_name + '_' + cam_type if len(cam_type) > 0 else dataset_name

    

    if at_step is None:
        checkpoint_file = u.get_checkpoint_basefilename(log_dir, False, latest=train_args.getint('Training', 'NUM_ITER'), joint=True)
    else:
        checkpoint_file = u.get_checkpoint_basefilename(log_dir, False, latest=at_step, joint=True)
    print(checkpoint_file)
    eval_dir = u.get_eval_dir(log_dir, evaluation_name, data)

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    shutil.copy2(eval_cfg_file_path, eval_dir)

    codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset = True, joint=True)
    dataset._kw['model_path'] = [model_path]
    dataset._kw['model'] = 'cad' if 'cad' in model_path else 'reconst'
    dataset._kw['model'] = 'reconst' if 'reconst' in model_path else 'cad'

    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.5)
    config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)

    t_errors = []
    R_errors = []
    all_test_visibs = []

    external_path = eval_args.get('BBOXES','EXTERNAL')

    test_embeddings = []  
    for scene_id in scenes:

        test_imgs = eval_utils.load_scenes(scene_id, eval_args)
        test_imgs_depth = eval_utils.load_scenes(scene_id, eval_args, depth=True) if icp else None

        if estimate_bbs:
            print(external_path)
            if external_path == 'False':
                bb_preds = {}
                for i,img in enumerate(test_imgs):
                    print((img.shape))
                    bb_preds[i] = ssd.detectSceneBBs(img, min_score=.05, nms_threshold=.45)
                print(bb_preds)
            else:
                if estimate_masks:
                    bb_preds = inout.load_yaml(os.path.join(external_path, '{:02d}/mask_rcnn_predict.yml'.format(scene_id)))
                    print(list(bb_preds[0][0].keys()))
                    mask_paths = glob.glob(os.path.join(external_path, '{:02d}/masks/*.npy'.format(scene_id)))
                    maskrcnn_scene_masks = [np.load(mp) for mp in mask_paths] 
                else:
                    maskrcnn_scene_masks = None
                    bb_preds = inout.load_yaml(os.path.join(external_path,'{:02d}.yml'.format(scene_id)))

            test_img_crops, test_img_depth_crops, bbs, bb_scores, visibilities = eval_utils.generate_scene_crops(test_imgs, test_imgs_depth, bb_preds, eval_args, (H_AE, W_AE), inst_masks = maskrcnn_scene_masks)
        else:
            test_img_crops, test_img_depth_crops, bbs, bb_scores, visibilities = eval_utils.get_gt_scene_crops(scene_id, eval_args, train_args, load_gt_masks = external_path if gt_masks else gt_masks)

        if len(test_img_crops) == 0:
            print(('ERROR: object %s not in scene %s' % (obj_id,scene_id)))
            exit()

        info = inout.load_info(data_params['scene_info_mpath'].format(scene_id))
        Ks_test = [np.array(v['cam_K']).reshape(3,3) for v in list(info.values())]

        ######remove
        gts = inout.load_gt(data_params['scene_gt_mpath'].format(scene_id))
        visib_gts = inout.load_yaml(data_params['scene_gt_stats_mpath'].format(scene_id, 15))
        #######
        W_test, H_test = data_params['test_im_size']

        icp_renderer = icp_utils.SynRenderer(train_args, dataset._kw['model_path'][0]) if icp else None
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
                print('no detections')
                continue

            print(view)
            preds = {}
            pred_views = []
            all_test_visibs.append(test_visibs[0])
            t_errors_crop = []
            R_errors_crop = []
            
            for i, (test_crop, test_bb, test_score) in enumerate(zip(test_crops, test_bbs, test_scores)):    

                start = time.time()
                if train_args.getint('Dataset','C') == 1:
                    test_crop = cv2.cvtColor(test_crop,cv2.COLOR_BGR2GRAY)[:,:,None]
                Rs_est, ts_est,_ = codebook.auto_pose6d(sess, 
                                                        test_crop, 
                                                        test_bb, 
                                                        Ks_test[view].copy(), 
                                                        top_nn, 
                                                        train_args,
                                                        codebook._get_codebook_name(model_path),
                                                        refine=iterative_code_refinement)
                Rs_est_old, ts_est_old = Rs_est.copy(), ts_est.copy()
                ae_time = time.time() - start

                if eval_args.getboolean('PLOT','EMBEDDING_PCA'):
                    test_embeddings[-1].append(codebook.test_embedding(sess,test_crop,normalized=True))

                if eval_args.getboolean('EVALUATION','gt_trans'):
                    ts_est = np.empty((top_nn,3))
                    for n in range(top_nn):
                        smallest_diff = np.inf
                        for visib_gt,gt in zip(visib_gts[view],gts[view]):
                            if gt['obj_id'] == obj_id:
                                diff = np.sum(np.abs(gt['obj_bb']-np.array(visib_gt['bbox_obj'])))
                                if diff < smallest_diff:
                                    smallest_diff = diff
                                    gt_obj = gt.copy()      
                                    print('Im there')                             
                        ts_est[n] = np.array(gt_obj['cam_t_m2c']).reshape(-1)

                try:
                    run_time = ae_time + bb_preds[view][0]['det_time'] if estimate_bbs else ae_time
                except:
                    run_time = ae_time

                for p in range(top_nn):
                    if icp:
                        # note: In the CVPR paper a different ICP was used
                        start = time.time()
                        # depth icp
                        R_est_refined, t_est_refined = icp_utils.icp_refinement(test_crops_depth[i], icp_renderer,Rs_est[p],
                            ts_est[p], Ks_test[view].copy(), (W_test, H_test),depth_only=True, max_mean_dist_factor=5.0)
                        print(t_est_refined)

                        # x,y update,does not change tz:
                        _, ts_est_refined, _ = codebook.auto_pose6d(sess, 
                                                        test_crop,
                                                        test_bb, 
                                                        Ks_test[view].copy(), 
                                                        top_nn, 
                                                        train_args, 
                                                        codebook._get_codebook_name(model_path), 
                                                        depth_pred=t_est_refined[2], 
                                                        refine=iterative_code_refinement)

                        t_est_refined = ts_est_refined[p]

                        # rotation icp, only accepted if below 20 deg change
                        R_est_refined, _ = icp_utils.icp_refinement(test_crops_depth[i], icp_renderer,R_est_refined,t_est_refined, Ks_test[view].copy(), (W_test, H_test), no_depth=True)
                        print((Rs_est[p]))
                        print(R_est_refined)


                        icp_time = time.time() - start
                        Rs_est[p], ts_est[p] = R_est_refined, t_est_refined
                    
                        
                    preds.setdefault('ests',[]).append({'score':test_score, 'R': Rs_est[p], 't':ts_est[p]})
                run_time = run_time + icp_time if icp else run_time

                min_t_err, min_R_err = eval_plots.print_trans_rot_errors(gts[view], obj_id, ts_est, ts_est_old, Rs_est, Rs_est_old)
                t_errors_crop.append(min_t_err)
                R_errors_crop.append(min_R_err)
                                       
                if eval_args.getboolean('PLOT','NEAREST_NEIGHBORS') and not icp:
                    for R_est, t_est in zip(Rs_est,ts_est):
                        pred_views.append(dataset.render_rot( R_est ,downSample = 2))
                    eval_plots.show_nearest_rotation(pred_views, test_crop, view)
                if eval_args.getboolean('PLOT','SCENE_WITH_ESTIMATE'):
                    eval_plots.plot_scene_with_estimate(test_imgs[view].copy(),icp_renderer.renderer if icp else dataset.renderer,Ks_test[view].copy(), Rs_est_old[0], 
                                                        ts_est_old[0], Rs_est[0], ts_est[0],test_bb, test_score, obj_id, gts[view], bb_preds[view] if estimate_bbs else None)


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
        eval_plots.plot_R_err_recall(eval_args, eval_dir, scenes)
        eval_plots.plot_R_err_hist2(np.array(R_errors), eval_dir)
    if eval_args.getboolean('PLOT','CUM_VSD_ERROR_HIST'):
        try:
            eval_plots.plot_vsd_err_hist(eval_args, eval_dir, scenes)
        except:
            pass
    if eval_args.getboolean('PLOT','VSD_OCCLUSION'):
        try:    
            eval_plots.plot_vsd_occlusion(eval_args, eval_dir, scenes, np.array(all_test_visibs))
        except:
            pass
    if eval_args.getboolean('PLOT','R_ERROR_OCCLUSION'):
        try:  
            eval_plots.plot_re_rect_occlusion(eval_args, eval_dir, scenes, np.array(all_test_visibs))
        except:
            pass
    if eval_args.getboolean('PLOT','ANIMATE_EMBEDDING_PCA'):
        eval_plots.animate_embedding_path(test_embeddings[0])

    report = latex_report.Report(eval_dir,log_dir)
    report.write_configuration(train_cfg_file_path,eval_cfg_file_path)
    report.merge_all_tex_files()
    report.include_all_figures()
    report.save(open_pdf=True)

if __name__ == '__main__':
    main()
