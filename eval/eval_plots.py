
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib2tikz import save as tikz_save
import cv2
import os
import numpy as np
import tensorflow as tf
from gl_utils import tiles
from sklearn.decomposition import PCA

from sixd_toolkit.pysixd import inout,pose_error


def plot_reconstruction_test(sess, encoder, decoder, x):

    if x.dtype == 'uint8':
        x = x/255.
        print 'converted uint8 to float type'

    if x.ndim == 3:
        x = np.expand_dims(x, 0)

    reconst = sess.run(decoder.x, feed_dict={encoder.x: x})
    cv2.imshow('reconst_test',cv2.resize(reconst[0],(256,256)))
    

def plot_reconstruction_test_batch(sess, encoder, decoder, test_img_crops, noof_scene_views, obj_id, eval_dir=None):
    
    sample_views = np.random.choice(noof_scene_views, np.min([100,noof_scene_views]), replace=False)
    
    sample_batch = []
    i=0
    j=0
    while i < 16:
        if test_img_crops[sample_views[j]].has_key(obj_id):
            sample_batch.append(test_img_crops[sample_views[j]][obj_id][0])
            i += 1
        j += 1
            
    x = np.array(sample_batch).squeeze()
    
    if x.dtype == 'uint8':
        x = x/255.
        print 'converted uint8 to float type'
    
    reconst = sess.run(decoder.x, feed_dict={encoder.x: x})

    reconstruction_imgs = np.hstack(( tiles(x, 4, 4), tiles(reconst, 4, 4)))
    cv2.imwrite(os.path.join(eval_dir,'figures','reconstruction_imgs.png'), reconstruction_imgs*255)

def plot_reconstruction_train(sess, decoder, train_code):
    if train_code.ndim == 1:
        train_code = np.expand_dims(train_code, 0)
    reconst = sess.run(decoder.x, feed_dict={decoder._latent_code: train_code})
    cv2.imshow('reconst_train',cv2.resize(reconst[0],(256,256)))
    


def show_nearest_rotation(pred_views, test_crop):

    nearest_views = tiles(np.array(pred_views),1,len(pred_views),10,10)
    cv2.imshow('nearest_views',cv2.resize(nearest_views/255.,(256,len(pred_views)*256)))
    cv2.imshow('test_crop',cv2.resize(test_crop,(256,256)))
    

def plot_scene_with_estimate(test_img,icp_renderer,K_test, R_est_old, t_est_old,R_est_ref, t_est_ref, test_bb, test_score, obj_id, gts=[]):   

    xmin = int(test_bb[0])
    ymin = int(test_bb[1])
    xmax = int(test_bb[0]+test_bb[2])
    ymax = int(test_bb[1]+test_bb[3])

    print ymin, xmin, ymax, xmax

    obj_in_scene = icp_renderer.render_trafo(K_test.copy(), R_est_old, t_est_old, test_img.shape)
    scene_view = test_img.copy()
    scene_view[obj_in_scene > 0] = obj_in_scene[obj_in_scene > 0]
    cv2.rectangle(scene_view, (xmin,ymin),(xmax,ymax), (0,255,0), 2)
    cv2.putText(scene_view, '%s: %1.3f' % (obj_id,test_score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, (0,255,0), 2)
    cv2.imshow('scene_estimation',scene_view)
    
    obj_in_scene_ref = icp_renderer.render_trafo(K_test.copy(), R_est_ref, t_est_ref,test_img.shape)
    scene_view_refined = test_img.copy()
    scene_view_refined[obj_in_scene_ref > 0] = obj_in_scene_ref[obj_in_scene_ref > 0]
    cv2.rectangle(scene_view_refined, (xmin,ymin),(xmax,ymax), (0,255,0), 2)
    cv2.putText(scene_view_refined,'%s: %1.3f' % (obj_id,test_score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, (0,255,0), 2)
    cv2.imshow('scene_estimation_refined',scene_view_refined)

    for gt in gts:
        if gt['obj_id'] == obj_id:
            obj_in_scene = icp_renderer.render_trafo(K_test.copy(), gt['cam_R_m2c'], gt['cam_t_m2c'],test_img.shape)
            scene_view = test_img.copy()
            scene_view[obj_in_scene > 0] = obj_in_scene[obj_in_scene > 0]
            cv2.imshow('ground truth scene_estimation',scene_view)

    

def compute_pca_plot_embedding(eval_dir, z_train, z_test=None, save=True):
    sklearn_pca = PCA(n_components=3)
    full_z_pca = sklearn_pca.fit_transform(z_train)
    if z_test is not None:
        full_z_pca_test = sklearn_pca.transform(z_test)

    fig = plt.figure()
    ax = Axes3D(fig)

    c=np.linspace(0, 1, len(full_z_pca))
    ax.scatter(full_z_pca[:,0],full_z_pca[:,1],full_z_pca[:,2], c=c, marker='.', label='train_z')
    if z_test is not None:
        ax.scatter(full_z_pca_test[:,0],full_z_pca_test[:,1],full_z_pca_test[:,2], c='red', marker='.', label='test_z')

    plt.title('Embedding Principal Components')
    plt.legend()
    if save:
        plt.savefig(os.path.join(eval_dir,'figures','pca_embedding.pdf'))


def plot_viewsphere_for_embedding(Rs_viewpoints, eval_dir):
    fig = plt.figure()
    ax = Axes3D(fig)
    c=np.linspace(0, 1, len(Rs_viewpoints))
    ax.scatter(Rs_viewpoints[:,2,0],Rs_viewpoints[:,2,1],Rs_viewpoints[:,2,2], c=c, marker='.', label='embed viewpoints')
    plt.title('Embedding Viewpoints')
    plt.legend()
    plt.savefig(os.path.join(eval_dir,'figures','embedding_viewpoints.pdf'))

def plot_t_err_hist(t_errors, eval_dir):

    x = np.sort(np.abs(t_errors[:,0]))
    y = np.sort(np.abs(t_errors[:,1]))
    z = np.sort(np.abs(t_errors[:,2]))
    
    recall = (np.arange(len(t_errors))+1.)/len(t_errors)
    
    fig = plt.figure()
    plt.title('Recall vs Translation Error')
    plt.grid()
    plt.plot(x,recall)
    plt.plot(y,recall)
    plt.plot(z,recall)
    plt.xlabel('translation err [mm]')
    plt.ylabel('recall')
    plt.legend(['cum x error','cum y error','cum z error'])
    tikz_save(os.path.join(eval_dir,'latex','t_err_hist.tex'), figurewidth ='0.45\\textheight', figureheight='0.45\\textheight', show_info=False)

def plot_R_err_hist(top_n, eval_dir, scene_ids):
    
    angle_errs = []
    for scene_id in scene_ids:
        if not os.path.exists(os.path.join(eval_dir,'error=re_ntop=%s' % top_n,'errors_{:02d}.yml'.format(scene_id))):
            print 'WARNING: ' + os.path.join(eval_dir,'error=re_ntop=%s' % top_n,'errors_{:02d}.yml'.format(scene_id)) + ' not found'
            continue
        angle_errs_dict = inout.load_yaml(os.path.join(eval_dir,'error=re_ntop=%s' % top_n,'errors_{:02d}.yml'.format(scene_id)))
        angle_errs += [angle_e['errors'].values()[0] for angle_e in angle_errs_dict]


    if len(angle_errs) == 0:
        return
        
    angle_errs = np.array(angle_errs)



    fig = plt.figure()
    plt.grid()
    plt.xlabel('angle err [deg]')
    plt.ylabel('recall')
    plt.title('Angle Error vs Recall')
    legend=[]


    for n in np.unique(np.array([top_n, 1])):
        
        total_views = len(angle_errs)/top_n
        min_angle_errs = np.empty((total_views,))
        min_angle_errs_rect = np.empty((total_views,))

        for view in xrange(total_views):
            top_n_errors = angle_errs[view*top_n:(view+1)*top_n]
            if n == 1:
                top_n_errors = top_n_errors[np.newaxis,0]
            min_angle_errs[view] = np.min(top_n_errors)
            min_angle_errs_rect[view] = np.min(np.hstack((top_n_errors, 180-top_n_errors)))

        min_angle_errs_sorted = np.sort(min_angle_errs)
        min_angle_errs_rect_sorted = np.sort(min_angle_errs_rect)
        recall = (np.arange(total_views)+1.)/total_views

        # fill curve
        min_angle_errs_sorted = np.hstack((min_angle_errs_sorted, np.array([180.])))
        min_angle_errs_rect_sorted = np.hstack((min_angle_errs_rect_sorted, np.array([90.])))
        recall = np.hstack((recall,np.array([1.])))

        AUC_angle = np.trapz(recall,min_angle_errs_sorted/180.)
        AUC_angle_rect = np.trapz(recall,min_angle_errs_rect_sorted/90.)
        
        plt.plot(min_angle_errs_sorted,recall)
        plt.plot(min_angle_errs_rect_sorted,recall)
        
        legend += ['top {0} angle err, AUC = {1:.4f}'.format(n,AUC_angle), 'top {0} rectified angle err, AUC = {1:.4f}'.format(n,AUC_angle_rect)]
    plt.legend(legend)
    tikz_save(os.path.join(eval_dir,'latex','R_err_hist.tex'), figurewidth ='0.45\\textheight', figureheight='0.45\\textheight', show_info=False)

def print_trans_rot_errors(gts, obj_id, ts_est, ts_est_old, Rs_est, Rs_est_old):      

    t_errs = []
    obj_gts = []

    for gt in gts:
        if gt['obj_id'] == obj_id:
            t_errs.append(ts_est[0]-gt['cam_t_m2c'].squeeze())
            obj_gts.append(gt)

    min_t_err_idx = np.argmin(np.linalg.norm(np.array(t_errs),axis=1))
    print min_t_err_idx
    print np.array(t_errs).shape
    print len(obj_gts)
    gt = obj_gts[min_t_err_idx].copy()   

    try:
        print 'Translation Error before refinement'
        print ts_est_old[0]-gt['cam_t_m2c'].squeeze()
        print 'Translation Error after refinement'
        print t_errs[min_t_err_idx]
        print 'Rotation Error before refinement'
        print pose_error.re(Rs_est_old[0],gt['cam_R_m2c'])
        print 'Rotation Error after refinement'
        print pose_error.re(Rs_est[0],gt['cam_R_m2c'])
    except:
        pass


        

    return t_errs[min_t_err_idx]
        
def plot_vsd_err_hist(eval_args, eval_dir, scene_ids):
    top_n = eval_args.getint('METRIC','TOP_N')
    delta = eval_args.getint('METRIC','VSD_DELTA')
    tau = eval_args.getint('METRIC','VSD_TAU')
    cost = eval_args.get('METRIC','VSD_COST')


    vsd_errs = []
    for scene_id in scene_ids:
        error_file_path = os.path.join(eval_dir,'error=vsd_ntop=%s_delta=%s_tau=%s_cost=%s' % (top_n, delta, tau, cost), 'errors_{:02d}.yml'.format(scene_id))

        if not os.path.exists(error_file_path):
            print 'WARNING: ' + error_file_path + ' not found'
            continue

        vsd_dict = inout.load_yaml(error_file_path)
        vsd_errs += [vsd_e['errors'].values()[0] for vsd_e in vsd_dict]

    if len(vsd_errs) == 0:
        return
    vsd_errs = np.array(vsd_errs)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim((0.0,1.0))
    plt.grid()
    plt.xlabel('vsd err')
    plt.ylabel('recall')
    plt.title('VSD Error vs Recall')
    legend=[]
    
    for n in np.unique(np.array([top_n, 1])):
        
        total_views = len(vsd_errs)/top_n
        min_vsd_errs = np.empty((total_views,))

        for view in xrange(total_views):
            top_n_errors = vsd_errs[view*top_n:(view+1)*top_n]
            if n == 1:
                top_n_errors = top_n_errors[np.newaxis,0]
            min_vsd_errs[view] = np.min(top_n_errors)

        min_vsd_errs_sorted = np.sort(min_vsd_errs)
        recall = (np.arange(total_views)+1.)/total_views

        # fill curve
        min_vsd_errs_sorted = np.hstack((min_vsd_errs_sorted, np.array([1.])))
        recall = np.hstack((recall,np.array([1.])))

        AUC_vsd = np.trapz(recall,min_vsd_errs_sorted)
        plt.plot(min_vsd_errs_sorted,recall)
        
        legend += ['top {0} vsd err, AUC = {1:.4f}'.format(n,AUC_vsd)]
    plt.legend(legend)
    tikz_save(os.path.join(eval_dir,'latex','vsd_err_hist.tex'), figurewidth ='0.45\\textheight', figureheight='0.45\\textheight', show_info=False)

def plot_vsd_occlusion(eval_args, eval_dir, scene_ids, all_test_visibs, bins = 10):

    top_n = eval_args.getint('METRIC','TOP_N')
    delta = eval_args.getint('METRIC','VSD_DELTA')
    tau = eval_args.getint('METRIC','VSD_TAU')
    cost = eval_args.get('METRIC','VSD_COST')

    all_vsd_errs = []
    for scene_id in scene_ids:
        error_file_path = os.path.join(eval_dir,'error=vsd_ntop=%s_delta=%s_tau=%s_cost=%s' % (top_n, delta, tau, cost), 'errors_{:02d}.yml'.format(scene_id))

        if not os.path.exists(error_file_path):
            print 'WARNING: ' + error_file_path + ' not found'
            continue

        vsd_dict = inout.load_yaml(error_file_path)
        all_vsd_errs += [vsd_e['errors'].values()[0] for vsd_e in vsd_dict]

    if len(all_vsd_errs) == 0:
        return
    all_vsd_errs = np.array(all_vsd_errs)

    # print all_vsd_errs

    fig = plt.figure()
    # ax = plt.axes()
    # ax.set_xlim((0.0,1.0))
    # ax.set_ylim((0.0,1.0))
    ax = plt.gca()
    ax.set_xlim((0.0,1.0))
    plt.grid()
    plt.ylabel('vsd err')
    plt.xlabel('visibility [percent]')
    # plt.xlim((0.0, 1.0))
    # plt.ylim((0.0, 1.0))
    
    total_views = len(all_vsd_errs)/top_n
    vsd_errs = np.empty((total_views,))

    for view in xrange(total_views):
        top_n_errors = all_vsd_errs[view*top_n:(view+1)*top_n]
        vsd_errs[view] = top_n_errors[0]

    bounds = np.linspace(0,1,bins+1)
    bin_vsd_errs = []
    bin_count = []

    for idx in xrange(bins):
        bin_idcs = np.where((all_test_visibs>bounds[idx]) & (all_test_visibs<bounds[idx+1]))
        bin_vsd_errs.append(vsd_errs[bin_idcs])
        bin_count.append(len(bin_idcs[0]))
    
    middle_bin_vis = bounds[:-1] + (bounds[1]-bounds[0])/2.
    # plt.bar(middle_bin_vis,mean_vsd_err,0.5/bins)

    plt.boxplot(bin_vsd_errs, positions = middle_bin_vis, widths=0.5/bins, sym='+')


    # count_str = 'bin count ' + bins * '%s ' 
    # count_str = count_str % tuple(bin_count)
    plt.title('Visibility vs Mean VSD Error' + str(bin_count))
    tikz_save(os.path.join(eval_dir,'latex','vsd_occlusion.tex'), figurewidth ='0.45\\textheight', figureheight='0.45\\textheight', show_info=False)
    
def plot_re_rect_occlusion(eval_args, eval_dir, scene_ids, all_test_visibs, bins = 10):

    top_n = eval_args.getint('METRIC','TOP_N')
    
    all_angle_errs = []
    for scene_id in scene_ids:

        if not os.path.exists(os.path.join(eval_dir,'error=re_ntop=%s' % top_n,'errors_{:02d}.yml'.format(scene_id))):
            print 'WARNING: ' + os.path.join(eval_dir,'error=re_ntop=%s' % top_n,'errors_{:02d}.yml'.format(scene_id)) + ' not found'
            continue

        angle_errs_dict = inout.load_yaml(os.path.join(eval_dir,'error=re_ntop=%s' % top_n,'errors_{:02d}.yml'.format(scene_id)))
        all_angle_errs += [angle_e['errors'].values()[0] for angle_e in angle_errs_dict]

    if len(all_angle_errs) == 0:
        return
    all_angle_errs = np.array(all_angle_errs)
    # print all_vsd_errs

    fig = plt.figure()
    plt.grid()
    plt.ylabel('rot err [deg]')
    plt.xlabel('visibility [percent]')
    # plt.axis((-0.1, 1.1, -0.1, 1.1))
    # plt.xlim((0.0, 1.0))
    # plt.ylim((0.0, 1.0))
    
    total_views = len(all_angle_errs)/top_n
    angle_errs_rect = np.empty((total_views,))

    for view in xrange(total_views):
        top_n_errors = all_angle_errs[view*top_n:(view+1)*top_n]
        angle_errs_rect[view] = np.min([top_n_errors[0], 180-top_n_errors[0]])

    bounds = np.linspace(0,1,bins+1)
    bin_angle_errs = []
    bin_count = []

    for idx in xrange(bins):
        bin_idcs = np.where((all_test_visibs>bounds[idx]) & (all_test_visibs<bounds[idx+1]))
        # median_angle_err[idx] = np.median(angle_errs_rect[bin_idcs])
        bin_angle_errs.append(angle_errs_rect[bin_idcs])
        bin_count.append(len(bin_idcs[0]))
    
    middle_bin_vis = bounds[:-1] + (bounds[1]-bounds[0])/2.
    # plt.bar(middle_bin_vis,median_angle_err,0.5/bins)
    plt.boxplot(bin_angle_errs, positions = middle_bin_vis, widths=0.5/bins, sym='+')

    # count_str = 'bin count ' + bins * '%s ' 
    # count_str = count_str % tuple(bin_count)
    plt.title('Visibility vs Median Rectified Rotation Error' + str(bin_count))
    tikz_save(os.path.join(eval_dir,'latex','R_err_occlusion.tex'), figurewidth ='0.45\\textheight', figureheight='0.45\\textheight', show_info=False)
    

def animate_embedding_path(z_test):
    pass
