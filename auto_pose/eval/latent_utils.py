import os
import numpy as np
import cv2
from collections import defaultdict
import hashlib
import glob
import time

import matplotlib.pyplot as plt
from sixd_toolkit.pysixd import transform, pose_error, inout
from sixd_toolkit.params import dataset_params

from auto_pose.ae.pysixd_stuff import view_sampler
from auto_pose.eval import eval_plots, eval_utils
from auto_pose.ae import utils as u


def compute_plot_emb_invariance(args_latent):
    Rs, lon_lat, pts = eval_plots.generate_view_points(noof=101)
    syn_crops = []
    z_train = np.zeros((len(Rs), encoder.latent_space_size))
    for R in Rs:
        syn_crops.append(dataset.render_rot(R, obj_id=1)/255.)
    for a, e in u.batch_iteration_indices(len(Rs), 200):
        print a
        z_train[a:e] = sess.run(encoder.z, feed_dict={
                                encoder._input: syn_crops[a:e]})

    aug = eval(args_latent.get('Emb_invariance', 'aug'))

    batch = []
    orig_img = (syn_crops[100]*255).astype(np.uint8)  # H, W, C,  C H W
    for i in xrange(200):
        print i
        img = aug.augment_image(orig_img.copy()).astype(np.float32) / 255.
        #img = img.transpose( (1, 2, 0) ) #C H, W 1, 2,
        batch.append(img)
    batch = np.array(batch)
    z_test = sess.run(encoder.z, feed_dict={encoder._input: batch})

    eval_plots.compute_pca_plot_embedding(
        '', z_train, z_test=z_test, lon_lat=None, save=False, inter_factor=1)
    from gl_utils import tiles
    import cv2
    mean_var = np.mean(np.var(z_test, axis=0))
    cv2.imshow('mean_var: %s' % mean_var, tiles(batch, 10, 20))
    cv2.waitKey(0)
    plt.show()



def plot_latent_revolutions(num_obj):
    # generate PCA directions from all objects
    Rs, lon_lat, _ = eval_plots.generate_view_points(noof=201, num_cyclo=5)
    all_ztrain = []
    for i in range(0, num_o*2, 4):
        syn_crops = []
        z_train = np.zeros((len(Rs), encoder.latent_space_size))
        for R in Rs:

            syn_crops.append(dataset.render_rot(R, obj_id=i)/255.)
        for a, e in u.batch_iteration_indices(len(Rs), 200):
            print e
            z_train[a:e] = sess.run(encoder.z, feed_dict={
                                    encoder._input: syn_crops[a:e]})
        all_ztrain.append(z_train)
    all_ztrain = np.array(all_ztrain).reshape(-1, 128)
    pca_all = eval_plots.compute_pca_plot_embedding('', all_ztrain, lon_lat=list(lon_lat)*5, save=False)


    Rs, lon_lat, _ = eval_plots.generate_azim_elev_points(noof=36*8)

    fig = plt.figure(figsize=(3*num_obj, 3*4))
    fig.subplots_adjust(top=0.95, bottom=0.05)
    # plt.title('Embedding Principal Components')
    imgs = []
    axes = []
    for o in xrange(0, num_obj*4):
        syn_crops = []
        for R in Rs:
            if o >= 2*num_obj and o < 3*num_obj:
                R_rot = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
                syn_crops.append(dataset.render_rot(
                    np.dot(R, R_rot), obj_id=o)/255.)
            else:
                syn_crops.append(dataset.render_rot(R, obj_id=o)/255.)
        syn_crops = np.array(syn_crops)
        imgs.append(syn_crops[np.linspace(
            0, len(syn_crops), 8, endpoint=False).astype(np.int32)])
        # im = u.tiles(np.array(syn_crops),12,18*4,scale=0.66)
        z_train = np.zeros((len(Rs), encoder.latent_space_size))
        # cv2.imshow('',im)
        # cv2.waitKey(1)

        for a, e in u.batch_iteration_indices(len(Rs), 200):
            print e
            z_train[a:e] = sess.run(encoder.z, feed_dict={
                                    encoder._input: syn_crops[a:e]})
        # eval_plots.compute_pca_plot_embedding('',z_train,lon_lat=lon_lat,save=False)

        ax = fig.add_subplot(4, num_obj, o+1, projection='3d')
        # if o>=3*num_obj:
        #     pca_all=None
        eval_plots.compute_pca_plot_azelin(
            36*8+1, z_train, pca=pca_all, save=False, inter_factor=1, normalize=False, fig=fig, ax=ax)
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

        im = u.tiles(np.array(imgs).reshape(-1, 128, 128, 3), num_obj*4, 8, scale=1)
        cv2.imshow('', im)
        cv2.waitKey(1)
        plt.show()


def relative_pose_refinement(sess, args_latent, dataset, codebook):

    budget = args_latent.getint('Refinement', 'budget_per_epoch')
    epochs = args_latent.getint('Refinement', 'epochs')
    sampling_interval_deg = args_latent.getint('Refinement', 'sampling_interval_deg')
    top_n_refine = args_latent.getint('Refinement', 'max_num_modalities')
    t_z = args_latent.getint('Refinement', 't_z')
    num_obj = args_latent.getint('Data', 'num_obj')
    num_views = args_latent.getint('Data', 'num_views')

    K = eval(dataset._kw['k'])
    K = np.array(K).reshape(3,3)
    render_dims = eval(dataset._kw['render_dims'])
    clip_near = float(dataset._kw['clip_near'])
    clip_far = float(dataset._kw['clip_far'])
    pad_factor = float(dataset._kw['pad_factor'])

    pose_errs = []
    pose_errs_refref = []
    pose_errs_trans = []

    for i in range(0, num_obj):
        for j in range(num_views):
            random_R = transform.random_rotation_matrix()[:3, :3]

            full_target_view, full_target_view_dep= dataset.renderer.render(obj_id=i,
                                                             W=render_dims[0],
                                                             H=render_dims[1],
                                                             K=K.copy(),
                                                             R=random_R,
                                                             t=np.array([0,0,t_z]),
                                                             near=clip_near,
                                                             far=clip_far,
                                                             random_light=False)
            ys, xs = np.nonzero(full_target_view_dep > 0)
            target_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)
            target_view = dataset.extract_square_patch(full_target_view, target_bb, pad_factor)


            angle_off = 2*np.pi
            while abs(angle_off) > 45/180.*np.pi:
                # rand_direction = transform.make_rand_vector(3)
                # rand_angle = np.random.normal(0, 45/180.*np.pi)
                # R_off = transform.rotation_matrix(rand_angle, rand_direction)[:3, :3]
                
                rand_angle_x = np.random.normal(0,15/180.*np.pi)
                rand_angle_y = np.random.normal(0,15/180.*np.pi)
                rand_angle_z = np.random.normal(0,15/180.*np.pi)

                R_off = transform.euler_matrix(rand_angle_x,rand_angle_y,rand_angle_z)
                angle_off,_,_ = transform.rotation_from_matrix(R_off)
                random_R_pert = np.dot(R_off[:3, :3], random_R)
                random_t_pert = np.array([0,0,t_z]) + np.array([np.random.normal(0,10),np.random.normal(0,10),np.random.normal(0,50)])


                print angle_off * 180 / np.pi
                print random_t_pert

            full_perturbed_view, _ = dataset.renderer.render(obj_id=i,
                                                    W=render_dims[0],
                                                    H=render_dims[1],
                                                    K=K.copy(),
                                                    R=random_R_pert,
                                                    t=random_t_pert,
                                                    near=clip_near,
                                                    far=clip_far,
                                                    random_light=False
                                                )
            init_perturbed_view = dataset.extract_square_patch(full_perturbed_view, target_bb, pad_factor)

            start_time = time.time()

            R_refined, _ = codebook.refined_nearest_rotation(sess, target_view, 1, R_init=random_R_pert, t_init=random_t_pert,
                                                            budget=budget, epochs=epochs, high = sampling_interval_deg/180.*np.pi, obj_id=i, 
                                                             top_n_refine=top_n_refine, target_bb=target_bb)
            
            refine_R_1 = time.time() -start_time
            full_perturbed_view_2, _ = dataset.renderer.render(obj_id=i,
                                                            W=render_dims[0],
                                                            H=render_dims[1],
                                                            K=K.copy(),
                                                            R=R_refined[0],
                                                            t=random_t_pert,
                                                            near=clip_near,
                                                            far=clip_far,
                                                            random_light=False
                                                            )
            perturbed_view_2 = dataset.extract_square_patch(full_perturbed_view_2, target_bb, pad_factor)

            x_target, y_target, real_scale = multi_scale_template_matching(full_perturbed_view_2, full_target_view, args_latent)
            t_refined = np.array([random_t_pert[0]-(x_target-K[0, 2])/K[0, 0]*random_t_pert[2]*real_scale, 
                                  random_t_pert[1]-(y_target-K[1, 2])/K[1, 1]*random_t_pert[2]*real_scale,
                                  random_t_pert[2]*real_scale])
            refine_t_1 = time.time() - start_time


            print x_target, y_target, real_scale
            print t_refined
            print 'error t: ', t_refined - np.array([0,0,t_z])
            


            R_refined_refined, _ = codebook.refined_nearest_rotation(sess, target_view, 1, R_init=R_refined[0], t_init=t_refined,
                                                             budget=budget, epochs=epochs, high=sampling_interval_deg/2./180.*np.pi, obj_id=i,
                                                             top_n_refine=top_n_refine, target_bb=target_bb)
            refine_R_2 = time.time() - start_time

            pose_errs_trans.append(pose_error.te(t_refined, np.array([0, 0, t_z])))
            pose_errs.append(pose_error.re(random_R, R_refined[0]))
            pose_errs_refref.append(pose_error.re(random_R, R_refined_refined[0]))
            # pose_errs[-1] = np.minimum(pose_errs[-1],np.abs(pose_errs[-1]-180))
            
            print 'timings:'
            print refine_R_1
            print refine_t_1
            print refine_R_2

            if args_latent.getboolean('Visualization','verbose'):
                full_perturbed_view_3, _ = dataset.renderer.render(obj_id=i,
                                                                W=render_dims[0],
                                                                H=render_dims[1],
                                                                K=K.copy(),
                                                                R=R_refined[0],
                                                                t=t_refined,
                                                                near=clip_near,
                                                                far=clip_far,
                                                                random_light=False
                                                                )
                perturbed_view_3 = dataset.extract_square_patch(full_perturbed_view_3, target_bb, pad_factor)
                full_est_view_final, _ = dataset.renderer.render(obj_id=i,
                                                                W=render_dims[0],
                                                                H=render_dims[1],
                                                                K=K.copy(),
                                                                R=R_refined_refined[0],
                                                                t=t_refined,
                                                                near=clip_near,
                                                                far=clip_far,
                                                                random_light=False
                                                                )
                est_view_final = dataset.extract_square_patch(full_est_view_final, target_bb, pad_factor)

                cv2.imshow('goal_view', target_view)
                cv2.imshow('pert_view', init_perturbed_view/255.)
                cv2.imshow('est_view_1', perturbed_view_2/255.)
                cv2.imshow('est_view_2', perturbed_view_3/255.)
                cv2.imshow('est_view_3', est_view_final/255.)
                cv2.waitKey(0)

    if args_latent.getboolean('Visualization', 'rot_err_histogram'):
        plt.figure(1)
        plt.hist(pose_errs, bins=180)
        pose_errs = np.array(pose_errs)
        plt.title('pose_errs: median: ' + str(np.median(pose_errs)) + ', mean: ' + str(np.mean(
            pose_errs)) + ', <5deg: ' + str(len(pose_errs[pose_errs < 5])/1.0/len(pose_errs)))
        
        plt.figure(2)
        plt.hist(pose_errs_refref, bins=180)
        pose_errs_refref = np.array(pose_errs_refref)
        plt.title('pose_errs_refref: median: ' + str(np.median(pose_errs_refref)) 
        + ', mean: ' + str(np.mean(pose_errs_refref)) + ', <5deg: ' + str(len(pose_errs_refref[pose_errs_refref <= 5])/1.0/len(pose_errs_refref)))

        plt.figure(3)
        plt.hist(pose_errs_trans, bins=180)
        pose_errs_trans = np.array(pose_errs_trans)
        plt.title('pose_errs_trans: median: ' + str(np.median(pose_errs_trans)) + ', mean: ' + str(np.mean(pose_errs_trans))
                  + ', <5cm: ' + str(len(pose_errs_trans[pose_errs_trans <= 50])/1.0/len(pose_errs_trans)))

        plt.show()

    return pose_errs
        

def multi_scale_template_matching(im1, im2, args_latent):

    min_scale = args_latent.getfloat('Refinement', 'min_scale')
    max_scale = args_latent.getfloat('Refinement', 'max_scale')
    num_scales = args_latent.getfloat('Refinement', 'num_scales')
    verbose = args_latent.getboolean('Visualization', 'verbose')

    #im2 is target and template
    print im1.dtype, np.min(im1), np.max(im1), im1.shape
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1_gray_pad = np.zeros((im1_gray.shape[0]*2, im1_gray.shape[1]*2),dtype=np.uint8)
    im1_gray_pad[im1_gray.shape[0]//2:im1_gray.shape[0]//2*3,
                   im1_gray.shape[1]//2:im1_gray.shape[1]//2*3] = im1_gray
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im2_gray_canny = cv2.Canny(im2_gray, 150, 200, apertureSize = 3)
    

    tH, tW = im2_gray.shape
    found = None
    for scale in np.linspace(min_scale, max_scale, num_scales)[::-1]:
        resized_im1_gray = cv2.resize(im1_gray_pad, (int(im1_gray_pad.shape[1] * scale),
                                                 int(im1_gray_pad.shape[0] * scale)))
        real_scale = float(im1_gray_pad.shape[0] + im1_gray_pad.shape[1]) / (resized_im1_gray.shape[0] + resized_im1_gray.shape[1])

        resized_im1_gray_canny = cv2.Canny(resized_im1_gray, 150, 200, apertureSize = 3)

        result = cv2.matchTemplate(resized_im1_gray_canny, im2_gray_canny, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if verbose:
            clone = np.dstack([resized_im1_gray_canny, resized_im1_gray_canny, resized_im1_gray_canny])
            clone_temp = np.dstack([im2_gray_canny, im2_gray_canny, im2_gray_canny])
            result = np.dstack([result, result, result])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)
            cv2.imshow("result", result/np.max(result))
            cv2.imshow("template", clone_temp)
            print maxVal
            cv2.waitKey(0)
        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, real_scale)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, real_scale) = found
    (startX, startY) = (int(maxLoc[0] * real_scale - im1_gray.shape[1]//2), 
                        int(maxLoc[1] * real_scale - im1_gray.shape[0]//2))
    (endX, endY) = (int((maxLoc[0] + tW) * real_scale - im1_gray.shape[1]//2), 
                    int((maxLoc[1] + tH) * real_scale - im1_gray.shape[0]//2))
    (X, Y) = (int((maxLoc[0] + tW/2) * real_scale - im1_gray.shape[1]//2),
              int((maxLoc[1] + tH/2) * real_scale - im1_gray.shape[0]//2))

    # draw a bounding box around the detected result and display the image
    if verbose:
        cv2.rectangle(im1, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.imshow("Image 1", im1)
        cv2.imshow("Image 2", im2)
        cv2.waitKey(0)
    
    return (X, Y, real_scale)

def align_images(im1, im2, random_t_pert, warp_mode=cv2.MOTION_AFFINE, termination_eps=1e-7, number_of_iterations=5000):

    print im1.dtype, np.min(im1), np.max(im1), im1.shape
    im1_gray = cv2.cvtColor(im1.astype(np.float32), cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2.astype(np.float32), cv2.COLOR_BGR2GRAY)
    # Find size of image1
    sz = im1.shape

    # Define the motion model

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,number_of_iterations,  termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    print('##########')
    print('warp_matrix')
    print(warp_matrix)
    warp_matrix_normalized = warp_matrix[:2,:2].copy()
    warp_matrix_normalized[:, 0] /= np.linalg.norm(warp_matrix[:, 0])
    warp_matrix_normalized[:, 1] /= np.linalg.norm(warp_matrix[:, 1])
    warp_matrix[:2, :2] = np.dot(warp_matrix_normalized.T, warp_matrix[:2,:2])
    print np.linalg.norm(warp_matrix[:,0])
    print np.linalg.norm(warp_matrix[:,1])

    print('##########')

    cv2.imshow("Image 1", im1)
    cv2.imshow("Image 2", im2)
    cv2.imshow("Aligned Image 2", im2_aligned)
    cv2.waitKey(0)

    # Use warpAffine for Translation, Euclidean and Affine
    scale_est = (np.array(warp_matrix)[0, 0] + np.array(warp_matrix)[1, 1])/2

    t_z_est = scale_est*random_t_pert[2]

    # t = random_t_pert + np.array([])

    return t_z_est
