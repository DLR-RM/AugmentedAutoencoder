import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 

from matplotlib2tikz import save as tikz_save
import cv2
import os
import time
import numpy as np
import tensorflow as tf
from auto_pose.ae.utils import tiles, get_config_file_path, get_eval_config_file_path, get_log_dir, get_eval_dir
from sklearn.decomposition import PCA
from scipy import interpolate 
import glob
import pickle as pl

from auto_pose.meshrenderer import box3d_renderer
from sixd_toolkit.pysixd import inout,pose_error

from sixd_toolkit.params import dataset_params

view_idx = 0

def plot_reconstruction_test(sess, encoder, decoder, x):

    if x.dtype == 'uint8':
        x = x/255.
        print('converted uint8 to float type')

    if x.ndim == 3:
        x = np.expand_dims(x, 0)

    reconst = sess.run(decoder.x, feed_dict={encoder.x: x})
    cv2.imshow('reconst_test',cv2.resize(reconst[0],(256,256)))
    

def plot_reconstruction_test_batch(sess, codebook, decoder, test_img_crops, noof_scene_views, obj_id, eval_dir=None):
    
    encoder = codebook._encoder
    dataset = codebook._dataset

    sample_views = np.random.choice(noof_scene_views, np.min([100,noof_scene_views]), replace=False)
    
    sample_batch = []
    i=0
    j=0
    while i < 16:
        if obj_id in test_img_crops[sample_views[j]]:
            sample_batch.append(test_img_crops[sample_views[j]][obj_id][0])
            i += 1
        j += 1
            
    x = np.array(sample_batch).squeeze()
    
    if x.dtype == 'uint8':
        x = x/255.
        print('converted uint8 to float type')
    
    reconst = sess.run(decoder.x, feed_dict={encoder.x: x})
    nearest_neighbors = []
    for xi in x:
        Rs_est = codebook.nearest_rotation(sess, xi, top_n=8)
        pred_views = []
        pred_views.append(xi*255)
        for R_est in Rs_est:
            pred_views.append(dataset.render_rot( R_est ,downSample = 1))
        nearest_neighbors.append(tiles(np.array(pred_views),1,len(pred_views),10,10))

    all_nns_img = tiles(np.array(nearest_neighbors),len(nearest_neighbors),1,10,10)

    reconstruction_imgs = np.hstack(( tiles(x, 4, 4), tiles(reconst, 4, 4)))
    cv2.imwrite(os.path.join(eval_dir,'figures','reconstruction_imgs.png'), reconstruction_imgs*255)
    cv2.imwrite(os.path.join(eval_dir,'figures','nearest_neighbors_imgs.png'), all_nns_img)

def plot_reconstruction_train(sess, decoder, train_code):
    if train_code.ndim == 1:
        train_code = np.expand_dims(train_code, 0)
    reconst = sess.run(decoder.x, feed_dict={decoder._latent_code: train_code})
    cv2.imshow('reconst_train',cv2.resize(reconst[0],(256,256)))
    


def show_nearest_rotation(pred_views, test_crop, view):
    print((np.array(pred_views).shape)) 
    nearest_views = tiles(np.array(pred_views),1,len(pred_views),10,10)
    cv2.imshow('nearest_views',cv2.resize(nearest_views/255.,(len(pred_views)*256,256)))
    cv2.imshow('test_crop',cv2.resize(test_crop,(256,256)))

    

def plot_scene_with_3DBoxes(scene_res_dirs,dataset_name='tless',scene_id=1,save=False):

        # inout.save_results_sixd17(res_path, preds, run_time=run_time)


    # obj_gts = []
    # obj_infos = []
    # for object_id in xrange(1,noofobjects+1):
    #     obj_gts.append(inout.load_gt(os.path.join(sixd_img_path,'{:02d}'.format(object_id),'gt.yml')))
    #     obj_infos.append(inout.load_info(os.path.join(sixd_img_path,'{:02d}'.format(object_id),'info.yml')))
    #     print len(obj_gts)

    # dataset_name = eval_args.get('DATA','DATASET')
    # cam_type = eval_args.get('DATA','CAM_TYPE')

    # data_params = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type='primesense', cam_type='primesense')
    data_params = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='')


    models_cad_files = sorted(glob.glob(os.path.join(os.path.dirname(data_params['model_mpath']),'*.ply')))
    W,H = data_params['test_im_size']

    renderer_line = box3d_renderer.Renderer(
        models_cad_files, 
        1,
        W,
        H
    )

    scene_result_dirs = sorted(glob.glob(scene_res_dirs))
    print((data_params['test_rgb_mpath']))
    print((data_params['scene_gt_mpath']))

    # for scene_id in xrange(1,21):
        # sixd_img_path = data_params['test_rgb_mpath'].format(scene_id)
    scene_gts = inout.load_gt(data_params['scene_gt_mpath'].format(scene_id))
    scene_infos = inout.load_info(data_params['scene_info_mpath'].format(scene_id))

    scene_dirs = [d for d in scene_result_dirs if '%02d' % scene_id == d.split('/')[-1]]
    print(scene_dirs)

    for view in range(len(scene_infos)):
        sixd_img_path = data_params['test_rgb_mpath'].format(scene_id,view)
        img = cv2.imread(sixd_img_path)
        box_img = img.copy()
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        K = scene_infos[view]['cam_K']

        for bb in scene_gts[view]:

            xmin = int(bb['obj_bb'][0])
            ymin = int(bb['obj_bb'][1])
            xmax = int(bb['obj_bb'][0]+bb['obj_bb'][2])
            ymax = int(bb['obj_bb'][1]+bb['obj_bb'][3])

            cv2.rectangle(box_img, (xmin,ymin),(xmax,ymax), (0,255,0), 2)
            cv2.putText(box_img, '%s' % (bb['obj_id']), (xmin, ymax+20), cv2.FONT_ITALIC, .5, (0,255,0), 2)
        # for gt in scene_gts[view]:
            # if gt['obj_id'] not in [1,8,9]:
            #     lines_gt = renderer_line.render(gt['obj_id']-1,K,gt['cam_R_m2c'],gt['cam_t_m2c'],10,5000)
            #     lines_gt_mask = (np.sum(lines_gt,axis=2) < 20)[:,:,None]
            #     print lines_gt.shape
            #     lines_gt = lines_gt[:,:,[1,0,2]]
            #     img = lines_gt_mask*img + lines_gt

        for scene_dir in scene_dirs: 
            try:
                res_path = glob.glob(os.path.join(scene_dir,'%04d_*.yml' % (view)))
                print(res_path)
                res_path = res_path[0]
                # print 'here', res_path
                obj_id = int(res_path.split('_')[-1].split('.')[0])
                results = inout.load_results_sixd17(res_path)
                print(results)
                e = results['ests'][0]
                R_est = e['R']
                t_est = e['t']
                K = scene_infos[view]['cam_K']
                lines = renderer_line.render(obj_id-1,K,R_est,t_est,10,5000)
                lines_mask_inv = (np.sum(lines,axis=2) < 20)[:,:,None]
                lines_mask = lines_mask_inv == False
                # img[lines>0] = lines[lines>0]

                col = np.array(cm.hsv((obj_id-1)*8))[:3]

                lines = col * np.dstack((lines_mask,lines_mask,lines_mask))

                # if obj_id % 7 == 1:
                #     lines[:,:,0] = lines[:,:,1] 
                # elif obj_id % 7 == 2:
                #     lines[:,:,2] = lines[:,:,1]
                # elif obj_id % 7 == 3:
                #     lines[:,:,0] = lines[:,:,1]
                #     lines[:,:,1] = lines[:,:,2]

                img = lines_mask_inv * img + (lines*255).astype(np.uint8)
            except:
                print(('undeteceted obj: ', scene_dir))
        cv2.imshow('',img)
        if cv2.waitKey(1) == 32:
            cv2.waitKey(0)

        if save:
            if 'icp' in scene_res_dirs:

                if not os.path.exists('%02d' % scene_id):
                    os.makedirs('%02d' % scene_id)
                cv2.imwrite(os.path.join('%02d' % scene_id,'%04d.png' % view), img)
            else:

                if not os.path.exists('%02d_rgb' % scene_id):
                    os.makedirs('%02d_rgb' % scene_id)
                cv2.imwrite(os.path.join('%02d_rgb' % scene_id,'%04d.png' % view), img)
            # cv2.imwrite(os.path.join('%02d' % scene_id,'%04d_boxes.png' % view), box_img)



def plot_scene_with_estimate(test_img,renderer,K_test, R_est_old, t_est_old,R_est_ref, t_est_ref, test_bb, test_score, obj_id, gts=[], bb_pred=None):   
    global view_idx
    if bb_pred is not None:
        scene_detect = test_img.copy()
        for bb in bb_pred:
            try:
                xmin = int(bb['obj_bb'][0])
                ymin = int(bb['obj_bb'][1])
                xmax = int(bb['obj_bb'][0]+bb['obj_bb'][2])
                ymax = int(bb['obj_bb'][1]+bb['obj_bb'][3])
                if obj_id == bb['obj_id']:
                    cv2.rectangle(scene_detect, (xmin,ymin),(xmax,ymax), (0,255,0), 2)
                    cv2.putText(scene_detect, '%s: %1.3f' % (bb['obj_id'],bb['score']), (xmin, ymax+20), cv2.FONT_ITALIC, .5, (0,255,0), 2)
                else:
                    cv2.rectangle(scene_detect, (xmin,ymin),(xmax,ymax), (0,0,255), 2)
                    # cv2.putText(scene_detect, '%s: %1.3f' % (bb['obj_id'],bb['score']), (xmin, ymax+20), cv2.FONT_ITALIC, .5, (0,0,255), 2)
            except:
                pass
            #cv2.putText(scene_detect, '%s: %1.3f' % (obj_id,test_score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, (0,255,0), 2)
        cv2.imshow('scene_detect',scene_detect)
        

    xmin = int(test_bb[0])
    ymin = int(test_bb[1])
    xmax = int(test_bb[0]+test_bb[2])
    ymax = int(test_bb[1]+test_bb[3])

    print('here')
    obj_in_scene, _ = renderer.render( obj_id=0, W=test_img.shape[1],H=test_img.shape[0], K=K_test.copy(), R=R_est_old, t=np.array(t_est_old),near=10,far=10000,random_light=False)
    scene_view = test_img.copy()
    scene_view[obj_in_scene > 0] = obj_in_scene[obj_in_scene > 0]
    cv2.rectangle(scene_view, (xmin,ymin),(xmax,ymax), (0,255,0), 2)
    cv2.putText(scene_view, '%s: %1.3f' % (obj_id,test_score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, (0,255,0), 2)
    cv2.imshow('scene_estimation',scene_view)

    if not t_est_old[2] == t_est_ref[2]:
        obj_in_scene_ref, _ = renderer.render( obj_id=0, W=test_img.shape[1],H=test_img.shape[0], K=K_test.copy(), R=R_est_ref, t=np.array(t_est_ref),near=10,far=10000,random_light=False)
        scene_view_refined = test_img.copy()
        g_y = np.zeros_like(obj_in_scene_ref)
        g_y[:,:,1]= obj_in_scene_ref[:,:,1]
        scene_view_refined[obj_in_scene_ref > 0] = g_y[obj_in_scene_ref > 0]*2./3. + scene_view_refined[obj_in_scene_ref > 0]*1./3.
        # scene_view_refined[obj_in_scene_ref > 0] = obj_in_scene_ref[obj_in_scene_ref > 0]
        cv2.rectangle(scene_view_refined, (xmin,ymin),(xmax,ymax), (0,255,0), 2)
        cv2.putText(scene_view_refined,'%s: %1.3f' % (obj_id,test_score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, (0,255,0), 2)
        cv2.imshow('scene_estimation_refined',scene_view_refined)
    cv2.waitKey(0)
    view_idx += 1

    # for gt in gts:
    #     if gt['obj_id'] == obj_id:
    #         obj_in_scene, _ = renderer.render( obj_id=0, W=test_img.shape[1],H=test_img.shape[0], K=K_test.copy(), R=gt['cam_R_m2c'], t=np.array(gt['cam_t_m2c']),near=10,far=10000,random_light=False)
    #         scene_view = test_img.copy()
    #         scene_view[obj_in_scene > 0] = obj_in_scene[obj_in_scene > 0]
    #         cv2.imshow('ground truth scene_estimation',scene_view)


def compute_pca_plot_embedding(eval_dir, z_train, lon_lat=None, z_test=None, save=True, inter_factor = 1):
    print(inter_factor)
    sklearn_pca = PCA(n_components=3)
    full_z_pca = sklearn_pca.fit_transform(z_train)
    if z_test is not None:
        full_z_pca_test = sklearn_pca.transform(z_test)

    for i in [0,1]:
        fig = plt.figure()
        ax = Axes3D(fig)
        if lon_lat is None:
            c=np.linspace(0, 1, len(full_z_pca))
            x_fine, y_fine, z_fine = full_z_pca.T
        else:
            lon_lat = np.array(lon_lat)
            sort_idcs=np.argsort(lon_lat[:,i])
            full_z_pca_s = full_z_pca[sort_idcs]

            tck, u = interpolate.splprep(full_z_pca_s.T, s=2)
            u_fine = np.linspace(0,1,inter_factor*len(full_z_pca_s))
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)


            if i==0:
                c=np.array(lon_lat)[:,i]/(2*np.pi)
            else:
                c=(np.array(lon_lat)[:,i] + np.pi/2)/np.pi
        incr = len(c)//10
        incr_int = incr * inter_factor
        jet = plt.get_cmap('jet')
        for j in range(10): 
            #ax.scatter(full_z_pca[:,0],full_z_pca[:,1],full_z_pca[:,2], c=c, marker='.', label='PCs of train viewsphere')
            
            label ='lat level %s' % j if i==1 else 'lat level %s' % j
            ax.plot(x_fine[j*incr_int:(j+1)*incr_int], y_fine[j*incr_int:(j+1)*incr_int], z_fine[j*incr_int:(j+1)*incr_int], label=label, markersize=10, marker='.', color=jet(c[j*incr])[:3]+(0.8,))
        if z_test is not None:
            ax.scatter(full_z_pca_test[:,0],full_z_pca_test[:,1],full_z_pca_test[:,2], c='black', marker='.', label='test_z')
        if i==0:
            plt.title('Embedding Principal Components Lon')
        else:
            plt.title('Embedding Principal Components Lat')
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')
        ax.set_zlabel('pc3')
        ax.legend()

        if save:
            pl.dump(fig,file(os.path.join(eval_dir,'figures','pca_embedding.pickle'),'wb'))
            plt.savefig(os.path.join(eval_dir,'figures','pca_embedding.pdf'))

    return sklearn_pca

def compute_pca_plot_azelin(noof, z_train, pca=None, z_test=None, save=True, inter_factor = 2, normalize=False,fig=None, ax=None):

    if pca is None:
        sklearn_pca = PCA(n_components=3)
        full_z_pca = sklearn_pca.fit_transform(z_train)
    else:
        full_z_pca = pca.transform(z_train)
    if normalize:
        full_z_pca = full_z_pca/np.linalg.norm(full_z_pca,axis=1)[:,np.newaxis]
    if z_test is not None:
        full_z_pca_test = pca.transform(z_test)

    # full_z_pca = np.concatenate((full_z_pca,full_z_pca[0:1]))

    incr = np.linspace(0,1,3)
    off = noof * inter_factor
    jet = plt.get_cmap('brg')
    labels = ['azimuth','elevation','inplane']

    for j in range(3)[::-1]: 
        #ax.scatter(full_z_pca[:,0],full_z_pca[:,1],full_z_pca[:,2], c=c, marker='.', label='PCs of train viewsphere')
        if inter_factor<=1:
            x_fine, y_fine, z_fine = full_z_pca.T
            
        else:
            tck, u = interpolate.splprep(full_z_pca.T,s=3)
            u_fine = np.linspace(0,1,inter_factor*len(full_z_pca))
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

        ax.plot(x_fine[j*off:(j+1)*off], y_fine[j*off:(j+1)*off], z_fine[j*off:(j+1)*off], label=labels[j], marker='.',markersize=4,color=jet(incr[j])[:3]+(0.8,))
    if z_test is not None:
        ax.scatter(full_z_pca_test[:,0],full_z_pca_test[:,1],full_z_pca_test[:,2], c='red', marker='.', label='test_z')
    ax.scatter(x_fine[0],y_fine[0],z_fine[0],c='black', marker='o')
    ax.set_xlabel('pc1')
    ax.set_ylabel('pc2')
    ax.set_zlabel('pc3')
    ax.grid(False)

    # ax.legend()

    if save:
        pl.dump(fig,file(os.path.join(eval_dir,'figures','pca_embedding.pickle'),'wb'))
        plt.savefig(os.path.join(eval_dir,'figures','pca_embedding.pdf'))


def plot_viewsphere_for_embedding(Rs_viewpoints, eval_dir, errors=None,save=True):

    fig = plt.figure()
    ax = Axes3D(fig)
    if errors is not None:
        c= errors/180.
    else:
        c=np.linspace(0, 1, len(Rs_viewpoints))
    ax.scatter(Rs_viewpoints[:,2,0],Rs_viewpoints[:,2,1],Rs_viewpoints[:,2,2], c=c, s=50, marker='.', label='embed viewpoints')
    # ax.plot_surface(Rs_viewpoints[:,2,0],Rs_viewpoints[:,2,1],Rs_viewpoints[:,2,2], rstride=1, cstride=1, color=c, shade=0)

    plt.title('Embedding Viewpoints')
    plt.legend()
    if save:
        plt.savefig(os.path.join(eval_dir,'figures','embedding_viewpoints.pdf'))
    else:
        plt.show()


def generate_view_points(noof=1001, lat=(-0.5*np.pi,0.5*np.pi), lon=(0,2*np.pi), num_cyclo=1, renderer=None):
    from sixd_toolkit.pysixd import view_sampler
    azimuth_range = lon
    elev_range = lat



    views, _, lon_lat, pts = view_sampler.sample_views(noof-1,
                                                    azimuth_range = lon,
                                                    elev_range = lat,
                                                    sampling='fibonacci')
    Rs = np.empty( (len(views)*num_cyclo, 3, 3) )
    i = 0

    for view in views:
        for cyclo in np.linspace(0, 2.*np.pi, num_cyclo, endpoint=False):
            rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
            Rs[i,:,:] = rot_z.dot(view['R'])
            i += 1

    return Rs,lon_lat,pts

def generate_azim_elev_points(noof=36, orig_az_el_in = (0,np.pi/2,0)):
    from sixd_toolkit.pysixd import view_sampler

    elevs = np.roll(np.linspace(-np.pi,np.pi, noof),noof//4)
    azims = np.linspace(0,2*np.pi,noof, endpoint=False)
    pts_el = []
    pts_az = []
    sign = 1
    for e,el in enumerate(elevs):
        # if e >= len(elevs)//4:
        #     sign=-1
        # if e >= len(elevs)//4*3:
        #     sign=1
        x_el = sign*np.sin(el) * np.cos(orig_az_el_in[0])
        y_el = np.sin(el) * np.sin(orig_az_el_in[0])
        z_el = np.cos(el)
        pts_el.append([x_el,y_el,z_el])
    pts_el = np.array(pts_el)

    for az in azims:
        x_az = np.sin(orig_az_el_in[1]) * np.cos(az)
        y_az = np.sin(orig_az_el_in[1]) * np.sin(az)
        z_az = np.cos(orig_az_el_in[1])
        pts_az.append([x_az,y_az,z_az])
    pts_az = np.array(pts_az)

    views_el, _, _, _ = view_sampler.sample_views(noof,
                                                    sampling='None',
                                                    pts=pts_el)
    Rs_el = np.array([view['R'] for view in views_el])
    for r,R_el in enumerate(Rs_el):
        if r>=len(Rs_el)//4 and r<len(Rs_el)//4*3:
            rot_z = np.array([[np.cos(np.pi), -np.sin(np.pi), 0], [np.sin(np.pi), np.cos(np.pi), 0], [0, 0, 1]])
            Rs_el[r,:,:] = rot_z.dot(R_el)
    views_az, _, _, _ = view_sampler.sample_views(noof,
                                                sampling='None',
                                                pts=pts_az)
    Rs_az = np.array([view['R'] for view in views_az])

    Rs_in = np.empty( (noof, 3, 3) )
    pts_in = []
    i = 0
    for cyclo in np.linspace(0, 2.*np.pi, noof, endpoint=False):
        rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
        Rs_in[i,:,:] = rot_z.dot(Rs_el[0])
        pts_in.append(rot_z.dot(pts_el[0]))
        i += 1
    pts_in = np.array(pts_in)
    
    pts = np.concatenate((pts_az, pts_el, pts_in))
    Rs = np.concatenate((Rs_az, Rs_az[0:1], Rs_el, Rs_el[0:1], Rs_in, Rs_in[0:1]))
    lon_lat = None
    return Rs,lon_lat,pts


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


def plot_t_err_hist_vis(eval_args, eval_dir, scene_ids, bins=20):
    top_n_eval = eval_args.getint('EVALUATION','TOP_N_EVAL')
    top_n = eval_args.getint('METRIC','TOP_N')
    cam_type = eval_args.get('DATA','cam_type')
    dataset_name = eval_args.get('DATA','dataset')
    obj_id = eval_args.getint('DATA','obj_id')


    # if top_n_eval < 1:
    #     return

    data_params = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)

    t_errs = []
    for scene_id in scene_ids:
        error_file_path = os.path.join(eval_dir,'error=te_ntop=%s' % top_n,'errors_{:02d}.yml'.format(scene_id))
        if not os.path.exists(error_file_path):
            print(('WARNING: ' + error_file_path + ' not found'))
            continue
        # t_errs_dict = inout.load_yaml(error_file_path)
        # t_errs += [angle_e['errors'].values()[0] for angle_e in t_errs_dict]
        
        gts = inout.load_gt(data_params['scene_gt_mpath'].format(scene_id))
        visib_gts = inout.load_yaml(data_params['scene_gt_stats_mpath'].format(scene_id, 15))
        te_dict = inout.load_yaml(error_file_path)

        for view in range(len(gts)):
            res = te_dict[view*top_n:(view+1)*top_n]
            for gt,visib_gt in zip(gts[view],visib_gts[view]):
                if gt['obj_id'] == obj_id:
                    if visib_gt['visib_fract'] > 0.1:
                        for te_e in res:
                            t_errs += [list(te_e['errors'].values())[0]]

    if len(t_errs) == 0:
        return
        
    t_errs = np.array(t_errs)

    plot_t_err_hist2(t_errs, eval_dir, bins=bins)

def plot_t_err_hist2(t_errors, eval_dir, bins=15):
    fig = plt.figure()
    plt.title('Translation Error Histogram')
    plt.xlabel('translation err [mm]')
    plt.ylabel('views')
    bounds = np.linspace(0,100,bins+1)
    bin_count = []
    print((t_errors.shape))
    # eucl_terr = np.linalg.norm(t_errors,axis=1)
    eucl_terr = t_errors
    for idx in range(bins):
        bin_idcs = np.where((eucl_terr>bounds[idx]) & (eucl_terr<bounds[idx+1]))
        bin_count.append(len(bin_idcs[0]))
    middle_bin = bounds[:-1] + (bounds[1]-bounds[0])/2.
    plt.bar(middle_bin,bin_count,100*0.5/bins)
    tikz_save(os.path.join(eval_dir,'latex','t_err_hist2.tex'), figurewidth ='0.45\\textheight', figureheight='0.45\\textheight', show_info=False)

def plot_R_err_hist2(R_errors, eval_dir, bins=15, save=True):

    fig = plt.figure()
    plt.title('Rotation Error Histogram')
    plt.xlabel('Rotation err [deg]')
    plt.ylabel('views')
    bounds = np.linspace(0,180,bins+1)
    bin_count = []
    for idx in range(bins):
        bin_idcs = np.where((R_errors>bounds[idx]) & (R_errors<bounds[idx+1]))
        bin_count.append(len(bin_idcs[0]))
    middle_bin = bounds[:-1] + (bounds[1]-bounds[0])/2.
    plt.bar(middle_bin,bin_count,180*0.5/bins)
    plt.xlim((0, 180))
    if save:
        tikz_save(os.path.join(eval_dir,'latex','R_err_hist2.tex'), figurewidth ='0.45\\textheight', figureheight='0.45\\textheight', show_info=False)

def plot_R_err_hist_vis(eval_args, eval_dir, scene_ids, bins=20):
    top_n_eval = eval_args.getint('EVALUATION','TOP_N_EVAL')
    top_n = eval_args.getint('METRIC','TOP_N')
    cam_type = eval_args.get('DATA','cam_type')
    dataset_name = eval_args.get('DATA','dataset')
    obj_id = eval_args.getint('DATA','obj_id')


    # if top_n_eval < 1:
    #     return

    data_params = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)

    angle_errs = []
    for scene_id in scene_ids:
        error_file_path = os.path.join(eval_dir,'error=re_ntop=%s' % top_n,'errors_{:02d}.yml'.format(scene_id))
        if not os.path.exists(error_file_path):
            print(('WARNING: ' + error_file_path + ' not found'))
            continue
        # angle_errs_dict = inout.load_yaml(error_file_path)
        # angle_errs += [angle_e['errors'].values()[0] for angle_e in angle_errs_dict]
        
        gts = inout.load_gt(data_params['scene_gt_mpath'].format(scene_id))
        visib_gts = inout.load_yaml(data_params['scene_gt_stats_mpath'].format(scene_id, 15))
        re_dict = inout.load_yaml(error_file_path)

        for view in range(len(gts)):
            res = re_dict[view*top_n:(view+1)*top_n]
            for gt,visib_gt in zip(gts[view],visib_gts[view]):
                if gt['obj_id'] == obj_id:
                    if visib_gt['visib_fract'] > 0.1:
                        for re_e in res:
                            angle_errs += [list(re_e['errors'].values())[0]]

    if len(angle_errs) == 0:
        return
        
    angle_errs = np.array(angle_errs)

    plot_R_err_hist2(angle_errs, eval_dir, bins=bins)


def plot_R_err_recall(eval_args, eval_dir, scene_ids):
    
    top_n_eval = eval_args.getint('EVALUATION','TOP_N_EVAL')
    top_n = eval_args.getint('METRIC','TOP_N')
    cam_type = eval_args.get('DATA','cam_type')
    dataset_name = eval_args.get('DATA','dataset')
    obj_id = eval_args.getint('DATA','obj_id')


    # if top_n_eval < 1:
    #     return

    data_params = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)

    angle_errs = []
    for scene_id in scene_ids:
        error_file_path = os.path.join(eval_dir,'error=re_ntop=%s' % top_n,'errors_{:02d}.yml'.format(scene_id))
        if not os.path.exists(error_file_path):
            print(('WARNING: ' + error_file_path + ' not found'))
            continue
        # angle_errs_dict = inout.load_yaml(error_file_path)
        # angle_errs += [angle_e['errors'].values()[0] for angle_e in angle_errs_dict]
        
        gts = inout.load_gt(data_params['scene_gt_mpath'].format(scene_id))
        visib_gts = inout.load_yaml(data_params['scene_gt_stats_mpath'].format(scene_id, 15))
        re_dict = inout.load_yaml(error_file_path)

        for view in range(len(gts)):
            res = re_dict[view*top_n:(view+1)*top_n]
            for gt,visib_gt in zip(gts[view],visib_gts[view]):
                if gt['obj_id'] == obj_id:
                    if visib_gt['visib_fract'] > 0.1:
                        for re_e in res:
                            angle_errs += [list(re_e['errors'].values())[0]]

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

        for view in range(total_views):
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
    print(min_t_err_idx)
    print((np.array(t_errs).shape))
    print((len(obj_gts)))
    gt = obj_gts[min_t_err_idx].copy()   

    try:
        print('Translation Error before refinement')
        print((ts_est_old[0]-gt['cam_t_m2c'].squeeze()))
        print('Translation Error after refinement')
        print((t_errs[min_t_err_idx]))
        print('Rotation Error before refinement')
        print((pose_error.re(Rs_est_old[0],gt['cam_R_m2c'])))
        print('Rotation Error after refinement')
        R_err = pose_error.re(Rs_est[0],gt['cam_R_m2c'])
        print(R_err)
    except:
        pass


        

    return (t_errs[min_t_err_idx], R_err)
        
def plot_vsd_err_hist(eval_args, eval_dir, scene_ids):
    top_n_eval = eval_args.getint('EVALUATION','TOP_N_EVAL')
    top_n = eval_args.getint('METRIC','TOP_N')
    delta = eval_args.getint('METRIC','VSD_DELTA')
    tau = eval_args.getint('METRIC','VSD_TAU')
    cost = eval_args.get('METRIC','VSD_COST')
    cam_type = eval_args.get('DATA','cam_type')
    dataset_name = eval_args.get('DATA','dataset')
    obj_id = eval_args.getint('DATA','obj_id')

    if top_n_eval < 1:
        return

    data_params = dataset_params.get_dataset_params(dataset_name, model_type='', train_type='', test_type=cam_type, cam_type=cam_type)
    
    vsd_errs = []
    for scene_id in scene_ids:
        error_file_path = os.path.join(eval_dir,'error=vsd_ntop=%s_delta=%s_tau=%s_cost=%s' % (top_n, delta, tau, cost), 'errors_{:02d}.yml'.format(scene_id))

        if not os.path.exists(error_file_path):
            print(('WARNING: ' + error_file_path + ' not found'))
            continue
        gts = inout.load_gt(data_params['scene_gt_mpath'].format(scene_id))
        visib_gts = inout.load_yaml(data_params['scene_gt_stats_mpath'].format(scene_id, 15))
        vsd_dict = inout.load_yaml(error_file_path)
        for view,vsd_e in enumerate(vsd_dict):
            vsds = vsd_dict[view*top_n:(view+1)*top_n]
            for gt,visib_gt in zip(gts[view],visib_gts[view]):
                if gt['obj_id'] == obj_id:
                    if visib_gt['visib_fract'] > 0.1:
                        for vsd_e in vsds:
                            vsd_errs += [list(vsd_e['errors'].values())[0]]


    if len(vsd_errs) == 0:
        return
    vsd_errs = np.array(vsd_errs)
    print((len(vsd_errs)))

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

        for view in range(total_views):
            top_n_errors = vsd_errs[view*top_n:(view+1)*top_n]
            if n == 1:
                top_n_errors = top_n_errors[np.newaxis,0]
            min_vsd_errs[view] = np.min(top_n_errors)

        min_vsd_errs_sorted = np.sort(min_vsd_errs)
        recall = np.float32(np.arange(total_views)+1.)/total_views

        # fill curve
        min_vsd_errs_sorted = np.hstack((min_vsd_errs_sorted, np.array([1.])))
        recall = np.hstack((recall,np.array([1.])))

        AUC_vsd = np.trapz(recall, min_vsd_errs_sorted)
        plt.plot(min_vsd_errs_sorted,recall)
        
        legend += ['top {0} vsd err, AUC = {1:.4f}'.format(n,AUC_vsd)]
    plt.legend(legend)
    tikz_save(os.path.join(eval_dir,'latex','vsd_err_hist.tex'), figurewidth ='0.45\\textheight', figureheight='0.45\\textheight', show_info=False)

def plot_vsd_occlusion(eval_args, eval_dir, scene_ids, all_test_visibs, bins = 10):

    top_n_eval = eval_args.getint('EVALUATION','TOP_N_EVAL')
    top_n = eval_args.getint('METRIC','TOP_N')
    delta = eval_args.getint('METRIC','VSD_DELTA')
    tau = eval_args.getint('METRIC','VSD_TAU')
    cost = eval_args.get('METRIC','VSD_COST')

    if top_n_eval < 1:
        return

    all_vsd_errs = []
    for scene_id in scene_ids:
        error_file_path = os.path.join(eval_dir,'error=vsd_ntop=%s_delta=%s_tau=%s_cost=%s' % (top_n, delta, tau, cost), 'errors_{:02d}.yml'.format(scene_id))

        if not os.path.exists(error_file_path):
            print(('WARNING: ' + error_file_path + ' not found'))
            continue

        vsd_dict = inout.load_yaml(error_file_path)
        all_vsd_errs += [list(vsd_e['errors'].values())[0] for vsd_e in vsd_dict]

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

    for view in range(total_views):
        top_n_errors = all_vsd_errs[view*top_n:(view+1)*top_n]
        vsd_errs[view] = top_n_errors[0]

    bounds = np.linspace(0,1,bins+1)
    bin_vsd_errs = []
    bin_count = []

    for idx in range(bins):
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

    top_n_eval = eval_args.getint('EVALUATION','TOP_N_EVAL')
    top_n = eval_args.getint('METRIC','TOP_N')
    if top_n_eval < 1:
        return
        
    all_angle_errs = []
    for scene_id in scene_ids:

        if not os.path.exists(os.path.join(eval_dir,'error=re_ntop=%s' % top_n,'errors_{:02d}.yml'.format(scene_id))):
            print(('WARNING: ' + os.path.join(eval_dir,'error=re_ntop=%s' % top_n,'errors_{:02d}.yml'.format(scene_id)) + ' not found'))
            continue

        angle_errs_dict = inout.load_yaml(os.path.join(eval_dir,'error=re_ntop=%s' % top_n,'errors_{:02d}.yml'.format(scene_id)))
        all_angle_errs += [list(angle_e['errors'].values())[0] for angle_e in angle_errs_dict]

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

    for view in range(total_views):
        top_n_errors = all_angle_errs[view*top_n:(view+1)*top_n]
        angle_errs_rect[view] = np.min([top_n_errors[0], 180-top_n_errors[0]])

    bounds = np.linspace(0,1,bins+1)
    bin_angle_errs = []
    bin_count = []

    for idx in range(bins):
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

def main():
    import argparse
    import configparser
    from . import eval_utils
    parser = argparse.ArgumentParser()
    
    parser.add_argument('experiment_name')
    parser.add_argument('evaluation_name')
    parser.add_argument('--eval_cfg', default='eval.cfg', required=False)
    arguments = parser.parse_args()
    full_name = arguments.experiment_name.split('/')
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    evaluation_name = arguments.evaluation_name
    eval_cfg = arguments.eval_cfg

    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    train_cfg_file_path = get_config_file_path(workspace_path, experiment_name, experiment_group)
    eval_cfg_file_path = get_eval_config_file_path(workspace_path, eval_cfg=eval_cfg)

    train_args = configparser.ConfigParser(inline_comment_prefixes="#")
    eval_args = configparser.ConfigParser(inline_comment_prefixes="#")
    train_args.read(train_cfg_file_path)
    eval_args.read(eval_cfg_file_path)

    dataset_name = eval_args.get('DATA','DATASET')
    scenes = eval(eval_args.get('DATA','SCENES')) if len(eval(eval_args.get('DATA','SCENES'))) > 0 else eval_utils.get_all_scenes_for_obj(eval_args)
    cam_type = eval_args.get('DATA','cam_type')
    data = dataset_name + '_' + cam_type if len(cam_type) > 0 else dataset_name

    log_dir = get_log_dir(workspace_path, experiment_name, experiment_group)
    eval_dir = get_eval_dir(log_dir, evaluation_name, data)
    
    plot_R_err_hist_vis(eval_args, eval_dir, scenes,bins=15)
    plot_t_err_hist_vis(eval_args, eval_dir, scenes,bins=15)
    plt.show()

def main2():
    R_errors = []
    
    for R in range(100000):
        R_gt = transform.random_rotation_matrix()[:3,:3]
        R_est = transform.random_rotation_matrix()[:3,:3]
        R_errors.append(pose_error.re(R_est,R_gt))
    plot_R_err_hist2(R_errors,'',bins=90,save=False)

    azimuth_range = (0, 2 * np.pi)
    elev_range = (-0.5 * np.pi, 0.5 * np.pi)
    views, _ = view_sampler.sample_views(
        2563, 
        100, 
        azimuth_range, 
        elev_range
    )
        
    Rs = []
    for view in views:
        R_errors.append(pose_error.re(view['R'],views[np.random.randint(0,len(views))]['R']))
        Rs.append(view['R'])
    plot_R_err_hist2(R_errors,'',bins=45,save=False)

    plot_viewsphere_for_embedding(np.array(Rs),'',np.array(R_errors),save=False)

    plt.show()

if __name__ == "__main__":
    main()