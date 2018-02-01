import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import os
from sklearn.neighbors import NearestNeighbors

from meshrenderer import meshrenderer, meshrenderer_phong
from ae.utils import lazy_property

from sixd_toolkit.pysixd import transform, misc, inout, pose_error
from sixd_toolkit.params import dataset_params

# Constants
N = 2000                                 # number of random points in the dataset
dim = 3                                     # number of dimensions of the points
verbose = False
max_mean_dist_factor = 1.5


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=100, tolerance=0.001, verbose=False):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    if verbose:
        plt.clf()
        fig = plt.figure(1)
        ax = Axes3D(fig)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']) 
        ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
        ax.scatter(A[:,0],A[:,1],A[:,2], label='initial', marker='.', c='green')
        ax.scatter(B[:,0],B[:,1],B[:,2], label='target', marker='.', c='blue')

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        if verbose:
            anim = ax.scatter(src[0,:],src[1,:],src[2,:], label='estimated',marker='.',c='red')
            plt.legend()
            plt.draw()
            plt.pause(0.001)
            anim.remove()
        # update the current source
        src = np.dot(T, src)

        
        mean_error = np.mean(distances)
        # print mean_error
        # check error
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    if verbose:
        anim = ax.scatter(src[0,:],src[1,:],src[2,:], label='estimated',marker='.',c='red')
        plt.legend()
        plt.show()

    return T, distances, i


class SynRenderer(object):
	def __init__(self,train_args):
		MODEL_PATH = train_args.get('Paths','MODEL_PATH')
		self.model_path = MODEL_PATH
		self.W, self.H = eval(train_args.get('Dataset','RENDER_DIMS'))
		self.renderer

	@lazy_property
	def renderer(self):
		return meshrenderer.Renderer([self.model_path],1,'.',1)

	def generate_synthetic_depth(self,K_test, R_est, t_est):

	    # renderer = meshrenderer.Renderer(['/net/rmc-lx0050/home_local/sund_ma/data/SLC_precise_blue.ply'],1,'.',1)
	    # R = transform.random_rotation_matrix()[:3,:3]

	    _, depth_x = self.renderer.render( 
	                    obj_id=0,
	                    W=self.W, 
	                    H=self.H,
	                    K=K_test, 
	                    R=R_est, 
	                    t=np.array([0,0,t_est[2]]),
	                    near=10,
	                    far=10000,
	                    random_light=False
	                )
	    # import cv2
	    # cv2.imshow('bgr_x',bgr_x)
	    # scene_mi = cv2.imread('/home_local/sund_ma/data/t-less/t-less_v2/test_primesense/02/rgb/0250.png')
	    # cv2.imshow('scene_mi',scene_mi)
	    # cv2.waitKey(0)

	    pts = misc.rgbd_to_point_cloud(K_test,depth_x)[0]

	    return pts


def icp_refinement(depth_crop, icp_renderer, R_est, t_est, K_test):

	synthetic_pts = icp_renderer.generate_synthetic_depth(K_test, R_est,t_est)
	centroid_synthetic_pts = np.mean(synthetic_pts, axis=0)
	max_mean_dist = np.max(np.linalg.norm(synthetic_pts - centroid_synthetic_pts,axis=1))
	# print 'max_mean_dist', max_mean_dist
	
	K_test[0,2] = depth_crop.shape[0]/2
	K_test[1,2] = depth_crop.shape[1]/2
	real_depth_pts = misc.rgbd_to_point_cloud(K_test,depth_crop)[0]

	real_synmean_dist = np.linalg.norm(real_depth_pts-centroid_synthetic_pts,axis=1)
	real_depth_pts = real_depth_pts[real_synmean_dist < max_mean_dist_factor*max_mean_dist]

	print len(real_depth_pts)
	if len(real_depth_pts) < 500:
		print 'not enough visible points'
		R_refined = R_est
		t_refined = t_est
	else:
		sub_idcs_real = np.random.choice(len(real_depth_pts),np.min([len(real_depth_pts),len(synthetic_pts),N]))
		sub_idcs_syn = np.random.choice(len(synthetic_pts),np.min([len(real_depth_pts),len(synthetic_pts),N]))

		T, distances, iterations = icp(synthetic_pts[sub_idcs_syn], real_depth_pts[sub_idcs_real], tolerance=0.000001, verbose=verbose)

		t_est_hom = np.ones((4,1))
		t_est_hom[:3] = t_est.reshape(3,1)
		R_refined = np.dot(T[:3,:3],R_est).squeeze()
		t_refined = np.dot(T,t_est_hom)[:3].squeeze()

	return (R_refined, t_refined)