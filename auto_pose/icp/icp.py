import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from auto_pose.ae.pysixd_stuff import transform, misc
from .renderer import SynRenderer

# Constants
N = 3000                                 # number of random points in the dataset
dim = 3                                     # number of dimensions of the points
max_mean_dist_factor = 4.0
angle_change_limit = 0.35 # = 20 deg #0.5236=30 deg


class ICP():
    def __init__(self, test_args):
        self.syn_renderer = SynRenderer(test_args)

    def best_fit_transform(self,A, B, depth_only=False, no_depth=False):
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


        if depth_only:
            R=np.eye(3)
            t = centroid_B.T - centroid_A.T
            t = np.array([0,0,t[2]])
        else:
            # rotation matrix
            H = np.dot(AA.T, BB)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)

            # special reflection case
            if np.linalg.det(R) < 0:
               Vt[m-1,:] *= -1
               R = np.dot(Vt.T, U.T)
            
            t = centroid_B.T - np.dot(R,centroid_A.T)
            if no_depth:
                #TODO
                #t = np.array([t[0],t[1],0])
                t = np.array([0,0,0])


        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t


    def nearest_neighbor(self,src, dst):
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


    def icp(self,A, B, init_pose=None, max_iterations=100, tolerance=0.001, depth_only=False,no_depth=False):
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

        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = self.nearest_neighbor(src[:m,:].T, dst[:m,:].T)

            # compute the transformation between the current source and nearest destination points
            T,_,_ = self.best_fit_transform(src[:m,:].T, dst[:m,indices].T, depth_only=depth_only, no_depth=no_depth)


            # if verbose:
            #     anim = ax.scatter(src[0,:],src[1,:],src[2,:], label='estimated',marker='.',c='red')
            #     plt.legend()
            #     plt.draw()
            #     plt.pause(0.001)
            #     anim.remove()

            # update the current source
            src = np.dot(T, src)

            
            mean_error = np.mean(distances)
            # print mean_error
            # check error
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        T,_,_ = self.best_fit_transform(A, src[:m,:].T, depth_only=depth_only, no_depth=no_depth)

        # if verbose:
        #     anim = ax.scatter(src[0,:],src[1,:],src[2,:], label='estimated',marker='.',c='red')
        #     # final_trafo = np.dot(T, orig_src)
        #     # anim2 = ax.scatter(final_trafo[0,:],final_trafo[1,:],final_trafo[2,:], label='final_trafo',marker='.',c='black')
        #     plt.legend()
        #     plt.show()

        return T, distances, i


    def icp_refinement(self,depth_crop, R_est, t_est, K_test, test_render_dims, depth_only=False, no_depth=False,clas_idx=0):
        synthetic_pts = self.syn_renderer.generate_synthetic_depth(K_test, R_est, t_est, test_render_dims,clas_idx=clas_idx)
        centroid_synthetic_pts = np.mean(synthetic_pts, axis=0)
        max_mean_dist = np.max(np.linalg.norm(synthetic_pts - centroid_synthetic_pts,axis=1))
        # print 'max_mean_dist', max_mean_dist
        K_test_crop = K_test.copy()
        K_test_crop[0,2] = depth_crop.shape[0]/2
        K_test_crop[1,2] = depth_crop.shape[1]/2
        real_depth_pts = misc.rgbd_to_point_cloud(K_test_crop,depth_crop)[0]

        print(('noofpoints real/syn: ', len(real_depth_pts)))
        real_synmean_dist = np.linalg.norm(real_depth_pts-centroid_synthetic_pts,axis=1)
        real_depth_pts = real_depth_pts[real_synmean_dist < max_mean_dist_factor*max_mean_dist]
        print(('noofpoints real/syn: ', len(real_depth_pts), len(synthetic_pts)))

        print(('real min max x', np.min(real_depth_pts[:,0]), np.max(real_depth_pts[:,0])))
        print(('real min max y', np.min(real_depth_pts[:,1]), np.max(real_depth_pts[:,1])))
        print(('real min max z', np.min(real_depth_pts[:,2]), np.max(real_depth_pts[:,2])))
        print(('syn min max x', np.min(synthetic_pts[:,0]), np.max(synthetic_pts[:,0])))
        print(('syn min max y', np.min(synthetic_pts[:,1]), np.max(synthetic_pts[:,1])))
        print(('syn min max z', np.min(synthetic_pts[:,2]), np.max(synthetic_pts[:,2])))


        if len(real_depth_pts) < len(synthetic_pts)/8.:
            print('not enough visible points')
            R_refined = R_est
            t_refined = t_est
        else:
            sub_idcs_real = np.random.choice(len(real_depth_pts),np.min([len(real_depth_pts),len(synthetic_pts),N]))
            sub_idcs_syn = np.random.choice(len(synthetic_pts),np.min([len(real_depth_pts),len(synthetic_pts),N]))

            T, distances, iterations = self.icp(synthetic_pts[sub_idcs_syn], real_depth_pts[sub_idcs_real], 
                                            tolerance=0.000001, depth_only=depth_only, no_depth=no_depth)


            # reject big angle changes
            if no_depth:
                angle,_,_ = transform.rotation_from_matrix(T)
                if np.abs(angle) > angle_change_limit:
                    T = np.eye(4)

            H_est = np.eye(4)
            # R_est, t_est is from model to camera
            H_est[:3,3] = t_est 
            H_est[:3,:3] = R_est
            H_est_refined = np.dot(T,H_est)

            R_refined = H_est_refined[:3,:3]
            t_refined = H_est_refined[:3,3]

        return (R_refined, t_refined)
