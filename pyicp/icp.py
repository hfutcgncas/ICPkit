# modifyed from 
# https://github.com/DLR-RM/AugmentedAutoencoder/blob/master/auto_pose/icp/icp.py

import os
import numpy as np
from sklearn.neighbors import NearestNeighbors


# Constants
N = 3000                                 # number of random points in the dataset
dim = 3                                     # number of dimensions of the points



class ICP():
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
                t = np.array([t[0],t[1],0])


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

        # not sure
        # assert src.shape == dst.shape

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
        # not sure?
        # assert A.shape == B.shape

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



