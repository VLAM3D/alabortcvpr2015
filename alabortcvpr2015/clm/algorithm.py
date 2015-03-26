from __future__ import division
import abc

import numpy as np

from alabortcvpr2015.clm.result import CLMAlgorithmResult
from alabortcvpr2015.correlationfilter.utils import build_grid

multivariate_normal = None  # expensive, from scipy.stats


class CLMAlgorithm(object):

    def __init__(self, clf_ensemble, search_shape, pdm, scale=10,
                 factor=100, eps=10**-5):

        self.clf_ensemble = clf_ensemble
        self.search_shape = search_shape
        self.transform = pdm
        self.eps = eps
        self.scale = scale
        self.factor = factor

        # pre-compute
        self._precompute()

    @abc.abstractmethod
    def _precompute(self, **kwargs):
        pass

    @abc.abstractmethod
    def run(self, image, initial_shape, max_iters=20, gt_shape=None,
            map_inference=False):
        pass


# Concrete Implementations of CLM Algorithm -----------------------------------

class RLMS(CLMAlgorithm):
    r"""
    Regularized Landmark Mean-Shift (RLMS)
    """
    def _precompute(self):
        r"""
        Pre-compute state for RLMS algorithm
        """
        # import multivariate normal distribution from scipy
        global multivariate_normal
        if multivariate_normal is None:
            from scipy.stats import multivariate_normal  # expensive

        # build sampling grid associated to patch shape
        self.grid = build_grid(self.search_shape)
        upsampled_shape = self.factor * (np.asarray(self.search_shape) + 1)
        self.upsampled_grid = build_grid(upsampled_shape)
        self.offset = np.mgrid[self.factor:upsampled_shape[0]:self.factor,
                               self.factor:upsampled_shape[1]:self.factor]
        self.offset = self.offset.swapaxes(0, 2).swapaxes(0, 1)

        # set rho2
        self.rho2 = self.transform.model.noise_variance()

        # compute Gaussian-KDE grid
        mean = np.zeros(self.transform.n_dims)
        cov = self.scale * self.rho2
        mvn = multivariate_normal(mean=mean, cov=cov)
        self.kernel_grid = mvn.pdf(self.upsampled_grid)
        n_landmarks = self.transform.model.mean().n_points
        self.kernel_grids = np.empty((n_landmarks, 1) + self.search_shape)

        # compute shape model prior
        sim_prior = np.zeros((4,))
        pdm_prior = self.rho2 / self.transform.model.eigenvalues
        self.rho2_inv_L = np.hstack((sim_prior, pdm_prior))

        # compute Jacobian
        J = np.rollaxis(self.transform.d_dp(None), -1, 1)
        self.J = J.reshape((-1, J.shape[-1]))
        # compute inverse Hessian
        self.JJ = self.J.T.dot(self.J)
        # compute Jacobian pseudo-inverse
        self.pinv_J = np.linalg.solve(self.JJ, self.J.T)
        self.inv_JJ_prior = np.linalg.inv(self.JJ + np.diag(self.rho2_inv_L))

    def run(self, image, initial_shape, max_iters=20, gt_shape=None,
            map_inference=False):
        r"""
        Run RLMS algorithm
        """
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop
        while k < max_iters and eps > self.eps:

            target = self.transform.target
            # get all (x, y) pairs being considered
            xys = (target.points[:, None, None, ...] + self.grid)

            diff = np.require(
                np.round(-np.round(target.points) + target.points,
                         decimals=np.int(self.factor/10)) *
                self.factor, dtype=int)

            offsets = diff[:, None, None, :] + self.offset
            for j, o in enumerate(offsets):
                self.kernel_grids[j, ...] = self.kernel_grid[o[..., 0],
                                                             o[..., 1]]

            # build parts image
            parts_image = image.extract_patches(
                target, patch_size=self.search_shape, as_single_array=True)
            parts_image = parts_image[:, 0, ...]

            # compute parts response
            parts_response = self.clf_ensemble.predict(parts_image)

            # normalize
            min_parts_response = np.min(parts_response,
                                        axis=(-2, -1))[..., None, None]
            parts_response -= min_parts_response
            parts_response /= np.max(parts_response,
                                     axis=(-2, -1))[..., None, None]

            # compute parts kernel
            parts_kernel = parts_response * self.kernel_grids
            parts_kernel /= np.sum(
                parts_kernel, axis=(-2, -1))[..., None, None]

            # compute mean shift target
            mean_shift_target = np.sum(parts_kernel[:, 0, ..., None] * xys,
                                       axis=(-3, -2))

            # compute (shape) error term
            e = mean_shift_target.ravel() - target.as_vector()

            # solve for increments on the shape parameters
            if map_inference:
                Je = (self.rho2_inv_L * self.transform.as_vector() -
                      self.J.T.dot(e))
                dp = -self.inv_JJ_prior.dot(Je)
            else:
                dp = self.pinv_J.dot(e)

            # update pdm
            s_k = self.transform.target.points
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return CLMAlgorithmResult(image, self, p_list,
                                  gt_shape=gt_shape)
