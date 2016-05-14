
import abc

import numpy as np
from numpy.fft import fft2, ifft2, fftshift

from menpofast.feature import gradient as fast_gradient
from menpofast.utils import build_parts_image
from menpofast.image import Image

from menpofit.base import build_sampling_grid

from alabortcvpr2015.aam.algorithm import PartsAAMInterface

from .result import CLMAlgorithmResult


multivariate_normal = None  # expensive, from scipy.stats


class CLMAlgorithm(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _precompute(self, **kwargs):
        pass

    @abc.abstractmethod
    def run(self, image, initial_shape, max_iters=20, gt_shape=None, **kwargs):
        pass


class RLMS(CLMAlgorithm):
    r"""
    Regularized Landmark Mean-Shift
    """

    def __init__(self, multiple_clf, parts_shape, normalize_parts, covariance,
                 pdm, scale=10, factor=1, eps=10**-5, **kwarg):

        self.multiple_clf = multiple_clf
        self.parts_shape = parts_shape
        self.normalize_parts = normalize_parts
        self.covariance = covariance
        self.transform = pdm
        self.eps = eps
        self.scale = scale
        self.factor = factor

        # pre-compute
        self._precompute()

    def _precompute(self):

        global multivariate_normal
        if multivariate_normal is None:
            from scipy.stats import multivariate_normal  # expensive

        # build sampling grid associated to patch shape
        self._sampling_grid = build_sampling_grid(self.parts_shape)
        up_sampled_shape = self.factor * (np.asarray(self.parts_shape) + 1)
        self._up_sampled_grid = build_sampling_grid(up_sampled_shape)
        self.offset = np.mgrid[self.factor:up_sampled_shape[0]:self.factor,
                               self.factor:up_sampled_shape[1]:self.factor]
        self.offset = self.offset.swapaxes(0, 2).swapaxes(0, 1)

        # set rho2
        self._rho2 = self.transform.model.noise_variance()

        # compute Gaussian-KDE grid
        mean = np.zeros(self.transform.n_dims)
        covariance = self.scale + self._rho2
        mvn = multivariate_normal(mean=mean, cov=covariance)
        self._kernel_grid = mvn.pdf(self._up_sampled_grid/self.factor)
        n_parts = self.transform.model.mean().n_points
        self._kernel_grids = np.empty((n_parts,) + self.parts_shape)

        # compute Jacobian
        j = np.rollaxis(self.transform.d_dp(None), -1, 1)
        self._j = j.reshape((-1, j.shape[-1]))

        # set Prior
        sim_prior = np.zeros((4,))
        pdm_prior = self._rho2 / self.transform.model.eigenvalues
        self._j_prior = np.hstack((sim_prior, pdm_prior))

        # compute Hessian inverse
        h = self._j.T.dot(self._j)
        self._pinv_jT = np.linalg.solve(h, self._j.T)
        self._inv_h_prior = np.linalg.inv(h + np.diag(self._j_prior))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]

        for _ in range(max_iters):

            target = self.transform.target
            # get all (x, y) pairs being considered
            xys = (target.points[:, None, None, ...] +
                   self._sampling_grid)

            diff = np.require(
                np.round((np.round(target.points) - target.points) *
                         self.factor),
                dtype=int)

            offsets = diff[:, None, None, :] + self.offset
            for j, o in enumerate(offsets):
                self._kernel_grids[j, ...] = self._kernel_grid[o[..., 0],
                                                               o[..., 1]]

            # build parts image
            parts_image = build_parts_image(
                image, target, parts_shape=self.parts_shape,
                normalize_parts=self.normalize_parts)

            # compute parts response
            parts_response = self.multiple_clf(parts_image)
            parts_response[np.logical_not(np.isfinite(parts_response))] = .5

            # compute parts kernel
            parts_kernel = parts_response * self._kernel_grids
            parts_kernel /= np.sum(
                parts_kernel, axis=(-2, -1))[..., None, None]

            # compute mean shift target
            mean_shift_target = np.sum(parts_kernel[..., None] * xys,
                                       axis=(-3, -2))

            # compute (shape) error term
            e = mean_shift_target.ravel() - target.as_vector()

            # compute gauss-newton parameter updates
            if prior:
                dp = -self._inv_h_prior.dot(
                    self._j_prior * self.transform.as_vector() -
                    self._j.T.dot(e))
            else:
                dp = self._pinv_jT.dot(e)

            # update pdm
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return algorithm result
        return CLMAlgorithmResult(image, self, shape_parameters,
                                  gt_shape=gt_shape)
