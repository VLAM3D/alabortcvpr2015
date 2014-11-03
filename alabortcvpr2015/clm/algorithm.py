from __future__ import division
import abc

import numpy as np

from menpofast.utils import build_parts_image

from alabortcvpr2015.utils import build_sampling_grid
from alabortcvpr2015.clm.result import CLMAlgorithmResult


multivariate_normal = None  # expensive, from scipy.stats


# Abstract Interface for CLM Algorithms ---------------------------------------

class CLMAlgorithm(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, multiple_clf, parts_shape, normalize_parts,
                 pdm,  eps=10**-5, scale=10):

        self.multiple_clf = multiple_clf
        self.parts_shape = parts_shape
        self.normalize_parts = normalize_parts
        self.transform = pdm
        self.eps = eps
        self._scale = scale

        # pre-compute
        self._precompute()

    @abc.abstractmethod
    def _precompute(self, **kwargs):
        pass

    @abc.abstractmethod
    def run(self, image, initial_shape, max_iters=20, gt_shape=None, **kwargs):
        pass


# Concrete Implementations of CLM Algorithm -----------------------------------

class RLMS(CLMAlgorithm):
    r"""
    Regularized Landmark Mean-Shift
    """

    def _precompute(self):

        global multivariate_normal
        if multivariate_normal is None:
            from scipy.stats import multivariate_normal  # expensive

        # build sampling grid associated to patch shape
        self._sampling_grid = build_sampling_grid(self.parts_shape)

        # compute Gaussian-KDE grid
        mean = np.zeros(self.transform.n_dims)
        covariance = self._scale * self.transform.model.noise_variance()
        mvn = multivariate_normal(mean=mean, cov=covariance)
        self._kernel_grid = mvn.pdf(self._sampling_grid)

        # compute Jacobian
        j = np.rollaxis(self.transform.d_dp(None), -1, 1)
        self._j = j.reshape((-1, j.shape[-1]))

        # compute Hessian inverse
        self._h = self._j.T.dot(self._j)
        self._inv_h = np.linalg.inv(self._h)
        self._inv_h_prior = np.linalg.inv(self.transform.h_prior + self._h)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]

        # n_clfs = len(self.classifiers)
        # n_channels = image.n_channels

        for _ in xrange(max_iters):

            target = self.transform.target
            rounded_target = target.copy()
            rounded_target.points = np.round(target.points)
            # get all (x, y) pairs being considered
            xys = (rounded_target.points[:, None, None, ...] +
                   self._sampling_grid)

            # build parts image
            parts_image = build_parts_image(
                image, target, parts_shape=self.parts_shape,
                normalize_parts=self.normalize_parts)

            # compute parts response
            parts_response = self.multiple_clf(parts_image)

            # compute parts kernel
            parts_kernel = parts_response * self._kernel_grid
            parts_kernel /= np.sum(
                parts_kernel, axis=(-2, -1))[..., None, None]

            # compute mean shift target
            mean_shift_target = np.sum(parts_kernel[..., None] * xys,
                                       axis=(-3, -2))

            # compute (shape) error term
            e = mean_shift_target.ravel() - target.as_vector()

            # compute gauss-newton parameter updates
            if prior:
                dp = self._inv_h_prior.dot(
                    self.transform.j_prior * self.transform.as_vector() +
                    self._j.T.dot(e))
            else:
                dp = self._inv_h.dot(self._j.T.dot(e))

            # update pdm
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return CLM algorithm result
        return CLMAlgorithmResult(image, self, shape_parameters,
                                  gt_shape=gt_shape)
