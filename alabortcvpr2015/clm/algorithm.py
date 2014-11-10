from __future__ import division
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


class CLMAlgorithm(object):

    __metaclass__ = abc.ABCMeta

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
                 pdm, scale=10, factor=100, eps=10**-5, **kwarg):

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
        covariance = self.scale * self._rho2
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

        for _ in xrange(max_iters):

            target = self.transform.target
            # get all (x, y) pairs being considered
            xys = (target.points[:, None, None, ...] +
                   self._sampling_grid)

            diff = np.require(
                np.round((-np.round(target.points) + target.points) *
                self.factor), dtype=int)

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
            parts_response[np.logical_not(np.isfinite(parts_response))] = 1

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


class LK(CLMAlgorithm):

    def __init__(self, multiple_clf, parts_shape, normalize_parts, covariance,
                 pdm, sampling_mask=None, eps=10**-5, **kwargs):

        self.multiple_clf = multiple_clf
        self.parts_shape = parts_shape
        self.normalize_parts = normalize_parts
        self.covariance = covariance
        self.transform = pdm
        self.eps = eps

        self.n_parts = self.multiple_clf.F.shape[0]
        self.n_channels = self.multiple_clf.F.shape[-3]

        global multivariate_normal
        if multivariate_normal is None:
            from scipy.stats import multivariate_normal  # expensive

        # compute template response
        mvn = multivariate_normal(mean=np.zeros(2), cov=self.covariance)
        grid = build_sampling_grid(self.parts_shape)
        self.template = np.require(
            fft2(np.tile(mvn.pdf(grid)[None, None, None],
                         (self.n_parts, 1, 1, 1, 1))), dtype=np.complex64)

        if sampling_mask is None:
            parts_shape = self.parts_shape
            sampling_mask = np.require(np.ones((parts_shape)), dtype=np.bool)

        template_shape = self.template.shape
        image_mask = np.tile(sampling_mask[None, None, None, ...],
                             template_shape[:2] + (self.n_channels, 1, 1))
        self.image_vec_mask = np.nonzero(image_mask.flatten())[0]
        self.gradient_mask = np.nonzero(np.tile(
            image_mask[None, ...], (2, 1, 1, 1, 1, 1)))

        template_mask = np.tile(sampling_mask[None, None, None, ...],
                                template_shape[:3] + (1, 1))
        self.template_vec_mask = np.nonzero(template_mask.flatten())[0]

        # pre-compute
        self._precompute()

    def gradient(self, image):
        g = fast_gradient(image.pixels.reshape((-1,) + self.parts_shape))
        return g.reshape((2,) + image.pixels.shape)

    def steepest_descent_images(self, gradient, dw_dp):
        # reshape gradient
        # gradient: n_dims x n_parts x offsets x n_ch x (h x w)
        gradient = gradient[self.gradient_mask].reshape(
            gradient.shape[:-2] + (-1,))
        # compute steepest descent images
        # gradient: n_dims x n_parts x offsets x n_ch x (h x w)
        # ds_dp:    n_dims x n_parts x                          x n_params
        # sdi:               n_parts x offsets x n_ch x (h x w) x n_params
        sdi = 0
        a = gradient[..., None] * dw_dp[..., None, None, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (n_parts x n_offsets x n_ch x w x h) x n_params
        return sdi[:, 0, ...]


class LKForward(LK):
    r"""
    Lucas-Kanade Forward Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = np.rollaxis(self.transform.d_dp(None), -1)

        # set rho2
        self._rho2 = self.transform.model.noise_variance()

        # set Prior
        sim_prior = np.zeros((4,))
        pdm_prior = self._rho2 / self.transform.model.eigenvalues
        self._j_prior = np.hstack((sim_prior, pdm_prior))
        self._h_prior = np.diag(self._j_prior)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]

        parts_template = self.template.ravel()[self.template_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            parts_image = build_parts_image(
                image, self.transform.target, parts_shape=self.parts_shape,
                normalize_parts=self.normalize_parts)

            # compute image response
            parts_fft2 = np.require(fft2(parts_image.pixels[:, 0, ...]),
                                    dtype=np.complex64)
            parts_response = np.sum(self.multiple_clf.F * parts_fft2, axis=-3)
            parts_response[np.logical_not(np.isfinite(parts_response))] = 0.5

            # compute error image
            e = (parts_template -
                 parts_response.ravel()[self.template_vec_mask])

            # compute image gradient
            nabla_i = self.gradient(parts_image)
            nabla_i = np.require(fft2(nabla_i), dtype=np.complex64)

            # compute image jacobian
            j = self.steepest_descent_images(nabla_i, self._dw_dp)
            h = np.sqrt(np.asarray(j.shape[-2]))
            w = h
            j = j.reshape((self.n_parts, self.n_channels, h, w, -1))

            # compute jacobian response
            j = np.sum(self.multiple_clf.F.ravel()[self.image_vec_mask].reshape(
                j.shape[:-1])[..., None] * j, axis=-4)
            j = j.reshape((-1, self.transform.n_parameters))
            conj_j = np.conj(j)

            # compute hessian
            h = conj_j.T.dot(j)

            # compute gauss-newton parameter updates
            if prior:
                dp = -np.real(np.linalg.solve(
                    self._h_prior + h,
                    self._j_prior * self.transform.as_vector() -
                    conj_j.T.dot(e)))
            else:
                dp = np.real(np.linalg.solve(h, conj_j.T.dot(e)))

            # update transform
            target = self.transform.target
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


class LKInverse(LK):
    r"""
    Lucas-Kanade Inverse Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = np.rollaxis(self.transform.d_dp(None), -1)

        # set rho2
        self._rho2 = self.transform.model.noise_variance()

        # set Prior
        sim_prior = np.zeros((4,))
        pdm_prior = self._rho2 / self.transform.model.eigenvalues
        self._j_prior = np.hstack((sim_prior, pdm_prior))
        self._h_prior = np.diag(self._j_prior)

        inv_F = np.real(ifft2(self.multiple_clf.F))[:, None, ...]

        # compute image gradient
        nabla_F = self.gradient(Image(inv_F))
        nabla_F = np.require(fft2(nabla_F), dtype=np.complex64)

        # compute image jacobian
        self.j = self.steepest_descent_images(nabla_F, self._dw_dp)
        h = np.sqrt(np.asarray(self.j.shape[-2]))
        w = h
        self.j = self.j.reshape((self.n_parts, self.n_channels, h, w, -1))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]

        parts_template = self.template.ravel()[self.template_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            parts_image = build_parts_image(
                image, self.transform.target, parts_shape=self.parts_shape,
                normalize_parts=self.normalize_parts)

            # compute image response
            parts_fft2 = np.require(fft2(parts_image.pixels[:, 0, ...]),
                                        dtype=np.complex64)
            parts_response = np.sum(self.multiple_clf.F * parts_fft2, axis=-3)
            parts_response[np.logical_not(np.isfinite(parts_response))] = 0.5

            # compute error image
            e = (parts_template -
                 parts_response.ravel()[self.template_vec_mask])

            # compute jacobian response
            j = np.sum(parts_fft2.ravel()[self.image_vec_mask].reshape(
                self.j.shape[:-1])[..., None] * self.j, axis=-4)
            j = j.reshape((-1, self.transform.n_parameters))
            conj_j = np.conj(j)

            # compute hessian
            h = conj_j.T.dot(j)

            # compute gauss-newton parameter updates
            if prior:
                dp = -np.real(np.linalg.solve(
                    self._h_prior + h,
                    self._j_prior * self.transform.as_vector() -
                    conj_j.T.dot(e)))
            else:
                dp = np.real(np.linalg.solve(h, conj_j.T.dot(e)))

            # update transform
            target = self.transform.target
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