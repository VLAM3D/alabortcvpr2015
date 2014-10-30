from __future__ import division
import abc
import numpy as np

from cvpr2015.utils import build_parts_image, build_sampling_grid

from .result import CLMAlgorithmResult

multivariate_normal = None  # expensive, from scipy.stats


# Abstract Interface for CLM Algorithms ---------------------------------------

class CLMAlgorithm(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, classifiers, parts_shape, normalize_parts,
                 pdm,  eps=10**-5, scale=10):

        self.classifiers = classifiers
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
        #covariance = self.scale * self.transform.model.noise_variance
        covariance = self._scale * self.transform.model.noise_variance
        mvn = multivariate_normal(mean=mean, cov=covariance)
        self._kernel_grid = mvn.pdf(self._sampling_grid)

        # compute Jacobian
        self._j = np.rollaxis(self.transform.d_dp(None), -1)
        #self._j = j.reshape((-1, j.shape[-1]))

        # compute Hessian
        self._h = np.einsum('ijk, ijl -> kl', self._j, self._j)
        #self._h = self._j.T.dot(self._j)

    def _solve(self, h, j, e, prior):
        t = self.transform

        if prior:
            dp = np.linalg.solve(t.h_prior + h,
                                 t.j_prior * t.as_vector() - j.T.dot(e))
        else:
            dp = -np.linalg.solve(h, j.T.dot(e))

        return dp

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]

        n_clfs = len(self.classifiers)
        n_channels = image.n_channels

        for _ in xrange(max_iters):

            target = self.transform.target
            # get all (x, y) pairs being considered
            xys = target.points[:, None, None, ...] + self._sampling_grid
            xys = xys.astype(int)

            # build parts image
            parts_image = build_parts_image(
                image, target, parts_shape=self.parts_shape,
                normalize_parts=self.normalize_parts)

            # compute parts response
            parts_pixels = parts_image.pixels
            parts_response = np.zeros((n_clfs,) + self.parts_shape)
            for j, clf in enumerate(self.classifiers):
                part_pixels = np.reshape(parts_pixels[:, j, 0, ...],
                                         (n_channels, -1)).T
                parts_response[j, :, :] = np.reshape(
                    clf(part_pixels), self.parts_shape)

            # compute parts kernel
            parts_kernel = parts_response * self._kernel_grid
            parts_kernel /= np.sum(
                parts_kernel, axis=(-2, -1))[..., None, None]

            mean_shift_target = np.sum(parts_kernel[..., None] * xys,
                                       axis=(-3, -2))

            # compute (shape) error term
            #e = mean_shift_target.ravel() - target.as_vector()
            e = mean_shift_target - target.points

            # compute gauss-newton parameter updates
            #dp = self._solve(self._h, self._j, e, prior)
            sd_delta_p = np.einsum('ikj, ki -> j', self._j, e)
            dp = np.linalg.solve(self._h, sd_delta_p)

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

# class RLMS(CLMAlgorithm):
#     r"""
#     Regularized Landmark Mean-Shift
#     """
#
#     def _precompute(self):
#
#         global multivariate_normal
#         if multivariate_normal is None:
#             from scipy.stats import multivariate_normal  # expensive
#
#         # build sampling grid associated to patch shape
#         self._sampling_grid = build_sampling_grid(self.parts_shape)
#
#         # compute Gaussian-KDE grid
#         mean = np.zeros(self.transform.n_dims)
#         #covariance = self.scale * self.transform.model.noise_variance
#         covariance = 100 * self.transform.model.noise_variance
#         mvn = multivariate_normal(mean=mean, cov=covariance)
#         self._kernel_grid = mvn.pdf(self._sampling_grid)
#
#         # concatenate all filters
#         n_channels, height, width = self.classifiers[0].f.shape
#         n_landmarks = len(self.classifiers)
#         self._F = np.zeros((n_channels, n_landmarks, height, width),
#                            dtype=np.complex64)
#         for j, clf in enumerate(self.classifiers):
#             self._F[:, j, ...] = clf.f
#
#         # compute Jacobian
#         self._j = np.rollaxis(self.transform.d_dp(None), -1)
#         #self._j = j.reshape((-1, j.shape[-1]))
#
#         # compute Hessian
#         self._h = np.einsum('ijk, ijl -> kl', self._j, self._j)
#         #self._h = self._j.T.dot(self._j)
#
#     def _solve(self, h, j, e, prior):
#         t = self.transform
#
#         if prior:
#             dp = np.linalg.solve(t.h_prior + h,
#                                  t.j_prior * t.as_vector() - j.T.dot(e))
#         else:
#             dp = -np.linalg.solve(h, j.T.dot(e))
#
#         return dp
#
#     def run(self, image, initial_shape, gt_shape=None, max_iters=20,
#             prior=False):
#
#         # initialize transform
#         self.transform.set_target(initial_shape)
#         shape_parameters = [self.transform.as_vector()]
#
#         for _ in xrange(max_iters):
#
#             target = self.transform.target
#
#             # build parts image
#             parts_image = build_parts_image(
#                 image, target, parts_shape=self.parts_shape,
#                 normalize_parts=self.normalize_parts)
#
#             # get all (x, y) pairs being considered
#             xys = target.points[:, None, None, ...] + self._sampling_grid
#
#             # compute parts response
#             parts_response = np.real(
#                 ifft2(self._F * fft2(parts_image.pixels[:, :, 0, ...])))
#             parts_response = np.sum(parts_response, axis=0)
#             min_parts_response = np.min(parts_response,
#                                         axis=(-2, -1))[..., None, None]
#             #    (parts_response - [..., None, None]) / (np.max(
#             # parts_response, axis=(-2, -1))- np.min(parts_response, axis=(-2, -1)))[..., None, None]
#             parts_response -= min_parts_response
#             parts_response /= np.max(parts_response,
#                                      axis=(-2, -1))[..., None, None]
#             # compute parts kernel
#             parts_kernel = parts_response * self._kernel_grid
#             parts_kernel /= np.sum(
#                 parts_kernel, axis=(-2, -1))[..., None, None]
#
#             mean_shift_target = np.sum(parts_kernel[..., None] * xys,
#                                        axis=(-3, -2))
#
#             # compute (shape) error term
#             #e = mean_shift_target.ravel() - target.as_vector()
#             e = mean_shift_target - target.points
#
#             # compute gauss-newton parameter updates
#             #dp = self._solve(self._h, self._j, e, prior)
#             sd_delta_p = np.einsum('ikj, ki -> j', self._j, e)
#             dp = np.linalg.solve(self._h, sd_delta_p)
#
#             # update pdm
#             self.transform.from_vector_inplace(self.transform.as_vector() + dp)
#             shape_parameters.append(self.transform.as_vector())
#
#             # test convergence
#             error = np.abs(np.linalg.norm(
#                 target.points - self.transform.target.points))
#             if error < self.eps:
#                 break
#
#         # return CLM algorithm result
#         return CLMAlgorithmResult(image, self, shape_parameters,
#                                   gt_shape=gt_shape)




