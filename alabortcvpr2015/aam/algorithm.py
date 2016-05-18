
import abc

import numpy as np

from menpo.feature import gradient as fast_gradient
from menpo.image import extract_patches, MaskedImage

from .result import AAMAlgorithmResult

def build_parts_image(image, centres, parts_shape, offsets=np.array([[0, 0]]),
                      normalize_parts=False):

    # extract patches
    parts = extract_patches(image.pixels, np.round(centres.points),
                            np.array(parts_shape), offsets)

    # build parts image
    # img.pixels: n_centres x n_offsets x n_channels x height x width
    img = Image(parts)

    if normalize_parts:
        # normalize parts if required
        img.normalize_norm_inplace(mode='per_channel')

    return img

# Abstract Interface for AAM Algorithms ---------------------------------------

class AAMAlgorithm(object, metaclass=abc.ABCMeta):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):

        # set common state for all AAM algorithms
        self.appearance_model = appearance_model
        self.template = appearance_model.mean()
        self.transform = transform
        self.eps = eps

        # set interface
        self.interface = aam_interface(self, **kwargs)

        self._U = self.appearance_model.components.T
        self._pinv_U = np.linalg.pinv(
            self._U[self.interface.image_vec_mask, :]).T

        # pre-compute
        self._precompute()

    @abc.abstractmethod
    def _precompute(self, **kwargs):
        pass

    @abc.abstractmethod
    def run(self, image, initial_shape, max_iters=20, gt_shape=None, **kwargs):
        pass


# Concrete Implementations of AAM Algorithm -----------------------------------

class PIC(AAMAlgorithm):
    r"""
    Project-Out Inverse Compositional Algorithm
    """

    def _precompute(self):

        # sample appearance model
        self._U = self._U[self.interface.image_vec_mask, :]

        # compute model's gradient
        nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        dw_dp = self.interface.dw_dp()

        self._inv_sigma2 = self.appearance_model.inverse_noise_variance()

        # compute steepest descent images
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)

        # project out appearance model from J
        self._j_po = self._inv_sigma2 * (j -
                                         self._U.dot(self._pinv_U.T.dot(j)))

        # compute inverse hessian
        self._h = self._j_po.T.dot(j)

        # set Prior
        sim_prior = np.zeros((4,))
        pdm_prior = 1 / self.interface.eigenvalues
        self._j_prior = np.hstack((sim_prior, pdm_prior))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in range(max_iters):

            # warp image
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute gauss-newton parameter updates
            dp = self.interface.solve(self._h, self._j_po, e, prior)

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
        return AAMAlgorithmResult(image, self, shape_parameters,
                                  gt_shape=gt_shape)


class AIC(AAMAlgorithm):
    r"""
    Alternating Inverse Compositional Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        # set inverse sigma2
        self._inv_sigma2 = self.appearance_model.inverse_noise_variance()

        # set Prior
        sim_prior = np.zeros((4,))
        pdm_prior = 1 / self.interface.eigenvalues
        self._j_prior = np.hstack((sim_prior, pdm_prior))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in range(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # reconstruct appearance
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            ja = self._inv_sigma2 * j

            # compute hessian
            h = ja.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, ja, e, prior)

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
        return AAMAlgorithmResult(image, self, shape_parameters,
                                  appearance_parameters=appearance_parameters,
                                  gt_shape=gt_shape)


# Abstract Interface for AAM interfaces ---------------------------------------

class AAMInterface(object, metaclass=abc.ABCMeta):

    def __init__(self, aam_algorithm):
        self.algorithm = aam_algorithm

    @abc.abstractmethod
    def dw_dp(self):
        pass

    @abc.abstractmethod
    def warp(self, image):
        pass

    @abc.abstractmethod
    def gradient(self, image):
        pass

    @abc.abstractmethod
    def steepest_descent_images(self, gradient, dw_dp):
        pass

    @abc.abstractmethod
    def solve(self, h, j, e, prior):
        pass


# Concrete Implementations of AAM Interfaces ----------------------------------

class GlobalAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling_step=None):
        super(GlobalAAMInterface, self). __init__(aam_algorithm)

        n_true_pixels = self.algorithm.template.n_true_pixels()
        n_channels = self.algorithm.template.n_channels
        n_parameters = self.algorithm.transform.n_parameters
        sampling_mask = np.require(np.zeros(n_true_pixels), dtype=np.bool)

        if sampling_step is None:
            sampling_step = 1
        sampling_pattern = range(0, n_true_pixels, sampling_step)
        sampling_mask[sampling_pattern] = 1

        self.image_vec_mask = np.nonzero(np.tile(
            sampling_mask[None, ...], (n_channels, 1)).flatten())[0]
        self.dw_dp_mask = np.nonzero(np.tile(
            sampling_mask[None, ..., None], (2, 1, n_parameters)))
        self.gradient_mask = np.nonzero(np.tile(
            sampling_mask[None, None, ...], (2, n_channels, 1)))
        self.gradient2_mask = np.nonzero(np.tile(
            sampling_mask[None, None, None, ...], (2, 2, n_channels, 1)))

        self.eigenvalues = self.algorithm.transform.pdm.model.eigenvalues

    def dw_dp(self):
        dw_dp = np.rollaxis(self.algorithm.transform.d_dp(
            self.algorithm.template.mask.true_indices()), -1)
        return dw_dp[self.dw_dp_mask].reshape((dw_dp.shape[0], -1,
                                               dw_dp.shape[2]))

    def warp(self, image):
        return image.warp_to_mask(self.algorithm.template.mask,
                                  self.algorithm.transform,
                                  warp_landmarks = False)

    def gradient(self, image):
        grad = fast_gradient(image)
        grad.set_boundary_pixels()
        return grad.as_vector().reshape((2, image.n_channels, -1))

    def steepest_descent_images(self, gradient, dw_dp):
        # reshape gradient
        # gradient: n_dims x n_channels x n_pixels
        gradient = gradient[self.gradient_mask].reshape(
            gradient.shape[:2] + (-1,))
        # compute steepest descent images
        # gradient: n_dims x n_channels x n_pixels
        # dw_dp:    n_dims x            x n_pixels x n_params
        # sdi:               n_channels x n_pixels x n_params
        sdi = 0
        a = gradient[..., None] * dw_dp[:, None, ...]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (n_channels x n_pixels) x n_params
        return sdi.reshape((-1, sdi.shape[2]))

    def solve(self, h, j, e, prior):
        t = self.algorithm.transform
        jp = t.jp()

        if prior:
            inv_h = np.linalg.inv(h)
            dp = inv_h.dot(j.T.dot(e))
            dp = -np.linalg.solve(t.h_prior + jp.dot(inv_h.dot(jp.T)),
                                  t.j_prior * t.as_vector() - jp.dot(dp))
        else:
            dp = np.linalg.solve(h, j.T.dot(e))
            dp = jp.dot(dp)

        return dp


class PartsAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling_mask=None):
        super(PartsAAMInterface, self). __init__(aam_algorithm)

        if sampling_mask is None:
            parts_shape = self.algorithm.appearance_model.parts_shape
            sampling_mask = np.require(np.ones((parts_shape)), dtype=np.bool)

        image_shape = self.algorithm.template.pixels.shape
        image_mask = np.tile(sampling_mask[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.image_vec_mask = np.nonzero(image_mask.flatten())[0]
        self.gradient_mask = np.nonzero(np.tile(
            image_mask[None, ...], (2, 1, 1, 1, 1, 1)))

        self.eigenvalues = self.algorithm.transform.model.eigenvalues

    def dw_dp(self):
        return np.rollaxis(self.algorithm.transform.d_dp(None), -1)

    def warp(self, image):
        return build_parts_image(
            image, self.algorithm.transform.target,
            parts_shape=self.algorithm.parts_shape,
            normalize_parts=self.algorithm.normalize_parts)

    def gradient(self, image):
        g = fast_gradient(image.pixels.reshape(
            (-1,) + self.algorithm.appearance_model.mean().shape[-2:]))
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
        return sdi.reshape((-1, sdi.shape[-1]))

    def solve(self, h, j, e, prior):
        t = self.algorithm.transform
        j_prior = self.algorithm._j_prior

        if prior:
            dp = -np.linalg.solve(np.diag(j_prior) + h,
                                  j_prior * t.as_vector() - j.T.dot(e))
        else:
            dp = np.linalg.solve(h, j.T.dot(e))

        return dp
