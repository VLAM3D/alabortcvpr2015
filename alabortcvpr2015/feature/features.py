import itertools
import numpy as np

from skimage.feature import daisy as skimage_daisy
from cyvlfeat.sift.dsift import dsift as cyvlfeat_dsift

from .base import ndfeature, winitfeature
from .cython import gradient as cython_gradient

scipy_gaussian_filter = None  # expensive


@ndfeature
def gradient(pixels):
    return cython_gradient(pixels)


@ndfeature
def gaussian_filter(pixels, sigma):
    global scipy_gaussian_filter
    if scipy_gaussian_filter is None:
        from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    output = np.empty(pixels.shape)
    for dim in range(pixels.shape[0]):
        scipy_gaussian_filter(pixels[dim, ...], sigma, output=output[dim, ...])
    return output


@ndfeature
def daisy(pixels, step=4, radius=15, rings=3, histograms=8, orientations=8,
          normalization='l1', sigmas=None, ring_radii=None):
    pixels = skimage_daisy(pixels[0, ...], step=step, radius=radius,
                           rings=rings, histograms=histograms,
                           orientations=orientations,
                           normalization=normalization, sigmas=sigmas,
                           ring_radii=ring_radii)

    return np.rollaxis(pixels, -1)


@winitfeature
def dsift(pixels, step=1, size=3, bounds=None, window_size=2, norm=False,
          fast=False, float_descriptors=False, geometry=(4, 4, 8)):
    centers, output = cyvlfeat_dsift(np.rot90(pixels[0, ..., ::-1]),
                                     step=step, size=size, bounds=bounds,
                                     window_size=window_size, norm=norm,
                                     fast=fast,
                                     float_descriptors=float_descriptors,
                                     geometry=geometry)
    shape = pixels.shape[1:] - 2 * centers[:2, 0]
    return (np.require(output.reshape((-1, shape[0], shape[1])),
                       dtype=np.double),
            np.require(centers[:2, ...].T[..., ::-1].reshape(
                (shape[0], shape[1], 2)), dtype=np.int))



@ndfeature
def no_op(image_data):
    r"""
    A no operation feature - does nothing but return a copy of the pixels
    passed in.
    """
    return image_data.copy()
