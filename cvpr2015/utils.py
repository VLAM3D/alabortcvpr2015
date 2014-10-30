from __future__ import division
import numpy as np

from skimage.filter import gaussian_filter

from menpo.shape import TriMesh
from menpo.transform import Translation

from .image import Image, MaskedImage, BooleanImage
from .image.cython import extract_patches


def convert_from_menpo(menpo_image):

    cls = eval(type(menpo_image).__name__)

    if cls is Image:
        image = cls(np.rollaxis(menpo_image.pixels, -1), copy=True)
    elif cls is MaskedImage:
        image = cls(np.rollaxis(menpo_image.pixels, -1),
                    mask=menpo_image.mask.pixels[..., 0], copy=True)
    elif cls is BooleanImage:
        image = cls(menpo_image.pixels[..., 0], copy=True)
    else:
        raise ValueError('{} is not a Menpo image class'.format(cls))

    if menpo_image.has_landmarks:
        image.landmarks = menpo_image.landmarks

    return image


def build_reference_frame(landmarks, boundary=3, group='source', trilist=None):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks : :map:`PointCloud`
        The landmarks that will be used to build the reference frame.

    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    group : `string`, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

    trilist : ``(t, 3)`` `ndarray`, optional
        Triangle list that will be used to build the reference frame.

        If ``None``, defaults to performing Delaunay triangulation on the
        points.

    Returns
    -------
    reference_frame : :map:`Image`
        The reference frame.
    """
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)
    if trilist is not None:
        reference_frame.landmarks[group] = TriMesh(
            reference_frame.landmarks[group].lms.points, trilist=trilist)

    # TODO: revise kwarg trilist in method constrain_mask_to_landmarks,
    # perhaps the trilist should be directly obtained from the group landmarks
    reference_frame.constrain_mask_to_landmarks(group=group, trilist=trilist)

    return reference_frame


def build_patch_reference_frame(landmarks, boundary=3, group='source',
                                patch_shape=(16, 16)):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks : :map:`PointCloud`
        The landmarks that will be used to build the reference frame.

    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    group : `string`, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

    patch_shape : tuple of ints, optional
        Tuple specifying the shape of the patches.

    Returns
    -------
    patch_based_reference_frame : :map:`Image`
        The patch based reference frame.
    """
    boundary = np.max(patch_shape) + boundary
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)

    # mask reference frame
    reference_frame.build_mask_around_landmarks(patch_shape, group=group)

    return reference_frame


def _build_reference_frame(landmarks, boundary=3, group='source'):
    # translate landmarks to the origin
    minimum = landmarks.bounds(boundary=boundary)[0]
    landmarks = Translation(-minimum).apply(landmarks)

    resolution = landmarks.range(boundary=boundary)
    reference_frame = MaskedImage.blank(resolution)
    reference_frame.landmarks[group] = landmarks

    return reference_frame


def build_parts_image(image, centres, parts_shape, offsets=np.array([[0, 0]]),
                      normalize_parts=False):

    # call cython module
    parts = extract_patches(image.pixels, np.round(centres.points),
                            np.array(parts_shape), offsets)

    # build parts image
    # img.pixels: n_channels x n_centres x n_offsets x height x width
    img = Image(parts)

    if normalize_parts:
        # normalize parts if required
        img.normalize_norm_inplace()

    return img


def build_sampling_grid(patch_shape):
    patch_shape = np.array(patch_shape)
    patch_half_shape = np.require(np.floor(patch_shape / 2), dtype=int)
    start = -patch_half_shape
    end = patch_half_shape + 1
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    sampling_grid = sampling_grid.swapaxes(0, 2).swapaxes(0, 1)
    return sampling_grid


def flatten_out(list_of_lists):
    return [i for l in list_of_lists for i in l]


fsmooth = lambda x, sigma: gaussian_filter(x, sigma, mode='constant')
