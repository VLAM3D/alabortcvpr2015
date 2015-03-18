from __future__ import division
import cPickle
from skimage.filter import gaussian_filter

from menpo.image import Image
from menpo.visualize import print_dynamic, progress_bar_str

from menpofit.aam.base import build_reference_frame


def pickle_load(path):
    with open(str(path), 'rb') as f:
        return cPickle.load(f)


def pickle_dump(obj, path):
    with open(str(path), 'wb') as f:
        cPickle.dump(obj, f, protocol=2)


def compute_features(images, features, verbose=None, level_str=""):
    feature_images = []
    for c, i in enumerate(images):
        if verbose:
            print_dynamic(
                '{}Computing feature space: {}'.format(
                    level_str, progress_bar_str((c + 1.) / len(images),
                                                show_bar=False)))
        i = features(i)
        feature_images.append(i)

    return feature_images


def scale_images(images, scale, verbose=None, level_str=""):
    scaled_images = []
    for c, i in enumerate(images):
        if verbose:
            print_dynamic(
                '{}Scaling features: {}'.format(
                    level_str, progress_bar_str((c + 1.) / len(images),
                                                show_bar=False)))
        scaled_images.append(i.rescale(scale))
    return scaled_images


def warp_images(images, shapes, ref_shape, transform, verbose=None,
                level_str=""):
    # compute transforms
    ref_frame = build_reference_frame(ref_shape)
    # warp images to reference frame
    warped_images = []
    for c, (i, s) in enumerate(zip(images, shapes)):
        if verbose:
            print_dynamic('{}Warping images - {}'.format(
                level_str,
                progress_bar_str(float(c + 1) / len(images),
                                 show_bar=False)))
        # compute transforms
        t = transform(ref_frame.landmarks['source'].lms, s)
        # warp images
        warped_i = i.warp_to_mask(ref_frame.mask, t)
        # attach reference frame landmarks to images
        warped_i.landmarks['source'] = ref_frame.landmarks['source']
        warped_images.append(warped_i)
    return warped_images


def extract_patches(images, shapes, parts_shape, verbose=None,
                    level_str=""):
    # extract parts
    parts_images = []
    for c, (i, s) in enumerate(zip(images, shapes)):
        if verbose:
            print_dynamic('{}Warping images - {}'.format(
                level_str,
                progress_bar_str(float(c + 1) / len(images),
                                 show_bar=False)))
        parts_image = Image(i.extract_patches(
            s, patch_size=parts_shape, as_single_array=True))
        parts_images.append(parts_image)
    return parts_images


def flatten_out(list_of_lists):
    return [i for l in list_of_lists for i in l]


fsmooth = lambda x, sigma: gaussian_filter(x, sigma, mode='constant')
