from __future__ import division
import abc
from copy import deepcopy

from menpo.model import PCAModel
from menpo.visualize import print_dynamic

from menpofit.transform import DifferentiablePiecewiseAffine
from menpofit.builder import (
    normalization_wrt_reference_shape, build_shape_model)
from menpofit.aam.base import build_reference_frame

from alabortcvpr2015.utils import (
    compute_features, scale_images, warp_images, extract_patches)


class AAMBuilder(object):

    def build(self, images, group=None, label=None, verbose=False):
        # normalize images and compute reference shape
        reference_shape, images = normalization_wrt_reference_shape(
            images, group, label, self.diagonal, verbose=verbose)

        # build models at each scale
        if verbose:
            print_dynamic('- Building models\n')
        shape_models = []
        appearance_models = []
        # for each pyramid level (high --> low)
        for j, s in enumerate(self.scales):
            if verbose:
                if len(self.scales) > 1:
                    level_str = '  - Level {}: '.format(j)
                else:
                    level_str = '  - '

            # obtain image representation
            if j == 0:
                # compute features at highest level
                feature_images = compute_features(images, self.features,
                                                  verbose=verbose,
                                                  level_str=level_str)
                level_images = feature_images
            elif self.scale_features:
                # scale features at other levels
                level_images = scale_images(feature_images, s, verbose,
                                            level_str)
            else:
                # scale images and compute features at other levels
                scaled_images = scale_images(images, s, verbose=verbose,
                                             level_str=level_str)
                level_images = compute_features(scaled_images, self.features,
                                                verbose=verbose,
                                                level_str=level_str)

            # extract potentially rescaled shapes ath highest level
            level_shapes = [i.landmarks[group][label]
                            for i in level_images]

            # obtain shape representation
            if j == 0 or self.scale_shapes:
                # obtain shape model
                if verbose:
                    print_dynamic('{}Building shape model'.format(level_str))
                shape_model = build_shape_model(
                    level_shapes, self.max_shape_components)
                # add shape model to the list
                shape_models.append(shape_model)
            else:
                # copy precious shape model and add it to the list
                shape_models.append(deepcopy(shape_model))

            # obtain warped images
            warped_images = self._warp_images(level_images, level_shapes,
                                              shape_model.mean(),
                                              verbose=verbose,
                                              level_str=level_str)

            # obtain appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(level_str))
            appearance_model = PCAModel(warped_images)
            # trim appearance model if required
            if self.max_appearance_components is not None:
                appearance_model.trim_components(
                    self.max_appearance_components)
            # add appearance model to the list
            appearance_models.append(appearance_model)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        appearance_models.reverse()
        self.scales.reverse()

        aam = self._build_aam(shape_models, appearance_models, reference_shape)

        return aam

    @abc.abstractmethod
    def _build_aam(self, shape_models, appearance_models, reference_shape):
        pass


class GlobalAAMBuilder(AAMBuilder):

    def __init__(self, features=None, transform=DifferentiablePiecewiseAffine,
                 trilist=None, diagonal=None, sigma=None, scales=(1, .5),
                 scale_shapes=True, scale_features=True,
                 max_shape_components=None, max_appearance_components=None,
                 boundary=3):

        self.features = features
        self.transform = transform
        self.trilist = trilist
        self.diagonal = diagonal
        self.sigma = sigma
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.boundary = boundary

    def _build_reference_frame(self, mean_shape):
        return build_reference_frame(mean_shape, boundary=self.boundary,
                                     trilist=self.trilist)

    def _warp_images(self, images, shapes, ref_shape, verbose, level_str):
        return warp_images(images, shapes, ref_shape, self.transform,
                           verbose=verbose, level_str=level_str)

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return GlobalAAM(shape_models, appearance_models, reference_shape,
                         self.transform, self.features, self.sigma,
                         self.scales, self.scale_shapes, self.scale_features)


class PartsAAMBuilder(AAMBuilder):

    def __init__(self, parts_shape=(17, 17), features=None,
                 normalize_parts=False, diagonal=None, sigma=None,
                 scales=(1, .5), scale_shapes=False, scale_features=True,
                 max_shape_components=None, max_appearance_components=None):

        self.parts_shape = parts_shape
        self.features = features
        self.normalize_parts = normalize_parts
        self.diagonal = diagonal
        self.sigma = sigma
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components

    def _warp_images(self, images, shapes, _, verbose, level_str):
        return extract_patches(images, shapes, self.parts_shape,
                               verbose=verbose, level_str=level_str)

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return PartsAAM(shape_models, appearance_models, reference_shape,
                        self.parts_shape, self.features,
                        self.normalize_parts, self.sigma, self.scales,
                        self.scale_shapes, self.scale_features)


from .base import GlobalAAM, PartsAAM


