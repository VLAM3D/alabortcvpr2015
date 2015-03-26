from __future__ import division
from copy import deepcopy

from menpo.feature import no_op
from menpo.visualize import print_dynamic, progress_bar_str

from menpofit.builder import (
    normalization_wrt_reference_shape, build_shape_model)

from alabortcvpr2015.correlationfilter import generate_gaussian_response
from alabortcvpr2015.utils import (
    compute_features, scale_images, extract_patches)

from .classifier import CF, CFEnsemble, LinearSVMLR, MultipleLinearSVMLR


class CLMBuilder(object):

    def __init__(self, classifier=CF, context_shape=(51, 51),
                 filter_shape=(17, 17), features=no_op,
                 covariance=3, diagonal=None, scales=(1, .5),
                 scale_shapes=True, scale_features=True,
                 max_shape_components=None):
        self.classifier = classifier
        self.context_shape = context_shape
        self.filter_shape = filter_shape
        self.covariance = covariance
        self.features = features
        self.diagonal = diagonal
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components

    def build(self, images, group=None, label=None, verbose=False, **kwargs):
        # normalize images and compute reference shape
        reference_shape, images = normalization_wrt_reference_shape(
            images, group, label, self.diagonal, verbose=verbose)

        # generate desired responses
        response = generate_gaussian_response(self.filter_shape,
                                              self.covariance)

        # build models at each scale
        if verbose:
            print_dynamic('- Building models\n')
        shape_models = []
        classifiers = []
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

            # obtain parts images
            parts_images = extract_patches(level_images, level_shapes,
                                           self.context_shape, level_str,
                                           verbose)

            # build classifiers
            n_landmarks = level_shapes[0].n_points
            level_classifiers = []
            for l in range(n_landmarks):
                if verbose:
                    print_dynamic('{}Building classifiers - {}'.format(
                        level_str,
                        progress_bar_str((l + 1.) / n_landmarks,
                                         show_bar=False)))

                X = [i.pixels[l][0] for i in parts_images]

                clf = self.classifier(X, response, **kwargs)
                level_classifiers.append(clf)

            # build Multiple classifier
            if self.classifier is CF:
                clf_ensemble = CFEnsemble(level_classifiers)
            elif self.classifier is LinearSVMLR:
                clf_ensemble = MultipleLinearSVMLR(level_classifiers)

            # add appearance model to the list
            classifiers.append(clf_ensemble)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape models and classifiers so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        classifiers.reverse()
        self.scales.reverse()

        clm = CLM(shape_models, classifiers, reference_shape, response,
                  self.features, self.scales, self.scale_shapes,
                  self.scale_features)

        return clm, level_classifiers


from .base import CLM
