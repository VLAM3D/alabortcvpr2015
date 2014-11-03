from __future__ import division

from serializablecallable import SerializableCallable


class CLM(object):

    def __init__(self, shape_models, classifiers, reference_shape,
                 parts_shape, features, normalize_parts, sigma, scales,
                 scale_shapes, scale_features):

        self.shape_models = shape_models
        self.classifiers = classifiers
        self.parts_shape = parts_shape
        self.features = features
        self.normalize_parts = normalize_parts
        self.sigma = sigma
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    def __getstate__(self):
        import menpo.feature as menpo_feature
        d = self.__dict__.copy()

        features = d.pop('features')
        if self.pyramid_on_features:
            # features is a single callable
            d['features'] = SerializableCallable(features, [menpo_feature])
        else:
            # features is a list of callables
            d['features'] = [SerializableCallable(f, [menpo_feature])
                             for f in features]
        return d

    def __setstate__(self, state):
        try:
            state['features'] = state['features'].callable
        except AttributeError:
            state['features'] = [f.callable for f in state['features']]
        self.__dict__.update(state)

    @property
    def n_levels(self):
        """
        The number of scale levels of the AAM.

        :type: `int`
        """
        return len(self.scales)
