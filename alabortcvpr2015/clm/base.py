class CLM(object):

    def __init__(self, shape_models, classifiers, reference_shape, response,
                 features, scales, scale_shapes, scale_features):

        self.shape_models = shape_models
        self.classifiers = classifiers
        self.response = response
        self.features = features
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    @property
    def n_levels(self):
        """
        The number of scale levels of the CLM.

        :type: `int`
        """
        return len(self.scales)
