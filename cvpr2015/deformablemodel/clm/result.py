from __future__ import division

from cvpr2015.deformablemodel.result import AlgorithmResult, FitterResult


# Concrete Implementations of CLM Algorithm Results #--------------------------

class CLMAlgorithmResult(AlgorithmResult):

    def __init__(self, image, fitter, shape_parameters,
                 gt_shape=None):
        super(CLMAlgorithmResult, self).__init__()
        self.image = image
        self.fitter = fitter
        self.shape_parameters = shape_parameters
        self._gt_shape = gt_shape


# Concrete Implementations of AAM Fitter Results # ----------------------------

class CLMFitterResult(FitterResult):

    @property
    def costs(self):
        r"""
        Returns a list containing the cost at each fitting iteration.

        :type: `list` of `float`
        """
        raise ValueError('costs not implemented yet.')

    @property
    def final_cost(self):
        r"""
        Returns the final fitting cost.

        :type: `float`
        """
        return self.algorithm_results[-1].final_cost

    @property
    def initial_cost(self):
        r"""
        Returns the initial fitting cost.

        :type: `float`
        """
        return self.algorithm_results[0].initial_cost
