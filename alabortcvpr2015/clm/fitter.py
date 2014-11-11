from __future__ import division

from alabortcvpr2015.fitter import Fitter
from alabortcvpr2015.pdm import OrthoPDM

from .algorithm import RLMS


# Concrete Implementations of AAM Fitters -------------------------------------

class CLMFitter(Fitter):

    def __init__(self, clm, algorithm_cls=RLMS, n_shape=None,
                 scale=10, **kwargs):

        super(CLMFitter, self).__init__()

        self.dm = clm
        self._algorithms = []
        self._check_n_shape(n_shape)
        scale = self._check_scale(scale)

        for j, (clf, sm, s) in enumerate(zip(self.dm.classifiers,
                                          self.dm.shape_models,
                                          scale)):

            pdm = OrthoPDM(sm, sigma2=sm.noise_variance())

            algorithm = algorithm_cls(clf, self.dm.parts_shape,
                                      self.dm.normalize_parts,
                                      self.covariance, pdm, scale=s,
                                      **kwargs)

            self._algorithms.append(algorithm)

    @property
    def covariance(self):
        return self.dm.covariance

    def _check_scale(self, s):
        scale = []
        if type(s) is int or type(s) is float:
            for _ in xrange(self.dm.n_levels):
                scale.append(s)
        elif len(s) == 1 and self.dm.n_levels > 1:
            for _ in xrange(self.dm.n_levels):
                scale.append(s)
        elif len(s) == self.dm.n_levels:
            scale = s
        else:
            raise ValueError('scale can be an integer or a float or None'
                             'or a list containing 1 or {} of '
                             'those'.format(self.dm.n_levels))
        return scale

