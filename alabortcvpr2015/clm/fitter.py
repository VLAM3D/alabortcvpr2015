from __future__ import division

from alabortcvpr2015.fitter import Fitter
from alabortcvpr2015.pdm import OrthoPDM

from .algorithm import RLMS


# Concrete Implementations of AAM Fitters -------------------------------------

class CLMFitter(Fitter):

    def __init__(self, clm, algorithm_cls=RLMS, n_shape=None, **kwargs):

        super(CLMFitter, self).__init__()

        self.dm = clm
        self._algorithms = []
        self._check_n_shape(n_shape)

        for j, (clf, sm) in enumerate(zip(self.dm.classifiers,
                                          self.dm.shape_models)):

            pdm = OrthoPDM(sm, sigma2=sm.noise_variance())

            algorithm = algorithm_cls(clf, self.dm.parts_shape,
                                      self.dm.normalize_parts,
                                      self.covariance, pdm, **kwargs)

            self._algorithms.append(algorithm)

    @property
    def covariance(self):
        return self.dm.covariance
