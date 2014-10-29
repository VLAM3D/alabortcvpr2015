from functools import partial
from .features import daisy, dsift


aam_daisy = partial(daisy, step=1, rings=1, radius=5, histograms=0,
                    normalization='off')
aam_dsift = partial(dsift, fast=True, window_size=5, geometry=(1, 1, 8))

aam_daisy.__name__ = 'aam_daisy'
aam_dsift.__name__ = 'aam_dsift'


