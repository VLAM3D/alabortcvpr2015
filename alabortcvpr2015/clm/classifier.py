from __future__ import division

import numpy as np
from numpy.fft import fft2, fftshift

from sklearn import svm
from sklearn import linear_model

from menpo.image import Image

from alabortcvpr2015.correlationfilters import (
    learn_mccf, normalizenorm_vec, generate_response)


class CF(object):
    r"""
    Correlation Filter
    """
    def __init__(self, samples, response, learn_filter=learn_mccf, l=0.01,
                 normalize=normalizenorm_vec, mask=True, boundary='constant'):

        self.normalize = normalize
        self.boundary = boundary

        n_samples = len(samples)
        k, h, w = samples[0].shape

        if mask:
            cy = np.hanning(h)
            cx = np.hanning(w)
            self.cosine_mask = cy[..., None].dot(cx[None, ...])

        X = np.empty((n_samples, k, h, w))
        for j, s in enumerate(samples):
            if self.normalize is not None:
                s = self.normalize(s)
            if mask is not None:
                s = self.cosine_mask * s
            X[j] = s

        self._filter, _, _ = learn_filter(X, response, l=l, boundary=boundary)

    @property
    def spatial_filter(self):
        return Image(self._filter[:, ::-1, ::-1])

    @property
    def frequency_filter_abs(self):
        return Image(np.abs(fftshift(fft2(self._filter[:, ::-1, ::-1]))))

    def predict(self, z):
        return generate_response(z, self._filter, normalize=self.normalize,
                                 boundary=self.boundary)


class CFEnsemble(object):
    r"""
    Ensemble of Correlation Filters
    """
    def __init__(self, cfs):

        self.normalize = cfs[0].normalize
        self.boundary = cfs[0].boundary

        k, h, w = cfs[0]._filter.shape
        n_filters = len(cfs)

        # concatenate all filters
        self._filter_ensemble = np.empty((n_filters, k, h, w))
        for j, cf in enumerate(cfs):
            self._filter_ensemble[j, ...] = cf._filter

    def predict(self, z):
        return generate_response(z, self._filter_ensemble,
                                 normalize=self.normalize,
                                 boundary=self.boundary, axis=1)


class LinearSVMLR(object):
    r"""
    Binary classifier that combines Linear Support Vector Machines and
    Logistic Regression.
    """
    def __init__(self, samples, mask, threshold=0.05, **kwarg):

        mask = mask[0]

        n_samples = len(samples)
        n_offsets, n_channels, height, width = samples[0].shape

        true_mask = mask >= threshold
        false_mask = mask < threshold

        n_true = len(mask[true_mask])
        n_false = len(mask[false_mask][::])

        pos_labels = np.ones((n_true * n_samples,))
        neg_labels = -np.ones((n_false * n_samples,))

        pos_samples = np.zeros((n_channels, n_true * n_samples))
        neg_samples = np.zeros((n_channels, n_false * n_samples))
        for j, x in enumerate(samples):
            pos_index = j*n_true
            pos_samples[:, pos_index:pos_index+n_true] = x[0, :, true_mask].T
            neg_index = j*n_false
            neg_samples[:, neg_index:neg_index+n_false] = x[0, :, false_mask].T

        X = np.vstack((pos_samples.T, neg_samples.T))
        t = np.hstack((pos_labels, neg_labels))

        self.clf1 = svm.LinearSVC(class_weight='auto', **kwarg)
        self.clf1.fit(X, t)
        t1 = self.clf1.decision_function(X)
        self.clf2 = linear_model.LogisticRegression(class_weight='auto')
        self.clf2.fit(t1[..., None], t)

    def __call__(self, x):
        t1_pred = self.clf1.decision_function(x)
        return self.clf2.predict_proba(t1_pred[..., None])[:, 1]


class MultipleLinearSVMLR(object):
    r"""
    Multiple Binary classifier that combines Linear Support Vector Machines
    and Logistic Regression.
    """
    def __init__(self, clfs):

        self.classifiers = clfs
        self.n_clfs = len(clfs)

    def __call__(self, parts_image):

        h, w = parts_image.shape[-2:]
        parts_pixels = parts_image.pixels

        parts_response = np.zeros((self.n_clfs, h, w))
        for j, clf in enumerate(self.classifiers):
            i = parts_pixels[j, ...].reshape((parts_image.shape[-3], -1))
            parts_response[j, ...] = clf(i.T).reshape((h, w))

        # normalize
        min_parts_response = np.min(parts_response,
                                    axis=(-2, -1))[..., None, None]
        parts_response -= min_parts_response
        parts_response /= np.max(parts_response,
                                 axis=(-2, -1))[..., None, None]

        return parts_response