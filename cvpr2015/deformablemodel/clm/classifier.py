import numpy as np
from scipy.signal import cosine
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from sklearn import svm
from sklearn import linear_model


class MCF(object):
    r"""
    Multi-channel Correlation Filter
    """
    def __init__(self, X, Y, l=0, cosine_mask=False):

        if X[0].shape[1:] != (len(Y),) + Y[0].shape:
            raise ValueError('')

        n_channels, n_offsets, height, width = X[0].shape

        self._cosine_mask = 1
        if cosine_mask:
            self._cosine_mask = np.sum(np.meshgrid(cosine(height),
                                                   cosine(width)), axis=0)

        X_hat = self._compute_fft2s(X)
        Y_hat = self._compute_fft2s(Y)

        # self.f = np.zeros((n_channels, height, width), dtype=np.complex64)
        # for i in xrange(n_channels):
        #     for j in xrange(height):
        #         for k in xrange(width):
        #             H_hat = 0
        #             J_hat = 0
        #             for x_hat in X_hat:
        #                 for o, y_hat in enumerate(Y_hat):
        #                     H_hat += (np.conj(x_hat[i, o, j, k]) *
        #                               x_hat[i, o, j, k])
        #                     J_hat += (np.conj(x_hat[i, o, j, k]) *
        #                               y_hat[j, k])
        #             H_hat += l
        #             self.f[i, j, k] = J_hat / H_hat

        self.f = np.zeros((n_channels, height, width), dtype=np.complex64)
        sxx_hat = 0
        syx_hat = 0
        for i in xrange(n_channels):
            for x_hat in X_hat:
                for o, y_hat in enumerate(Y_hat):
                    sxx_hat = sxx_hat + x_hat[i, o] * np.conj(x_hat[i, o])
                    syx_hat = syx_hat + y_hat * np.conj(x_hat[i, o])
            sxx_hat += l * np.eye(sxx_hat.shape[1])
            self.f[i] = syx_hat / (sxx_hat + l * np.eye(sxx_hat.shape[1]))

    def _compute_fft2s(self, X):
        X_hat = []
        for x in X:
            x_hat = np.require(fft2(self._cosine_mask * x),
                               dtype=np.complex64)
            X_hat.append(x_hat)
        return X_hat

    def __call__(self, x):
        return np.real(
            ifft2(self.f * fft2(self._cosine_mask * x)))


class MultipleMCF(object):
    r"""
    Multiple of Multi-channel Correlation Filter
    """
    def __init__(self, mcfs):

        self._cosine_mask = mcfs[0]._cosine_mask
        # concatenate all filters
        n_channels, height, width = mcfs[0].f.shape
        n_landmarks = len(mcfs)
        self.F = np.zeros((n_channels, n_landmarks, height, width),
                           dtype=np.complex64)
        for j, clf in enumerate(mcfs):
            self.F[:, j, ...] = clf.f

    def __call__(self, parts_image):

        return np.sum(np.real(ifft2(
            self.F * fft2(self._cosine_mask *
                           parts_image.pixels[:, :, 0, ...]))), axis=0)


class LinearSVMLR(object):
    r"""
    Binary classifier that combines Linear Support Vector Machines and
    Logistic Regression.
    """
    def __init__(self, samples, mask, threshold=0.05):

        n_samples = len(samples)
        n_channels, n_offsets, height, width = samples[0].shape

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
            pos_samples[:, pos_index:pos_index+n_true] = x[:, 0, true_mask]
            neg_index = j*n_false
            neg_samples[:, neg_index:neg_index+n_false] = x[:, 0, false_mask]

        X = np.vstack((pos_samples.T, neg_samples.T))
        t = np.hstack((pos_labels, neg_labels))

        self.clf1 = svm.LinearSVC(class_weight='auto')
        self.clf1.fit(X, t)
        t1 = self.clf1.decision_function(X)
        self.clf2 = linear_model.LogisticRegression(class_weight='auto')
        self.clf2.fit(t1[..., None], t)

    def __call__(self, x):
        t1_pred = self.clf1.decision_function(x)
        return self.clf2.predict_proba(t1_pred[..., None])[:, 1]
