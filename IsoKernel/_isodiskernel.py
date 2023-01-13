# Copyright 2022 Xin Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.base import BaseEstimator, TransformerMixin
from IsoKernel._isokernel import IsoKernel
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import numpy as np
import math


class IsoDisKernel(BaseEstimator):
    """Isolation Distributional Kernel is a new way to measure the similarity between two distributions.

    It addresses two key issues of kernel mean embedding, where the kernel employed has: 
    (i) a feature map with intractable dimensionality which leads to high computational cost; 
    and (ii) data independency which leads to poor detection accuracy in anomaly detection.

    Parameters
    ----------
    n_estimators : int, default=200
        The number of base estimators in the ensemble.

    max_samples : int, default="auto"
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples` * X.shape[0]` samples.
            - If "auto", then `max_samples=min(8, n_samples)`.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    References
    ----------
    .. [1] Kai Ming Ting, Bi-Cun Xu, Takashi Washio, and Zhi-Hua Zhou. 2020. 
    "Isolation Distributional Kernel: A New Tool for Kernel based Anomaly Detection".
    In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20).
    Association for Computing Machinery, New York, NY, USA, 198-206.

    Examples
    --------
    >>> from IsoKernel import IsoDisKernel
    >>> import numpy as np
    >>> X = [[0.4,0.3], [0.3,0.8], [0.5,0.4], [0.5,0.1]]
    >>> idk = IsoDisKernel.fit(X)
    >>> D_i = [[0.4,0.3], [0.3,0.8]]
    >>> D_j = [[0.5, 0.4], [0.5, 0.1]]
    >>> idk.similarity(D_j, D_j)
    """

    def __init__(self, n_estimators=200, max_samples="auto", random_state=None) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X):
        """ Fit the model on data X.
        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        Returns
        -------
        self : object
        """
        X = check_array(X)
        iso_kernel = IsoKernel(
            self.n_estimators, self.max_samples, self.random_state)
        self.iso_kernel = iso_kernel.fit(X)
        self.is_fitted_ = True
        return self

    def kernel_mean_embedding(self, X):
        return np.mean(X, axis=0)

    def kme_similarity(self, kme_D_i, kme_D_j,  is_normalize=False):
        if is_normalize:
            return np.dot(kme_D_i, kme_D_j)/(math.sqrt(np.dot(kme_D_i, kme_D_i)) * math.sqrt(np.dot(kme_D_j, kme_D_j)))
        return np.dot(kme_D_i, kme_D_j) / self.n_estimators

    def similarity(self, D_i, D_j, is_normalize=False):
        """ Compute the isolation distribution kernel of D_i and D_j.
        Parameters
        ----------
        D_i: array-like of shape (n_instances, n_features)
            The input instances.
        D_j: array-like of shape (n_instances, n_features)
            The input instances.
        is_normalize: whether return the normalized similarity matrix ranged of [0,1]. Default: False
        Returns
        -------
        The Isolation distribution similarity of given two dataset.
        """
        emb_D_i, emb_D_j = self.transform(D_i, D_j)
        kme_D_i, kme_D_j = self.kernel_mean_embedding(
            emb_D_i), self.kernel_mean_embedding(emb_D_j)
        return self.kme_similarity(kme_D_i, kme_D_j, is_normalize=is_normalize)

    def transform(self, D_i, D_j):
        """ Compute the isolation kernel feature of D_i and D_j.
        Parameters
        ----------
        D_i: array-like of shape (n_instances, n_features)
            The input instances.
        D_j: array-like of shape (n_instances, n_features)
            The input instances.
        Returns
        -------
        The finite binary features based on the kernel feature map.
        The features are organised as a n_instances by psi*t matrix.
        """
        check_is_fitted(self)
        D_i = check_array(D_i)
        D_j = check_array(D_j)
        return self.iso_kernel.transform(D_i), self.iso_kernel.transform(D_j)
