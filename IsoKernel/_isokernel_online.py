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


import random

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps


class IsoKernelOnline(TransformerMixin, BaseEstimator):
    """  Build Isolation Kernel feature vector representations via the feature map
    for a given dataset.

    Isolation kernel is a data dependent kernel measure that is
    adaptive to local data distribution and has more flexibility in capturing
    the characteristics of the local data distribution. It has been shown promising
    performance on density and distance-based classification and clustering problems.

    This version uses Voronoi diagrams to split the data space and calculate Isolation
    kernel Similarity. Based on this implementation, the feature
    in the Isolation kernel space is the index of the cell in Voronoi diagrams. Each
    point is represented as a binary vector such that only the cell the point falling
    into is 1.

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
    .. [1] Qin, X., Ting, K.M., Zhu, Y. and Lee, V.C.
    "Nearest-neighbour-induced isolation similarity and its impact on density-based clustering".
    In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33, 2019, July, pp. 4755-4762

    Examples
    --------
    >>> from IsoKernel import IsoKernel
    >>> import numpy as np
    >>> X = [[0.4,0.3], [0.3,0.8], [0.5, 0.4], [0.5, 0.1]]
    >>> ik = IsoKernel.fit(X)
    >>> ik.transform(X)
    >>> ik.similarity(X)
    """

    def __init__(self, n_estimators=200, max_samples="auto", random_state=None) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_instance = 0
        self.center_data = None

    def add_observation(self, x):

        if not self.center_data:
            self.center_data = [[]for i in range(self.n_estimators)]
        for i in range(self.n_estimators):
            self.center_data[i] = self.update_center(
                self.center_data[i], self.max_samples, x)
        self.n_instance += 1

        return self

    def update_center(self, center_data, m, x):
        if self.n_instance <= m-1:
            center_data.append(x)
        else:
            i = random.randint(1, self.n_instance)
            if i <= m:
                j = random.randint(1, m) - 1
                center_data[j] = x
        return center_data

    def similarity(self, X):
        """ Compute the isolation kernel similarity matrix of X.
        Parameters
        ----------
        X: array-like of shape (n_instances, n_features)
            The input instances.
        Returns
        -------
        The simalarity matrix are organised as a n_instances * n_instances matrix.
        """

        embed_X = self.transform(X)
        return np.inner(embed_X, embed_X) / self.n_estimators

    def transform(self, X):
        """ Compute the isolation kernel feature of X.
        Parameters
        ----------
        X: array-like of shape (n_instances, n_features)
            The input instances.
        Returns
        -------
        The finite binary features based on the kernel feature map.
        The features are organised as a n_instances by psi*t matrix.
        """

        # check_is_fitted(self)
        X = check_array(X)
        for i in range(self.n_estimators):
            x_center_dist = euclidean_distances(X, self.center_data[i])
            nearest_center_index = np.argmin(x_center_dist, axis=1)
            ik_value = np.eye(self.max_samples, dtype=int)[
                nearest_center_index]
            if i == 0:
                embedding = ik_value
            else:
                embedding = np.append(embedding, ik_value, axis=1)
        return embedding
    
    
