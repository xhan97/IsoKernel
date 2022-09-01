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


from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# temporary solution for relative imports in case isokernel is not installed
# if isokernel is installed, no need to use the following line
from IsoKernel import IsoKernel
from IsoKernel import IsoDisKernel
from IsoKernel._isokernel_online import IsoKernelOnline

@pytest.fixture
def data():
    return load_wine(return_X_y=True)


def test_isokernel_transform(data):
    X, y = data
    ik = IsoKernel(n_estimators=200, max_samples=3)
    emb_X = ik.fit_transform(X)
    assert emb_X.shape[0] == X.shape[0]
    assert emb_X.shape[1] == 200*3


def test_isokernel_similarity(data):
    X, y = data
    ik = IsoKernel(n_estimators=200, max_samples=3)
    ik = ik.fit(X)
    print(ik.similarity(X))


def test_isodiskernel_transform(data):
    X, y = data
    idk = IsoDisKernel(n_estimators=200, max_samples=3)
    idk = idk.fit(X)
    D_i = X[:10]
    D_j = X[-10:]
    print(idk.transform(D_i, D_j))


def test_isodiskernel_similarity(data):
    X, y = data
    idk = IsoDisKernel(n_estimators=200, max_samples=3)
    idk = idk.fit(X)
    D_i = X[:10]
    D_j = X[-10:]
    print(idk.similarity(D_i, D_j))

def test_isodiskernel_online(data):

    X, y = data
    ol_ik = IsoKernelOnline(n_estimators=200, max_samples=3)
    for i, item in enumerate(X):
        ol_ik.add_observation(item)
	
    print(ol_ik.similarity(X))
