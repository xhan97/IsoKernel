
.. -*- mode: rst -*-

IsoKernel
======================================================================

IsoKernel - IsoKernel is a python library for Isolation Kernel. It includes several kernel methods using isolation mechanism.


Supported Kernel methods
-----------------------------

* Isolation Kernel (IsoKernel)
* Isolation Distribution Kernel (IsoDisKernel)

----------
Installing
----------

PyPI install, presuming you have an up to date pip.

.. .. code:: bash

..    pip install IsoKernel

For a manual install of the latest code directly from GitHub:

.. code:: bash

    pip install git+https://github.com/xhan97/IsoKernel.git


Alternatively download the package, install requirements, and manually run the installer:

.. code:: bash

    wget https://codeload.github.com/xhan97/IsoKernel/zip/refs/heads/master
    unzip IsoKernel-master.zip
    rm IsoKernel-master.zip
    cd IsoKernel-master

    pip install -r requirements.txt

    python setup.py install

------------------
How to use IsoKernel
------------------

The IsoKernel package inherits from sklearn classes, and thus drops in neatly
next to other sklearn  with an identical calling API. Similarly it
supports input in a variety of formats: an array (or pandas dataframe) of shape ``(num_samples x num_features)``.

.. code:: python

    from IsoKernel import IsoKernel
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(1000)

    ik = IsoKernel(n_estimators=200, max_samples=16)
    ik = ik.fit(data)
    # get Isolation Kernel feature vector
    ik.transform(data)
    # get Isolation Kernel similarity
    ik.similarity(data)

------------------
How to use IsoDisKernel
------------------
Isolation Distributional Kernel is a new way to measure the similarity between two distributions.
It addresses two key issues of kernel mean embedding, where the kernel employed has:
    (i) a feature map with intractable dimensionality which leads to high computational cost;
    (ii) data independency which leads to poor accuracy.
.. code:: python

    from IsoKernel import IsoDisKernel
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(1000)

    idk = IsoDisKernel(n_estimators=200, max_samples=16)
    idk = idk.fit(data)
    D_i = data[:10]
    D_j = data[-10:]
    # get similarity of two distributions
    sim = idk.similarity(D_i, D_j)
    # get ik feature
    ikm_D_i, ikm_D_j = idk.ik_feature
    # get kernel mean embedding
    kme_D_i, kme_D_j = idk.kme

-----------------
Running the Tests
-----------------

The package tests can be run after installation using the command:

.. code:: bash

    pip install pytest 

or, if ``pytest`` is installed:

.. code:: bash

    pytest  IsoKernel/tests

If one or more of the tests fail, please report a bug at https://github.com/xhan97/IsoKernel/issues

--------------
Python Version
--------------

Python 3  is recommend  the better option if it is available to you.

------
Citing
------

If you have used this codebase in a scientific publication and wish to
cite it, please use the following publication (Bibtex format):

.. code:: bibtex

   @inproceedings{10.1145/3219819.3219990,
        author = {Ting, Kai Ming and Zhu, Yue and Zhou, Zhi-Hua},
        title = {Isolation Kernel and Its Effect on SVM},
        year = {2018},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        booktitle = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
        pages = {2329–2337},
        numpages = {9},
        location = {London, United Kingdom},
        series = {KDD '18}
        }

.. code:: bibtex

    @inproceedings{ting2020Isolation,
        author = {Ting, Kai Ming and Xu, Bi-Cun and Washio, Takashi and Zhou, Zhi-Hua},
        title = {Isolation Distributional Kernel: A New Tool for Kernel Based Anomaly Detection},
        year = {2020},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        doi = {10.1145/3394486.3403062},
        pages = {198-206},
        numpages = {9},
        series = {KDD '20}
    }

.. code:: bibtex

    @inproceedings{HZTZL22Streaming,
     author = {Han, Xin and Zhu, Ye and Ting, Kai Ming and Zhan, De-Chuan and Li, Gang},
     title = {Streaming Hierarchical Clustering Based on Point-Set Kernel},
     year = {2022},
     isbn = {9781450393850},
     publisher = {Association for Computing Machinery},
     address = {New York, NY, USA},
     url = {https://doi.org/10.1145/3534678.3539323},
     doi = {10.1145/3534678.3539323},
     pages = {525–533},
     numpages = {9},
     keywords = {streaming data, hierarchical clustering, isolation kernel},
     location = {Washington DC, USA},
     series = {KDD '22}
}

License
-------

Apache license
