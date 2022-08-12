
.. -*- mode: rst -*-

|PyPI|_  |Downloads|_  |Codecov|_ |CircleCI|_ 


.. |PyPI| image:: https://badge.fury.io/py/inne.svg
.. _PyPI: https://badge.fury.io/py/inne

.. |Codecov| image:: https://codecov.io/gh/xhan97/inne/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/xhan97/inne

.. |CircleCI| image:: https://circleci.com/gh/xhan97/inne.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/xhan97/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/inne/badge/?version=latest
.. _ReadTheDocs: https://inne.readthedocs.io/en/latest/?badge=latest

.. |Downloads| image:: https://pepy.tech/badge/inne
.. _Downloads: https://pepy.tech/project/inne


isoKernel
======================================================================

isoKernel - Isolation Kernel.

----------
Installing
----------

PyPI install, presuming you have an up to date pip.

.. code:: bash

   pip install isoKernel

For a manual install of the latest code directly from GitHub:

.. code:: bash

    pip install git+https://github.com/xhan97/isoKernel.git


Alternatively download the package, install requirements, and manually run the installer:

.. code:: bash

    wget https://codeload.github.com/xhan97/isoKernel/zip/refs/heads/master
    unzip isoKernel-master.zip
    rm isoKernel-master.zip
    cd isoKernel-master

    pip install -r requirements.txt

    python setup.py install

------------------
How to use isoKernel
------------------

The isoKernel package inherits from sklearn classes, and thus drops in neatly
next to other sklearn  with an identical calling API. Similarly it
supports input in a variety of formats: an array (or pandas dataframe) of shape ``(num_samples x num_features)``.

.. code:: python

    from inne import IsolationNNE
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(1000)

    clf = IsolationNNE(n_estimators=200, max_samples=16)
    clf.fit(data)
    anomaly_labels = clf.predict(data)

-----------------
Running the Tests
-----------------

The package tests can be run after installation using the command:

.. code:: bash

    pip install pytest 

or, if ``pytest`` is installed:

.. code:: bash

    pytest  inne/tests

If one or more of the tests fail, please report a bug at https://github.com/xhan97/inne/issues

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

License
-------

Apache license
