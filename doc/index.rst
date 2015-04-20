pyasdf
======

``pyasdf`` is a Python API to read and write seismological ASDF files.


.. admonition:: Contact Us

    If you encounter a bug or another error, please open a new
    `issue <https://github.com/SeismicData/pyasdf/issues>`_ on Github.


Installation
------------

``pyasdf`` supports Python version 2.7 and 3.4 and it depends on the following
Python modules: ``NumPy``, ``ObsPy``, ``h5py``, ``mpi4py``, ``colorama``, and
``pytest``. Keep in mind that ``h5py`` must be compiled with parallel I/O
support and that it is linked against the same MPI as ``mpi4py`` which of
course should be the same that is used by your computer.

If you know what you are doing, install it any way you see fit. Otherwise do
yourself a favor and download the
`Anaconda Python distribution <https://store.continuum.io/cshop/anaconda/>`_
for Python 2.7 or 3.4. After downloading update it, and install the
dependencies.

.. code-block:: bash

    $ conda update conda
    $ conda install -c obspy obspy colorama pytest pip


Installing the parallel libraries is a bit more difficult unfortunately. For
all the following make sure that the MPI package of your local
supercomputer/cluster is loaded. The ``mpi4py`` potentially shipping with
Anaconda might not work on your cluster so uninstall it and reinstall with
``pip`` at which point it should link against your cluster's MPI
implementation.

.. code-block:: bash

    $ conda uninstall mpi4py
    $ pip install mpi4py

Then install parallel ``h5py`` according to
`these instructions <http://docs.h5py.org/en/latest/mpi.html>`_.




Testing
^^^^^^^

To assert that your installation is working properly, execute

.. code-block:: bash

    $ python -m pyasdf.tests

and make sure all tests pass. Otherwise please contact the developers.

Build the Documentation
^^^^^^^^^^^^^^^^^^^^^^^

The documentation requires ``sphinx`` and the Bootstrap theme. Install both
with

.. code-block:: bash

    $ pip install sphinx sphinx-bootstrap-theme

Build the doc with

.. code-block:: bash

    $ cd doc
    $ make html

Finally open the ``doc/_build/html/index.html`` file with the browser of your
choice.


Tutorial
--------

Learning Python and ObsPy
^^^^^^^^^^^^^^^^^^^^^^^^^

``pyasdf`` is written in `Python <http://www.python.org>`_ and utilizes the
data structures of `ObsPy <http://obspy.org>`_ to allow the construction of
modern and efficient workflows. Python is an easy to learn and powerful
interactive programming language with an exhaustive scientific ecosystem. The
following resources are useful if you are starting out with Python and ObsPy:

* `Good, general Python tutorial <http://learnpythonthehardway.org/book/>`_
* `IPython Notebook in Nature <http://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261>`_
* `Introduction to the scientific Python ecosystem <https://scipy-lectures.github.io/>`_
* `The ObsPy Documentation <http://docs.obspy.org/master>`_
* `The ObsPy Tutorial <http://docs.obspy.org/master/tutorial/index.html>`_

Using pyasdf
^^^^^^^^^^^^

to be written ...


Acknowledgements
----------------

We gratefully acknowledge support from the EU-FP7 725 690 **VERCE** project
(number 283543, `www.verce.eu <http://www.verce.eu>`_).

Detailed Documentation
----------------------

to be written ...
