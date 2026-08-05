Installation
============

pyasdf and Dependencies
-----------------------

``pyasdf`` supports Python versions >= 3.7 and it depends on the
following Python modules: ``NumPy``, ``ObsPy``, ``h5py``, ``colorama``,
``pytest``, ``prov``, ``dill``, and optionally ``mpi4py``. You can
install ``pyasdf`` with or without parallel I/O support; the later requires
``mpi4py`` and parallel versions of ``hdf5`` and ``h5py``.

``pyasdf`` itself is available on pypi and also as a conda package in the
``conda-forge`` channel (non-parallel only).

If you know what you are doing, install it any way you see fit. Otherwise do
yourself a favor and download the
`Anaconda Python distribution <https://store.continuum.io/cshop/anaconda/>`_
for your chosen Python version.


.. note:: **When to choose the parallel I/O version?**

    Please note that in most cases it is not worth it to install the parallel
    I/O version. For one most machines (aside from actual HPC machines)
    don't even have the necessary hardware to do actually parallel I/O. Also
    seismological waveform data is usually not that big in volume so a single
    reading/writing thread might be sufficient. Furthermore modern SSDs can
    write at very high speeds.

    But if your application does indeed benefit from parallel I/O follow the
    instructions below.


Non-parallel ``pyasdf``
^^^^^^^^^^^^^^^^^^^^^^^

This is very easy - just execute this one line and it will install all the
dependencies including ``pyasdf`` (assuming you installed the
`Anaconda Python distribution <https://store.continuum.io/cshop/anaconda/>`_
as recommended above).

.. code-block:: bash

    $ conda install -c conda-forge pyasdf



``pyasdf`` with parallel I/O
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You only have to make sure to have a parallel version of ``h5py`` installed. A
simple way is to just use one on conda, e.g.:

.. code-block:: bash

    $ conda install -c spectraldns h5py-parallel


Additionally you need ``mpi4py``. The one on ``conda`` might work, if not,
read on.

For all of the following steps make sure that the MPI package of your local
supercomputer/cluster is loaded. The ``mpi4py`` potentially shipping with
Anaconda might not work on your cluster - if that is the case uninstall it
and reinstall with ``pip`` at which point it should link against your
cluster's MPI implementation.

.. code-block:: bash

    $ conda uninstall mpi4py
    $ pip install mpi4py


After everything is installed, you can run the following command to print
information about the current system.

.. code-block:: bash

    $ python -c "import pyasdf; pyasdf.print_sys_info()"

which will print something along the following lines::

    pyasdf version 0.1.4
    ===============================================================================
    CPython 2.7.9, compiler: GCC 4.2.1 (Apple Inc. build 5577)
    Darwin 14.3.0 64bit
    Machine: x86_64, Processor: i386 with 8 cores
    ===============================================================================
    HDF5 version 1.8.17, h5py version: 2.5.0
    MPI: Open MPI, version: 1.10.1, mpi4py version: 2.0.0
    Parallel I/O support: True
    Problematic multiprocessing: False
    ===============================================================================
    Other_modules:
        dill: 0.2.5
        lxml: 3.7.2
        numpy: 1.11.3
        obspy: 1.0.3
        prov: 1.4.0
        scipy: 0.18.1


Testing
-------

To assert that your installation is working properly, execute

.. code-block:: bash

    $ python -m pyasdf.tests

and make sure all tests pass. Otherwise please contact the developers.


Building the Documentation
--------------------------

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
