Installation
============

pyasdf and Dependencies
-----------------------

``pyasdf`` supports Python version 2.7, 3.4, 3.5, and 3.6 and it depends on the
following Python modules: ``NumPy``, ``ObsPy``, ``h5py``, ``colorama``,
``flake8``, ``pytest``, ``prov``, ``dill``, and optionally ``mpi4py``. You can
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
    reading/writing thread might be sufficient.

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

.. note::

    There currently might be some issues in combination of Python 3 and MPI
    so if you want to utilize parallel I/O best stick to Python 2.7 for now.


The version with parallel I/O support is a bit more difficult as the ``h5py``
installable via ``conda`` has no parallel I/O support.

.. code-block:: bash

    $ conda update conda
    $ conda install -c conda-forge obspy colorama pytest pip flake8 dill prov


For all of the following steps make sure that the MPI package of your local
supercomputer/cluster is loaded. The ``mpi4py`` potentially shipping with
Anaconda might not work on your cluster so uninstall it and reinstall with
``pip`` at which point it should link against your cluster's MPI
implementation.

.. code-block:: bash

    $ conda uninstall mpi4py
    $ pip install mpi4py

Keep in mind that ``h5py`` must be compiled with parallel I/O support and that
it is linked against the same MPI as ``mpi4py`` which of course should be the
same that is used by your computer.

Install parallel ``h5py`` according to
`these instructions <http://docs.h5py.org/en/latest/mpi.html>`_.

A further thing to keep in mind is that ``mpi4py`` changed some of their
internal API for version 2.0. This has to be accounted for when installing the
parallel ``h5py`` version. See here for more details:
https://github.com/SeismicData/pyasdf/issues/11

Finally install ``pyasdf`` with

.. code-block:: bash

    $ pip install pyasdf

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


This should enable you to judge if ``pyasdf`` can run on your system.
Especially important is the *Parallel I/O support* line. If multiprocessing
is problematic, ``pyasdf`` will not be able to run on more than one machine
without MPI. Please see
`here <https://github.com/obspy/obspy/wiki/Notes-on-Parallel-Processing-with-Python-and-ObsPy>`_
for information about why and how to fix it.



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
