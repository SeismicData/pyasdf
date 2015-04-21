Installation
============

pyasdf and Dependencies
-----------------------

``pyasdf`` supports Python version 2.7 and 3.4 and it depends on the following
Python modules: ``NumPy``, ``ObsPy``, ``h5py``, ``mpi4py``, ``colorama``,
``flake8``, and ``pytest``. Keep in mind that ``h5py`` must be compiled with
parallel I/O support and that it is linked against the same MPI as ``mpi4py``
which of course should be the same that is used by your computer.

If you know what you are doing, install it any way you see fit. Otherwise do
yourself a favor and download the
`Anaconda Python distribution <https://store.continuum.io/cshop/anaconda/>`_
for Python 2.7 or 3.4. After downloading update it, and install the
dependencies.

.. code-block:: bash

    $ conda update conda
    $ conda install -c obspy obspy colorama pytest pip flake8


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


After everything is installed, run the following command to print
information about the current system.

.. code-block:: bash

    $ python -c "import pyasdf; pyasdf.print_sys_info()"

which will print something along the following lines::

    pyasdf version 0.1.x
    ===============================================================================
    CPython 2.7.9, compiler: GCC 4.2.1 (Apple Inc. build 5577)
    Darwin 14.3.0 64bit
    Machine: x86_64, Processor: i386 with 8 cores
    ===============================================================================
    HDF5 version 1.8.13, h5py version: 2.3.1
    MPI: Open MPI, version: 1.8.3, mpi4py version: 1.3.1
    Parallel I/O support: True
    Problematic multiprocessing: False
    ===============================================================================
    Other_modules:
        obspy: 0.10.1
        lxml: 3.4.2
        scipy: 0.15.1
        numpy: 1.9.2


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


Build the Documentation
-----------------------

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
