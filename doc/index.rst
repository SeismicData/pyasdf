pyasdf
======

``pyasdf`` is a Python API to read and write seismological ASDF files.


.. admonition:: Contact Us

    If you encounter a bug or another error, please open a new
    `issue <https://github.com/SeismicData/pyasdf/issues>`_ on Github.


Installation
------------

To get started, please follow the :doc:`installation` instructions.

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

This section will teach you the basics of creating and working with. For
more detailed information, please see the :doc:`api` documentation or have a
look at the examples that demonstrate more complex workflows.

.. admonition:: Notes on using MPI

    ``pyasdf`` will behave slightly different depending on whether it is being
    called with MPI or not. If called with MPI, the underlying HDF5 files
    will be opened with MPI I/O and fully parallel I/O will be utilized for
    the processing functions. Keep the scripts reasonably short if using MPI
    or place appropriate barriers.

    The module will detect if it has been called with MPI if ``mpi4py`` can
    be imported and the size of the communicator is greater than one. Thus
    calling it with ``mpirun/mpiexec`` with only one core will not be
    recognized as being called with MPI.

    Due to a limitation of Python you should always delete all references to
    the :class:`~pyasdf.asdf_data_set.ASDFDataSet` objects at the end.
    Otherwise the program will potentially not finish when called with MPI.


Creating an ASDF data set
*************************

The first step is always to import ``pyasdf`` and create a
:class:`~pyasdf.asdf_data_set.ASDFDataSet` object. If the file does not
exists, it will be created, otherwise the existing one will be opened.

.. code-block:: python

    import pyasdf
    ds = pyasdf.ASDFDataSet("new_file.h5", compression="szip-nn-10")

Compression will help to reduce the size of files without affecting the
quality as all offered compression options are lossless. Only new data will
be compressed and compression will be disable for files created with
parallel I/O as these two features are not compatible. See the documentation
of the  :class:`~pyasdf.asdf_data_set.ASDFDataSet` object for more details.

ASDF can optionally store events and associate waveforms with a given event.
To add an event with ``pyasdf`` use the
:meth:`~pyasdf.asdf_data_set.ASDFDataSet.add_quakeml` method.




Acknowledgements
----------------

We gratefully acknowledge support from the EU-FP7 725 690 **VERCE** project
(number 283543, `www.verce.eu <http://www.verce.eu>`_).

Detailed Documentation
----------------------

.. toctree::
    :maxdepth: 2

    installation
    examples
    api
    asdf_data_set
