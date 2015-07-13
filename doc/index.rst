pyasdf
======

``pyasdf`` is a Python API to read and write seismological ASDF files.


.. admonition:: Contact Us

    If you encounter a bug or another error, please open a new
    `issue <https://github.com/SeismicData/pyasdf/issues>`_ on Github.


This document is organized as follows. Other documents can be reached via
the navigation bar up top.


.. contents::
    :local:
    :depth: 3


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

This section will teach you the basics of creating and working with ASDF
files with `pyasdf`. For more detailed information, please see the
:doc:`api` documentation or have a look at the :doc:`examples` section that
demonstrates more complex workflows.

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

    >>> import pyasdf
    >>> ds = pyasdf.ASDFDataSet("new_file.h5", compression="gzip-3")

Compression will help to reduce the size of files without affecting the
quality as all offered compression options are lossless. Only new data will
be compressed and compression will be disable for files created with
parallel I/O as these two features are not compatible. See the documentation
of the  :class:`~pyasdf.asdf_data_set.ASDFDataSet` object for more details.

At any point you can get an overview of the contents by printing the object

.. code-block:: python

    >>> print(ds)
    ASDF file [format version: 0.0.1b]: 'new_file.h5' (7.7 KB)
    Contains 0 event(s)
    Contains waveform data from 0 station(s).


**Adding Information about an Earthquake:**

ASDF can optionally store events and associate waveforms with a given event.
To add an event with ``pyasdf`` use the
:meth:`~pyasdf.asdf_data_set.ASDFDataSet.add_quakeml` method. Be aware that
all operations will directly write to the file without an explicit
*save/write* step. This enables ``pyasdf`` to deal with arbitrarily big data
sets.

.. code-block:: python

    >>> ds.add_quakeml("/path/to/quake.xml")
    >>> print(ds)
    ASDF file [format version: 0.0.1b]: 'new_file.h5' (14.7 KB)
    Contains 1 event(s)
    Contains waveform data from 0 station(s).

The event can later be accessed again by using the ``event`` attribute.
Please be careful that this might contain multiple events and not only one.


.. code-block:: python

    >>> event = ds.events[0]


**Adding Waveforms:**

The next step is to add waveform data. In this example we will add all SAC
files in one folder with the help of the
:meth:`~pyasdf.asdf_data_set.ASDFDataSet.add_waveforms` method.
Printing the progress is unnecessary but useful when dealing with large
amounts of data. There are a couple of subtleties to keep in mind here:

* The data will be compressed with the compression settings given during the
  initialization of the data set object.
* It is possible to optionally associate waveforms with a specific event.
* You must give a tag. The tag is an additional identifier of that particular
  waveform. The ``"raw_recording"`` tag is by convention only given to raw,
  unprocessed data.
* The actual type of the data will not change. This example uses SAC which
  means data is in single precision floating point, MiniSEED will typically
  be in single precision integer if coming from a data center. If you desire
  a different data type to be stored for whatever reason you have to
  manually convert it and pass the ObsPy :class:`~obspy.core.stream.Stream`
  object.


.. code-block:: python

    >>> import glob
    >>> files = glob.glob("/path/to/data/*.mseed")
    >>> for _i, filename in enumerate(files):
    ...     print("Adding file %i of %i ..." % (_i + 1, len(files)))
    ...     ds.add_waveforms(filename, tag="raw_recording", event_id=event)
    Adding file 1 of 588 ...
    ...
    >>> print(ds)
    ASDF file [format version: 0.0.1b]: 'new_file.h5' (169.7 MB)
    Contains 1 event(s)
    Contains waveform data from 196 station(s).


**Adding Station Information:**

The last step to create a very basic ASDF file is to add station information
which is fairly straightforward.


.. note::

    Please keep in mind that this will potentially rearrange and split the
    StationXML files to fit within the structure imposed by the ASDF format.
    StationXML can store any number and combination of networks, stations,
    and channels whereas ASDF requires a **separate StationXML file per
    station**. ``pyasdf`` will thus split up files if necessary and also
    merge them with any possibly already existing StationXML files.


.. code-block:: python

    >>> files = glob.glob("/path/to/stations/*.xml")
    >>> for _i, filename in enumerate(files):
    ...     print("Adding file %i of %i ..." % (_i + 1, len(files)))
    ...     ds.add_stationxml(filename)
    Adding file 1 of 196 ...
    ...
    >>> print(ds)
    ASDF file [format version: 0.0.1b]: 'new_file.h5' (188.3 MB)
    Contains 1 event(s)
    Contains waveform data from 196 station(s).


**Adding Auxiliary Data:**

The ASDF format has the capability to store generic non-waveform data called
*auxiliary data*. Auxiliary data are data arrays with associcated
attributes that can represent anything; each sub-community is supposed to come
up with conventions of how to use this.

.. code-block:: python

    >>> import numpy as np
    >>> data = np.random.random(100)
    # The type always should be camel case.
    >>> data_type = "RandomArrays"
    # Name to identify the particular piece of data.
    >>> tag = "example_array"
    # Any additional parameters as a Python dictionary which will end up as
    # attributes of the array.
    >>> parameters = {
    ...     "a": 1,
    ...     "b": 2.0}
    >>> ds.add_auxiliary_data(data=data, data_type=data_type, tag=tag,
    ...                       parameters=parameters)
    >>> print(ds)
    ASDF file [format version: b'0.0.1b']: 'new_file.h5' (188.3 MB)
    Contains 1 event(s)
    Contains waveform data from 196 station(s).
    Contains 1 type(s) of auxiliary data: RandomArrays


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
    station_data
