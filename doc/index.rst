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


Reading an existing ASDF data set
*********************************

Once you have acquired an ASDF file by whatever means it is time to read the
data. To a large extent reading works by attribute access (meaning tab
completion in interactive environments is available). At access time the
data is read from an ASDF file an parsed to an ObsPy object.

As always, the first step is to open a file. Note that this does not yet
read any data.

.. code-block:: python

    >>> import pyasdf
    >>> ds = pyasdf.ASDFDataSet("example.h5")


**Reading Events**

To read event data, simply access the ``event`` attribute. At access time the
events will be parsed to an :class:`obspy.core.event.Catalog` object which can
used for further analysis.

.. code-block:: python

    >>> type(ds.events)
    obspy.core.event.Catalog
    >>> print(ds.events)
    4 Event(s) in Catalog:
    1998-09-01T10:29:54.500000Z | -58.500,  -26.100 | 5.5 Mwc
    2012-04-04T14:21:42.300000Z | +41.818,  +79.689 | 4.4 mb | manual
    2012-04-04T14:18:37.000000Z | +39.342,  +41.044 | 4.3 ML | manual
    2012-04-04T14:08:46.000000Z | +38.017,  +37.736 | 3.0 ML | manual


**Reading Waveforms and StationXML**

Waveforms and station meta information can be accessed at a per-station
granularity under the ``waveforms`` attribute. **Note that tab completion
works everywhere so please play around in the IPython shell which is the
best way to learn how to use pyasdf.**

In the following we will access the data for the ``IU.ANMO`` station. Note
that the dot is replaced by an underscore to work around syntax limitations
imposed by Python. Once again, at attribute access everything is read from
the ASDF file and parsed to an ObsPy object.

.. code-block:: python

    >>> print(ds.waveforms.IU_ANMO)
    Contents of the data set for station IU.ANMO:
        - Has a StationXML file
        - 2 Waveform Tag(s):
            observed_processed
            synthetic)

    >>> type(ds.waveform.IU_ANMO.StationXML))
    obspy.core.inventory.inventory.Inventory
    >>> print(ds.waveform.IU_ANMO.StationXML)
    Inventory created at 2014-12-10T17:04:09.000000Z
    Created by: IRIS WEB SERVICE: fdsnws-station | version: 1.1.9
            http://service.iris.edu/fdsnws/station/1/query?network=IU&level=res...
    Sending institution: IRIS-DMC (IRIS-DMC)
    Contains:
        Networks (1):
            IU
        Stations (1):
            IU.ANMO (Albuquerque, New Mexico, USA)
        Channels (6):
            IU.ANMO..BH1, IU.ANMO..BH2, IU.ANMO..BHU, IU.ANMO..BHV,
            IU.ANMO..BHW, IU.ANMO..BHZ

    >>> type(ds.waveforms.IU_ANMO.synthetic)
    obspy.core.stream.Stream
    >>> print(ds.waveforms.IU_ANMO.synthetic)
    3 Trace(s) in Stream:
    IU.ANMO.S3.MXE | 1998-09-01T10:29:52.250000Z - 1998-09-01T12:10:19.857558Z | 7.0 Hz, 42300 samples
    IU.ANMO.S3.MXN | 1998-09-01T10:29:52.250000Z - 1998-09-01T12:10:19.857558Z | 7.0 Hz, 42300 samples
    IU.ANMO.S3.MXZ | 1998-09-01T10:29:52.250000Z - 1998-09-01T12:10:19.857558Z | 7.0 Hz, 42300 samples


Now attribute access is convenient for interactive use, but not that much for
programmatic access. A number of better ways are available for that case:


.. code-block:: python

    >>> ds.waveforms.list()
    ['IU.ADK', 'IU.AFI', 'IU.ANMO', 'IU.ANTO', 'IU.BBSR']

    >>> print(ds.waveforms["IU.ANMO"])
    Contents of the data set for station IU.ANMO:
        - Has a StationXML file
        - 2 Waveform Tag(s):
            observed_processed
            synthetic)

    >>> for station in ds.waveforms:
    ...     print(station)
    ...     break
    Contents of the data set for station IU.ADK:
        - Has a StationXML file
        - 2 Waveform Tag(s):
            observed_processed
            synthetic)


The object returned for each station is just a helper object easing the
access to the waveform and station data.


.. code-block:: python

    >>> sta = ds.waveforms["IU.ANMO"]
    >>> type(sta)
    pyasdf.utils.WaveformAccessor


ASDF distinguishes waveforms by tags. All waveforms belonging to a certain
tag can be accessed with either attribute or key access.

.. code-block:: python

    >>> print(sta.synthetic)
    3 Trace(s) in Stream:
    IU.ANMO.S3.MXE | 1998-09-01T10:29:52.250000Z - 1998-09-01T12:10:19.857558Z | 7.0 Hz, 42300 samples
    IU.ANMO.S3.MXN | 1998-09-01T10:29:52.250000Z - 1998-09-01T12:10:19.857558Z | 7.0 Hz, 42300 samples
    IU.ANMO.S3.MXZ | 1998-09-01T10:29:52.250000Z - 1998-09-01T12:10:19.857558Z | 7.0 Hz, 42300 samples
    >>> sta.synthetic == sta["synthetic"]
    True


Get a list of all tags for a certain station:

.. code-block:: python

    >>> sta.get_waveform_tags()
    ['observed_processed', 'synthetic']


Sometimes more granular access is required. To that end one can also access
waveforms at the individual trace level.

.. code-block:: python

    >>> sta.list()
    ['IU.ANMO..BHZ__1998-09-01T10:24:49__1998-09-01T12:09:48__observed_processed',
     'IU.ANMO.S3.MXE__1998-09-01T10:29:52__1998-09-01T12:10:19__synthetic',
     'IU.ANMO.S3.MXN__1998-09-01T10:29:52__1998-09-01T12:10:19__synthetic',
     'IU.ANMO.S3.MXZ__1998-09-01T10:29:52__1998-09-01T12:10:19__synthetic',
     'StationXML']
    >>> print(sta["IU.ANMO.S3.MXE__1998-09-01T10:29:52__1998-09-01T12:10:19__synthetic"])
    1 Trace(s) in Stream:
    IU.ANMO.S3.MXE | 1998-09-01T10:29:52.250000Z - 1998-09-01T12:10:19.857558Z | 7.0 Hz, 42300 samples


The advantage of the ASDF format is that it is also able to store relations
so it can for example tell what event a certain waveform is associated with:

.. code-block:: python

    >>> cat = ds.events  # The events have to be in memory for the reference to work.
    >>> print(sta.synthetic[0].stats.asdf.event_id.getReferredObject())
    Event:	1998-09-01T10:29:54.500000Z | -58.500,  -26.100 | 5.5 Mwc

            resource_id: ResourceIdentifier(id="smi:service.iris.edu/fdsnws/event/1/query?eventid=656970")
             event_type: 'earthquake'
    ---------
     event_descriptions: 1 Elements
       focal_mechanisms: 1 Elements
                origins: 1 Elements
             magnitudes: 1 Elements


Additionally it can also store the provenance associated with a certain
waveform trace:

.. code-block:: python

    >>> prov_id = sta.synthetic[0].stats.asdf.provenance_id
    >>> ds.provenance.get_provenance_document_for_id(prov_id)
    {'document': <ProvDocument>, 'name': '373da5fe_d424_4f44_9bca_4334d77ed10b'}
    >>> ds.provenance.get_provenance_document_for_id(prov_id)["document"].plot()

.. graphviz:: _static/example_waveform_simulation.dot


**Reading Auxiliary Data**

ASDF can additionally also store non-waveform data. Access is via the
``auxiliary_data`` attribute. Once again, dictionary access is possible.

.. code-block:: python

    >>> print(ds.auxiliary_data)
    Data set contains the following auxiliary data types:
        AdjointSource (12 item(s))
        File (1 item(s))

    >>> print(ds.auxiliary_data.AdjointSource)
    print(ds.auxiliary_data.AdjointSource)
    12 auxiliary data item(s) of type 'AdjointSource' available:
        BW_ALFO_EHE
        BW_ALFO_EHN
        BW_ALFO_EHZ
        BW_BLA_EHE
        BW_BLA_EHN
        BW_BLA_EHZ
        IU_ANMO_EHE
        IU_ANMO_EHN
        IU_ANMO_EHZ
        TA_A023_BHE
        TA_A023_BHN
        TA_A023_BHZ

    >>> ds.auxiliary_data.list()
    ['AdjointSource', 'File']

    >>> ds.auxiliary_data.AdjointSource == ds.auxiliary_data["AdjointSource"]
    True

Accessing a single item once again by attribute or key access.

.. code-block:: python

    >>> ds.auxiliary_data.AdjointSource.BW_ALFO_EHE
    Auxiliary Data of Type 'AdjointSource'
        Tag: 'BW_ALFO_EHE'
        Provenance ID: '{http://seisprov.org/seis_prov/0.1/#}sp012_as_cd84e87'
        Data shape: '(20000,)', dtype: 'float64'
        Parameters:
            adjoint_source_type: multitaper
            elevation_in_m: 473.232036071
            latitude: 57.9589770294
            local_depth_in_m: 0.0
            longitude: 170.352381909
            misfit_value: 1.45e-05
            orientation: E
            sampling_rate_in_hz: 10.0
            station_id: BW.ALFO..EHE
            units: m

    >>> ds.auxiliary_data.AdjointSource.list()
    ['BW_ALFO_EHE', 'BW_ALFO_EHN', 'BW_ALFO_EHZ', 'BW_BLA_EHE', 'BW_BLA_EHN', 'BW_BLA_EHZ',
     'IU_ANMO_EHE', 'IU_ANMO_EHN', 'IU_ANMO_EHZ', 'TA_A023_BHE', 'TA_A023_BHN', 'TA_A023_BHZ']

    >>> ds.auxiliary_data.AdjointSource.BW_ALFO_EHE == ds.auxiliary_data.AdjointSource["BW_ALFO_EHE"]
    True

    >>> adj_src = ds.auxiliary_data.AdjointSource.BW_ALFO_EHE
    >>> adj_src.data
    <HDF5 dataset "BW_ALFO_EHE": shape (20000,), type "<f8">
    >>> adj_src.parameters
    {'adjoint_source_type': 'multitaper',
     'elevation_in_m': 473.23203607130199,
     'latitude': 57.958977029361542,
     'local_depth_in_m': 0.0,
     'longitude': 170.35238190943915,
     'misfit_value': 1.45e-05,
     'orientation': 'E',
     'sampling_rate_in_hz': 10.0,
     'station_id': 'BW.ALFO..EHE',
     'units': 'm'}


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
    querying_data
