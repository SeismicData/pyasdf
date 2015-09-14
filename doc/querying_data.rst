Querying a Data Set
===================

The potential size of ASDF data sets demands efficient and easy ways to loop
over and work on certain subsets of it. This functionality is available in
the :meth:`pyasdf.asdf_data_set.ASDFDataSet.ifilter` method which is very
powerful and explained in the rest of this section.

Quick Example
-------------

In this introductory example we will loop over a data set and extract all
waveform data that has a network starting with ``T``, a ``raw_recording`` tag,
and a sampling rate of more than 10 *Hz*. The
:meth:`~pyasdf.asdf_data_set.ASDFDataSet.ifilter` method returns an iterator
meaning one has to loop over it to cause it to extract the required
information. It yields a :class:`~pyasdf.utils.FilteredWaveformAccessor` which
works the same as the normal waveform accessor object but with a potentially
limited dataset. Have a look at the syntax and it should be pretty
self-explanatory.

.. code-block:: python

    import pyasdf

    ds = pyasdf.ASDFDataSet("example.h5")

    for station in ds.ifilter(ds.q.network == "T*",
                              ds.q.tag == "raw_recording",
                              ds.q.sampling_rate > 10.0):
        # Extract the StationXML information in the same way as always. Might
        # be `None` if it does not exists.
        inv = station.StationXML
        if inv is None:
            continue

        # You can also get the coordinates.
        coords = station.coordinates

        # Waveform access also works the normal way by tag.
        st = station.raw_recording

        # Process it and do whatever you want.
        st.detrend("linear")
        st.taper(type="hann", max_percentage=0.05)
        st.attach_response(inv)
        st.remove_response()
        ...


More Example Queries
--------------------

We'll show some more example to give you a feel for how to construct queries
and what they mean. The syntax is not as powerful as a full SQL implementation
but for practical purposes it gets pretty close and it can express and resolve
some fairly complex queries. You can chain an arbitrary number of constraints
which must all be fulfilled at the same time, each constraint is based on
the rich comparison operators from Python: ``==``, ``!=``, ``<``, ``<=``,
``>``, and ``>=``.

Separate arguments are chained by a logical ``AND``, so **every comma can be
interpreted as a logical** ``AND`` **.** The following query will iterate over
all waveforms with a network starting with ``T``, but not the ``TA``
(USArray) network. Wildcards are resolved by the :mod:`fnmatch` module.

.. code-block:: python

    for station in ds.ifilter(ds.q.network == "T*",
                              ds.q.network != "TA"):
        ...

String attributes (``network``, ``station``, ``channel``, ...) can also be
queried by passing a list of potentially wildcarded strings. **Each element in
such a list is joined by a logical** ``OR``, e.g. the next query will select
all stations starting with ``B``, or the ``CA`` station, or any two letter
station starting with ``T``:


.. code-block:: python

    for station in ds.ifilter(ds.q.station == ["B*", "ALTM", "T?"]):
        ...


You can also search over geographical constraints based on the ``latitude``,
``longitude``, and ``elevation_in_m`` keys. As soon as one of these keys is
given it will only return stations that have coordinate information in the
form of StationXML.


.. code-block:: python

    for station in ds.ifilter(ds.q.latitude >= 10.0, ds.q.latitude <= 20,
                              ds.q.longitude <= -101.2,
                              ds.q.elevation_in_m > 200.0)
        ...


Get all vertical component channels from USArray with a sampling rate of at
least 5 *Hz* that are not processed (``raw_recording`` tag by convention) in
January 2015:

.. code-block:: python

    for station in ds.ifilter(ds.q.network == "TA",
                              ds.q.channel == "*Z",
                              ds.q.sampling_rate >= 5,
                              ds.q.tag == "raw_recording",
                              ds.q.starttime >= obspy.UTCDateTime(2015, 1, 1),
                              ds.q.endtime <= obspy.UTCDateTime(2015, 2, 1))
        ...


Query Types
-----------


.. warning::

    Most of the queries work as one would intuitively expect, the exception are
    ``!=`` queries for ``latitude``, ``longitude``, and  ``elevation_in_m``.
    These are all optional pieces of meta information for a waveform trace. If
    any of these keys is part of a query and a given trace does not have that
    piece of information that trace will not be returned no matter what the
    query actually asks for. Assume that a trace has no associated StationXML,
    the following query will not return it even though it is logically true:

    .. code-block:: python

        for station in ds.ifilter(ds.q.latitude != 10.0):
            ...

    This is a consequence of how the queries are internally implemented.
    Working around this would require much more code or less flexibility in
    other areas. Just be aware of this and it should prove no issue.

    The ``ifilter()`` method will furthermore see missing ``event``, ``origin``,
    ``magnitude``, and ``focal_mechanism`` ids as an empty string.


.. raw:: html

    <div style="height:10px"></div>

The available query keys alongside other information regarding their usage are
listed in the following table:

.. raw:: html

    <div style="height:10px"></div>

+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| Name                | Allowed Types                                                        | Description                                                                       |
+=====================+======================================================================+===================================================================================+
| **String Parameters:**                                                                                                                                                         |
| These can be given as a single (wildcarded) string or as a list of (wildcarded) strings which will be connected via a logical ``OR``.                                          |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``network``         | ``str`` or list of ``str``                                           | The network code.                                                                 |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``station``         | ``str`` or list of ``str``                                           | The station code.                                                                 |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``location``        | ``str`` or list of ``str``                                           | The location code.                                                                |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``channel``         | ``str`` or list of ``str``                                           | The channel code.                                                                 |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``tag``             | ``str`` or list of ``str``                                           | The hierarchical tag associated with the trace.                                   |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| **Geographical Parameters:**                                                                                                                                                   |
| Search over geographical parameters stored in the StationXML files. If any of these three parameter is given: A station that has no StationXML file (or no coordinates),       |
| will not be returned, no matter what the query actually asks for.                                                                                                              |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``longitude``       | ``float``                                                            | The longitude of the recording station.                                           |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``latitude``        | ``float``                                                            | The latitude of the recording station.                                            |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``elevation_in_m``  |  ``float``                                                           | The elevation in meters of the recording station.                                 |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| **Temporal Parameters:**                                                                                                                                                       |
| Pass as a :class:`~obspy.core.utcdatetime.UTCDateTime` object or any string or number that can be parsed to one.                                                               |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``starttime``       | :class:`~obspy.core.utcdatetime.UTCDateTime`, ``str``, or ``float``  | The start time of the waveform.                                                   |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``endtime``         | :class:`~obspy.core.utcdatetime.UTCDateTime`, ``str``, or ``float``  | The end time of the waveform.                                                     |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| **Waveform Attribute Parameters:**                                                                                                                                             |
| Evaluated per waveform trace. Don't use ``==`` for floats but rather a combination of ``>=`` and ``<=``.                                                                       |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``sampling_rate``   | ``float``                                                            | The sampling rate of the waveform in *Hz*.                                        |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``npts``            | ``int``                                                              | The number of samples of the waveform.                                            |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| **Event Relation Parameters:**                                                                                                                                                 |
| If a trace does not have any of these, the ``ifilter()`` method will see it as though it is an empty string. **These are not wildcarded** as ``?`` and                         |
| ``*`` are perfectly valid URL components and most IDs are URLs.                                                                                                                |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``event``           | :class:`~obspy.core.event.Event`,                                    | The event associated with the waveform.                                           |
|                     | :class:`~obspy.core.event.ResourceIdentifier`, or ``str``            |                                                                                   |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``magnitude``       | :class:`~obspy.core.event.Magnitude`,                                | The magnitude associated with the waveform.                                       |
|                     | :class:`~obspy.core.event.ResourceIdentifier`, or ``str``            |                                                                                   |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``origin``          | :class:`~obspy.core.event.Origin`,                                   | The origin associated with the waveform.                                          |
|                     | :class:`~obspy.core.event.ResourceIdentifier`, or ``str``            |                                                                                   |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
| ``focal_mechanism`` | :class:`~obspy.core.event.FocalMechanism`,                           | The focal mechanism associated with the waveform.                                 |
|                     | :class:`~obspy.core.event.ResourceIdentifier`, or ``str``            |                                                                                   |
+---------------------+----------------------------------------------------------------------+-----------------------------------------------------------------------------------+
