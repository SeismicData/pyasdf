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
limited dataset.

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

String attributes (network, station, channel, ...) can also be queried by
passing a list of potentially wildcarded strings. **Each element in such a
list is joined by a logical** ``OR``, e.g. the next query will select all
stations starting with ``B``, or the ``CA`` station, or any two letter station
starting with ``T``.


.. code-block:: python

    for station in ds.ifilter(ds.q.station == ["B*", "ALTM", "T?"]):
        ...


You can also search over geographical constraints based on the ``latitude``,
``longitude``, and ``elevation_in_m`` keys. As soon as one of these keys is
given it will only return stations that have coordinate information in the
form of StationXML.


.. code-block:: python

    for station in ds.ifilter(ds.q.latitude >= 10.0, ds.q.latitude <= 20,
                              ds.q.longitude)
        ...