Iterating Over Data
===================

The potential size of ASDF data sets demands efficient and easy ways to loop
over and work on certains subsets of it.

Iterating over Tags
-------------------

Each waveform has exactly one associated tag which is used as a further
hierarchy layer but they oftentimes also serve a descriptive purpose. To loop
all waveforms with a certain tag, use the
:meth:`pyasdf.asdf_data_set.ASDFDataSet.iter_tag` method which returns an
:class:`obspy.core.stream.Stream` and a
:class:`obspy.core.station.inventory.Inventory` object (which might ``None``)
for each station that has at least one waveform with the desired tag:

.. code-block:: python

    import pyasdf

    ds = pyasdf.ASDFDataSet("example.h5")
    for st, inv in ds.iter_tag("raw_recording"):
        # The inventory might be None if no station information is available.
        # Include this check if you always need inventory information.
        if inv is None:
            continue
        st.detrend("linear")
        st.taper(type="hann", max_percentage=0.05)
        st.attach_response(inv)
        st.remove_response()
        ...


Iterating over Event/Origin/Magnitude/Focal Mechanism IDs
---------------------------------------------------------

The ASDF format allows the association of waveforms with event, origin,
magnitude, or focal mechanism IDs from a QuakeML files. It is possible
