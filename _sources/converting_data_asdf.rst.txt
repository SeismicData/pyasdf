Converting Data to ASDF
=======================

To convert generic data to ASDF please just use ObsPy to read it and add it
to ASDF files as written in the tutorial: :doc:`tutorial`.

Converting a folder of SAC files to ASDF
----------------------------------------

As this is a common use case we ship a utility for this. Note that this is
not a silver bullet and it cannot cover all cases, so please check your data
afterwards.

.. code-block:: bash

    $ python -m pyasdf.scripts.sac2asdf folder_with_sac_files out.h5 synthetics

This will search (not recursively) for all SAC files in
``folder_with_sac_files`` and write them to the `out.h5` ASDF file. All
waveforms will (in this case) have the tag `synthetics`.

It will honor the following SAC header fields and convert them to corresponding
StationXML and QuakeML documents:

* ``evla``
* ``evlo``
* ``evdp``
* ``o``
* ``stla``
* ``stlo``
* ``stel``
* ``stdp``

The waveforms will be correctly linked to the corresponding QuakeML files
with ASDF's internal reference system. Please see the help of the script for
more details.


.. code-block:: bash

    $ python -m pyasdf.scripts.sac2asdf --help


The file can then be read as follows:

.. code-block:: python

    >>> import pyasdf
    >>> ds = pyasdf.ASDFDataSet("./out.h5")
    >>> ds
    ASDF file [format version: 1.0.0]: 'out.h5' (615.1 MB)
        Contains 1 event(s)
        Contains waveform data from 1829 station(s).

    # Get all events.
    >>> cat = ds.events
    >>> cat
    1 Event(s) in Catalog:
    2005-02-16T20:28:06.438750Z | -35.390,  -16.000

    # Get waveforms.
    >>> st = ds.waveforms.IU_ANMO.synthetics
    >>> st
    3 Trace(s) in Stream:
    IU.ANMO.S3.MXE | 2005-02-16T20:28:06.438750Z - ... | 6.2 Hz, 37200 samples
    IU.ANMO.S3.MXN | 2005-02-16T20:28:06.438750Z - ... | 6.2 Hz, 37200 samples
    IU.ANMO.S3.MXZ | 2005-02-16T20:28:06.438750Z - ... | 6.2 Hz, 37200 samples

    # Get station coordinates.
    >>> ds.waveforms.IU_ANMO.coordinates
    {'elevation_in_m': 1720.0,
     'latitude': 34.945899963378906,
     'longitude': -106.45719909667969}

    # Get event associated with a certain trace
    >>> event = st[0].stats.asdf.event_ids[0].get_referred_object()
    >>> event.origins[0].latitude, event.origins[0].longitude
    (-35.38999938964844, -16.0)
