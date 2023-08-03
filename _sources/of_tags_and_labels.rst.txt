Of Tags and Labels
==================

The ``ASDF`` data format has tags and labels which, while similar to a certain
extent, serve a different purpose. This is sometimes a bit confusing to
people new to the format - this page explains and clarifies everything.


Tags
----

**Tags are used as an additional hierarchical layer.** They are for example
used to distinguish observed and synthetic data or two synthetic waveforms
calculated with slightly different earth models. Each waveform trace must
have a tag - it is used as part of the arrays' names in the HDF5 file.
There are little rules to them but they should be pretty short.

**The** ``raw_recording`` **tag is by convention reserved for raw data counts
straight from a digitizer**.

Other names depend on the use case - common choices are ``synthetic_prem`` or
``processed_1_10_s``.


The following example shows how tags are used and how to work with them.


.. code-block:: python

    >>> import obspy
    >>> import pyasdf

    >>> ds = pyasdf.ASDFDataSet("./example.h5")

    # The tag always has to be set explicitly.
    >>> ds.add_waveforms(obspy.read(), tag="example")

    # Data is again retrieved by tag.
    >>> st = ds.waveforms.BW_RJOB.example
    >>> st[0].stats.asdf.tag
    'example'

    # These are the names of the arrays in the HDF5 file. Note how it is
    # used as an additional hierarchy layer.
    >>> ds.waveforms.BW_RJOB.list()
    ['BW.RJOB..EHE__2009-08-24T00:20:03__2009-08-24T00:20:32__example',
     'BW.RJOB..EHN__2009-08-24T00:20:03__2009-08-24T00:20:32__example',
     'BW.RJOB..EHZ__2009-08-24T00:20:03__2009-08-24T00:20:32__example']

    # Loop over all waveforms in the file with the "example" tag.
    >>> for st in ds.ifilter(ds.q.tag == "example"):
    ...     print(st)
    Filtered contents of the data set for station BW.RJOB:
        - Has no StationXML file
        - 1 Waveform Tag(s):
            example


Labels
------

Labels on the other hand are an optional list of words potentially assigned to
a waveform. They can be used to describe and label similar
waveforms without influencing how they are stored on disc. **They are a
piece of meta information** useful to organize the data a bit better. Its
always a list of simple UTF-8 encoded words. It's best illustrated with an
example.


.. code-block:: python

    >>> import obspy
    >>> import pyasdf

    >>> ds = pyasdf.ASDFDataSet("./example.h5")

    # Tags must be given. Labels are optional. Note how unicode is possible.
    >>> ds.add_waveforms(obspy.read(), tag="example",
    ...                  labels=["example", u"öä∂ß"])

    # After retrieving the data again, the labels are attached to the stats
    # objects.
    >>> st = ds.waveforms.BW_RJOB.example
    >>> st[0].stats.asdf.labels
    ['example', 'öä∂ß']

    # It is also possible to search for waveforms with a certain label
    # including wildcard searches.
    >>> for st in ds.ifilter(ds.q.labels == u"*ä*"):
    ...    print(st)
    Filtered contents of the data set for station BW.RJOB:
        - Has no StationXML file
        - 1 Waveform Tag(s):
            example
