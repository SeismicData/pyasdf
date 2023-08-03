Usage as a Context Manager
==========================

The :class:`~pyasdf.asdf_data_set.ASDFDataSet` class can also be used as a
context manager which guarantees that the file is always closed after it is
used.


.. code-block:: python

    import pyasdf

    with pyasdf.ASDFDataSet("example.h5") as ds:
        ds.add_waveforms(...)
