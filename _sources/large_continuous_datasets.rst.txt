Large Continuous Data Sets
==========================

ASDF can also be used for large continuous data sets. This tutorial
demonstrates it with one year of data from the Bavarian seismic network, all of
which will be stored in a single ASDF file.

When designing an application around ASDF please take a couple of minutes to
think about the granularity at which you want to store the data. It would be
possible to store all waveforms for a single day, or a month, or even several
years from multiple stations, all in the same ASDF file. The best granularity
depends on the specific application. Very large files might be a bit easier to
work with (less bookkeeping) but are harder to move around, backup, and more
prone to corruption through bit rot.

We start with a folder full of MiniSEED files, one per day.

.. code-block:: bash


    $ du -sh EHZ.D
    7.3G    EHZ.D

    $ ls EHZ.D | wc -l
    365


A simple script writes everything into a single ASDF file. `tqdm
<https://pypi.python.org/pypi/tqdm>`_ (install with ``pip install tqdm``) is a
very simple way to add a progress bar to a loop. Feel free to leave it out.

.. code-block:: bash

    import glob
    from tqdm import tqdm
    from pyasdf import ASDFDataSet

    ds = ASDFDataSet("out_gzip3.h5", compression="gzip-3")

    for filename in tqdm(glob.glob("./EHZ.D/BW.ALTM.*")):
        ds.append_waveforms(filename, tag="raw_recording")


Note that this script uses the
:meth:`~pyasdf.asdf_data_set.ASDFDataSet.append_waveforms` method. This will,
assuming two files are exactly adjacent (last sample of first file and first
sample of next file are exactly one delta apart), store both in a single, large
array which might be advantageous for some access patterns. It can be
completely replaced by the
:meth:`~pyasdf.asdf_data_set.ASDFDataSet.add_waveforms` method which will not
do the merging. A case by case choice has to be made to determine which is more
suitable for any given application.

Also note that compression can have quite a strong effect but the writing and
reading speed will naturally decrease. The compression option can be set when
initializing the ASDF file. Here are some results for this particular data set:

.. code-block:: bash

    du -sh *.h5
    4.8G    out_gzip3.h5
    23G     out_uncompressed.h5

The compression is even more efficient than the original STEIM encoding. The
uncompressed version is faster to create though (of course this, like all
benchmarks, depends on a large number of factors).


The normal access pattern for ASDF breaks a bit here as it would potentially
load a lot of data into memory at one go. ``pyasdf`` prevents accidental use
when trying to load a full year into memory at once:


.. code-block:: python

    >>> ds.waveforms.BW_ALTM.raw_recording
    ---------------------------------------------------------------------------
    ASDFValueError                            Traceback (most recent call last)

    ...

    ASDFValueError: All waveforms for station 'BW.ALTM' and item 'raw_recording'
    would require '23897.74 MB of memory. The current limit is 1024.00 MB. Adjust
    by setting 'ASDFDataSet.single_item_read_limit_in_mb' or use a different
    method to read the waveform data.

Thus either adjust the setting as suggested, or, much better, use a different
method to access the data:

.. code-block:: python

    >>> ds.get_waveforms(network="BW", station="ALTM", location="", channel="EHZ",
                         starttime=obspy.UTCDateTime(2015, 3, 11),
                         endtime=obspy.UTCDateTime(2015, 3, 12), tag="raw_recording")
    1 Trace(s) in Stream:
    BW.ALTM..EHZ | 2015-03-10T23:59:59... - 2015-03-12T00:00:00... | 200.0 Hz, 17280002 samples


This will only extract sample at exactly the required times making it very
suitable for very big data arrays.

Another example of this is to write a simple loop over every hour in the year:


.. code-block:: python

    starttime = obspy.UTCDateTime(2015, 1, 1)
    endtime = obspy.UTCDateTime(2016, 1, 1)

    while starttime + 3600 < endtime:
        st = ds.get_waveforms(network="BW", station="ALTM",
                              location="", channel="EHZ",
                              starttime=starttime, endtime=starttime + 3600,
                              tag="raw_recording")
        print(st)
        starttime += 3600
