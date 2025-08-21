Deleting Pieces of a Data Set
=============================

*HDF5* files are per-se mutable. ``pyadsf`` currently allows the deletion of
any piece of information in an ``ASDF`` file. Thus overwriting arrays
currently only works as a combination of deleting and writing again. Please
contact the developers if you require additional functionalities.

Deleting Things
---------------

Almost anything can be deleted by using the ``del`` operator. The following
code snippets illustrates the different possibilities to delete data. Please
be careful - **this directly modifies the file and is not revertible**.

.. code-block:: python

    import pyasdf

    with pyasdf.ASDFDataSet("example.h5") as ds:
        # Delete all events.
        del ds.events
        # Delete all information from a particular station.
        del ds.waveforms.BW_RJOB
        del ds.waveforms["BW.RJOB"]
        # Delete all waveforms with a certain tag from a particular station.
        del ds.waveforms.BW_RJOB.example
        del ds.waveforms["BW.RJOB"]["example"]
        # Delete the StationXML file for a certain station.
        del ds.waveforms.BW_RJOB.StationXML
        del ds.waveforms["BW.RJOB"]["StationXML"]
        # Directly delete a certain piece of waveform information.
        del ds.waveforms["BW.RJOB"][
            "BW.RJOB..EHE__2009-08-24T00:20:03__2009-08-24T00:20:32__example"]
        # Delete a provenance document.
        del ds.provenance.example_document
        del ds.provenance["example_document"]
        # Delete an auxiliary data group.
        del ds.auxiliary_data.RandomArrays
        del ds.auxiliary_data["RandomArrays"]
        # Delete a certain piece of auxiliary data.
        del ds.auxiliary_data.RandomArrays.array_a
        del ds.auxiliary_data["RandomArrays"]["array_a"]
        # Also works with nested paths.
        del ds.auxiliary_data.RandomArrays.nested.path.array_a
        del ds.auxiliary_data["RandomArrays"]["nested"]["path"]["array_a"]


Freeing Space
-------------

Deleting data sets or groups within an *HDF5* file does in general not
physically delete the data from the file. It just removes the item from the
index. To actually regain the now not needed space use the ``h5repack``
program that ships with *HDF5*.


Assuming a file has been created with the following code snippet:

.. code-block:: python

    import pyasdf

    with pyasdf.ASDFDataSet("example.h5") as ds:
        ds.add_waveforms(..., tag="example")

Current file size:

.. code-block:: bash

    $ ls -l example.h5
    -rw-r--r--  1 lion  staff  144424 Jan 19 15:47 example.h5


Delete some waveform data.

.. code-block:: python

    import pyasdf

    with pyasdf.ASDFDataSet("example.h5") as ds:
        del ds.waveforms.BW_RJOB


The physical space is only regained after ``h5repack`` is used.

.. code-block:: bash

    $ ls -l example.h5
    -rw-r--r--  1 lion  staff  144424 Jan 19 15:47 example.h5

    $ h5repack example.h5 example_repacked.h5

    $ ls -l example*
    -rw-r--r--  1 lion  staff  144424 Jan 19 15:47 example.h5
    -rw-r--r--  1 lion  staff   20092 Jan 19 15:48 example_repacked.h5




