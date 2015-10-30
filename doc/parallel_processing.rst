Parallel Processing
===================

An explicit design goal of the ``ASDF`` format has been to enable
high-performance and parallel processing. ``pyasdf`` aids in that regard by
providing a couple of different functions. If you require functions that
operate in different patterns please open a Github issue and we can discuss it.


process() method
----------------

The :meth:`~pyasdf.asdf_data_set.ASDFDataSet.process` method is conceptually
very simple: It applies a function to each station in one data set and writes
the results to a new data set. It can run in parallel with MPI. In that case
parallel I/O will be used if available. Alternatively it can be run in parallel
without MPI on shared memory architectures but in that case I/O is not parallel
and only one process will write at any point in time.

A simple example best illustrates how to use it.


.. code-block:: python

    import pyasdf

    # The first step is to define a function. You can do whatever you want in
    # that function to process the data.
    def process(st, inv):
        # st is an ObsPy stream object with a certain tag and inv is an ObsPy
        # inventory object which might be None.

        # This example will not write anything for data which has no
        # corresponding inventory information.
        if inv is None:
            return

        # This very simple example will deconvolve the instrument response.
        st.attach_response(inv)
        st.remove_response(output="VEL")

        # The new file will contain whatever is returned here. If nothing is
        # returned, nothing will be written. In case something is returned, the
        # inventory data will also be copied from the input to the output file.
        return st


    # Open an existing data set. Setting the mode is not strictly necessary but
    # might improve the speed a bit.

    # Make sure to either use a with statement or delete the reference to the
    # data set object at the end. Otherwise it might not be able to properly
    # close the file which will stall MPI.
    with pyasdf.ASDFDataSet("example.h5", mode="r") as ds:
        ds.process(
            # Pass the processing function here.
            process_function=process,
            # The output filename. Must not yet exist.
            output_filename="processed_example.h5",
            # Maps the tags. The keys are the tags in the input file and traces
            # with that tag will end up in the output file with the corresponding
            # value.
            # Also note that only data with tags present in this map will be
            # processed. Others will be ignored.
            tag_map={"raw_recording": "processed"})


Run it with MPI
^^^^^^^^^^^^^^^

Save the above example to file and run it with MPI. Make sure ``h5py`` has been
compiled with parallel I/O support. Consult the :doc:`installation` document on
how do that and making sure it works.

.. code-block:: bash

    $ mpirun -n 64 python script.py

As soon as it is run with more then 1 core and an MPI communicator is available
``pyasdf`` will recognize its being called with MPI and chose the correct
internal functions.


Run it with multiprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If MPI is not available ``pyasdf`` will attempt to run the processing in
parallel using Python's native ``multiprocessing`` module. In that case only
one core will be able to write at any given point in time: the cores will take
turns. Nonetheless this speeds up things quite a bit if your processing is
reasonably heavy.

.. code-block:: bash

    $ python script.py


It will attempt to use all available cores by default. Additional parameters
available for processing with multiprocessing:

.. code-block:: python

    ...

    ds.process(...,
               # The length of the traceback shown if an error is raised during
               # the processing of a trace. Defaults to 3.
               traceback_limit=10,
               # The number of cores to run it on. Defaults to -1 which is
               # equal to the number of cores on your system.
               cpu_count=5)

process_two_files_without_parallel_output() method
--------------------------------------------------

The
:meth:`~pyasdf.asdf_data_set.ASDFDataSet.process_two_files_without_parallel_output`
is useful to compare data in two separate data sets for example for misfit or
window selection procedures. One once again has to provide a function which
will be called for each station that is common in both data sets. The results
are collected in memory and gathered on rank 0.

The function can currently only be run with MPI and is best explained by a
short example. The following script will calculate the squared sum for all
traces at each station in both data sets.

.. code-block:: python

    from mpi4py import MPI
    import pyasdf

    # Open two data sets. Setting the mode is optional but might speed up things a
    # bit.
    ds_1 = pyasdf.ASDFDataSet("/Users/lion/asdf_example.h5", mode="r")
    ds_2 = pyasdf.ASDFDataSet("/Users/lion/asdf_example.h5", mode="r")


    # The function takes two station groups which contain waveform and inventory
    # information for the same station in both data sets. The function will only be
    # called for stations that are available in both data sets. Keep in mind that
    # each station can contain data from an arbitrary number of tags and can also
    # contain inventory information.
    def process(s_group_1, s_group_2):
        energy_ds_1 = 0
        energy_ds_2 = 0

        # Get energy for data in data set one. Don't deal with the sampling rate
        # for now...
        for tag in s_group_1.get_waveform_tags():
            for tr in s_group_1[tag]:
                energy_ds_1 += (tr.data ** 2).sum()

        # Do the same for the other group.
        for tag in s_group_2.get_waveform_tags():
            for tr in s_group_2[tag]:
                energy_ds_2 += (tr.data ** 2).sum()

        # Just return what you want to collect. Make sure this is not too big as it
        # will be stored in memory and send over MPI to the process with rank 0.
        return {
            "energy_ds_1": energy_ds_1,
            "energy_ds_2": energy_ds_2}

    # Launch it.
    results = ds_1.process_two_files_without_parallel_output(ds_2, process)

    # Results are available on rank 0.
    if MPI.COMM_WORLD.rank == 0:
        print(results)

Save to a file an run with

.. code-block:: bash

    $ mpirun -n 64 python script.py

The result is something akin to


.. code-block:: python

    {'AF.CVNA': {'energy_ds_1': 9.228861452825754e-09,
                 'energy_ds_2': 9.228861452825754e-09},
     'AF.DODT': {'energy_ds_1': 4.879421311443366e-09,
                 'energy_ds_2': 4.879421311443366e-09},
     'AF.EKNA': {'energy_ds_1': 3.5441928281088053e-09,
                 'energy_ds_2': 3.5441928281088053e-09},
     'AF.GRM': {'energy_ds_1': 2.5369817358011915e-08,
                'energy_ds_2': 2.5369817358011915e-08},
     ...}


