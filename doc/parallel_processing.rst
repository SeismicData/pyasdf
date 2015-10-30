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
