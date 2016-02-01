#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Processing utilities for ASDF files.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2016
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import traceback

from . import ASDFException


def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage
    depending on the use case.
    """
    return [container[_i::count] for _i in range(count)]


def process_each_with_each(process_function, ds_a, ds_b, traceback_limit=3):
    """
    Process each station with each other station across two data sets.

    It must be called with MPI and at least two cores. It is useful for many
    kinds of cross-correlation and double-difference applications.

    Please keep in mind that it distributes the work in a very simplistic
    manner and that the load balancing is far from optimal. If you require a
    more optimized scheme either implement it yourself or talk to the authors
    of this Python package.

    :param process_function: A callback function taking three arguments:
        * ``this_station_ds_a``: A station group for one station in data set A.
        * ``this_station_ds_b``: Station group for the same station in data
            set B.
        * ``other_stations_iterator``: An iterator yielding station sets for
            all other stations.
        This function will be called for all stations that are common across
        both data sets.
    :type process_function: function
    :param ds_a: One data set.
    :type ds_a: :class:`pyasdf.asdf_data_set.ASDFDataSet`
    :param ds_b: Another data set.
    :type ds_b: :class:`pyasdf.asdf_data_set.ASDFDataSet`
    :param traceback_limit: If an error occurs within the given callback
        function a traceback will be printed. Set the maximum size of it here.
    :type traceback_limit: int
    """
    if not ds_a.mpi:
        raise ASDFException("Currently only works with MPI.")

    mpi = ds_a.mpi

    # Collect the work that needs to be done on rank 0.
    if mpi.comm.rank == 0:

        this_stations = set(ds_a.waveforms.list())
        other_stations = set(ds_b.waveforms.list())

        # Usable stations are those that are part of both.
        usable_stations = list(this_stations.intersection(other_stations))
        all_stations = usable_stations
        stations_for_this_rank = split(usable_stations, mpi.comm.size)
    else:
        all_stations = None
        stations_for_this_rank = None

    # Scatter jobs. Each rank will now have a certain amount of stations.
    stations_for_this_rank = mpi.comm.scatter(stations_for_this_rank,
                                              root=0)
    # But each rank will also know of all other stations.
    all_stations = mpi.comm.bcast(all_stations, root=0)

    mpi.comm.barrier()

    results = {}

    for _i, station in enumerate(stations_for_this_rank):
        if mpi.rank == 0:
            print(" -> Processing approximately task %i of %i ..." % (
                (_i * mpi.size + 1), len(all_stations)))

        # Create a list of all stations, except this one.
        other_stations = list(set(all_stations).difference({station}))

        def station_iterator(stations):
            for _i in stations:
                yield ds_a.waveforms[_i], ds_b.waveforms[_i]

        try:
            result = process_function(
                getattr(ds_a.waveforms, station),
                getattr(ds_b.waveforms, station),
                station_iterator(other_stations))
        except Exception:
            # If an exception is raised print a good error message
            # and traceback to help diagnose the issue.
            msg = ("\nError during the processing of station '%s' "
                   "on rank %i:" % (station, mpi.rank))

            # Extract traceback from the exception.
            exc_info = sys.exc_info()
            stack = traceback.extract_stack(
                    limit=traceback_limit)
            tb = traceback.extract_tb(exc_info[2])
            full_tb = stack[:-1] + tb
            exc_line = traceback.format_exception_only(
                    *exc_info[:2])
            tb = ("Traceback (At max %i levels - most recent call "
                  "last):\n" % traceback_limit)
            tb += "".join(traceback.format_list(full_tb))
            tb += "\n"
            tb += "".join(exc_line)

            # These potentially keep references to the HDF5 file
            # which in some obscure way and likely due to
            # interference with internal HDF5 and Python references
            # prevents it from getting garbage collected. We
            # explicitly delete them here and MPI can finalize
            # afterwards.
            del exc_info
            del stack

            print(msg)
            print(tb)
        else:
            results[station] = result

    # Gather and create a final dictionary of results.
    gathered_results = mpi.comm.gather(results, root=0)

    results = {}
    if mpi.rank == 0:
        for result in gathered_results:
            results.update(result)

    # Likely not necessary as the gather two line above implies a
    # barrier but better be safe than sorry.
    mpi.comm.barrier()

    return results
