#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2014
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import collections
import io
import os
import sys
import time
import warnings
import weakref

# Py2k/3k compat.
try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

from lxml.etree import QName
import numpy as np
import obspy

from .exceptions import (WaveformNotInFileException, NoStationXMLForStation,
                         ASDFValueError, ASDFAttributeError)
from .header import MSG_TAGS
from .inventory_utils import get_coordinates

# Tuple holding a the body of a received message.
ReceivedMessage = collections.namedtuple("ReceivedMessage", ["data"])
# Tuple denoting a single worker.
Worker = collections.namedtuple("Worker", ["active_jobs",
                                           "completed_jobs_count"])


def get_multiprocessing():
    """
    Helper function returning the multiprocessing module or the threading
    version of it.
    """
    if is_multiprocessing_problematic():
        msg = ("NumPy linked against 'Accelerate.framework'. Multiprocessing "
               "will be disabled. See "
               "https://github.com/obspy/obspy/wiki/Notes-on-Parallel-"
               "Processing-with-Python-and-ObsPy for more information.")
        warnings.warn(msg)
        # Disable by replacing with dummy implementation using threads.
        import multiprocessing as mp
        from multiprocessing import dummy  # NOQA
        multiprocessing = dummy
        multiprocessing.cpu_count = mp.cpu_count
    else:
        import multiprocessing  # NOQA
    return multiprocessing


def is_multiprocessing_problematic():
    """
    Return True if multiprocessing is known to have issues on the given
    platform.

    Mainly results from the fact that some BLAS/LAPACK implementations
    cannot deal with forked processing.
    """
    # Handling numpy linked against accelerate.
    config_info = str([value for key, value in
                       np.__config__.__dict__.items()
                       if key.endswith("_info")]).lower()

    if "accelerate" in config_info or "veclib" in config_info:
        return True
    elif "openblas" in config_info:
        # Most openBLAS can only operate with one thread...
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    else:
        return False


def sizeof_fmt(num):
    """
    Handy formatting for human readable filesize.

    From http://stackoverflow.com/a/1094933/1657047
    """
    for x in ["bytes", "KB", "MB", "GB"]:
        if num < 1024.0 and num > -1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0
    return "%3.1f %s" % (num, "TB")


class ProvenanceAccessor(object):
    """
    Accessor helper for the provenance records.
    """
    def __init__(self, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__data_set = weakref.ref(asdf_data_set)

    def __getattr__(self, item):
        _records = self.__data_set()._provenance_group
        if item not in _records:
            raise AttributeError
        return self.__data_set().get_provenance_document(item)

    def __dir__(self):
        return sorted((self.__data_set()._provenance_group.keys()))

    def __len__(self):
        return len(self.__dir__())

    def __iter__(self):
        for _i in self.__dir__():
            yield getattr(self, _i)

    def __str__(self):
        if not len(self):
            return "No provenance document in file."
        ret_str = "%i Provenance Document(s):\n\t%s" % (
            len(self), "\n\t".join(dir(self)))
        return ret_str

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())


class AuxiliaryDataContainer(object):
    def __init__(self, data, data_type, tag, parameters):
        self.data = data
        self.data_type = data_type
        self.tag = tag
        if "provenance_id" in parameters:
            parameters = copy.deepcopy(parameters)
            self.provenance_id = parameters.pop("provenance_id")
        else:
            self.provenance_id = None
        self.parameters = parameters

        self.__file_cache = None

    def __del__(self):
        try:
            self.__file_cache.close()
        except:
            pass

    @property
    def file(self):
        if self.data_type != "File":
            raise ASDFAttributeError(
                "The 'file' property is only available "
                "for auxiliary data with the data type 'File'.")
        if self.__file_cache is not None:
            return self.__file_cache

        self.__file_cache = io.BytesIO(self.data.value.tostring())
        return self.__file_cache

    def __str__(self):
        return (
            "Auxiliary Data of Type '{data_type}'\n"
            "\tTag: '{tag}'\n"
            "{provenance}"
            "\tData shape: '{data_shape}', dtype: '{dtype}'\n"
            "\tParameters:\n\t\t{parameters}"
            .format(data_type=self.data_type, data_shape=self.data.shape,
                    dtype=self.data.dtype, tag=self.tag,
                    provenance="" if self.provenance_id is None else
                    "\tProvenance ID: '%s'\n" % self.provenance_id,
                    parameters="\n\t\t".join([
                        "%s: %s" % (_i[0], _i[1]) for _i in
                        sorted(self.parameters.items(), key=lambda x: x[0])])))


class AuxiliaryDataAccessor(object):
    """
    Helper class facilitating access to the actual waveforms and stations.
    """
    def __init__(self, auxiliary_data_type, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__auxiliary_data_type = auxiliary_data_type
        self.__data_set = weakref.ref(asdf_data_set)

    def __getattr__(self, item):
        return self.__data_set()._get_auxiliary_data(
            self.__auxiliary_data_type, item.replace("___", "."))

    def __dir__(self):
        __group = self.__data_set()._auxiliary_data_group[
            self.__auxiliary_data_type]
        return sorted([_i.replace(".", "___") for _i in __group.keys()])


class AuxiliaryDataGroupAccessor(object):
    """
    Helper class to facilitate access to the auxiliary data types.
    """
    def __init__(self, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__data_set = weakref.ref(asdf_data_set)

    def __getattr__(self, item):
        __auxiliary_data_group = self.__data_set()._auxiliary_data_group
        if item not in __auxiliary_data_group:
            raise AttributeError
        return AuxiliaryDataAccessor(item, self.__data_set())

    def __dir__(self):
        __auxiliary_group = self.__data_set()._auxiliary_data_group
        return sorted(__auxiliary_group.keys())

    def __len__(self):
        return len(self.__dir__())


class StationAccessor(object):
    """
    Helper class to facilitate access to the waveforms and stations.
    """
    def __init__(self, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__data_set = weakref.ref(asdf_data_set)

    def __getattr__(self, item):
        __waveforms = self.__data_set()._waveform_group
        if item.replace("_", ".") not in __waveforms:
            raise AttributeError
        return WaveformAccessor(item.replace("_", "."), self.__data_set())

    def __dir__(self):
        __waveforms = self.__data_set()._waveform_group
        return sorted(set(
            [_i.replace(".", "_") for _i in __waveforms.keys()]))

    def __len__(self):
        return len(self.__dir__())

    def __iter__(self):
        for _i in self.__dir__():
            yield getattr(self, _i.replace("_", "."))


class WaveformAccessor(object):
    """
    Helper class facilitating access to the actual waveforms and stations.
    """
    def __init__(self, station_name, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self._station_name = station_name
        self.__data_set = weakref.ref(asdf_data_set)

    @property
    def coordinates(self):
        """
        Get coordinates of the station if any.
        """
        coords = self.__get_coordinates(level="station")
        if self._station_name not in coords:
            raise ASDFValueError("StationXML file has no coordinates for "
                                 "station '%s'." % self._station_name)
        return coords[self._station_name]

    @property
    def channel_coordinates(self):
        """
        Get coordinates of the station at the channel level if any.
        """
        coords = self.__get_coordinates(level="channel")
        # Filter to only keep channels with the current station name.
        coords = {key: value for key, value in coords.items()
                  if key.startswith(self._station_name + ".")}
        if not(coords):
            raise ASDFValueError("StationXML file has no coordinates at "
                                 "the channel level for station '%s'." %
                                 self._station_name)
        return coords

    def __get_coordinates(self, level):
        """
        Helper function.
        """
        station = self.__data_set()._waveform_group[self._station_name]
        if "StationXML" not in station:
            raise NoStationXMLForStation("Station '%s' has no StationXML "
                                         "file." % self._station_name)
        try:
            with io.BytesIO(station["StationXML"].value.tostring()) as buf:
                coordinates = get_coordinates(buf, level=level)
        finally:
            # HDF5 reference are tricky...
            del station

        return coordinates

    def __getattr__(self, item):
        if item != "StationXML":
            __station = self.__data_set()._waveform_group[self._station_name]
            keys = [_i for _i in __station.keys()
                    if _i.endswith("__" + item)]

            if not keys:
                # Important as __del__() for the waveform group is otherwise
                # not always called.
                del __station
                raise WaveformNotInFileException(
                    "Tag '%s' not part of the data set for station '%s'." % (
                        item, self._station_name))

            traces = [self.__data_set()._get_waveform(_i) for _i in keys]
            return obspy.Stream(traces=traces)
        else:
            return self.__data_set()._get_station(self._station_name)

    def __dir__(self):
        __station = self.__data_set()._waveform_group[self._station_name]
        directory = ["_station_name", "coordinates", "channel_coordinates"]
        if "StationXML" in __station:
            directory.append("StationXML")
        directory.extend([_i.split("__")[-1]
                          for _i in __station.keys()
                          if _i != "StationXML"])
        return sorted(set(directory))

    def __str__(self):
        contents = self.__dir__()
        waveform_contents = [_i for _i in contents if _i not in [
                             "StationXML", "_station_name", "coordinates",
                             "channel_coordinates"]]

        ret_str = (
            "Contents of the data set for station {station}:\n"
            "    - {station_xml}\n"
            "    - {count} Waveform Tag(s):\n"
            "         {waveforms}"
        )
        return ret_str.format(
            station=self._station_name,
            station_xml="Has a StationXML file" if "StationXML" in contents
            else "Has no StationXML file",
            count=len(waveform_contents),
            waveforms="\n        ".join(waveform_contents)
        )

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())


def is_mpi_env():
    """
    Returns True if the current environment is an MPI environment.
    """
    try:
        import mpi4py
    except ImportError:
        return False

    try:
        import mpi4py.MPI
    except ImportError:
        return False

    if mpi4py.MPI.COMM_WORLD.size == 1 and mpi4py.MPI.COMM_WORLD.rank == 0:
        return False
    return True


class StreamBuffer(collections.MutableMapping):
    """
    Very simple key value store for obspy stream object with the additional
    ability to approximate the size of all stored stream objects.
    """
    def __init__(self):
        self.__streams = {}

    def __getitem__(self, key):
        return self.__streams[key]

    def __setitem__(self, key, value):
        if not isinstance(value, obspy.Stream):
            raise TypeError
        self.__streams[key] = value

    def __delitem__(self, key):
        del self.__streams[key]

    def keys(self):
        return self.__streams.keys()

    def __len__(self):
        return len(self.__streams)

    def __iter__(self):
        return iter(self.__streams)

    def get_size(self):
        """
        Try to approximate the size of all stores Stream object.
        """
        cum_size = 0
        for stream in self.__streams.values():
            cum_size += sys.getsizeof(stream)
            for trace in stream:
                cum_size += sys.getsizeof(trace)
                cum_size += sys.getsizeof(trace.stats)
                cum_size += sys.getsizeof(trace.stats.__dict__)
                cum_size += sys.getsizeof(trace.data)
                cum_size += trace.data.nbytes
        # Add one percent buffer just in case.
        return cum_size * 1.01


# Two objects describing a job and a worker.
class Job(object):
    __slots__ = "arguments", "result"

    def __init__(self, arguments, result=None):
        self.arguments = arguments
        self.result = result

    def __repr__(self):
        return "Job(arguments=%s, result=%s)" % (str(self.arguments),
                                                 str(self.result))


class JobQueueHelper(object):
    """
    A simple helper class managing job distribution to workers.
    """
    def __init__(self, jobs, worker_names):
        """
        Init with a list of jobs and a list of workers.

        :type jobs: List of arguments distributed to the jobs.
        :param jobs: A list of jobs that will be distributed to the workers.
        :type: list of integers
        :param workers: A list of usually integers, each denoting a worker.
        """
        self._all_jobs = [Job(_i) for _i in jobs]
        self._in_queue = self._all_jobs[:]
        self._finished_jobs = []
        self._poison_pills_received = 0

        self._workers = {_i: Worker([], [0]) for _i in worker_names}

        self._starttime = time.time()

    def poison_pill_received(self):
        """
        Increment the point pills received counter.
        """
        self._poison_pills_received += 1

    def get_job_for_worker(self, worker_name):
        """
        Get a job for a worker.

        :param worker_name: The name of the worker requesting work.
        """
        job = self._in_queue.pop(0)
        self._workers[worker_name].active_jobs.append(job)
        return job.arguments

    def received_job_from_worker(self, arguments, result, worker_name):
        """
        Call when a worker returned a job.

        :param arguments: The arguments the jobs was called with.
        :param result: The result of the job
        :param worker_name: The name of the worker.
        """
        # Find the correct job.
        job = [_i for _i in self._workers[worker_name].active_jobs
               if _i.arguments == arguments]
        if len(job) == 0:
            msg = ("MASTER: Job %s from worker %i not found. All jobs: %s\n" %
                   (str(arguments), worker_name,
                    str(self._workers[worker_name].active_jobs)))
            raise ValueError(msg)
        if len(job) > 1:
            raise ValueError("WTF %i %s %s" % (
                worker_name, str(arguments),
                str(self._workers[worker_name].active_jobs)))
        job = job[0]
        job.result = result

        self._workers[worker_name].active_jobs.remove(job)
        self._workers[worker_name].completed_jobs_count[0] += 1
        self._finished_jobs.append(job)

    def __str__(self):
        workers = "\n\t".join([
            "Worker %s: %i active, %i completed jobs" %
            (str(key), len(value.active_jobs), value.completed_jobs_count[0])
            for key, value in self._workers.items()])

        return (
            "Jobs (running %.2f seconds): "
            "queued: %i | finished: %i | total: %i\n"
            "\t%s\n" % (time.time() - self._starttime, len(self._in_queue),
                        len(self._finished_jobs), len(self._all_jobs),
                        workers))

    @property
    def queue_empty(self):
        return not bool(self._in_queue)

    @property
    def finished(self):
        return len(self._finished_jobs)

    @property
    def all_done(self):
        return len(self._all_jobs) == len(self._finished_jobs)

    @property
    def all_poison_pills_received(self):
        return len(self._workers) == self._poison_pills_received


def pretty_sender_log(rank, destination, tag, payload):
    import colorama
    prefix = colorama.Fore.RED + "sent to      " + colorama.Fore.RESET
    _pretty_log(prefix, destination, rank, tag, payload)


def pretty_receiver_log(source, rank, tag, payload):
    import colorama
    prefix = colorama.Fore.GREEN + "received from" + colorama.Fore.RESET
    _pretty_log(prefix, rank, source, tag, payload)


def _pretty_log(prefix, first, second, tag, payload):
    import colorama

    colors = (colorama.Back.WHITE + colorama.Fore.MAGENTA,
              colorama.Back.WHITE + colorama.Fore.BLUE,
              colorama.Back.WHITE + colorama.Fore.GREEN,
              colorama.Back.WHITE + colorama.Fore.YELLOW,
              colorama.Back.WHITE + colorama.Fore.BLACK,
              colorama.Back.WHITE + colorama.Fore.RED,
              colorama.Back.WHITE + colorama.Fore.CYAN)

    tag_colors = (
        colorama.Fore.RED,
        colorama.Fore.GREEN,
        colorama.Fore.BLUE,
        colorama.Fore.YELLOW,
        colorama.Fore.MAGENTA,
    )

    # Deterministic colors also on Python 3.
    msg_tag_keys = sorted(MSG_TAGS.keys(), key=lambda x: str(x))
    tags = [i for i in msg_tag_keys if isinstance(i, (str, bytes))]

    tag = MSG_TAGS[tag]
    tag = tag_colors[tags.index(tag) % len(tag_colors)] + tag + \
        colorama.Style.RESET_ALL

    first = colorama.Fore.YELLOW + "MASTER  " + colorama.Fore.RESET \
        if first == 0 else colors[first % len(colors)] + \
        ("WORKER %i" % first) + colorama.Style.RESET_ALL
    second = colorama.Fore.YELLOW + "MASTER  " + colorama.Fore.RESET \
        if second == 0 else colors[second % len(colors)] + \
        ("WORKER %i" % second) + colorama.Style.RESET_ALL

    print("%s %s %s [%s] -- %s" % (first, prefix, second, tag, str(payload)))


def split_qualified_name(name):
    """
    Takes a qualified name and returns a tuple of namespace, localpart.

    If namespace is not a valid URL, an error will be raised.

    :param name: The qualified name.
    """
    try:
        q_name = QName(name)
    except ValueError:
        raise ASDFValueError("Not a valid qualified name.")
    url, localname = q_name.namespace, q_name.localname
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ASDFValueError("Not a valid qualified name.")
    return url, localname
