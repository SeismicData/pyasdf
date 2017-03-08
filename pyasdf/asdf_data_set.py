#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python implementation of the Adaptable Seismic Data Format (ASDF).

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Import ObsPy first as import h5py on some machines will some reset paths
# and lxml cannot be loaded anymore afterwards...
import obspy

import collections
import copy
import io
import itertools
import math
import multiprocessing
import os
import re
import sys
import time
import traceback
import uuid
import warnings

import dill
import h5py
import lxml.etree
import numpy as np
import prov
import prov.model


# Minimum compatibility wrapper between Python 2 and 3.
try:
    filter = itertools.ifilter
except AttributeError:
    # Python 3 is a bit more aggressive when buffering warnings but here it
    # is fairly important that they are shown, thus we monkey-patch it to
    # flush stderr afterwards.
    def get_warning_fct():
        closure_warn = warnings.warn

        def __warn(self, *args, **kwargs):
            closure_warn(self, *args, **kwargs)
            sys.stderr.flush()

        return __warn

    warnings.warn = get_warning_fct()


from .exceptions import ASDFException, ASDFWarning, ASDFValueError, \
    NoStationXMLForStation
from .header import COMPRESSIONS, FORMAT_NAME, \
    FORMAT_VERSION, MSG_TAGS, MAX_MEMORY_PER_WORKER_IN_MB, POISON_PILL, \
    PROV_FILENAME_REGEX, TAG_REGEX, VALID_SEISMOGRAM_DTYPES
from .query import Query, merge_query_functions
from .utils import is_mpi_env, StationAccessor, sizeof_fmt, ReceivedMessage,\
    pretty_receiver_log, pretty_sender_log, JobQueueHelper, StreamBuffer, \
    AuxiliaryDataGroupAccessor, AuxiliaryDataContainer, get_multiprocessing, \
    ProvenanceAccessor, split_qualified_name, _read_string_array, \
    FilteredWaveformAccessor, label2string, labelstring2list, \
    AuxiliaryDataAccessor, wf_name2tag, to_list_of_resource_identifiers
from .inventory_utils import isolate_and_merge_station, merge_inventories


class ASDFDataSet(object):
    """
    Object dealing with Adaptable Seismic Data Format (ASDF).

    Central object of this Python package.
    """
    q = Query()

    def __init__(self, filename, compression="gzip-3", shuffle=True,
                 debug=False, mpi=None, mode="a",
                 single_item_read_limit_in_mb=1024.0):
        """
        :type filename: str
        :param filename: The filename of the HDF5 file (to be).
        :type compression: str
        :param compression: The compression to use. Defaults to
            ``"gzip-3"`` which yielded good results in the past. Will
            only be applied to newly created data sets. Existing ones are not
            touched. Using parallel I/O will also disable compression as it
            is not possible to use both at the same time.

            **Available compressions choices (all of them are lossless):**

            * ``None``: No compression
            * ``"gzip-0"`` - ``"gzip-9"``: Gzip compression level 0  (worst
              but fast) to 9 (best but slow)
            * ``"lzf"``: LZF compression
            * ``"szip-ec-8"``: szip compression
            * ``"szip-ec-10"``: szip compression
            * ``"szip-nn-8"``: szip compression
            * ``"szip-nn-10"``: szip compression

        :type shuffle: bool
        :param shuffle: Turn the shuffle filter on or off. Turning it on
            oftentimes increases the compression ratio at the expense of
            some speed.
        :type debug: bool
        :param debug: If True, print debug messages. Potentially very verbose.
        :param mpi: Force MPI on/off. Don't touch this unless you have a
            reason.
        :type mpi: bool
        :param mode: The mode the file is opened in. Passed to the
            underlying :class:`h5py.File` constructor. pyasdf expects to be
            able to write to files for many operations so it might result in
            strange errors if a read-only mode is used. Nonetheless this is
            quite useful for some use cases as long as one is aware of the
            potential repercussions.
        :type mode: str
        :type single_item_read_limit_in_mb: float
        :param single_item_read_limit_in_mb: This limits the amount of waveform
            data that can be read with a simple attribute or dictionary
            access. Some ASDF files can get very big and this raises an
            error if one tries to access more then the specified value. This
            is mainly to guard against accidentally filling ones memory on
            the interactive command line when just exploring an ASDF data
            set. There are other ways to still access data and even this
            setting can be overwritten.
        """
        self.__force_mpi = mpi
        self.debug = debug

        # The limit on how much data can be read with a single item access.
        self.single_item_read_limit_in_mb = single_item_read_limit_in_mb

        # Deal with compression settings.
        if compression not in COMPRESSIONS:
            msg = "Unknown compressions '%s'. Available compressions: \n\t%s" \
                % (compression, "\n\t".join(sorted(
                    [str(i) for i in COMPRESSIONS.keys()])))
            raise Exception(msg)

        self.__compression = COMPRESSIONS[compression]
        self.__shuffle = shuffle
        # Turn off compression for parallel I/O. Any already written
        # compressed data will be fine. Don't need to raise it if file is
        # opened in read-only mode.
        if self.__compression[0] and self.mpi and mode != "r":
            msg = "Compression will be disabled as parallel HDF5 does not " \
                  "support compression"
            warnings.warn(msg, ASDFWarning)
            self.__compression = COMPRESSIONS[None]
            self.__shuffle = False
        # No need to shuffle if no compression is used.
        elif self.__compression[0] is None:
            self.__shuffle = False

        # Open file or take an already open HDF5 file object.
        if not self.mpi:
            self.__file = h5py.File(filename, mode=mode)
        else:
            self.__file = h5py.File(filename, mode=mode, driver="mpio",
                                    comm=self.mpi.comm)

        # Workaround to HDF5 only storing the relative path by default.
        self.__original_filename = os.path.abspath(filename)

        # Write file format and version information to the file.
        if "file_format" in self.__file.attrs:
            if self.__file.attrs["file_format"].decode() != FORMAT_NAME:
                msg = "Not a '%s' file." % FORMAT_NAME
                raise ASDFException(msg)
            if "file_format_version" not in self.__file.attrs:
                msg = ("No file format version given for file '%s'. The "
                       "program will continue but the result is undefined." %
                       self.filename)
                warnings.warn(msg, ASDFWarning)
            elif self.__file.attrs["file_format_version"].decode() != \
                    FORMAT_VERSION:
                msg = ("The file '%s' has version number '%s'. The reader "
                       "expects version '%s'. The program will continue but "
                       "the result is undefined." % (
                           self.filename,
                           self.__file.attrs["file_format_version"],
                           FORMAT_VERSION))
                warnings.warn(msg, ASDFWarning)
        else:
            self.__file.attrs["file_format"] = \
                self._zeropad_ascii_string(FORMAT_NAME)
            self.__file.attrs["file_format_version"] = \
                self._zeropad_ascii_string(FORMAT_VERSION)

        # Create the waveform and provenance groups.
        if "Waveforms" not in self.__file:
            self.__file.create_group("Waveforms")
        if "Provenance" not in self.__file:
            self.__file.create_group("Provenance")
        if "AuxiliaryData" not in self.__file:
            self.__file.create_group("AuxiliaryData")

        # Easy access to the waveforms.
        self.waveforms = StationAccessor(self)
        self.auxiliary_data = AuxiliaryDataGroupAccessor(self)
        self.provenance = ProvenanceAccessor(self)

        # Force synchronous init if run in an MPI environment.
        if self.mpi:
            self.mpi.comm.barrier()

    def __del__(self):
        """
        Cleanup. Force flushing and close the file.

        If called with MPI this will also enable MPI to cleanly shutdown.
        """
        try:
            self.flush()
            self._close()
        except (ValueError, TypeError, AttributeError):
            pass

    def __eq__(self, other):
        """
        More or less comprehensive equality check. Potentially quite slow as
        it checks all data.

        :type other:`~pyasdf.asdf_data_set.ASDFDDataSet`
        """
        if type(self) != type(other):
            return False
        if self._waveform_group.keys() != other._waveform_group.keys():
            return False
        if self._provenance_group.keys() != other._provenance_group.keys():
            return False
        if self.events != other.events:
            return False
        for station, group in self._waveform_group.items():
            other_group = other._waveform_group[station]
            for tag, data_set in group.items():
                other_data_set = other_group[tag]
                try:
                    if tag == "StationXML":
                        np.testing.assert_array_equal(data_set.value,
                                                      other_data_set.value)
                    else:
                        np.testing.assert_allclose(
                            data_set.value, other_data_set.value)
                except AssertionError:
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __enter__(self):
        """
        Enable usage as a context manager.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Enable usage as a context manager.
        """
        self.__del__()
        return False

    def __delattr__(self, key):
        # Only the events can be deleted.
        if key == "events":
            del self.__file["QuakeML"]
        # Otherwise try to get the item and if that succeed, raise an error
        # that it cannot be deleted.
        else:
            # Triggers an AttributeError if the attribute does not exist.
            getattr(self, key)
            raise AttributeError("Attribute '%s' cannot be deleted." % key)

    def flush(self):
        """
        Flush the underlying HDF5 file.
        """
        self.__file.flush()

    def _close(self):
        """
        Attempt to close the underlying HDF5 file.
        """
        try:
            self.__file.close()
        except:
            pass

    def _zeropad_ascii_string(self, text):
        """
        Returns a zero padded ASCII string in the most compatible way possible.

        Might later need to handle bytes/unicode.

        :param text: The text to be converted.
        """
        return np.string_((text + "\x00").encode())

    @property
    def _waveform_group(self):
        return self.__file["Waveforms"]

    @property
    def _provenance_group(self):
        return self.__file["Provenance"]

    @property
    def _auxiliary_data_group(self):
        return self.__file["AuxiliaryData"]

    @property
    def asdf_format_version(self):
        """
        Returns the version of the ASDF file.
        """
        return self.__file.attrs["file_format_version"].decode()

    @property
    def filename(self):
        """
        Get the path of the underlying file on the filesystem. Works in most
        circumstances.
        """
        return self.__original_filename

    @property
    def mpi(self):
        """
        Returns a named tuple with ``comm``, ``rank``, ``size``, and ``MPI``
        if run with MPI and ``False`` otherwise.
        """
        # Simple cache as this is potentially accessed a lot.
        if hasattr(self, "__is_mpi"):
            return self.__is_mpi

        if self.__force_mpi is True:
            self.__is_mpi = True
        elif self.__force_mpi is False:
            self.__is_mpi = False
        else:
            self.__is_mpi = is_mpi_env()

        # If it actually is an mpi environment, set the communicator and the
        # rank.
        if self.__is_mpi:

            # Check if HDF5 has been complied with parallel I/O.
            c = h5py.get_config()
            if not hasattr(c, "mpi") or not c.mpi:
                is_parallel = False
            else:
                is_parallel = True

            if not is_parallel:
                msg = "Running under MPI requires HDF5/h5py to be complied " \
                      "with support for parallel I/O."
                raise RuntimeError(msg)

            import mpi4py

            # This is not needed on most mpi4py installations.
            if not mpi4py.MPI.Is_initialized():
                mpi4py.MPI.Init()

            # Set mpi tuple to easy class wide access.
            mpi_ns = collections.namedtuple("mpi_ns", ["comm", "rank",
                                                       "size", "MPI"])
            comm = mpi4py.MPI.COMM_WORLD
            self.__is_mpi = mpi_ns(comm=comm, rank=comm.rank,
                                   size=comm.size, MPI=mpi4py.MPI)

        return self.__is_mpi

    @property
    def events(self):
        """
        Get all events stored in the data set.

        :rtype: An ObsPy :class:`~obspy.core.event.Catalog` object.
        """
        if "QuakeML" not in self.__file:
            return obspy.core.event.Catalog()
        data = self.__file["QuakeML"]
        if not len(data.value):
            return obspy.core.event.Catalog()

        with io.BytesIO(_read_string_array(data)) as buf:
            try:
                cat = obspy.read_events(buf, format="quakeml")
            except:
                # ObsPy is not able to read empty QuakeML files but they are
                # still valid QuakeML files.
                buf.seek(0, 0)
                result = None
                try:
                    result = obspy.core.quakeml._validate(buf)
                except:
                    pass
                # If validation is successful, but the initial read failed,
                # it must have been an empty QuakeML object.
                if result is True:
                    cat = obspy.core.event.Catalog()
                else:
                    # Re-raise exception
                    raise
        return cat

    @events.setter
    def events(self, event):
        """
        Set the events of the dataset.

        :param event: One or more events. Will replace all existing ones.
        :type event: :class:`~obspy.core.event.Event` or
            :class:`~obspy.core.event.Catalog`
        """
        if isinstance(event, obspy.core.event.Event):
            cat = obspy.core.event.Catalog(events=[event])
        elif isinstance(event, obspy.core.event.Catalog):
            cat = event
        else:
            raise TypeError("Must be an ObsPy event or catalog instance")

        with io.BytesIO() as buf:
            cat.write(buf, format="quakeml")
            buf.seek(0, 0)
            data = np.frombuffer(buf.read(), dtype=np.dtype("byte"))

        # Create the QuakeML data set if it does not exist.
        if "QuakeML" not in self.__file:
            self.__file.create_dataset("QuakeML", dtype=np.dtype("byte"),
                                       compression=self.__compression[0],
                                       compression_opts=self.__compression[1],
                                       shuffle=self.__shuffle,
                                       shape=(0,), maxshape=(None,),
                                       fletcher32=not bool(self.mpi))

        self.__file["QuakeML"].resize(data.shape)
        self.__file["QuakeML"][:] = data

    def add_auxiliary_data_file(
            self, filename_or_object, path, parameters=None,
            provenance_id=None):
        """
        Special function adding a file or file like object as an auxiliary
        data object denoting a file. This is very useful to store arbitrary
        files in ASDF.

        :param filename_or_object: Filename, open-file or file-like object.
            File should really be opened in binary mode, but this i not
            checked.
        :param path: The path of the file. Has the same limitations as normal
            tags.
        :param parameters: Any additional options, as a Python dictionary.
        :param provenance_id: The id of the provenance of this data. The
            provenance information itself must be added separately. Must be
            given as a qualified name, e.g. ``'{namespace_uri}id'``.
        """
        if hasattr(filename_or_object, "read"):
            data = np.frombuffer(filename_or_object.read(),
                                 dtype=np.dtype("byte"))
        else:
            with io.open(filename_or_object, "rb") as fh:
                data = np.frombuffer(fh.read(),
                                     dtype=np.dtype("byte"))

        if parameters is None:
            parameters = {}

        self.add_auxiliary_data(data=data, data_type="Files", path=path,
                                parameters=parameters,
                                provenance_id=provenance_id)

    def add_auxiliary_data(self, data, data_type, path, parameters,
                           provenance_id=None):
        """
        Adds auxiliary data to the file.

        :param data: The actual data as a n-dimensional numpy array.
        :param data_type: The type of data, think of it like a subfolder.
        :param path: The path of the data. Must be unique per data_type. Can
            be a path separated by forward slashes at which point it will be
            stored in a nested structure.
        :param parameters: Any additional options, as a Python dictionary.
        :param provenance_id: The id of the provenance of this data. The
            provenance information itself must be added separately. Must be
            given as a qualified name, e.g. ``'{namespace_uri}id'``.


        The data type is the category of the data and the path the name of
        that particular piece of data within that category.

        >>> ds.add_auxiliary_data(numpy.random.random(10),
        ...                       data_type="RandomArrays",
        ...                       path="test_data",
        ...                       parameters={"a": 1, "b": 2})
        >>> ds.auxiliary_data.RandomArrays.test_data
        Auxiliary Data of Type 'RandomArrays'
        Path: 'test_data'
        Data shape: '(10, )', dtype: 'float64'
        Parameters:
            a: 1
            b: 2

        The path can contain forward slashes to create a nested hierarchy of
        auxiliary data.

        >>> ds.add_auxiliary_data(numpy.random.random(10),
        ...                       data_type="RandomArrays",
        ...                       path="some/nested/path/test_data",
        ...                       parameters={"a": 1, "b": 2})
        >>> ds.auxiliary_data.RandomArrays.some.nested.path.test_data
        Auxiliary Data of Type 'RandomArrays'
        Path: 'some/nested/path/test_data'
        Data shape: '(10, )', dtype: 'float64'
        Parameters:
            a: 1
            b: 2
        """
        # Assert the data type name.
        pattern = r"^[A-Z][A-Za-z0-9]*$"
        if re.match(pattern, data_type) is None:
            raise ASDFValueError(
                "Data type name '{name}' is invalid. It must validate "
                "against the regular expression '{pattern}'.".format(
                    name=data_type, pattern=pattern))

        # Split the path.
        tag_path = path.strip("/").split("/")

        for path in tag_path:
            # Assert each path piece.
            tag_pattern = r"^[a-zA-Z0-9][a-zA-Z0-9_]*[a-zA-Z0-9]$"
            if re.match(tag_pattern, path) is None:
                raise ASDFValueError(
                    "Tag name '{name}' is invalid. It must validate "
                    "against the regular expression '{pattern}'.".format(
                        name=path, pattern=tag_pattern))

        if provenance_id is not None:
            # Will raise an error if not a valid qualified name.
            split_qualified_name(provenance_id)
        # Complicated multi-step process but it enables one to use
        # parallel I/O with the same functions.
        info = self._add_auxiliary_data_get_collective_information(
            data=data, data_type=data_type, tag_path=tag_path,
            parameters=parameters, provenance_id=provenance_id)
        if info is None:
            return
        self._add_auxiliary_data_write_collective_information(info=info)
        self._add_auxiliary_data_write_independent_information(info=info,
                                                               data=data)

    def _add_auxiliary_data_get_collective_information(
            self, data, data_type, tag_path, parameters, provenance_id=None):
        """
        The information required for the collective part of adding some
        auxiliary data.

        This will extract the group name, the parameters of the dataset to
        be created, and the attributes of the dataset.
        """
        if "provenance_id" in parameters:
            raise ASDFValueError("'provenance_id' is a reserved parameter "
                                 "and cannot be used as an arbitrary "
                                 "parameters.")
        # If the provenance id is set, add it to the parameters. At this
        # point it is assumed, that the id is valid.
        if provenance_id is not None:
            parameters = copy.deepcopy(parameters)
            parameters.update({"provenance_id":
                              self._zeropad_ascii_string(provenance_id)})

        group_name = "%s/%s" % (data_type, "/".join(tag_path))
        if group_name in self._auxiliary_data_group:
            msg = "Data '%s' already exists in file. Will not be added!" % \
                  group_name
            warnings.warn(msg, ASDFWarning)
            return

        # XXX: Figure out why this is necessary. It should work according to
        # the specs.
        if self.mpi:
            fletcher32 = False
        else:
            fletcher32 = True

        info = {
            "data_name": group_name,
            "data_type": data_type,
            "dataset_creation_params": {
                "name": "/".join(tag_path),
                "shape": data.shape,
                "dtype": data.dtype,
                "compression": self.__compression[0],
                "compression_opts": self.__compression[1],
                "shuffle": self.__shuffle,
                "fletcher32": fletcher32,
                "maxshape": tuple([None] * len(data.shape))
            },
            "dataset_attrs": parameters,
        }
        return info

    def _add_auxiliary_data_write_independent_information(self, info, data):
        """
        Writes the independent part of auxiliary data to the file.
        """
        self._auxiliary_data_group[info["data_name"]][:] = data

    def _add_auxiliary_data_write_collective_information(self, info):
        """
        Writes the collective part of auxiliary data to the file.
        """
        data_type = info["data_type"]
        if data_type not in self._auxiliary_data_group:
            self._auxiliary_data_group.create_group(data_type)
        group = self._auxiliary_data_group[data_type]

        ds = group.create_dataset(**info["dataset_creation_params"])
        for key, value in info["dataset_attrs"].items():
            ds.attrs[key] = value

    def add_quakeml(self, event):
        """
        Adds a QuakeML file or an existing ObsPy event to the data set.

        An exception will be raised if an event is attempted to be added
        that already exists within the data set. Duplicates are detected
        based on the public ids of the events.

        :param event: Filename or existing ObsPy event object.
        :type event: :class:`~obspy.core.event.Event` or
            :class:`~obspy.core.event.Catalog`
        :raises: ValueError

        .. rubric:: Example

        For now we will create a new ASDF file but one can also use an
        existing one.

        >>> impory pyasdf
        >>> import obspy
        >>> ds = pyasdf.ASDFDataSet("new_file.h5")

        One can add an event either by passing a filename ...

        >>> ds.add_quakeml("/path/to/quake.xml")

        ... or by passing an existing event or catalog object.

        >>> cat = obspy.read_events("/path/to/quakem.xml")
        >>> ds.add_quakeml(cat)
        """
        if isinstance(event, obspy.core.event.Event):
            cat = obspy.core.event.Catalog(events=[event])
        elif isinstance(event, obspy.core.event.Catalog):
            cat = event
        else:
            cat = obspy.read_events(event, format="quakeml")

        old_cat = self.events
        existing_resource_ids = set([_i.resource_id.id for _i in old_cat])
        new_resource_ids = set([_i.resource_id.id for _i in cat])
        intersection = existing_resource_ids.intersection(new_resource_ids)
        if intersection:
            msg = ("Event id(s) %s already present in ASDF file. Adding "
                   "events cancelled")
            raise ValueError(msg % ", ".join(intersection))
        old_cat.extend(cat)

        self.events = old_cat

    def get_data_for_tag(self, station_name, tag):
        """
        Returns the waveform and station data for the requested station and
        path.

        :param station_name: A string with network id and station id,
            e.g. ``"IU.ANMO"``
        :type station_name: str
        :param tag: The path of the waveform.
        :type tag: str
        :return: tuple of the waveform and the inventory.
        :rtype: (:class:`~obspy.core.stream.Stream`,
                 :class:`~obspy.core.inventory.inventory.Inventory`)
        """
        station_name = station_name.replace(".", "_")
        station = getattr(self.waveforms, station_name)
        st = getattr(station, tag)
        inv = getattr(station, "StationXML") \
            if "StationXML" in station else None
        return st, inv

    def _get_idx_and_size_estimate(self, waveform_name, starttime, endtime):
        network, station = waveform_name.split(".")[:2]
        data = self.__file["Waveforms"]["%s.%s" % (network, station)][
            waveform_name]

        idx_start = 0
        idx_end = data.shape[0]
        dt = 1.0 / data.attrs["sampling_rate"]

        # Starttime is a timestamp in nanoseconds.
        # Get time of first and time of last sample.
        data_starttime = obspy.UTCDateTime(
            float(data.attrs["starttime"]) / 1.0E9)
        data_endtime = data_starttime + (idx_end - 1) * dt

        # Modify the data indices to restrict the data if necessary.
        if starttime is not None and starttime > data_starttime:
            offset = max(0, int((starttime - data_starttime) // dt))
            idx_start = offset
            # Also modify the data_starttime here as it changes the actually
            # read data.
            data_starttime += offset * dt
        if endtime is not None and endtime < data_endtime:
            offset = max(0, int((data_endtime - endtime) // dt))
            idx_end -= offset

        # Check the size against the limit.
        array_size_in_mb = \
            (idx_end - idx_start) * data.dtype.itemsize / 1024.0 / 1024.0

        del data

        return idx_start, idx_end, data_starttime, array_size_in_mb

    def _get_waveform(self, waveform_name, starttime=None, endtime=None):
        """
        Retrieves the waveform for a certain path name as a Trace object. For
        internal use only, use the dot accessors for outside access.
        """
        idx_start, idx_end, data_starttime, array_size_in_mb = \
            self._get_idx_and_size_estimate(waveform_name,
                                            starttime, endtime)

        if array_size_in_mb > self.single_item_read_limit_in_mb:
            msg = ("The current selection would read %.2f MB into memory from "
                   "'%s'. The current limit is %.2f MB. Adjust by setting "
                   "'ASDFDataSet.single_item_read_limit_in_mb' or use a "
                   "different method to read the waveform data." % (
                    array_size_in_mb, waveform_name,
                    self.single_item_read_limit_in_mb))
            raise ASDFValueError(msg)

        network, station, location, channel = waveform_name.split(".")[:4]
        channel = channel[:channel.find("__")]
        data = self.__file["Waveforms"]["%s.%s" % (network, station)][
            waveform_name]

        tr = obspy.Trace(data=data[idx_start: idx_end])
        tr.stats.starttime = data_starttime
        tr.stats.sampling_rate = data.attrs["sampling_rate"]
        tr.stats.network = network
        tr.stats.station = station
        tr.stats.location = location
        tr.stats.channel = channel
        # Set some format specific details.
        tr.stats._format = FORMAT_NAME
        details = obspy.core.util.AttribDict()
        setattr(tr.stats, FORMAT_NAME.lower(), details)
        details.format_version = FORMAT_VERSION

        # Read all the ids if they are there.
        ids = ["event_id", "origin_id", "magnitude_id", "focal_mechanism_id"]
        for name in ids:
            if name in data.attrs:
                setattr(details, name + "s",
                        [obspy.core.event.ResourceIdentifier(_i) for _i in
                         data.attrs[name].tostring().decode().split(",")])

        if "provenance_id" in data.attrs:
            details.provenance_id = \
                data.attrs["provenance_id"].tostring().decode()

        if "labels" in data.attrs:
            details.labels = labelstring2list(data.attrs["labels"])

        # Add the tag to the stats dictionary.
        details.tag = wf_name2tag(waveform_name)

        return tr

    def _get_auxiliary_data(self, data_type, tag):
        group = self._auxiliary_data_group[data_type][tag]

        if isinstance(group, h5py.Group):
            return AuxiliaryDataAccessor(
                auxiliary_data_type=re.sub(r"^/AuxiliaryData/", "",
                                           group.name),
                asdf_data_set=self)
        return AuxiliaryDataContainer(
            data=group, data_type=data_type, tag=tag,
            parameters={i: j for i, j in group.attrs.items()})

    @property
    def filesize(self):
        """
        Return the current size of the file.
        """
        return os.path.getsize(self.filename)

    @property
    def pretty_filesize(self):
        """
        Return a pretty string representation of the size of the file.
        """
        return sizeof_fmt(self.filesize)

    def __str__(self):
        """
        Pretty string formatting.
        """
        ret = "{format} file [format version: {version}]: '{filename}' ({" \
              "size})".format(
                  format=FORMAT_NAME,
                  version=self.asdf_format_version,
                  filename=os.path.relpath(self.filename),
                  size=self.pretty_filesize)
        ret += "\n\tContains %i event(s)" % len(self.events)
        ret += "\n\tContains waveform data from {len} station(s).".format(
            len=len(self.__file["Waveforms"])
        )
        if len(self.auxiliary_data):
            ret += "\n\tContains %i type(s) of auxiliary data: %s" % (
                len(self.auxiliary_data),
                ", ".join(sorted(self.auxiliary_data.list())))
        return ret

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())

    def append_waveforms(self, waveform, tag):
        """
        Append waveforms to an existing data array if possible.

        .. note::

            The :meth:`.add_waveforms` method is the better choice in most
            cases. This function here is only useful for special cases!

        The main purpose of this function is to enable the construction of
        ASDF files with large single arrays. The classical example is all
        recordings of a station for a year or longer. If the
        :meth:`.add_waveforms` method is used ASDF will internally store
        each file in one or more data sets. This function here will attempt
        to enlarge existing data arrays and append to them creating larger
        ones that are a bit more efficient to read. It is slower to write
        this way but it can potentially be faster to read.

        This is only possible if the to be appended waveform traces
        seamlessly fit after an existing trace with a tolerance of half a
        sampling interval.

        Please note that a single array in ASDF cannot have any gaps and/or
        overlaps so even if this function is used it might still result in
        several data sets in the HDF5 file.

        This additionally requires that the data-sets being appended to have
        chunks as non-chunked data cannot be resized. MPI is consequently
        not allowed for this function as well.

        .. warning::

            Any potentially set `event_id`, `origin_id`, `magnitude_id`,
            `focal_mechanism_id`, `provenance_id`, or `labels` will carry over
            from the trace that is being appended to so please only use this
            method if you known what you are doing.


        .. rubric:: Example

        Assuming three trace in three different files, ``A``, ``B``,
        and ``C``, that could seamlessly be merged to one big trace
        without producing a gap or overlap. Using :meth:`.add_waveforms`
        will create three seperate data arrays in the ASDF file, e.g.

        .. code-block:: python

            ds.add_waveforms(A, tag="random")
            ds.add_waveforms(B, tag="random")
            ds.add_waveforms(C, tag="random")

        results in:

        .. code-block:: python

            /Waveforms
                /XX.YYYY
                    A
                    B
                    C

        Using this method here on the other hand will (if possible) create a
        single large array which might be a bit faster to read or to iterate
        over.

        .. code-block:: python

            ds.append_waveforms(A, tag="random")
            ds.append_waveforms(B, tag="random")
            ds.append_waveforms(C, tag="random")

        This results in:

        .. code-block:: python

            /Waveforms
                /XX.YYYY
                    A + B + C

        :param waveform: The waveform to add. Can either be an ObsPy Stream
            or Trace object or something ObsPy can read.
        :type waveform: :class:`obspy.core.stream.Stream`,
            :class:`obspy.core.trace.Trace`, str, ...
        :param tag: The path that will be given to all waveform files. It is
            mandatory for all traces and facilitates identification of the data
            within one ASDF volume. The ``"raw_record"`` path is,
            by convention, reserved to raw, recorded, unprocessed data.
        :type tag: str
        """
        if self.mpi:
            raise ASDFException("This function cannot work with parallel "
                                "MPI I/O.")

        tag = self.__parse_and_validate_tag(tag)
        waveform = self.__parse_waveform_input_and_validate(waveform)

        def _get_dataset_within_tolerance(station_group, trace):
            # Tolerance.
            min_t = trace.stats.starttime.timestamp - \
                0.5 * trace.stats.delta
            max_t = min_t + trace.stats.delta

            i = trace.id + "__"
            for ds_name in station_group.list():
                if not ds_name.startswith(i):
                    continue
                ds = station_group._WaveformAccessor__hdf5_group[ds_name]
                t = ds.attrs["starttime"] / 1e9 + ds.shape[0] * \
                    1.0 / ds.attrs["sampling_rate"]
                del ds
                if min_t <= t <= max_t:
                    return ds_name

        for trace in waveform:
            # The logic is quite easy - find an existing data-set that is
            # within the allowed tolerance and append, otherwise just pass
            # to the add_waveforms() method.
            sta_name = "%s.%s" % (trace.stats.network, trace.stats.station)
            if sta_name in self.__file["Waveforms"]:
                ds_name = _get_dataset_within_tolerance(
                    self.waveforms[sta_name], trace=trace)
                if ds_name:
                    # Append!
                    sta_group = self.__file["Waveforms"][sta_name]
                    ds = sta_group[ds_name]

                    # Make sure it actually can be resized.
                    if ds.maxshape[0] is not None:
                        msg = ("'maxshape' of '%s' is not set to None which "
                               "prevents it from being resized." % ds_name)
                        raise ASDFValueError(msg)
                    if ds.chunks is None:
                        msg = ("Data set '%s' is not chunked which "
                               "prevents it from being resized." % ds_name)
                        raise ASDFValueError(msg)

                    existing = ds.shape[0]
                    # Resize.
                    ds.resize((existing + trace.stats.npts, ))
                    # Add data.
                    ds[existing:] = trace.data

                    # Rename.
                    new_data_name = self.__get_waveform_ds_name(
                        net=trace.stats.network, sta=trace.stats.station,
                        loc=trace.stats.location, cha=trace.stats.channel,
                        start=obspy.UTCDateTime(ds.attrs["starttime"] / 1e9),
                        end=trace.stats.endtime,
                        tag=tag)
                    del ds

                    # This does not copy data but just changes the name.
                    sta_group[new_data_name] = sta_group[ds_name]
                    del sta_group[ds_name]

                    del sta_group
                    continue

            # If this did not work - append.
            self.add_waveforms(waveform=trace, tag=tag)

    def add_waveforms(self, waveform, tag, event_id=None, origin_id=None,
                      magnitude_id=None, focal_mechanism_id=None,
                      provenance_id=None, labels=None):
        """
        Adds one or more waveforms to the current ASDF file.

        :param waveform: The waveform to add. Can either be an ObsPy Stream or
            Trace object or something ObsPy can read.
        :type waveform: :class:`obspy.core.stream.Stream`,
            :class:`obspy.core.trace.Trace`, str, ...
        :param tag: The path that will be given to all waveform files. It is
            mandatory for all traces and facilitates identification of the data
            within one ASDF volume. The ``"raw_record"`` path is,
            by convention, reserved to raw, recorded, unprocessed data.
        :type tag: str
        :param event_id: The event or id which the waveform is associated
            with. This is useful for recorded data if a clear association is
            given, but also for synthetic data. Can also be a list of items.
        :type event_id: :class:`obspy.core.event.Event`,
            :class:`obspy.core.event.ResourceIdentifier`, str, or list
        :param origin_id: The particular origin this waveform is associated
            with. This is mainly useful for synthetic data where the origin
            is precisely known. Can also be a list of items.
        :type origin_id: :class:`obspy.core.event.Origin`,
            :class:`obspy.core.event.ResourceIdentifier`, str, or list
        :param magnitude_id: The particular magnitude this waveform is
            associated with. This is mainly useful for synthetic data where
            the magnitude is precisely known. Can also be a list of items.
        :type magnitude_id: :class:`obspy.core.event.Magnitude`,
            :class:`obspy.core.event.ResourceIdentifier`, str, or list
        :param focal_mechanism_id: The particular focal mechanism this
            waveform is associated with. This is mainly useful for synthetic
            data where the mechanism is precisely known. Can also be a list of
            items.
        :type focal_mechanism_id: :class:`obspy.core.event.FocalMechanism`,
            :class:`obspy.core.event.ResourceIdentifier`, str, or list
        :param provenance_id: The id of the provenance of this data. The
            provenance information itself must be added separately. Must be
            given as a qualified name, e.g. ``'{namespace_uri}id'``.
        :type labels: list of str
        :param labels: A list of labels to associate with all the added
            traces. Must not contain a comma as that is used as a separator.

        .. rubric:: Examples

        We first setup an example ASDF file with a single event.

        >>> from pyasdf import ASDFDataSet
        >>> ds = ASDFDataSet("event_tests.h5")
        >>> ds.add_quakeml("quake.xml")
        >>> event = ds.events[0]

        Now assume we have a MiniSEED file that is an unprocessed
        observation of that earthquake straight from a datacenter called
        ``recording.mseed``. We will now add it to the file, give it the
        ``"raw_recording"`` path (which is reserved for raw, recorded,
        and unproceseed data) and associate it with the event. Keep in mind
        that this association is optional. It can also be associated with
        multiple events - in that case just pass a list of objects.

        >>> ds.add_waveforms("recording.mseed", path="raw_recording",
        ...                  event_id=event)

        It is also possible to directly add
        :class:`obspy.core.stream.Stream` objects containing an arbitrary
        number of :class:`obspy.core.trace.Trace` objects.

        >>> import obspy
        >>> st = obspy.read()  # Reads an example file without argument.
        >>> print(st)
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.00Z - ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.00Z - ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.00Z - ... | 100.0 Hz, 3000 samples
        >>> ds.add_waveforms(st, path="obspy_example")

        Just to demonstrate that all waveforms can also be retrieved again.

        >>> print(print(ds.waveforms.BW_RJOB.obspy_example))
        3 Trace(s) in Stream:
        BW.RJOB..EHZ | 2009-08-24T00:20:03.00Z - ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHN | 2009-08-24T00:20:03.00Z - ... | 100.0 Hz, 3000 samples
        BW.RJOB..EHE | 2009-08-24T00:20:03.00Z - ... | 100.0 Hz, 3000 samples

        For the last example lets assume we have the result of a simulation
        stored in the ``synthetics.sac`` file. In this case we know the
        precise source parameters (we specified them before running the
        simulation) so it is a good idea to add that association to the
        waveform. Please again keep in mind that they are all optional and
        depending on your use case they might or might not be
        useful/meaningful. You can again pass lists of all of these objects
        in which case multiple associations will be stored in the file.

        >>> origin = event.preferred_origin()
        >>> magnitude = event.preferred_magnitude()
        >>> focal_mechanism = event.preferred_focal_mechansism()
        >>> ds.add_waveforms("synthetics.sac", event_id=event,
        ...                  origin_id=origin, magnitude_id=magnitude,
        ...                  focal_mechanism_id=focal_mechanism)

        """
        if provenance_id is not None:
            # Will raise an error if not a valid qualified name.
            split_qualified_name(provenance_id)

        # Parse labels to a single comma separated string.
        labels = label2string(labels)

        tag = self.__parse_and_validate_tag(tag)
        waveform = self.__parse_waveform_input_and_validate(waveform)

        # Actually add the data.
        for trace in waveform:
            # Complicated multi-step process but it enables one to use
            # parallel I/O with the same functions.
            info = self._add_trace_get_collective_information(
                trace, tag, event_id=event_id, origin_id=origin_id,
                magnitude_id=magnitude_id,
                focal_mechanism_id=focal_mechanism_id,
                provenance_id=provenance_id, labels=labels)
            if info is None:
                continue
            self._add_trace_write_collective_information(info)
            self._add_trace_write_independent_information(info, trace)

    def __parse_and_validate_tag(self, tag):
        tag = tag.strip()
        if tag.lower() == "stationxml":
            msg = "Tag '%s' is invalid." % tag
            raise ValueError(msg)
        return tag

    def __parse_waveform_input_and_validate(self, waveform):
        # The next function expects some kind of iterable that yields traces.
        if isinstance(waveform, obspy.Trace):
            waveform = [waveform]
        elif isinstance(waveform, obspy.Stream):
            pass
        # Delegate to ObsPy's format/input detection.
        else:
            waveform = obspy.read(waveform)

        for trace in waveform:
            if trace.data.dtype in VALID_SEISMOGRAM_DTYPES:
                continue
            else:
                raise TypeError("The trace's dtype ('%s') is not allowed "
                                "inside ASDF. Allowed are little and big "
                                "endian 4 and 8 byte signed integers and "
                                "floating point numbers." %
                                trace.data.dtype.name)
        return waveform

    def get_provenance_document(self, document_name):
        """
        Retrieve a provenance document with a certain name.

        :param document_name: The name of the provenance document to retrieve.
        """
        if document_name not in self._provenance_group:
            raise ASDFValueError(
                "Provenance document '%s' not found in file." % document_name)

        data = self._provenance_group[document_name]

        with io.BytesIO(_read_string_array(data)) as buf:
            doc = prov.read(buf, format="xml")
        return doc

    def add_provenance_document(self, document, name=None):
        """
        Add a provenance document to the current ASDF file.

        :type document: Filename, file-like objects or prov document.
        :param document: The document to add.
        :type name: str
        :param name: The name of the document within ASDF. Must be lowercase
            alphanumeric with optional underscores. If not given, it will
            autogenerate one. If given and it already exists, it will be
            overwritten.
        """
        # Always open it. We will write it anew everytime. This enables
        # pyasdf to store everything the prov package can read.
        if not isinstance(document, prov.model.ProvDocument):
            document = prov.read(document)

        # Autogenerate name if not given.
        if not name:
            name = str(uuid.uuid4()).replace("-", "_")

        # Assert the name against the regex.
        if PROV_FILENAME_REGEX.match(name) is None:
            raise ASDFValueError("Name '%s' is invalid. It must validate "
                                 "against the regular expression '%s'." %
                                 (name, PROV_FILENAME_REGEX.pattern))

        with io.BytesIO() as buf:
            document.serialize(buf, format="xml")
            buf.seek(0, 0)
            data = np.frombuffer(buf.read(), dtype=np.dtype("byte"))

        # If it already exists, overwrite the existing one.
        if name in self._provenance_group:
            self._provenance_group[name].resize(data.shape)
            self._provenance_group[name][:] = data
        else:
            # maxshape takes care to create an extendable data set.
            self._provenance_group.create_dataset(
                name, data=data,
                compression=self.__compression[0],
                compression_opts=self.__compression[1],
                shuffle=self.__shuffle,
                maxshape=(None,),
                fletcher32=True)

    def _add_trace_write_independent_information(self, info, trace):
        """
        Writes the independent part of a trace to the file.

        :param info:
        :param trace:
        :return:
        """
        self._waveform_group[info["data_name"]][:] = trace.data

    def _add_trace_write_collective_information(self, info):
        """
        Writes the collective part of a trace to the file.

        :param info:
        :return:
        """
        station_name = info["station_name"]
        if station_name not in self._waveform_group:
            self._waveform_group.create_group(station_name)
        group = self._waveform_group[station_name]

        ds = group.create_dataset(**info["dataset_creation_params"])
        for key, value in info["dataset_attrs"].items():
            ds.attrs[key] = value

    def __get_waveform_ds_name(self, net, sta, loc, cha, start, end, tag):
        return "{net}.{sta}.{loc}.{cha}__{start}__{end}__{tag}".format(
            net=net, sta=sta, loc=loc, cha=cha,
            start=start.strftime("%Y-%m-%dT%H:%M:%S"),
            end=end.strftime("%Y-%m-%dT%H:%M:%S"),
            tag=tag)

    def _add_trace_get_collective_information(
            self, trace, tag, event_id=None, origin_id=None,
            magnitude_id=None, focal_mechanism_id=None,
            provenance_id=None, labels=None):
        """
        The information required for the collective part of adding a trace.

        This will extract the group name, the parameters of the dataset to
        be created, and the attributes of the dataset.

        :param trace: The trace to add.
        :param tag: The path of the trace.
        """
        # Assert the tag.
        if not re.match(TAG_REGEX, tag):
            raise ValueError("Invalid tag: '%s' - Must satisfy the regex "
                             "'%s'." % (tag, TAG_REGEX.pattern))

        station_name = "%s.%s" % (trace.stats.network, trace.stats.station)
        # Generate the name of the data within its station folder.
        data_name = self.__get_waveform_ds_name(
            net=trace.stats.network, sta=trace.stats.station,
            loc=trace.stats.location, cha=trace.stats.channel,
            start=trace.stats.starttime, end=trace.stats.endtime, tag=tag)

        group_name = "%s/%s" % (station_name, data_name)
        if group_name in self._waveform_group:
            msg = "Data '%s' already exists in file. Will not be added!" % \
                  group_name
            warnings.warn(msg, ASDFWarning)
            return

        # Checksumming cannot be used when writing with MPI I/O.
        if self.mpi:
            fletcher32 = False
        else:
            fletcher32 = True

        info = {
            "station_name": station_name,
            "data_name": group_name,
            "dataset_creation_params": {
                "name": data_name,
                "shape": (trace.stats.npts,),
                "dtype": trace.data.dtype,
                "compression": self.__compression[0],
                "compression_opts": self.__compression[1],
                "shuffle": self.__shuffle,
                "fletcher32": fletcher32,
                "maxshape": (None,)
            },
            "dataset_attrs": {
                # Starttime is the epoch time in nanoseconds.
                "starttime":
                    int(round(trace.stats.starttime.timestamp * 1.0E9)),
                "sampling_rate": trace.stats.sampling_rate
            }
        }

        # Add the labels if they are given. Given labels overwrite the one
        # in the attributes.
        if labels:
            info["dataset_attrs"]["labels"] = labels
        elif hasattr(trace.stats, "asdf") and \
                hasattr(trace.stats.asdf, "labels") and \
                trace.stats.asdf.labels:
            labels = label2string(trace.stats.asdf.labels)
            info["dataset_attrs"]["labels"] = labels

        # The various ids can either be given as the objects themselves,
        # e.g. an event, an origin, a magnitude, or a focal mechansism.
        # Alternatively they can be passed as ResourceIdentifier objects or
        # anything that can be converted to a resource identifier. Always
        # either 0, 1, or more of them, in that case as part of an iterator.
        # After the next step they are all lists of ResourceIdentifiers.
        event_id = to_list_of_resource_identifiers(
            event_id, name="event_id", obj_type=obspy.core.event.Event)
        origin_id = to_list_of_resource_identifiers(
            origin_id, name="origin_id", obj_type=obspy.core.event.Origin)
        magnitude_id = to_list_of_resource_identifiers(
            magnitude_id, name="magnitude_id",
            obj_type=obspy.core.event.Magnitude)
        focal_mechanism_id = to_list_of_resource_identifiers(
            focal_mechanism_id, name="focal_mechanism_id",
            obj_type=obspy.core.event.FocalMechanism)

        # Add all the event ids.
        ids = {
            "event_id": event_id,
            "origin_id": origin_id,
            "magnitude_id": magnitude_id,
            "focal_mechanism_id": focal_mechanism_id}
        for name, obj in ids.items():
            if obj is None and \
                    hasattr(trace.stats, "asdf") and \
                    hasattr(trace.stats.asdf, name + "s"):
                obj = trace.stats.asdf[name + "s"]
            if obj:
                info["dataset_attrs"][name] = \
                    self._zeropad_ascii_string(
                        ",".join(str(_i.id) for _i in obj))

        # Set the provenance id. Either get the one from the arguments or
        # use the one already set in the trace.stats attribute.
        if provenance_id is None and \
                hasattr(trace.stats, "asdf") and \
                hasattr(trace.stats.asdf, "provenance_id"):
            provenance_id = trace.stats.asdf["provenance_id"]
        if provenance_id:
            info["dataset_attrs"]["provenance_id"] = \
                self._zeropad_ascii_string(str(provenance_id))

        return info

    def _get_station(self, station_name):
        """
        Retrieves the specified StationXML as an ObsPy Inventory object. For
        internal use only, use the dot accessors for external access.

        :param station_name: A string with network id and station id,
            e.g. ``"IU.ANMO"``
        :type station_name: str
        """
        if station_name not in self.__file["Waveforms"] or \
                "StationXML" not in self.__file["Waveforms"][station_name]:
            return None

        data = self.__file["Waveforms"][station_name]["StationXML"]

        with io.BytesIO(_read_string_array(data)) as buf:
            try:
                inv = obspy.read_inventory(buf, format="stationxml")
            except lxml.etree.XMLSyntaxError:
                raise ValueError(
                    "Invalid XML file stored in the StationXML group for "
                    "station %s (HDF5 path '/Waveforms/%s/StationXML'). Talk "
                    "to the person/program that created the file!" % (
                        station_name, station_name))

        return inv

    def _add_inventory_object(self, inv, network_id, station_id):
        station_name = "%s.%s" % (network_id, station_id)

        # Write the station information to a numpy array that will then be
        # written to the HDF5 file.
        with io.BytesIO() as buf:
            inv.write(buf, format="stationxml")
            buf.seek(0, 0)
            data = np.frombuffer(buf.read(), dtype=np.dtype("byte"))

        if station_name not in self._waveform_group:
            self._waveform_group.create_group(station_name)
        station_group = self._waveform_group[station_name]

        # If it already exists, overwrite the existing one.
        if "StationXML" in station_group:
            station_group["StationXML"].resize(data.shape)
            station_group["StationXML"][:] = data
        else:
            # maxshape takes care to create an extendable data set.
            station_group.create_dataset(
                "StationXML", data=data,
                compression=self.__compression[0],
                compression_opts=self.__compression[1],
                shuffle=self.__shuffle,
                maxshape=(None,),
                fletcher32=True)

    def add_stationxml(self, stationxml):
        """
        Adds the StationXML to the data set object.

        This does some fairly exhaustive processing and will happily
        split the StationXML file and merge it with existing ones.

        If you care to have an a more or less unchanged StationXML file in
        the data set object be sure to add it one station at a time.

        :param stationxml: Filename of StationXML file or an ObsPy inventory
            object containing the same.
        :type stationxml: str or
            :class:`~obspy.core.inventory.inventory.Inventory`
        """
        # If not already an inventory object, delegate to ObsPy and see if
        # it can read it.
        if not isinstance(stationxml,
                          obspy.core.inventory.inventory.Inventory):
            stationxml = obspy.read_inventory(stationxml, format="stationxml")

        # Now we essentially walk the whole inventory, see what parts are
        # already available and add only the new ones. This involved quite a
        # bit of splitting and merging of the inventory objects.
        network_station_codes = set()
        for network in stationxml:
            for station in network:
                network_station_codes.add((network.code, station.code))

        for network_id, station_id in network_station_codes:
            station_name = "%s.%s" % (network_id, station_id)

            # Get any existing station information.
            existing_inventory = self._get_station(station_name)
            # If it does not exist yet, make sure its well behaved and add it.
            if existing_inventory is None:
                self._add_inventory_object(
                    inv=isolate_and_merge_station(
                        stationxml, network_id=network_id,
                        station_id=station_id),
                    network_id=network_id, station_id=station_id)
            # Otherwise merge with the existing one and overwrite the
            # existing one.
            else:
                self._add_inventory_object(
                    inv=merge_inventories(
                        inv_a=existing_inventory, inv_b=stationxml,
                        network_id=network_id, station_id=station_id),
                    network_id=network_id, station_id=station_id)

    def validate(self):
        """
        Validate and ASDF file. It currently checks that each waveform file
        has a corresponding station file.

        This does not (by far) replace the actual ASDF format validator.
        """
        summary = {"no_station_information": 0, "no_waveforms": 0,
                   "good_stations": 0}
        for station in self.waveforms:
            has_stationxml = "StationXML" in station
            has_waveforms = bool(station.get_waveform_tags())

            if has_stationxml is False and has_waveforms is False:
                continue

            elif has_stationxml is False:
                print("No station information available for station '%s'" %
                      station._station_name)
                summary["no_station_information"] += 1
                continue
            elif has_waveforms is False:
                print("Station with no waveforms: '%s'" %
                      station._station_name)
                summary["no_waveforms"] += 1
                continue
            summary["good_stations"] += 1

        print("\nChecked %i station(s):" % len(self.waveforms))
        print("\t%i station(s) have no available station information" %
              summary["no_station_information"])
        print("\t%i station(s) with no waveforms" %
              summary["no_waveforms"])
        print("\t%i good station(s)" % summary["good_stations"])

    def ifilter(self, *query_objects):
        """
        Return an iterator containing only the filtered information. Usage
        is fairly complex, a separate documentation page for
        :doc:`querying_data` is available - here is just a quick example:

        >>> for station in ds.ifilter(ds.q.network == "B?",
        ...                           ds.q.channel == "*Z",
        ...                           ds.q.starttime >= "2015-01-01")
        ...     ...
        """
        queries = merge_query_functions(query_objects)

        for station in self.waveforms:
            # Cheap checks first.
            # Test network and station codes if either is given.
            if queries["network"] or queries["station"]:
                net_code, sta_code = station._station_name.split(".")
                if queries["network"]:
                    if queries["network"](net_code) is False:
                        continue
                if queries["station"]:
                    if queries["station"](sta_code) is False:
                        continue

            # Check if the coordinates have to be parsed. Only station level
            # coordinates are used.
            if queries["latitude"] or queries["longitude"] or \
                    queries["elevation_in_m"]:
                # This will parse the StationXML files in a very fast manner
                # (but this is still an I/O heavy process!)
                try:
                    coords = station.coordinates
                    latitude = coords["latitude"]
                    longitude = coords["longitude"]
                    elevation_in_m = coords["elevation_in_m"]
                except NoStationXMLForStation:
                    latitude = None
                    longitude = None
                    elevation_in_m = None

                if queries["latitude"]:
                    if queries["latitude"](latitude) is False:
                        continue

                if queries["longitude"]:
                    if queries["longitude"](longitude) is False:
                        continue

                if queries["elevation_in_m"]:
                    if queries["elevation_in_m"](elevation_in_m) \
                            is False:
                        continue

            wfs = station.filter_waveforms(queries)

            if not wfs:
                continue

            yield FilteredWaveformAccessor(
                station_name=station._station_name,
                asdf_data_set=station._WaveformAccessor__data_set(),
                filtered_items=wfs)

        raise StopIteration

    def process_two_files_without_parallel_output(
            self, other_ds, process_function, traceback_limit=3):
        """
        Process data in two data sets.

        This is mostly useful for comparing data in two data sets in any
        number of scenarios. It again takes a function and will apply it on
        each station that is common in both data sets. Please see the
        :doc:`parallel_processing` document for more details.

        Can only be run with MPI.

        :type other_ds: :class:`.ASDFDataSet`
        :param other_ds: The data set to compare to.
        :param process_function: The processing function takes two
            parameters: The station group from this data set and the matching
            station group from the other data set.
        :type traceback_limit: int
        :param traceback_limit: The length of the traceback printed if an
            error occurs in one of the workers.
        :return: A dictionary for each station with gathered values. Will
            only be available on rank 0.
        """
        if not self.mpi:
            raise ASDFException("Currently only works with MPI.")

        # Collect the work that needs to be done on rank 0.
        if self.mpi.comm.rank == 0:

            def split(container, count):
                """
                Simple function splitting a container into equal length chunks.
                Order is not preserved but this is potentially an advantage
                depending on the use case.
                """
                return [container[_i::count] for _i in range(count)]

            this_stations = set(self.waveforms.list())
            other_stations = set(other_ds.waveforms.list())

            # Usable stations are those that are part of both.
            usable_stations = list(this_stations.intersection(other_stations))
            total_job_count = len(usable_stations)
            jobs = split(usable_stations, self.mpi.comm.size)
        else:
            jobs = None

        # Scatter jobs.
        jobs = self.mpi.comm.scatter(jobs, root=0)

        # Dictionary collecting results.
        results = {}

        for _i, station in enumerate(jobs):

            if self.mpi.rank == 0:
                print(" -> Processing approximately task %i of %i ..." % (
                      (_i * self.mpi.size + 1), total_job_count))

            try:
                result = process_function(
                    getattr(self.waveforms, station),
                    getattr(other_ds.waveforms, station))
            except Exception:
                # If an exception is raised print a good error message
                # and traceback to help diagnose the issue.
                msg = ("\nError during the processing of station '%s' "
                       "on rank %i:" % (station, self.mpi.rank))

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
                # A bit convoluted but compatible with Python 2 and
                # 3 and hopefully all encoding problems.
                tb += "".join(
                    _i.decode(errors="ignore")
                    if hasattr(_i, "decode") else _i
                    for _i in exc_line)

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
        gathered_results = self.mpi.comm.gather(results, root=0)

        results = {}
        if self.mpi.rank == 0:
            for result in gathered_results:
                results.update(result)

        # Likely not necessary as the gather two line above implies a
        # barrier but better be safe than sorry.
        self.mpi.comm.barrier()

        return results

    def process(self, process_function, output_filename, tag_map,
                traceback_limit=3, **kwargs):
        """
        Process the contents of an ``ASDF`` file in parallel.

        Applies a function to the contents of the current data set and
        writes the output to a new ``ASDF`` file. Can be run with and
        without MPI. Please see the :doc:`parallel_processing` document for
        more details.

        :type process_function: function
        :param process_function: A function with two argument:
            An :class:`obspy.core.stream.Stream` object and an
            :class:`obspy.core.inventory.inventory.Inventory` object. It should
            return a :class:`obspy.core.stream.Stream` object which will
            then be written to the new file.
        :type output_filename: str
        :param output_filename: The output filename. Must not yet exist.
        :type tag_map: dict
        :param tag_map: A dictionary mapping the input tags to output tags.
        :type traceback_limit: int
        :param traceback_limit: The length of the traceback printed if an
            error occurs in one of the workers.
        """
        # Check if the file exists.
        msg = "Output file '%s' already exists." % output_filename
        if not self.mpi:
            if os.path.exists(output_filename):
                raise ValueError(msg)
        else:
            # Only check file on one core to improve performance.
            if self.mpi.rank == 0:
                file_exists = os.path.exists(output_filename)
            else:
                file_exists = False

            # Make sure the exception is still raised on every core.
            file_exists = self.mpi.comm.bcast(file_exists, root=0)
            if file_exists:
                raise ValueError(msg)

        station_tags = []

        # Only rank 0 zero requires all that information.
        if not self.mpi or (self.mpi and self.mpi.rank == 0):
            stations = self.waveforms.list()

            # Get all possible station and waveform path combinations and let
            # each process read the data it needs.
            for station in stations:
                # Get the station and all possible tags.
                waveforms = self.__file["Waveforms"][station].keys()

                tags = set()
                for waveform in waveforms:
                    if waveform == "StationXML":
                        continue
                    tags.add(waveform.split("__")[-1])

                for tag in tags:
                    if tag not in tag_map.keys():
                        continue
                    station_tags.append((station, tag))
            has_station_tags = bool(station_tags)
        else:
            has_station_tags = False

        if self.mpi:
            has_station_tags = self.mpi.comm.bcast(has_station_tags, root=0)

        if not has_station_tags:
            raise ValueError("No data matching the path map found.")

        # Copy the station and event data only on the master process.
        if not self.mpi or (self.mpi and self.mpi.rank == 0):
            # Deactivate MPI even if active to not run into any barriers.
            output_data_set = ASDFDataSet(output_filename, mpi=False)
            for station_name, station_group in self._waveform_group.items():
                for tag, data in station_group.items():
                    if tag != "StationXML":
                        continue
                    if station_name not in output_data_set._waveform_group:
                        group = output_data_set._waveform_group.create_group(
                            station_name)
                    else:
                        group = output_data_set._waveform_group[station_name]
                    station_group.copy(source=data, dest=group,
                                       name="StationXML")

            # Copy the events.
            if self.events:
                output_data_set.events = self.events
            del output_data_set

        if self.mpi:
            self.mpi.comm.barrier()

        if not self.mpi:
            compression = self.__compression
        else:
            compression = None

        output_data_set = ASDFDataSet(output_filename, compression=compression)

        # Check for MPI, if yes, dispatch to MPI worker, if not dispatch to
        # the multiprocessing handler.
        if self.mpi:
            self._dispatch_processing_mpi(process_function, output_data_set,
                                          station_tags, tag_map,
                                          traceback_limit=traceback_limit)
        else:
            self._dispatch_processing_multiprocessing(
                process_function, output_data_set, station_tags, tag_map,
                traceback_limit=traceback_limit, **kwargs)

    def _dispatch_processing_mpi(self, process_function, output_data_set,
                                 station_tags, tag_map, traceback_limit):
        # Make sure all processes enter here.
        self.mpi.comm.barrier()

        if self.mpi.rank == 0:
            self._dispatch_processing_mpi_master_node(output_data_set,
                                                      station_tags, tag_map)
        else:
            self._dispatch_processing_mpi_worker_node(
                process_function, output_data_set, tag_map,
                traceback_limit=traceback_limit)

    def _dispatch_processing_mpi_master_node(self, output_dataset,
                                             station_tags, tag_map):
        """
        The master node. It distributes the jobs and takes care that
        metadata modifying actions are collective.
        """
        from mpi4py import MPI

        worker_nodes = range(1, self.mpi.comm.size)
        workers_requesting_write = []

        jobs = JobQueueHelper(jobs=station_tags,
                              worker_names=worker_nodes)

        __last_print = time.time()

        print("Launching processing using MPI on %i processors." %
              self.mpi.comm.size)

        # Barrier to indicate we are ready for looping. Must be repeated in
        # each worker node!
        self.mpi.comm.barrier()

        # Reactive event loop.
        while not jobs.all_done:
            time.sleep(0.01)

            # Informative output.
            if time.time() - __last_print > 2.0:
                print(jobs)
                __last_print = time.time()

            if (len(workers_requesting_write) >= 0.5 * self.mpi.comm.size) or \
                    (len(workers_requesting_write) and
                     jobs.all_poison_pills_received):
                if self.debug:
                    print("MASTER: initializing metadata synchronization.")

                # Send force write msgs to all workers and block until all
                # have been sent. Don't use blocking send cause then one
                # will have to wait each time anew and not just once for each.
                # The message will ready each worker for a collective
                # operation once its current operation is ready.
                requests = [self._send_mpi(None, rank, "MASTER_FORCES_WRITE",
                                           blocking=False)
                            for rank in worker_nodes]
                self.mpi.MPI.Request.waitall(requests)

                self._sync_metadata(output_dataset, tag_map=tag_map)

                # Reset workers requesting a write.
                workers_requesting_write[:] = []
                if self.debug:
                    print("MASTER: done with metadata synchronization.")
                continue

            # Retrieve any possible message and "dispatch" appropriately.
            status = MPI.Status()
            msg = self.mpi.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                     status=status)
            tag = MSG_TAGS[status.tag]
            source = status.source

            if self.debug:
                pretty_receiver_log(source, self.mpi.rank, status.tag, msg)

            if tag == "WORKER_REQUESTS_ITEM":
                # Send poison pill if no more work is available. After
                # that the worker should not request any more jobs.
                if jobs.queue_empty:
                    self._send_mpi(POISON_PILL, source, "MASTER_SENDS_ITEM")
                else:
                    # And send a new station path to process it.
                    station_tag = jobs.get_job_for_worker(source)
                    self._send_mpi(station_tag, source, "MASTER_SENDS_ITEM")

            elif tag == "WORKER_DONE_WITH_ITEM":
                station_tag, result = msg
                jobs.received_job_from_worker(station_tag, result, source)

            elif tag == "WORKER_REQUESTS_WRITE":
                workers_requesting_write.append(source)

            elif tag == "POISON_PILL_RECEIVED":
                jobs.poison_pill_received()

            else:
                raise NotImplementedError

        print("Master done, shutting down workers...")
        # Shutdown workers.
        for rank in worker_nodes:
            self._send_mpi(None, rank, "ALL_DONE")

        # Collect any stray messages that a poison pill has been received.
        # Does not matter for the flow but for aestetic reasons this is nicer.
        for _ in range(self.mpi.size * 3):
            time.sleep(0.01)
            if not self.mpi.comm.Iprobe(source=MPI.ANY_SOURCE,
                                        tag=MSG_TAGS["POISON_PILL_RECEIVED"]):
                continue
            self.mpi.comm.recv(source=MPI.ANY_SOURCE,
                               tag=MSG_TAGS["POISON_PILL_RECEIVED"])

        self.mpi.comm.barrier()
        print(jobs)

    def _dispatch_processing_mpi_worker_node(self, process_function,
                                             output_dataset, tag_map,
                                             traceback_limit):
        """
        A worker node. It gets jobs, processes them and periodically waits
        until a collective metadata update operation has happened.
        """
        self.stream_buffer = StreamBuffer()

        worker_state = {
            "poison_pill_received": False,
            "waiting_for_write": False,
            "waiting_for_item": False
        }

        # Barrier to indicate we are ready for looping. Must be repeated in
        # the master node!
        self.mpi.comm.barrier()

        # Loop until the 'ALL_DONE' message has been sent.
        while not self._get_msg(0, "ALL_DONE"):
            time.sleep(0.01)

            # Check if master requested a write.
            if self._get_msg(0, "MASTER_FORCES_WRITE"):
                self._sync_metadata(output_dataset, tag_map=tag_map)
                for key, value in self.stream_buffer.items():
                    if value is not None:
                        for trace in value:
                            output_dataset.\
                                _add_trace_write_independent_information(
                                    trace.stats.__info, trace)
                    self._send_mpi((key, str(value)), 0,
                                   "WORKER_DONE_WITH_ITEM",
                                   blocking=False)
                self.stream_buffer.clear()
                worker_state["waiting_for_write"] = False

            # Keep looping and wait for either the next write or that the
            # loop terminates.
            if worker_state["waiting_for_write"] or \
                    worker_state["poison_pill_received"]:
                continue

            if not worker_state["waiting_for_item"]:
                # Send message that the worker requires work.
                self._send_mpi(None, 0, "WORKER_REQUESTS_ITEM", blocking=False)
                worker_state["waiting_for_item"] = True
                continue

            msg = self._get_msg(0, "MASTER_SENDS_ITEM")
            if not msg:
                continue

            # Beyond this point it will always have received a new item.
            station_tag = msg.data
            worker_state["waiting_for_item"] = False

            # If no more work to be done, store state and keep looping as
            # stuff still might require to be written.
            if station_tag == POISON_PILL:
                if self.stream_buffer:
                    self._send_mpi(None, 0, "WORKER_REQUESTS_WRITE",
                                   blocking=False)
                    worker_state["waiting_for_write"] = True
                worker_state["poison_pill_received"] = True
                self._send_mpi(None, 0, "POISON_PILL_RECEIVED",
                               blocking=False)
                continue

            # Otherwise process the data.
            stream, inv = self.get_data_for_tag(*station_tag)
            try:
                stream = process_function(stream, inv)
            except Exception:
                # If an exception is raised print a good error message
                # and traceback to help diagnose the issue.
                msg = ("\nError during the processing of station '%s' "
                       "and tag '%s' on rank %i:" % (
                           station_tag[0], station_tag[1],
                           self.mpi.rank))

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
                # A bit convoluted but compatible with Python 2 and
                # 3 and hopefully all encoding problems.
                tb += "".join(
                    _i.decode(errors="ignore")
                    if hasattr(_i, "decode") else _i
                    for _i in exc_line)

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

                # Make sure synchronization works.
                self.stream_buffer[station_tag] = None
            else:
                # Add stream to buffer only if no error occured.
                self.stream_buffer[station_tag] = stream

            # If the buffer is too large, request from the master to stop
            # the current execution.
            if self.stream_buffer.get_size() >= \
                    MAX_MEMORY_PER_WORKER_IN_MB * 1024 ** 2:
                self._send_mpi(None, 0, "WORKER_REQUESTS_WRITE",
                               blocking=False)
                worker_state["waiting_for_write"] = True

        print("Worker %i shutting down..." % self.mpi.rank)

        # Collect any left-over messages. This can happen if the node
        # requests a new item (which would be a poison pill in any case) but
        # the master in the meanwhile forces a write which might cause
        # everything to be finished so the message is never collected. This
        # is troublesome if more than one ASDF file is processed in sequence
        # as the message would then spill over to the next file.
        for _ in range(3):
            time.sleep(0.01)
            self._get_msg(0, "MASTER_SENDS_ITEM")

        self.mpi.comm.barrier()

    def _sync_metadata(self, output_dataset, tag_map):
        """
        Method responsible for synchronizing metadata across all processes
        in the HDF5 file. All metadata changing operations must be collective.
        """
        if hasattr(self, "stream_buffer"):
            sendobj = []
            for key, stream in self.stream_buffer.items():
                if stream is None:
                    continue
                for trace in stream:
                    info = \
                        output_dataset._add_trace_get_collective_information(
                            trace, tag_map[key[1]])
                    trace.stats.__info = info
                    sendobj.append(info)
        else:
            sendobj = [None]

        data = self.mpi.comm.allgather(sendobj=sendobj)
        # Chain and remove None.
        trace_info = filter(lambda x: x is not None,
                            itertools.chain.from_iterable(data))
        # Write collective part.
        for info in trace_info:
            output_dataset._add_trace_write_collective_information(info)

        # Make sure all remaining write requests are processed before
        # proceeding.
        if self.mpi.rank == 0:
            for rank in range(1, self.mpi.size):
                msg = self._get_msg(rank, "WORKER_REQUESTS_WRITE")
                if self.debug and msg:
                    print("MASTER: Ignoring write request by worker %i" %
                          rank)

        self.mpi.comm.barrier()

    def _dispatch_processing_multiprocessing(
            self, process_function, output_data_set, station_tags, tag_map,
            traceback_limit, cpu_count=-1):
        multiprocessing = get_multiprocessing()

        input_filename = self.filename
        output_filename = output_data_set.filename

        # Make sure all HDF5 file handles are closed before fork() is called.
        # Might become irrelevant if the HDF5 library sees some changes but
        # right now it is necessary.
        self.flush()
        self._close()
        output_data_set.flush()
        output_data_set._close()
        del output_data_set

        # Lock for input and output files. Probably not needed for the input
        # files but better be safe.
        input_file_lock = multiprocessing.Lock()
        output_file_lock = multiprocessing.Lock()

        # Also lock the printing on screen to not mangle the output.
        print_lock = multiprocessing.Lock()

        # Use either the given number of cores or the maximum number of cores.
        if cpu_count == -1:
            cpu_count = multiprocessing.cpu_count()

        # Don't use more cores than jobs.
        cpu_count = min(cpu_count, len(station_tags))

        # Create the input queue containing the jobs.
        input_queue = multiprocessing.JoinableQueue(
            maxsize=int(math.ceil(1.1 * (len(station_tags) + cpu_count))))

        for _i in station_tags:
            input_queue.put(_i)

        # Put some poison pills.
        for _ in range(cpu_count):
            input_queue.put(POISON_PILL)

        # Give a short time for the queues to play catch-up.
        time.sleep(0.1)

        # The output queue will collect the reports from the jobs.
        output_queue = multiprocessing.Queue()

        # Create n processes, with n being the number of available CPUs.
        processes = []
        for i in range(cpu_count):
            processes.append(_Process(
                in_queue=input_queue, out_queue=output_queue,
                in_filename=input_filename, out_filename=output_filename,
                in_lock=input_file_lock, out_lock=output_file_lock,
                print_lock=print_lock,
                processing_function=dill.dumps(process_function),
                process_name=i, total_task_count=len(station_tags),
                cpu_count=cpu_count, traceback_limit=traceback_limit,
                tag_map=tag_map))

        print("Launching processing using multiprocessing on %i cores ..." %
              cpu_count)
        _start = time.time()

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        _end = time.time()
        _dur = (_end - _start)
        print("Finished processing in %.2f seconds (%.2g sec/station)." % (
            _dur, _dur / len(station_tags)))

        ASDFDataSet.__init__(self, self.__original_filename)

        return

    def _get_msg(self, source, tag):
        """
        Helper method to get a message if available, returns a
        ReceivedMessage instance in case a message is available, None
        otherwise.
        """
        tag = MSG_TAGS[tag]
        if not self.mpi.comm.Iprobe(source=source, tag=tag):
            return
        msg = ReceivedMessage(self.mpi.comm.recv(source=source, tag=tag))
        if self.debug:
            pretty_receiver_log(source, self.mpi.rank, tag, msg.data)
        return msg

    def _send_mpi(self, obj, dest, tag, blocking=True):
        """
        Helper method to send a message via MPI.
        """
        tag = MSG_TAGS[tag]
        if blocking:
            value = self.mpi.comm.send(obj=obj, dest=dest, tag=tag)
        else:
            value = self.mpi.comm.isend(obj=obj, dest=dest, tag=tag)
        if self.debug:
            pretty_sender_log(dest, self.mpi.rank, tag, obj)
        return value

    def _recv_mpi(self, source, tag):
        """
        Helper method to receive a message via MPI.
        """
        tag = MSG_TAGS[tag]
        msg = self.mpi.comm.recv(source=source, tag=tag)
        if self.debug:
            pretty_receiver_log(source, self.mpi.rank, tag, msg)
        return msg

    def get_all_coordinates(self):
        """
        Get a dictionary of the coordinates of all stations.
        """
        coords = {}
        for station in self.waveforms:
            try:
                coords[station._station_name] = station.coordinates
            except NoStationXMLForStation:
                pass
        return coords

    def get_waveforms(self, network, station, location, channel, starttime,
                      endtime, tag, automerge=True):
        """
        Directly access waveforms.

        This enables a more exact specification of what one wants to
        retrieve from an ASDF file. Most importantly it honors the start and
        end time and only reads those samples that are actually desired -
        this is especially important for large, continuous data sets.

        :type network: str
        :param network: The network code. Can contain wildcards.
        :type station: str
        :param station: The station code. Can contain wildcards.
        :type location: str
        :param location: The location code. Can contain wildcards.
        :type channel: str
        :param channel: The channel code. Can contain wildcards.
        :type starttime: :class:`obspy.core.utcdatetime.UTCDateTime`.
        :param starttime: The time of the first sample.
        :type endtime: :class:`obspy.core.utcdatetime.UTCDateTime`.
        :param endtime: The time of the last sample.
        :type tag: str
        :param tag: The tag to extract.
        :type automerge: bool
        :param automerge: Automatically merge adjacent traces if they are
            exactly adjacent (e.g. last sample from previous trace + first
            sample of next trace are one delta apart).
        """
        st = obspy.Stream()

        for i in self.ifilter(self.q.network == network,
                              self.q.station == station,
                              self.q.location == location,
                              self.q.channel == channel,
                              self.q.tag == tag,
                              self.q.starttime <= endtime,
                              self.q.endtime >= starttime):
            for t in i.get_waveform_tags():
                st.extend(i.get_item(t, starttime=starttime,
                                     endtime=endtime))

        # Cleanup merge - will only merge exactly adjacent traces.
        if automerge:
            st.merge(method=-1)
        return st


class _Process(multiprocessing.Process):
    """
    Internal process used for the processing data in parallel with the
    multi-processing module.
    """
    def __init__(self, in_queue, out_queue, in_filename,
                 out_filename, in_lock, out_lock, print_lock,
                 processing_function, process_name,
                 total_task_count, cpu_count, traceback_limit, tag_map):
        super(_Process, self).__init__()
        self.input_queue = in_queue
        self.output_queue = out_queue
        self.input_filename = in_filename
        self.output_filename = out_filename
        self.input_file_lock = in_lock
        self.output_file_lock = out_lock
        self.print_lock = print_lock
        self.processing_function = processing_function
        self.__process_name = process_name
        self.__task_count = 0
        self.__total_task_count = total_task_count
        self.__cpu_count = cpu_count
        self.__traceback_limit = traceback_limit
        self.__tag_map = tag_map

    def run(self):
        while True:
            self.__task_count += 1
            stationtag = self.input_queue.get(timeout=1)
            if stationtag == POISON_PILL:
                self.input_queue.task_done()
                break

            # Only print on "rank" 0.
            if self.__process_name == 0:
                with self.print_lock:
                    print(" -> Processing approximately task %i of "
                          "%i ..." % (min((self.__task_count - 1) *
                                          self.__cpu_count,
                                          self.__total_task_count),
                                      self.__total_task_count))

            station, tag = stationtag

            with self.input_file_lock:
                input_data_set = ASDFDataSet(self.input_filename)
                stream, inv = \
                    input_data_set.get_data_for_tag(station, tag)
                input_data_set.flush()
                del input_data_set

            # Using dill as it works on more systems.
            func = dill.loads(self.processing_function)

            try:
                output_stream = func(stream, inv)
            except Exception:
                msg = ("\nError during the processing of station '%s' "
                       "and tag '%s' on CPU %i:" % (
                           stationtag[0], stationtag[1],
                           self.__process_name))

                # Extract traceback from the exception.
                exc_info = sys.exc_info()
                stack = traceback.extract_stack(
                    limit=self.__traceback_limit)
                tb = traceback.extract_tb(exc_info[2])
                full_tb = stack[:-1] + tb
                exc_line = traceback.format_exception_only(
                    *exc_info[:2])
                tb = ("Traceback (At max %i levels - most recent call "
                      "last):\n" % self.__traceback_limit)
                tb += "".join(traceback.format_list(full_tb))
                tb += "\n"
                # A bit convoluted but compatible with Python 2 and
                # 3 and hopefully all encoding problems.
                tb += "".join(
                    _i.decode(errors="ignore")
                    if hasattr(_i, "decode") else _i
                    for _i in exc_line)

                with self.print_lock:
                    print(msg)
                    print(tb)

                self.input_queue.task_done()
                continue

            if output_stream:
                with self.output_file_lock:
                    output_data_set = ASDFDataSet(self.output_filename)
                    output_data_set.add_waveforms(
                        output_stream, tag=self.__tag_map[tag])
                    del output_data_set

            self.input_queue.task_done()
