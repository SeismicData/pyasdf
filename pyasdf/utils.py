#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2013-2020
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import copy
import collections
import hashlib
import io
import itertools
import os
import re
import sys
import time
import weakref

# Py2k/3k compat.
try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

import h5py
from lxml.etree import QName
import numpy as np
import obspy

from .compat import string_types
from .exceptions import (
    WaveformNotInFileException,
    NoStationXMLForStation,
    ASDFValueError,
    ASDFAttributeError,
)
from .header import MSG_TAGS
from .inventory_utils import get_coordinates

# Tuple holding a the body of a received message.
ReceivedMessage = collections.namedtuple("ReceivedMessage", ["data"])
# Tuple denoting a single worker.
Worker = collections.namedtuple(
    "Worker", ["active_jobs", "completed_jobs_count"]
)


def is_multiprocessing_problematic():  # pragma: no cover
    """
    Return True if multiprocessing is known to have issues on the given
    platform.

    Mainly results from the fact that some BLAS/LAPACK implementations
    cannot deal with forked processing.
    """
    # Handling numpy linked against accelerate.
    config_info = str(
        [
            value
            for key, value in np.__config__.__dict__.items()
            if key.endswith("_info")
        ]
    ).lower()

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


def _read_string_array(data):
    """
    Helper function taking a string data set and preparing it so it can be
    read to a BytesIO object.
    """
    return data[()].tobytes().strip()


class SimpleBuffer(object):
    """
    Object that can be used as a cache.

    Will never contain more then the specified number of items. If more then
    then that are used, it will remove the item with the oldest last access
    time.
    """

    def __init__(self, limit=10):
        self._limit = limit
        self._buffer = collections.OrderedDict()

    def __setitem__(self, key, value):
        self._buffer[key] = value
        self._check_size_limit()

    def __getitem__(self, key):
        # Reorder on access.
        value = self._buffer.pop(key)
        self._buffer[key] = value
        return value

    def __len__(self):
        return len(self._buffer)

    def __contains__(self, item):
        return item in self._buffer

    def values(self):
        return self._buffer.values()

    def _check_size_limit(self):
        while len(self._buffer) > self._limit:
            self._buffer.popitem(last=False)


class ProvenanceAccessor(object):
    """
    Accessor helper for the provenance records.
    """

    # Cache up to 20 documents. Uses the sha1 hash of the documents.
    _cache = SimpleBuffer(limit=20)

    def __init__(self, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__data_set = weakref.ref(asdf_data_set)

    def __getattr__(self, item):
        item = str(item)
        _records = self.list()
        if item not in _records:
            raise AttributeError

        hash = hashlib.sha1(
            self.__data_set()._provenance_group[item][()].tobytes()
        ).hexdigest()
        if hash not in self._cache:
            self._cache[hash] = self.__data_set().get_provenance_document(item)
        return copy.deepcopy(self._cache[hash])

    def __delitem__(self, key):
        key = str(key)
        if key not in self.list():
            raise KeyError("Key/Attribute '%s' not known." % key)

        ds = self.__data_set()
        del ds._provenance_group[key]
        del ds

    def __delattr__(self, item):
        try:
            self.__delitem__(item)
        except KeyError:
            raise AttributeError(str(item))

    def __getitem__(self, item):
        try:
            return self.__getattr__(item)
        except AttributeError as e:
            raise KeyError(str(e))

    def __setitem__(self, key, value):
        self.__data_set().add_provenance_document(
            document=value, name=str(key)
        )

    def __dir__(self):
        return self.list() + [
            "list",
            "keys",
            "values",
            "items",
            "get_provenance_document_for_id",
        ]

    def list(self):
        """
        Return a list of available provenance documents.
        """
        return sorted((self.__data_set()._provenance_group.keys()))

    def get_provenance_document_for_id(self, provenance_id):
        """
        Get the provenance document containing a record with a certain id.

        :param provenance_id: The id of the provenance record whose
            containing document is searched. Must be given as a qualified name,
            e.g. ``'{namespace_uri}id'``.
        """
        # Will raise if not a proper qualified name.
        url, localname = split_qualified_name(provenance_id)
        name = "{%s}%s" % (url, localname)

        for key, document in self.items():
            all_ids = get_all_ids_for_prov_document(document)
            if name in all_ids:
                return {"name": key, "document": document}
        raise ASDFValueError(
            "Document containing id '%s' not found in the data set."
            % provenance_id
        )

    def __len__(self):
        return len(self.list())

    def __iter__(self):
        for _i in self.list():
            yield _i

    def keys(self):
        return self.__iter__()

    def values(self):
        for _i in self.list():
            yield self[_i]

    def items(self):
        for _i in self.list():
            yield (_i, self[_i])

    def __str__(self):
        if not len(self):
            return "No provenance document in file."
        ret_str = "%i Provenance Document(s):\n\t%s" % (
            len(self),
            "\n\t".join(self.list()),
        )
        return ret_str

    def _repr_pretty_(self, p, cycle):  # pragma: no cover
        p.text(self.__str__())


class AuxiliaryDataContainer(object):
    def __init__(self, data, data_type, tag, parameters):
        self.data = data
        self.data_type = data_type
        self.path = tag
        if "provenance_id" in parameters:
            parameters = copy.deepcopy(parameters)
            self.provenance_id = parameters.pop("provenance_id")
            # It might be returned as a byte string on some systems.
            try:
                self.provenance_id = self.provenance_id.decode()
            except Exception:  # pragma: no cover
                pass
        else:
            self.provenance_id = None
        self.parameters = parameters

        self.__file_cache = None

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        if (
            self.path != other.path
            or self.data_type != other.data_type
            or self.provenance_id != self.provenance_id
            or self.parameters != self.parameters
        ):
            return False

        try:
            np.testing.assert_equal(self.data, other.data)
        except AssertionError:
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __del__(self):
        try:
            self.__file_cache.close()
        except Exception:
            pass

    @property
    def file(self):
        if self.data_type.lower() not in ["files", "file"]:
            raise ASDFAttributeError(
                "The 'file' property is only available "
                "for auxiliary data with the data type 'Files' or 'File'."
            )
        if self.__file_cache is not None:
            return self.__file_cache

        self.__file_cache = io.BytesIO(_read_string_array(self.data))
        return self.__file_cache

    def __str__(self):
        # Deal with nested paths.
        split_dt = self.data_type.split("/")
        path = split_dt[1:]
        path.append(self.path)

        return (
            "Auxiliary Data of Type '{data_type}'\n"
            "\tPath: '{path}'\n"
            "{provenance}"
            "\tData shape: '{data_shape}', dtype: '{dtype}'\n"
            "\tParameters:\n\t\t{parameters}".format(
                data_type=split_dt[0],
                data_shape=self.data.shape,
                dtype=self.data.dtype,
                path="/".join(path),
                provenance=""
                if self.provenance_id is None
                else "\tProvenance ID: '%s'\n" % self.provenance_id,
                parameters="\n\t\t".join(
                    [
                        "%s: %s" % (_i[0], _i[1])
                        for _i in sorted(
                            self.parameters.items(), key=lambda x: x[0]
                        )
                    ]
                ),
            )
        )

    def _repr_pretty_(self, p, cycle):  # pragma: no cover
        p.text(self.__str__())


class AuxiliaryDataAccessor(object):
    """
    Helper class to access auxiliary data items.
    """

    def __init__(self, auxiliary_data_type, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__auxiliary_data_type = auxiliary_data_type
        self.__data_set = weakref.ref(asdf_data_set)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        if self.__auxiliary_data_type != other.__auxiliary_data_type:
            return False

        ds1 = self.__data_set()
        ds2 = other.__data_set()
        try:
            if ds1.filename != ds2.filename:
                return False
        finally:
            del ds1
            del ds2

        return True

    def __delattr__(self, item):
        try:
            self.__delitem__(item)
        except KeyError as e:
            raise AttributeError(str(e))

    def __delitem__(self, key):
        key = str(key)
        _l = self.list()
        if key not in _l:
            raise KeyError(
                "Key/Item '%s' not known or cannot be " "deleted." % key
            )
        ds = self.__data_set()
        try:
            del ds._auxiliary_data_group[self.__auxiliary_data_type][key]
        finally:
            del ds

    def __contains__(self, item):
        return item in self.list()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getattr__(self, item):
        if item not in self:
            raise AttributeError("Key/Item '%s' not known." % item)
        return self.__data_set()._get_auxiliary_data(
            self.__auxiliary_data_type, str(item)
        )

    def __getitem__(self, item):
        try:
            return self.__getattr__(item)
        except AttributeError as e:
            raise KeyError(str(e))

    @property
    def auxiliary_data_type(self):
        return self.__auxiliary_data_type

    def list(self):
        return sorted(
            self.__data_set()
            ._auxiliary_data_group[self.__auxiliary_data_type]
            .keys()
        )

    def __dir__(self):
        return self.list() + ["list", "auxiliary_data_type"]

    def __len__(self):
        return len(self.list())

    def __iter__(self):
        for _i in self.list():
            yield self[_i]

    def __str__(self):
        items = self.list()

        groups = []
        data_arrays = []

        ds = self.__data_set()

        try:
            for item in items:
                reference = ds._auxiliary_data_group[
                    self.__auxiliary_data_type
                ][item]

                if isinstance(reference, h5py.Group):
                    groups.append(item)
                else:
                    data_arrays.append(item)

                del reference
        finally:
            del ds

        ret_str = ""

        if groups:
            ret_str += (
                "{count} auxiliary data sub group(s) of type '{type}' "
                "available:\n"
                "\t{items}".format(
                    count=len(groups),
                    type=self.__auxiliary_data_type,
                    items="\n\t".join(groups),
                )
            )
        if data_arrays:
            if ret_str:
                ret_str += "\n"
            ret_str += (
                "{count} auxiliary data item(s) of type '{type}' available:\n"
                "\t{items}".format(
                    count=len(data_arrays),
                    type=self.__auxiliary_data_type,
                    items="\n\t".join(data_arrays),
                )
            )
        if not groups and not data_arrays:
            ret_str += "Empty auxiliary data group."

        return ret_str

    def _repr_pretty_(self, p, cycle):  # pragma: no cover
        p.text(self.__str__())


class AuxiliaryDataGroupAccessor(object):
    """
    Helper class to facilitate access to the auxiliary data types.
    """

    def __init__(self, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__data_set = weakref.ref(asdf_data_set)

    def __delitem__(self, item):
        try:
            self.__delattr__(item)
        except AttributeError as e:
            raise KeyError(str(e))

    def __delattr__(self, item):
        item = str(item)
        if item not in self:
            raise AttributeError("Key/Attribute '%s' not known." % item)
        ds = self.__data_set()
        try:
            del ds._auxiliary_data_group[item]
        finally:
            del ds

    def __getattr__(self, item):
        item = str(item)
        __auxiliary_data_group = self.__data_set()._auxiliary_data_group
        if item not in __auxiliary_data_group:
            raise AttributeError
        return AuxiliaryDataAccessor(item, self.__data_set())

    def __getitem__(self, item):
        try:
            return self.__getattr__(item)
        except AttributeError as e:
            raise KeyError(str(e))

    def __contains__(self, item):
        item = str(item)
        __auxiliary_data_group = self.__data_set()._auxiliary_data_group
        if item not in __auxiliary_data_group:
            return False
        return True

    def list(self):
        __auxiliary_group = self.__data_set()._auxiliary_data_group
        return sorted(__auxiliary_group.keys())

    def __dir__(self):
        return self.list() + ["list"]

    def __len__(self):
        return len(self.list())

    def __str__(self):
        if not self.list():
            return "Data set contains no auxiliary data."

        return (
            "Data set contains the following auxiliary data types:\n"
            "\t{items}".format(
                items="\n\t".join(
                    [
                        "%s (%i item(s))" % (_i, len(self[_i]))
                        for _i in self.list()
                    ]
                )
            )
        )

    def _repr_pretty_(self, p, cycle):  # pragma: no cover
        p.text(self.__str__())


class StationAccessor(object):
    """
    Helper class to facilitate access to the waveforms and stations.
    """

    def __init__(self, asdf_data_set):
        # Use weak references to not have any dangling references to the HDF5
        # file around.
        self.__data_set = weakref.ref(asdf_data_set)

    def __getattr__(self, item, replace=True):
        if replace:
            item = str(item).replace("_", ".")
        if item not in self.__data_set()._waveform_group:
            raise AttributeError("Attribute '%s' not found." % item)
        return WaveformAccessor(item, self.__data_set())

    def __getitem__(self, item):
        # Item access with replaced underscore and without. This is not
        # strictly valid ASDF but it helps to be flexible.
        try:
            return self.__getattr__(item, replace=False)
        except AttributeError:
            try:
                return self.__getattr__(item, replace=True)
            except AttributeError as e:
                raise KeyError(str(e))

    def __contains__(self, item):
        return item in [_i._station_name for _i in self]

    def __delattr__(self, item):
        # Triggers an AttributeError if the item does not exist.
        attr = getattr(self, item)
        # Only delete waveform accessors.
        if isinstance(attr, WaveformAccessor):
            ds = self.__data_set()
            del ds._waveform_group[attr._station_name]
            del ds
        else:
            raise AttributeError("Attribute '%s' cannot be deleted." % item)

    def __delitem__(self, key):
        # Triggers a KeyError if the key does not exist.
        attr = self[key]
        # Only delete waveform accessors.
        ds = self.__data_set()
        del ds._waveform_group[attr._station_name]
        del ds

    def list(self):
        return sorted(self.__data_set()._waveform_group.keys())

    def __dir__(self):
        return [_i.replace(".", "_") for _i in self.list()]

    def __len__(self):
        return len(self.list())

    def __iter__(self):
        for _i in self.list():
            yield self[_i]


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
    def __hdf5_group(self):
        """
        For internal use only! Returns the hdf5 group associated with this
        station. Make sure to delete the reference afterwards as HDF5 is
        very picky about dangling references especially with parallel I/O.
        """
        return self.__data_set()._waveform_group[self._station_name]

    def filter_waveforms(self, queries):
        wfs = []

        for wf in [_i for _i in self.list() if _i != "StationXML"]:
            # Location and channel codes.
            if queries["location"] or queries["channel"]:
                _, _, loc_code, cha_code = wf_name2seed_codes(wf)
                if queries["location"] is not None:
                    if queries["location"](loc_code) is False:
                        continue
                if queries["channel"] is not None:
                    if queries["channel"](cha_code) is False:
                        continue
            # Tag
            if queries["tag"]:
                tag = wf_name2tag(wf)
                if queries["tag"](tag) is False:
                    continue

            # Any of the other queries requires parsing the attribute path.
            if (
                queries["starttime"]
                or queries["endtime"]
                or queries["sampling_rate"]
                or queries["npts"]
                or queries["event"]
                or queries["magnitude"]
                or queries["origin"]
                or queries["focal_mechanism"]
                or queries["labels"]
            ):

                group = self.__hdf5_group
                try:
                    attrs = group[wf].attrs

                    if queries["sampling_rate"]:
                        if (
                            queries["sampling_rate"](attrs["sampling_rate"])
                            is False
                        ):
                            continue

                    if queries["npts"]:
                        if queries["npts"](len(group[wf])) is False:
                            continue

                    if queries["starttime"] or queries["endtime"]:
                        starttime = obspy.UTCDateTime(
                            float(attrs["starttime"]) / 1.0e9
                        )
                        endtime = starttime + (
                            len(group[wf]) - 1
                        ) * 1.0 / float(attrs["sampling_rate"])

                        if queries["starttime"]:
                            if queries["starttime"](starttime) is False:
                                continue

                        if queries["endtime"]:
                            if queries["endtime"](endtime) is False:
                                continue

                    if queries["labels"]:
                        if "labels" in attrs:
                            labels = labelstring2list(attrs["labels"])
                        else:
                            labels = None
                        if queries["labels"](labels) is False:
                            continue

                    if (
                        queries["event"]
                        or queries["magnitude"]
                        or queries["origin"]
                        or queries["focal_mechanism"]
                    ):
                        any_fails = False
                        for id in [
                            "event",
                            "origin",
                            "magnitude",
                            "focal_mechanism",
                        ]:
                            if queries[id]:
                                key = id + "_id"

                                # Defaults to an empty id. Each can
                                # potentially be a list of values.
                                if key in attrs:
                                    values = (
                                        attrs[key]
                                        .tobytes()
                                        .decode()
                                        .split(",")
                                    )
                                else:
                                    values = [None]

                                if not any(queries[id](_i) for _i in values):
                                    any_fails = True
                                    break
                        if any_fails is True:
                            continue

                finally:
                    del group
            wfs.append(wf)

        return wfs

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self._station_name != other._station_name:
            return False

        ds1 = self.__data_set()
        ds2 = other.__data_set()
        try:
            if ds1.filename != ds2.filename:
                return False
        finally:
            del ds1
            del ds2

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_waveform_attributes(self):
        """
        Get a dictionary of the attributes of all waveform data sets.

        This works purely on the meta data level and is thus fast. Event,
        origin, arrival, and focmec ids will be returned as lists as they can
        be set multiple times per data-set.
        """
        plural_keys = [
            "event_id",
            "origin_id",
            "magnitude_id",
            "focal_mechanism_id",
        ]

        attributes = {}
        for _i in self.list():
            if _i == "StationXML":
                continue

            attrs = {}
            for k, v in self.__hdf5_group[_i].attrs.items():
                # Make sure its an actual string.
                try:
                    v = v.decode()
                except Exception:
                    pass

                if k in plural_keys:
                    k += "s"
                    v = v.split(",")
                attrs[k] = v
            if attrs:
                attributes[_i] = attrs

        return attributes

    @property
    def coordinates(self):
        """
        Get coordinates of the station if any.
        """
        coords = self.__get_coordinates(level="station")
        # Such a file actually cannot be created with pyasdf but maybe with
        # other codes. Thus we skip the coverage here.
        if self._station_name not in coords:  # pragma: no cover
            raise ASDFValueError(
                "StationXML file has no coordinates for "
                "station '%s'." % self._station_name
            )
        return coords[self._station_name]

    @property
    def channel_coordinates(self):
        """
        Get coordinates of the station at the channel level if any.
        """
        coords = self.__get_coordinates(level="channel")
        # Filter to only keep channels with the current station name.
        coords = {
            key: value
            for key, value in coords.items()
            if key.startswith(self._station_name + ".")
        }
        if not coords:
            raise ASDFValueError(
                "StationXML file has no coordinates at "
                "the channel level for station '%s'." % self._station_name
            )
        return coords

    def __get_coordinates(self, level):
        """
        Helper function.
        """
        station = self.__data_set()._waveform_group[self._station_name]
        if "StationXML" not in station:
            raise NoStationXMLForStation(
                "Station '%s' has no StationXML " "file." % self._station_name
            )
        try:
            with io.BytesIO(_read_string_array(station["StationXML"])) as buf:
                coordinates = get_coordinates(buf, level=level)
        finally:
            # HDF5 reference are tricky...
            del station

        return coordinates

    def __getitem__(self, item):
        try:
            return self.__getattr__(item)
        except (AttributeError, WaveformNotInFileException) as e:
            raise KeyError(str(e))

    def get_waveform_tags(self):
        """
        Get all available waveform tags for this station.
        """
        return sorted(
            set(_i.split("__")[-1] for _i in self.list() if _i != "StationXML")
        )

    def __contains__(self, item):
        contents = self.list()
        contents.extend(self.__dir__())
        return item in contents

    def __delitem__(self, key):
        try:
            self.__delete_data(key)
        # This cannot really happen as it will be caught by the
        # __filter_data() method.
        except AttributeError as e:  # pragma: nocover
            raise KeyError(str(e))

    def __delattr__(self, item):
        self.__delete_data(item)

    def __delete_data(self, item):
        """
        Internal delete method.
        """
        items = self.__filter_data(item)

        h5 = self.__hdf5_group
        try:
            for item in items:
                if item in h5:
                    del h5[item]
        finally:
            del h5

    def __filter_data(self, item):
        """
        Internal filtering for item access and deletion.
        """
        # StationXML access.
        if item == "StationXML":
            return [item]

        _l = self.list()

        # Single trace access
        if item in _l:
            return [item]
        # Tag access.
        elif "__" not in item:
            __station = self.__data_set()._waveform_group[self._station_name]
            keys = [_i for _i in __station.keys() if _i.endswith("__" + item)]
            # Important as __del__() for the waveform group is otherwise
            # not always called.
            del __station

            # Filter by everything in self.list(). Once to make sure its
            # up-to-date and also for the filtered subclasses of this.
            keys = [_i for _i in keys if _i in _l]

            if not keys:
                raise WaveformNotInFileException(
                    "Tag '%s' not part of the data set for station '%s'."
                    % (item, self._station_name)
                )
            return keys

        raise AttributeError("Item '%s' not found." % item)

    def __getattr__(self, item):
        return self.get_item(item=item)

    def get_item(self, item, starttime=None, endtime=None):
        items = self.__filter_data(item)
        # StationXML access.
        if items == ["StationXML"]:
            station = self.__data_set()._get_station(self._station_name)
            if station is None:
                raise AttributeError(
                    "'%s' object has no attribute '%s'"
                    % (self.__class__.__name__, str(item))
                )
            return station

        # Get an estimate of the total require memory. But only estimate it
        # if the file is actually larger than the memory as the test is fairly
        # expensive but we don't really care for small files.
        if (
            self.__data_set().filesize
            > self.__data_set().single_item_read_limit_in_mb * 1024 ** 2
        ):
            total_size = sum(
                [
                    self.__data_set()._get_idx_and_size_estimate(
                        _i, starttime=starttime, endtime=endtime
                    )[3]
                    for _i in items
                ]
            )

            # Raise an error to not read an extreme amount of data into memory.
            if total_size > self.__data_set().single_item_read_limit_in_mb:
                msg = (
                    "All waveforms for station '%s' and item '%s' would "
                    "require '%.2f MB of memory. The current limit is %.2f "
                    "MB. Adjust by setting "
                    "'ASDFDataSet.single_item_read_limit_in_mb' or use a "
                    "different method to read the waveform data."
                    % (
                        self._station_name,
                        item,
                        total_size,
                        self.__data_set().single_item_read_limit_in_mb,
                    )
                )
                raise ASDFValueError(msg)

        traces = [
            self.__data_set()._get_waveform(
                _i, starttime=starttime, endtime=endtime
            )
            for _i in items
        ]
        return obspy.Stream(traces=traces)

    def list(self):
        """
        Get a list of all data sets for this station.
        """
        return sorted(
            self.__data_set()._waveform_group[self._station_name].keys()
        )

    def __dir__(self):
        """
        The dir method will list all this object's methods, the StationXML
        if it has one, and all tags.
        """
        # Python 3.
        if hasattr(object, "__dir__"):  # pragma: no cover
            directory = object.__dir__(self)
        # Python 2.
        else:  # pragma: no cover
            directory = [
                _i
                for _i in self.__dict__.keys()
                if not _i.startswith("_" + self.__class__.__name__)
            ]

        directory.extend(self.get_waveform_tags())
        if "StationXML" in self.list():
            directory.append("StationXML")
        directory.extend(
            ["_station_name", "coordinates", "channel_coordinates"]
        )
        return sorted(set(directory))

    def __str__(self):
        contents = self.__dir__()
        waveform_contents = sorted(self.get_waveform_tags())

        ret_str = (
            "Contents of the data set for station {station}:\n"
            "    - {station_xml}\n"
            "    - {count} Waveform Tag(s):\n"
            "        {waveforms}"
        )
        return ret_str.format(
            station=self._station_name,
            station_xml="Has a StationXML file"
            if "StationXML" in contents
            else "Has no StationXML file",
            count=len(waveform_contents),
            waveforms="\n        ".join(waveform_contents),
        )

    def _repr_pretty_(self, p, cycle):  # pragma: no cover
        p.text(self.__str__())


class FilteredWaveformAccessor(WaveformAccessor):
    """
    A version of the waveform accessor returning a limited number of waveforms.
    """

    def __init__(self, station_name, asdf_data_set, filtered_items):
        WaveformAccessor.__init__(
            self, station_name=station_name, asdf_data_set=asdf_data_set
        )
        self._filtered_items = filtered_items

    def list(self):
        # Get all keys and accept only the filtered items and the station
        # information. This makes sure the filtered waveforms are somewhat
        # up-to-date.
        items = sorted(
            self._WaveformAccessor__data_set()
            ._waveform_group[self._station_name]
            .keys()
        )

        return [
            _i
            for _i in items
            if _i in self._filtered_items or _i == "StationXML"
        ]

    def __str__(self):
        content = WaveformAccessor.__str__(self)
        return re.sub("^Contents", "Filtered contents", content)


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


class StreamBuffer(collections.abc.MutableMapping):
    """
    Very simple key value store for obspy stream object with the additional
    ability to approximate the size of all stored stream objects.
    """

    def __init__(self):
        self.__streams = {}

    def __getitem__(self, key):
        return self.__streams[key]

    def __setitem__(self, key, value):
        if not isinstance(value, obspy.Stream) and value is not None:
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
            # Skip None values.
            if stream is None:
                continue
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
        return "Job(arguments=%s, result=%s)" % (
            str(self.arguments),
            str(self.result),
        )


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
        job = [
            _i
            for _i in self._workers[worker_name].active_jobs
            if _i.arguments == arguments
        ]
        if len(job) == 0:
            msg = "MASTER: Job %s from worker %i not found. All jobs: %s\n" % (
                str(arguments),
                worker_name,
                str(self._workers[worker_name].active_jobs),
            )
            raise ValueError(msg)
        if len(job) > 1:
            raise ValueError(
                "WTF %i %s %s"
                % (
                    worker_name,
                    str(arguments),
                    str(self._workers[worker_name].active_jobs),
                )
            )
        job = job[0]
        job.result = result

        self._workers[worker_name].active_jobs.remove(job)
        self._workers[worker_name].completed_jobs_count[0] += 1
        self._finished_jobs.append(job)

    def __str__(self):
        workers = "\n\t".join(
            [
                "Worker %s: %i active, %i completed jobs"
                % (
                    str(key),
                    len(value.active_jobs),
                    value.completed_jobs_count[0],
                )
                for key, value in self._workers.items()
            ]
        )

        return (
            "Jobs (running %.2f seconds): "
            "queued: %i | finished: %i | total: %i\n"
            "\t%s\n"
            % (
                time.time() - self._starttime,
                len(self._in_queue),
                len(self._finished_jobs),
                len(self._all_jobs),
                workers,
            )
        )

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

    colors = (
        colorama.Back.WHITE + colorama.Fore.MAGENTA,
        colorama.Back.WHITE + colorama.Fore.BLUE,
        colorama.Back.WHITE + colorama.Fore.GREEN,
        colorama.Back.WHITE + colorama.Fore.YELLOW,
        colorama.Back.WHITE + colorama.Fore.BLACK,
        colorama.Back.WHITE + colorama.Fore.RED,
        colorama.Back.WHITE + colorama.Fore.CYAN,
    )

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
    tag = (
        tag_colors[tags.index(tag) % len(tag_colors)]
        + tag
        + colorama.Style.RESET_ALL
    )

    first = (
        colorama.Fore.YELLOW + "MASTER  " + colorama.Fore.RESET
        if first == 0
        else colors[first % len(colors)]
        + ("WORKER %i" % first)
        + colorama.Style.RESET_ALL
    )
    second = (
        colorama.Fore.YELLOW + "MASTER  " + colorama.Fore.RESET
        if second == 0
        else colors[second % len(colors)]
        + ("WORKER %i" % second)
        + colorama.Style.RESET_ALL
    )

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
    try:
        parsed_url = urlparse(url)
    except AttributeError:
        raise ASDFValueError("Not a valid qualified name.")
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ASDFValueError("Not a valid qualified name.")
    return url, localname


def get_all_ids_for_prov_document(document):
    """
    Gets a all ids from a prov document as qualified names in the lxml style.
    """
    ids = _get_ids_from_bundle(document)
    for bundle in document.bundles:
        ids.extend(_get_ids_from_bundle(bundle))

    return sorted(set(ids))


def _get_ids_from_bundle(bundle):
    all_ids = []
    for record in bundle._records:
        if not hasattr(record, "identifier") or not record.identifier:
            continue
        identifier = record.identifier
        all_ids.append(
            "{%s}%s" % (identifier.namespace.uri, identifier.localpart)
        )
    return all_ids


def wf_name2seed_codes(tag):
    """
    Converts an ASDF waveform name to the corresponding SEED identifiers.

    >>> wf_name2seed_codes("BW.ALTM.00.EHE__2012-01-..__2012_01-...__synth")
    ("BW", "ALTM", "00", "EHE")
    """
    tag = tag.split(".")[:4]
    tag[-1] = tag[-1].split("__")[0]
    return tag


def wf_name2tag(tag):
    """
    Extract the path from an ASDF waveform name.

    >>> wf_name2seed_codes("BW.ALTM.00.EHE__2012-01-..__2012_01-...__synth")
    "synth"
    """
    return tag.split("__")[-1]


def label2string(labels):
    """
    List of labels to a comma-saperated string.
    """
    if labels is not None:
        for _i in labels:
            if "," in _i:
                raise ValueError(
                    "The labels must not contain a comma as it is used "
                    "as the separator for the different values."
                )
        labels = ",".join([_i.strip() for _i in labels])
    return labels


def labelstring2list(labels):
    """
    String to a list of labels.
    """
    return [_i.strip() for _i in labels.split(",")]


def is_list(obj):
    """
    True if the returned object is an iterable but not a string.

    :param obj: The object to test.
    """
    return isinstance(obj, collections.abc.Iterable) and not isinstance(
        obj, string_types
    )


def to_list_of_resource_identifiers(obj, name, obj_type):
    if not obj:
        return obj

    if is_list(obj) and not isinstance(obj, obj_type):
        return list(
            itertools.chain.from_iterable(
                [
                    to_list_of_resource_identifiers(_i, name, obj_type)
                    for _i in obj
                ]
            )
        )

    if isinstance(obj, obj_type):
        obj = obj.resource_id
    elif isinstance(obj, obspy.core.event.ResourceIdentifier):
        obj = obj
    else:
        try:
            obj = obspy.core.event.ResourceIdentifier(obj)
        except Exception:
            msg = "Invalid type for %s." % name
            raise TypeError(msg)
    return [obj]
