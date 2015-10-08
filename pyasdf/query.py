#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query helpers.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import fnmatch
import functools
import operator

import obspy

from .compat import string_types


def _wildcarded_list(value):
    if isinstance(value, string_types):
        value = [_i.strip() for _i in str(value).split(",")]
    if not isinstance(value, collections.Iterable):
        raise TypeError
    return list(value)


def _event_or_id(value):
    if isinstance(value, obspy.core.event.Event):
        return str(value.resource_id)
    elif isinstance(value, obspy.core.event.ResourceIdentifier):
        return str(value)
    elif not isinstance(value, string_types):
        raise TypeError
    return str(value)


def _magnitude_or_id(value):
    if isinstance(value, obspy.core.event.Magnitude):
        return str(value.resource_id)
    elif isinstance(value, obspy.core.event.ResourceIdentifier):
        return str(value)
    elif not isinstance(value, string_types):
        raise TypeError
    return str(value)


def _origin_or_id(value):
    if isinstance(value, obspy.core.event.Origin):
        return str(value.resource_id)
    elif isinstance(value, obspy.core.event.ResourceIdentifier):
        return str(value)
    elif not isinstance(value, string_types):
        raise TypeError
    return str(value)


def _focmec_or_id(value):
    if isinstance(value, obspy.core.event.FocalMechanism):
        return str(value.resource_id)
    elif isinstance(value, obspy.core.event.ResourceIdentifier):
        return str(value)
    elif not isinstance(value, string_types):
        raise TypeError
    return str(value)


keywords = {
    "network": _wildcarded_list,
    "station": _wildcarded_list,
    "location": _wildcarded_list,
    "channel": _wildcarded_list,
    "path": _wildcarded_list,
    # Coordinates; require a StationXML file to be around.
    "longitude": float,
    "latitude": float,
    "elevation_in_m": float,
    # Temporal constraints.
    "starttime": obspy.UTCDateTime,
    "endtime": obspy.UTCDateTime,
    "sampling_rate": float,
    # Having integers complicates some things quite a bit and all IEEE
    # floating points can exactly represent integers up to a very large
    # number. All values are autoconverted from any input type so everything
    # just works.
    "npts": float,
    "event": _event_or_id,
    "magnitude": _magnitude_or_id,
    "origin": _origin_or_id,
    "focal_mechanism": _focmec_or_id}


def compose_and(funcs):
    def __temp(value):
        # Make a copy to turn a generator into a list to be able to persist.
        functions = list(funcs)
        return functools.reduce(lambda x, y: x and y,
                                [_i(value) for _i in functions])
    return __temp


def compose_or(funcs):
    def __temp(value):
        # Make a copy to turn a generator into a list to be able to persist.
        functions = list(funcs)
        if not functions:
            return True
        return functools.reduce(lambda x, y: x or y,
                                [_i(value) for _i in functions])
    return __temp


class Query(object):
    def __init__(self):
        pass

    def __getattr__(self, item):
        if item not in keywords:
            raise ValueError

        return QueryObject(name=item, type=keywords[item])

    def __dir__(self):
        """
        Makes tab completion work for nice interactive usage.
        """
        return sorted(list(keywords.keys()))


def merge_query_functions(functions):
    query = {}

    for item in keywords:
        fcts = []
        for name, fct in functions:
            if name != item:
                continue
            fcts.append(fct)
        if not fcts:
            query[item] = None
        else:
            query[item] = compose_and(fcts)

    return query


class QueryObject(object):
    __slots__ = ["name", "type"]
    numeric_types = (float, obspy.UTCDateTime)

    def __init__(self, name, type):
        self.name = name
        self.type = type

    def _get_comp_fct(self, op, comp_value):
        def __temp(value):
            return op(self.type(value), comp_value)
        return __temp

    def _numeric_type(self, other, op):
        if self.type not in self.numeric_types:
            raise TypeError("Only valid for numeric types.")
        other = self.type(other)
        return self.name, self._get_comp_fct(op, other)

    def __lt__(self, other):
        return self._numeric_type(other, operator.lt)

    def __le__(self, other):
        return self._numeric_type(other, operator.le)

    def __gt__(self, other):
        return self._numeric_type(other, operator.gt)

    def __ge__(self, other):
        return self._numeric_type(other, operator.ge)

    def __eq__(self, other):
        other = self.type(other)

        if self.type is _wildcarded_list:
            def get_match_fct(_i):
                def match(value):
                    return fnmatch.fnmatch(value, _i)
                return match

            return self.name, \
                compose_or([get_match_fct(_i) for _i in other])

        return self.name, self._get_comp_fct(operator.eq, other)

    def __ne__(self, other):
        _, fct = self.__eq__(other)

        def negate_fct(fct):
            def __temp(value):
                return not fct(value)
            return __temp

        return self.name, negate_fct(fct)
