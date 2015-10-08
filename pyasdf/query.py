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

from .compat import string_types, unicode_type


def _wildcarded_list(value):
    if isinstance(value, string_types):
        value = [_i.strip() for _i in unicode_type(value).split(",")]
    if not isinstance(value, collections.Iterable):
        raise TypeError
    return list(value)


def _type_or_none(type):
    def __temp(value):
        if value is None:
            return None
        else:
            return type(value)
    __temp._original_type = type
    return __temp


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

_event_or_id = _type_or_none(_event_or_id)
_origin_or_id = _type_or_none(_origin_or_id)
_magnitude_or_id = _type_or_none(_magnitude_or_id)
_focmec_or_id = _type_or_none(_focmec_or_id)


keywords = {
    "network": _wildcarded_list,
    "station": _wildcarded_list,
    "location": _wildcarded_list,
    "channel": _wildcarded_list,
    "tag": _wildcarded_list,
    "labels": _type_or_none(_wildcarded_list),
    # Coordinates; require a StationXML file to be around.
    "longitude": _type_or_none(float),
    "latitude": _type_or_none(float),
    "elevation_in_m": _type_or_none(float),
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

    def _get_comp_fct(self, op, comp_value, none_allowed):
        def __temp(value):
            if none_allowed and comp_value is None and value is None:
                return True

            value = self.type(value)

            # Return False if comparing a value of None for an already created
            # valid query object.
            if value is None:
                return False

            return op(value, comp_value)
        return __temp

    def _numeric_type(self, other, op, none_allowed):
        if self.type not in self.numeric_types and \
                (self.type is _type_or_none and self.type._original_type not in
                 self.numeric_types):
            raise TypeError("Only valid for numeric types.")

        if other is None and none_allowed is False:
            raise TypeError("Comparison not defined with None.")

        other = self.type(other)
        return self.name, self._get_comp_fct(op, other,
                                             none_allowed=none_allowed)

    def __lt__(self, other):
        return self._numeric_type(other, operator.lt, none_allowed=False)

    def __le__(self, other):
        return self._numeric_type(other, operator.le, none_allowed=False)

    def __gt__(self, other):
        return self._numeric_type(other, operator.gt, none_allowed=False)

    def __ge__(self, other):
        return self._numeric_type(other, operator.ge, none_allowed=False)

    def __eq__(self, other):
        other = self.type(other)

        if self.type is _wildcarded_list:
            def get_match_fct(_i):
                def match(value):
                    return fnmatch.fnmatch(value, _i)
                return match

            return self.name, \
                compose_or([get_match_fct(_i) for _i in other])

        return self.name, self._get_comp_fct(operator.eq, other,
                                             none_allowed=True)

    def __ne__(self, other):
        _, fct = self.__eq__(other)

        def negate_fct(fct):
            def __temp(value):
                return not fct(value)
            return __temp

        return self.name, negate_fct(fct)
