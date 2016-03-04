#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the internal logic of the query helper module.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import obspy
import pytest

from .. import query as q


def test_query_object_float():
    # First return value is the name.
    assert (q.QueryObject(name="_r", type=float) < 2.0)[0] == "_r"

    assert (q.QueryObject(name="_r", type=float) < 2.0)[1](2.1) is False
    assert (q.QueryObject(name="_r", type=float) < 2.0)[1](2.0) is False
    assert (q.QueryObject(name="_r", type=float) < 2.0)[1](1.0) is True

    assert (q.QueryObject(name="_r", type=float) <= 2.0)[1](2.1) is False
    assert (q.QueryObject(name="_r", type=float) <= 2.0)[1](2.0) is True
    assert (q.QueryObject(name="_r", type=float) <= 2.0)[1](1.0) is True

    assert (q.QueryObject(name="_r", type=float) > 2.0)[1](2.1) is True
    assert (q.QueryObject(name="_r", type=float) > 2.0)[1](2.0) is False
    assert (q.QueryObject(name="_r", type=float) > 2.0)[1](1.0) is False

    assert (q.QueryObject(name="_r", type=float) >= 2.0)[1](2.1) is True
    assert (q.QueryObject(name="_r", type=float) >= 2.0)[1](2.0) is True
    assert (q.QueryObject(name="_r", type=float) >= 2.0)[1](1.0) is False

    assert (q.QueryObject(name="_r", type=float) == 2.0)[1](2.1) is False
    assert (q.QueryObject(name="_r", type=float) == 2.0)[1](2.0) is True

    assert (q.QueryObject(name="_r", type=float) != 2.0)[1](2.1) is True
    assert (q.QueryObject(name="_r", type=float) != 2.0)[1](2.0) is False

    # Many values are autoconverted.
    assert (q.QueryObject(name="_r", type=float) < 2)[1](3.0) is False
    assert (q.QueryObject(name="_r", type=float) < 2)[1](2.0) is False
    assert (q.QueryObject(name="_r", type=float) < 2)[1](1.0) is True

    assert (q.QueryObject(name="_r", type=float) < "2")[1](3.0) is False
    assert (q.QueryObject(name="_r", type=float) < "2")[1](2.0) is False
    assert (q.QueryObject(name="_r", type=float) < "2")[1](1.0) is True

    assert (q.QueryObject(name="_r", type=float) < "2.0")[1](3.0) is False
    assert (q.QueryObject(name="_r", type=float) < "2.0")[1](2.0) is False
    assert (q.QueryObject(name="_r", type=float) < "2.0")[1](1.0) is True


def test_query_object_utcdatetime():
    utc = obspy.UTCDateTime

    a = obspy.UTCDateTime(2012, 1, 1, 1)
    ref = obspy.UTCDateTime(2012, 1, 1, 2)
    c = obspy.UTCDateTime(2012, 1, 1, 3)
    # First return value is the name.
    assert (q.QueryObject(name="_r", type=utc) < ref)[0] == "_r"

    assert (q.QueryObject(name="_r", type=utc) < ref)[1](c) is False
    assert (q.QueryObject(name="_r", type=utc) < ref)[1](ref) is False
    assert (q.QueryObject(name="_r", type=utc) < ref)[1](a) is True

    assert (q.QueryObject(name="_r", type=utc) <= ref)[1](c) is False
    assert (q.QueryObject(name="_r", type=utc) <= ref)[1](ref) is True
    assert (q.QueryObject(name="_r", type=utc) <= ref)[1](a) is True

    assert (q.QueryObject(name="_r", type=utc) > ref)[1](c) is True
    assert (q.QueryObject(name="_r", type=utc) > ref)[1](ref) is False
    assert (q.QueryObject(name="_r", type=utc) > ref)[1](a) is False

    assert (q.QueryObject(name="_r", type=utc) >= ref)[1](c) is True
    assert (q.QueryObject(name="_r", type=utc) >= ref)[1](ref) is True
    assert (q.QueryObject(name="_r", type=utc) >= ref)[1](a) is False

    assert (q.QueryObject(name="_r", type=utc) == ref)[1](c) is False
    assert (q.QueryObject(name="_r", type=utc) == ref)[1](ref) is True

    assert (q.QueryObject(name="_r", type=utc) != ref)[1](c) is True
    assert (q.QueryObject(name="_r", type=utc) != ref)[1](ref) is False

    # Many values are autoconverted.
    assert (q.QueryObject(name="_r", type=utc) < str(ref))[1](c) is False
    assert (q.QueryObject(name="_r", type=utc) < str(ref))[1](ref) is False
    assert (q.QueryObject(name="_r", type=utc) < str(ref))[1](a) is True

    assert (q.QueryObject(name="_r", type=utc) < ref.timestamp)[1](c) is False
    assert (q.QueryObject(name="_r", type=utc) < ref.timestamp)[1](ref) \
        is False
    assert (q.QueryObject(name="_r", type=utc) < ref.timestamp)[1](a) \
        is True


def test_query_strings():
    # string types need a special type setting.
    _w = q._wildcarded_list

    # First return value is the name.
    assert (q.QueryObject(name="_r", type=_w) == "aa")[0] == "_r"

    # Standard queries.
    assert (q.QueryObject(name="_r", type=_w) == "test")[1]("test") is True
    assert (q.QueryObject(name="_r", type=_w) == "test")[1]("test2") is False

    assert (q.QueryObject(name="_r", type=_w) != "test")[1]("test") is False
    assert (q.QueryObject(name="_r", type=_w) != "test")[1]("test2") is True

    # Works with unix wildcards.
    assert (q.QueryObject(name="_r", type=_w) == "tes?")[1]("test") is True
    assert (q.QueryObject(name="_r", type=_w) == "test")[1]("test2") is False

    assert (q.QueryObject(name="_r", type=_w) != "tes?")[1]("test") is False
    assert (q.QueryObject(name="_r", type=_w) != "test")[1]("test2") is True

    assert (q.QueryObject(name="_r", type=_w) == "tes*")[1]("test") is True
    assert (q.QueryObject(name="_r", type=_w) == "test")[1]("test2") is False

    assert (q.QueryObject(name="_r", type=_w) != "tes*")[1]("test") is False
    assert (q.QueryObject(name="_r", type=_w) != "test")[1]("test2") is True

    assert (q.QueryObject(name="_r", type=_w) == "te?t")[1]("test") is True
    assert (q.QueryObject(name="_r", type=_w) == "test")[1]("test2") is False

    assert (q.QueryObject(name="_r", type=_w) != "te?t")[1]("test") is False
    assert (q.QueryObject(name="_r", type=_w) != "test")[1]("test2") is True

    # Also works with list of strings.
    assert (q.QueryObject(name="_r", type=_w) ==
            ["aa?", "b*b", "hallo"])[1]("aa1") is True
    assert (q.QueryObject(name="_r", type=_w) !=
            ["aa?", "b*b", "hallo"])[1]("aa1") is False

    assert (q.QueryObject(name="_r", type=_w) ==
            ["aa?", "b*b", "hallo"])[1]("basdb") is True
    assert (q.QueryObject(name="_r", type=_w) !=
            ["aa?", "b*b", "hallo"])[1]("basdb") is False

    assert (q.QueryObject(name="_r", type=_w) ==
            ["aa?", "b*b", "hallo"])[1]("hallo") is True
    assert (q.QueryObject(name="_r", type=_w) !=
            ["aa?", "b*b", "hallo"])[1]("hallo") is False

    assert (q.QueryObject(name="_r", type=_w) ==
            ["aa?", "b*b", "hallo"])[1]("hallo2") is False
    assert (q.QueryObject(name="_r", type=_w) !=
            ["aa?", "b*b", "hallo"])[1]("hallo2") is True


def test_event_id():
    # Wildcards don't work here as a question mark and asterix are perfectly
    # valid URL components and the question mark is very very common.
    _t = q._event_or_id

    a = "smi:local/dummy_a"
    b = "smi:local/dummy_b"

    r_a = obspy.core.event.ResourceIdentifier(a)
    r_b = obspy.core.event.ResourceIdentifier(b)

    ev_a = obspy.read_events()[0]
    ev_a.resource_id = r_a

    ev_b = obspy.read_events()[0]
    ev_b.resource_id = r_b

    # First return value is the name.
    assert (q.QueryObject(name="_r", type=_t) == a)[0] == "_r"

    assert (q.QueryObject(name="_r", type=_t) == a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == a)[1](r_a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](r_b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](r_a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](r_b) is True

    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == a)[1](ev_a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](ev_b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](ev_a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](ev_b) is True

    assert (q.QueryObject(name="_r", type=_t) == ev_a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == ev_a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != ev_a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != ev_a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](ev_a) is True
    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](ev_b) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](ev_a) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](ev_b) is True

    assert (q.QueryObject(name="_r", type=_t) == ev_a)[1](r_a) is True
    assert (q.QueryObject(name="_r", type=_t) == ev_a)[1](r_b) is False
    assert (q.QueryObject(name="_r", type=_t) != ev_a)[1](r_a) is False
    assert (q.QueryObject(name="_r", type=_t) != ev_a)[1](r_b) is True

    # Test queries with None.
    assert (q.QueryObject(name="_r", type=_t) == None)[1](None) is True  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1](None) is False  # noqa

    assert (q.QueryObject(name="_r", type=_t) == None)[1]("random") is False  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1]("random") is True  # noqa

    assert (q.QueryObject(name="_r", type=_t) == "random")[1](None) is False
    assert (q.QueryObject(name="_r", type=_t) != "random")[1](None) is True


def test_origin_id():
    # Wildcards don't work here as a question mark and asterix are perfectly
    # valid URL components and the question mark is very very common.
    _t = q._origin_or_id

    a = "smi:local/dummy_a_origin"
    b = "smi:local/dummy_b_origin"

    r_a = obspy.core.event.ResourceIdentifier(a)
    r_b = obspy.core.event.ResourceIdentifier(b)

    org_a = obspy.read_events()[0].origins[0]
    org_a.resource_id = r_a

    org_b = obspy.read_events()[0].origins[0]
    org_b.resource_id = r_b

    # First return value is the name.
    assert (q.QueryObject(name="_r", type=_t) == a)[0] == "_r"

    assert (q.QueryObject(name="_r", type=_t) == a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == a)[1](r_a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](r_b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](r_a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](r_b) is True

    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == a)[1](org_a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](org_b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](org_a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](org_b) is True

    assert (q.QueryObject(name="_r", type=_t) == org_a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == org_a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != org_a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != org_a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](org_a) is True
    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](org_b) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](org_a) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](org_b) is True

    assert (q.QueryObject(name="_r", type=_t) == org_a)[1](r_a) is True
    assert (q.QueryObject(name="_r", type=_t) == org_a)[1](r_b) is False
    assert (q.QueryObject(name="_r", type=_t) != org_a)[1](r_a) is False
    assert (q.QueryObject(name="_r", type=_t) != org_a)[1](r_b) is True

    # Test queries with None.
    assert (q.QueryObject(name="_r", type=_t) == None)[1](None) is True  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1](None) is False  # noqa

    assert (q.QueryObject(name="_r", type=_t) == None)[1]("random") is False  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1]("random") is True  # noqa

    assert (q.QueryObject(name="_r", type=_t) == "random")[1](None) is False
    assert (q.QueryObject(name="_r", type=_t) != "random")[1](None) is True


def test_magnitude_id():
    # Wildcards don't work here as a question mark and asterix are perfectly
    # valid URL components and the question mark is very very common.
    _t = q._magnitude_or_id

    a = "smi:local/dummy_a_magnitude"
    b = "smi:local/dummy_b_magnitude"

    r_a = obspy.core.event.ResourceIdentifier(a)
    r_b = obspy.core.event.ResourceIdentifier(b)

    mag_a = obspy.read_events()[0].magnitudes[0]
    mag_a.resource_id = r_a

    mag_b = obspy.read_events()[0].magnitudes[0]
    mag_b.resource_id = r_b

    # First return value is the name.
    assert (q.QueryObject(name="_r", type=_t) == a)[0] == "_r"

    assert (q.QueryObject(name="_r", type=_t) == a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == a)[1](r_a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](r_b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](r_a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](r_b) is True

    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == a)[1](mag_a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](mag_b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](mag_a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](mag_b) is True

    assert (q.QueryObject(name="_r", type=_t) == mag_a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == mag_a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != mag_a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != mag_a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](mag_a) is True
    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](mag_b) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](mag_a) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](mag_b) is True

    assert (q.QueryObject(name="_r", type=_t) == mag_a)[1](r_a) is True
    assert (q.QueryObject(name="_r", type=_t) == mag_a)[1](r_b) is False
    assert (q.QueryObject(name="_r", type=_t) != mag_a)[1](r_a) is False
    assert (q.QueryObject(name="_r", type=_t) != mag_a)[1](r_b) is True

    # Test queries with None.
    assert (q.QueryObject(name="_r", type=_t) == None)[1](None) is True  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1](None) is False  # noqa

    assert (q.QueryObject(name="_r", type=_t) == None)[1]("random") is False  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1]("random") is True  # noqa

    assert (q.QueryObject(name="_r", type=_t) == "random")[1](None) is False
    assert (q.QueryObject(name="_r", type=_t) != "random")[1](None) is True


def test_focmec_id():
    # Wildcards don't work here as a question mark and asterix are perfectly
    # valid URL components and the question mark is very very common.
    _t = q._focmec_or_id

    a = "smi:local/dummy_a_focmec"
    b = "smi:local/dummy_b_focmec"

    r_a = obspy.core.event.ResourceIdentifier(a)
    r_b = obspy.core.event.ResourceIdentifier(b)

    focmec_a = obspy.core.event.FocalMechanism(resource_id=r_a)
    focmec_b = obspy.core.event.FocalMechanism(resource_id=r_b)

    # First return value is the name.
    assert (q.QueryObject(name="_r", type=_t) == a)[0] == "_r"

    assert (q.QueryObject(name="_r", type=_t) == a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == a)[1](r_a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](r_b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](r_a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](r_b) is True

    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == a)[1](focmec_a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](focmec_b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](focmec_a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](focmec_b) is True

    assert (q.QueryObject(name="_r", type=_t) == focmec_a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == focmec_a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != focmec_a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != focmec_a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](focmec_a) is True
    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](focmec_b) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](focmec_a) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](focmec_b) is True

    assert (q.QueryObject(name="_r", type=_t) == focmec_a)[1](r_a) is True
    assert (q.QueryObject(name="_r", type=_t) == focmec_a)[1](r_b) is False
    assert (q.QueryObject(name="_r", type=_t) != focmec_a)[1](r_a) is False
    assert (q.QueryObject(name="_r", type=_t) != focmec_a)[1](r_b) is True

    # Test queries with None.
    assert (q.QueryObject(name="_r", type=_t) == None)[1](None) is True  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1](None) is False  # noqa

    assert (q.QueryObject(name="_r", type=_t) == None)[1]("random") is False  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1]("random") is True  # noqa

    assert (q.QueryObject(name="_r", type=_t) == "random")[1](None) is False
    assert (q.QueryObject(name="_r", type=_t) != "random")[1](None) is True


def test_none_float():
    """
    Test queries with floats that can also be None.
    """
    _t = q._type_or_none(float)

    assert (q.QueryObject(name="_f", type=_t) == 1.0)[1](1.0) is True
    assert (q.QueryObject(name="_f", type=_t) == 1.0)[1](0.0) is False
    assert (q.QueryObject(name="_f", type=_t) != 1.0)[1](1.0) is False
    assert (q.QueryObject(name="_f", type=_t) != 1.0)[1](0.0) is True

    # Identity for None.
    assert (q.QueryObject(name="_f", type=_t) == None)[1](None) is True  # noqa
    assert (q.QueryObject(name="_f", type=_t) != None)[1](None) is False  # noqa

    # Zero is not None.
    assert (q.QueryObject(name="_f", type=_t) == 0.0)[1](None) is False
    assert (q.QueryObject(name="_f", type=_t) != 0.0)[1](None) is True
    assert (q.QueryObject(name="_f", type=_t) == None)[1](0.0) is False  # noqa
    assert (q.QueryObject(name="_f", type=_t) != None)[1](0.0) is True  # noqa

    # Other queries still for for normal floats.
    assert (q.QueryObject(name="_f", type=_t) <= 0.0)[1](0.0) is True
    assert (q.QueryObject(name="_f", type=_t) < 0.0)[1](0.0) is False
    assert (q.QueryObject(name="_f", type=_t) >= 0.0)[1](0.0) is True
    assert (q.QueryObject(name="_f", type=_t) > 0.0)[1](0.0) is False

    # These make no sense for None and thus their creation should fail.
    with pytest.raises(TypeError):
        q.QueryObject(name="_f", type=_t) <= None
    with pytest.raises(TypeError):
        q.QueryObject(name="_f", type=_t) < None
    with pytest.raises(TypeError):
        q.QueryObject(name="_f", type=_t) >= None
    with pytest.raises(TypeError):
        q.QueryObject(name="_f", type=_t) > None

    # To be able to keep the rest working, evaluation still has to work.
    assert (q.QueryObject(name="_f", type=_t) <= 0.0)[1](None) is False
    assert (q.QueryObject(name="_f", type=_t) < 0.0)[1](None) is False
    assert (q.QueryObject(name="_f", type=_t) >= 0.0)[1](None) is False
    assert (q.QueryObject(name="_f", type=_t) > 0.0)[1](None) is False

    # If a normal not None float is used a type error should be raised no
    # matter where None is used.
    with pytest.raises(TypeError):
        q.QueryObject(name="_f", type=float) <= None
    with pytest.raises(TypeError):
        q.QueryObject(name="_f", type=float) < None
    with pytest.raises(TypeError):
        q.QueryObject(name="_f", type=float) >= None
    with pytest.raises(TypeError):
        q.QueryObject(name="_f", type=float) > None
    with pytest.raises(TypeError):
        q.QueryObject(name="_f", type=float) == None  # noqa
    with pytest.raises(TypeError):
        q.QueryObject(name="_f", type=float) != None  # noqa

    with pytest.raises(TypeError):
        (q.QueryObject(name="_f", type=float) <= 1.0)[1](None)
    with pytest.raises(TypeError):
        (q.QueryObject(name="_f", type=float) < 1.0)[1](None)
    with pytest.raises(TypeError):
        (q.QueryObject(name="_f", type=float) >= 1.0)[1](None)
    with pytest.raises(TypeError):
        (q.QueryObject(name="_f", type=float) > 1.0)[1](None)
    with pytest.raises(TypeError):
        (q.QueryObject(name="_f", type=float) == 1.0)[1](None)
    with pytest.raises(TypeError):
        (q.QueryObject(name="_f", type=float) != 1.0)[1](None)


def test_event_id_none():
    """
    Tests that event ids can be none if desired.
    """
    _t = q._type_or_none(q._event_or_id)

    a = "smi:local/dummy_a"
    b = "smi:local/dummy_b"

    r_a = obspy.core.event.ResourceIdentifier(a)
    r_b = obspy.core.event.ResourceIdentifier(b)

    ev_a = obspy.read_events()[0]
    ev_a.resource_id = r_a

    ev_b = obspy.read_events()[0]
    ev_b.resource_id = r_b

    # First return value is the name.
    assert (q.QueryObject(name="_r", type=_t) == a)[0] == "_r"

    # All normal queries should still work.
    assert (q.QueryObject(name="_r", type=_t) == a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == a)[1](r_a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](r_b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](r_a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](r_b) is True

    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == a)[1](ev_a) is True
    assert (q.QueryObject(name="_r", type=_t) == a)[1](ev_b) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](ev_a) is False
    assert (q.QueryObject(name="_r", type=_t) != a)[1](ev_b) is True

    assert (q.QueryObject(name="_r", type=_t) == ev_a)[1](a) is True
    assert (q.QueryObject(name="_r", type=_t) == ev_a)[1](b) is False
    assert (q.QueryObject(name="_r", type=_t) != ev_a)[1](a) is False
    assert (q.QueryObject(name="_r", type=_t) != ev_a)[1](b) is True

    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](ev_a) is True
    assert (q.QueryObject(name="_r", type=_t) == r_a)[1](ev_b) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](ev_a) is False
    assert (q.QueryObject(name="_r", type=_t) != r_a)[1](ev_b) is True

    assert (q.QueryObject(name="_r", type=_t) == ev_a)[1](r_a) is True
    assert (q.QueryObject(name="_r", type=_t) == ev_a)[1](r_b) is False
    assert (q.QueryObject(name="_r", type=_t) != ev_a)[1](r_a) is False
    assert (q.QueryObject(name="_r", type=_t) != ev_a)[1](r_b) is True

    # But queries with None also should yield something.
    assert (q.QueryObject(name="_r", type=_t) == None)[1](a) is False  # noqa
    assert (q.QueryObject(name="_r", type=_t) == None)[1](b) is False  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1](a) is True  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1](b) is True  # noqa

    assert (q.QueryObject(name="_r", type=_t) == None)[1](None) is True  # noqa
    assert (q.QueryObject(name="_r", type=_t) == None)[1](None) is True  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1](None) is False  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1](None) is False  # noqa


def test_queries_with_optional_and_wildcarded_lists():
    _t = q._type_or_none(q._wildcarded_list)

    # Simple tests.
    assert (q.QueryObject(name="_r", type=_t) == "a")[1]("a") is True
    assert (q.QueryObject(name="_r", type=_t) == "a")[1]("b") is False
    assert (q.QueryObject(name="_r", type=_t) != "a")[1]("a") is False
    assert (q.QueryObject(name="_r", type=_t) != "a")[1]("b") is True

    assert (q.QueryObject(name="_r", type=_t) == None)[1](None) is True  # noqa
    assert (q.QueryObject(name="_r", type=_t) == None)[1]("a") is False  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1](None) is False  # noqa
    assert (q.QueryObject(name="_r", type=_t) != None)[1]("a") is True  # noqa

    # Try some lists.
    assert (q.QueryObject(name="_r", type=_t) == "a")[1](["a"]) is True
    assert (q.QueryObject(name="_r", type=_t) == "a")[1](["b"]) is False
    assert (q.QueryObject(name="_r", type=_t) != "a")[1](["a"]) is False
    assert (q.QueryObject(name="_r", type=_t) != "a")[1](["b"]) is True
    assert (q.QueryObject(name="_r", type=_t) == ["a"])[1](["a"]) is True
    assert (q.QueryObject(name="_r", type=_t) == ["a"])[1](["b"]) is False
    assert (q.QueryObject(name="_r", type=_t) != ["a"])[1](["a"]) is False
    assert (q.QueryObject(name="_r", type=_t) != ["a"])[1](["b"]) is True
    assert (q.QueryObject(name="_r", type=_t) == ["a"])[1](["a"]) is True
    assert (q.QueryObject(name="_r", type=_t) == ["a"])[1](["b"]) is False
    assert (q.QueryObject(name="_r", type=_t) != ["a"])[1](["a"]) is False
    assert (q.QueryObject(name="_r", type=_t) != ["a"])[1](["b"]) is True

    assert (q.QueryObject(name="_r", type=_t) == ["a"])[1](None) is False
    assert (q.QueryObject(name="_r", type=_t) != ["a"])[1](None) is True

    # None in lists should raise an error.
    with pytest.raises(TypeError) as e:
        (q.QueryObject(name="_r", type=_t) == ["a"])[1]([None])
    assert e.value.args[0] == "List cannot contain a None value."

    with pytest.raises(TypeError) as e:
        (q.QueryObject(name="_r", type=_t) != ["a"])[1]([None])
    assert e.value.args[0] == "List cannot contain a None value."

    with pytest.raises(TypeError) as e:
        (q.QueryObject(name="_r", type=_t) != [None])[1]("a")
    assert e.value.args[0] == "List cannot contain a None value."

    with pytest.raises(TypeError) as e:
        (q.QueryObject(name="_r", type=_t) == [None])[1]("a")
    assert e.value.args[0] == "List cannot contain a None value."

    with pytest.raises(TypeError) as e:
        (q.QueryObject(name="_r", type=_t) == ["a"])[1](["a", None])
    assert e.value.args[0] == "List cannot contain a None value."

    with pytest.raises(TypeError) as e:
        (q.QueryObject(name="_r", type=_t) == ["a", None])[1]("a")
    assert e.value.args[0] == "List cannot contain a None value."

    # A bit more complex queries.
    assert (q.QueryObject(name="_r", type=_t) == ["a", "b"])[1]("a") is True
    assert (q.QueryObject(name="_r", type=_t) == "a")[1](["a", "b"]) is True
    assert (q.QueryObject(name="_r", type=_t) != ["a", "b"])[1]("a") is False
    assert (q.QueryObject(name="_r", type=_t) != "a")[1](["a", "b"]) is False

    # Wildcards.
    assert (q.QueryObject(name="_r", type=_t) == "a*")[1]("ab") is True
    assert (q.QueryObject(name="_r", type=_t) != "a*")[1]("ab") is False
    assert (q.QueryObject(name="_r", type=_t) == ["a*", "x"])[1]("ab") is True
    assert (q.QueryObject(name="_r", type=_t) != ["a*", "x"])[1]("ab") is False

    # List againts lists.
    assert (q.QueryObject(name="_r", type=_t) == ["a*", "x"])[1](
        ["b", "c"]) is False
    assert (q.QueryObject(name="_r", type=_t) != ["a*", "x"])[1](
        ["b", "c"]) is True
    assert (q.QueryObject(name="_r", type=_t) == ["a*", "x"])[1](
        ["ab", "c"]) is True
    assert (q.QueryObject(name="_r", type=_t) != ["a*", "x"])[1](
        ["ab", "c"]) is False


def test_query_merging():
    # Names are fixed here and must be as in the query files. This is not a
    # general purpose library but specific for this module here.

    merged_queries = q.merge_query_functions([
        q.QueryObject(name="sampling_rate", type=float) < 2,
        q.QueryObject(name="sampling_rate", type=float) >= 0
    ])

    fct = merged_queries["sampling_rate"]

    assert fct(1) is True
    assert fct(0) is True
    assert fct(1.99) is True
    assert fct(2) is False
    assert fct(-0.1) is False

    # Not-given parameters are None.
    assert merged_queries["tag"] is None

    # One more test combining complex string queries.
    _w = q._wildcarded_list

    merged_queries = q.merge_query_functions([
        q.QueryObject(name="network", type=_w) == ["BW", "AL?", "B*A"],
        q.QueryObject(name="network", type=_w) != "AL2"
    ])
    fct = merged_queries["network"]

    assert fct("BW") is True
    assert fct("AL1") is True
    assert fct("AL3") is True
    assert fct("BABA") is True

    assert fct("Hello") is False
    assert fct("AL2") is False
    assert fct("BAB") is False


def test_query_object():
    """
    Basic tests about the behaviour of the query object.
    """
    query = q.Query()
    # Makes tab completion work.
    assert "network" in dir(query)

    with pytest.raises(ValueError):
        _ = query.random  # NOQA
