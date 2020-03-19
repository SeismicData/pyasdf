#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test cases for the inventory utils.

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

import inspect
import os

import prov
import pytest

from ..exceptions import ASDFValueError
from ..utils import (
    split_qualified_name,
    get_all_ids_for_prov_document,
    SimpleBuffer,
    sizeof_fmt,
)

data_dir = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
    "data",
)


def test_split_qualified_name():
    url, localpart = split_qualified_name("{http://example.org}bla")
    assert url == "http://example.org"
    assert localpart == "bla"

    with pytest.raises(ASDFValueError) as err:
        split_qualified_name("{bla}bla")
    assert err.value.args[0] == "Not a valid qualified name."

    with pytest.raises(ASDFValueError) as err:
        split_qualified_name("{http:///example.org}bla")
    assert err.value.args[0] == "Not a valid qualified name."

    with pytest.raises(ASDFValueError) as err:
        split_qualified_name("{http://example.orgbla")
    assert err.value.args[0] == "Not a valid qualified name."


def test_get_ids_from_prov_document():
    filename = os.path.join(data_dir, "example_schematic_processing_chain.xml")
    doc = prov.read(filename, format="xml")
    ids = get_all_ids_for_prov_document(doc)
    assert ids == [
        "{http://seisprov.org/seis_prov/0.1/#}sp001_wf_a34j4didj3",
        "{http://seisprov.org/seis_prov/0.1/#}sp002_dt_f87sf7sf78",
        "{http://seisprov.org/seis_prov/0.1/#}sp003_wf_js83hf34aj",
        "{http://seisprov.org/seis_prov/0.1/#}sp004_lp_f87sf7sf78",
        "{http://seisprov.org/seis_prov/0.1/#}sp005_wf_378f8ks8kd",
        "{http://seisprov.org/seis_prov/0.1/#}sp006_dc_f87sf7sf78",
        "{http://seisprov.org/seis_prov/0.1/#}sp007_wf_jude89du8l",
    ]


def test_simple_cache_dictionary():
    xx = SimpleBuffer(limit=2)
    xx[1] = 1
    xx[2] = 2

    assert sorted(list(xx.values())) == [1, 2]

    xx[3] = 3

    assert sorted(list(xx.values())) == [2, 3]

    xx = SimpleBuffer(limit=2)
    xx[1] = 1
    xx[2] = 2

    # Access reorders the items.
    xx[1]
    xx[3] = 3
    assert sorted(list(xx.values())) == [1, 3]

    xx = SimpleBuffer(limit=2)
    xx[1] = 1
    xx[2] = 2
    xx[3] = 3
    xx[4] = 4
    assert sorted(list(xx.values())) == [3, 4]
    assert 3 in xx
    assert 4 in xx
    assert len(xx) == 2


def test_sizeof_fmt_function():
    assert sizeof_fmt(1024) == "1.0 KB"
    assert sizeof_fmt(1024 ** 2) == "1.0 MB"
    assert sizeof_fmt(1024 ** 3) == "1.0 GB"
    assert sizeof_fmt(1024 ** 4) == "1.0 TB"
