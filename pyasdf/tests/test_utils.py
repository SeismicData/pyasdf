#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test cases for the inventory utils.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest

from ..exceptions import ASDFValueError
from ..utils import split_qualified_name


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
