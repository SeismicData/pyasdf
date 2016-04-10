#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compatibility between Python version.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

if sys.version_info.major == 3:  # pragma: no cover
    string_types = (bytes, str)
    unicode_type = str
else:  # pragma: no cover
    string_types = (bytes, str, unicode)  # NOQA
    unicode_type = unicode  # NOQA
