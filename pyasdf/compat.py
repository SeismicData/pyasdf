#!/usr/bin/env python
"""
Compatibility between Python version.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2013-2021
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import sys

if sys.version_info.major == 3:  # pragma: no cover
    string_types = (bytes, str)
    unicode_type = str
else:  # pragma: no cover
    string_types = (bytes, str, unicode)  # NOQA
    unicode_type = unicode  # NOQA
