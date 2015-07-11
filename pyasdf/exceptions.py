#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class ASDFException(Exception):
    """
    Generic exception for the Python ASDF implementation.
    """
    pass


class WaveformNotInFileException(ASDFException):
    """
    Raised when a non-existent waveform is accessed.
    """
    pass


class ASDFWarning(UserWarning):
    """
    Generic ASDF warning.
    """
    pass
