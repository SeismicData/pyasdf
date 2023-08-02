#!/usr/bin/env python
"""
:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2013-2021
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""


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


class NoStationXMLForStation(ASDFException):
    """
    Raised when a station has no associated StationXML file.
    """

    pass


class ASDFValueError(ASDFException, ValueError):
    """
    ASDF specific value error.
    """

    pass


class ASDFAttributeError(ASDFException, AttributeError):
    """
    ASDF specific attribute error.
    """

    pass


class ASDFWarning(UserWarning):
    """
    Generic ASDF warning.
    """

    pass
