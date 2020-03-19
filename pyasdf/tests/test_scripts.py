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
import io
import os
import sys

import numpy as np
import obspy
from obspy.io.sac import SACTrace
import pytest

import pyasdf
from pyasdf.scripts import sac2asdf


def test_sac2asdf_failure_cases(tmpdir):
    tmpdir = tmpdir.strpath
    output_file = os.path.join(tmpdir, "out.h5")

    sys_argv_backup = copy.copy(sys.argv)
    try:
        sys.argv = sys.argv[:1]
        sys.argv.append(tmpdir)
        sys.argv.append(output_file)
        # Invalid tag.
        sys.argv.append("RaNdOm")
        with pytest.raises(ValueError):
            sac2asdf.__main__()
    finally:
        # Restore to not mess with any of pytests logic.
        sys.argv = sys_argv_backup

    sys_argv_backup = copy.copy(sys.argv)
    try:
        sys.argv = sys.argv[:1]
        # Not a folder
        sys.argv.append(os.path.join(tmpdir, "12345"))
        sys.argv.append(output_file)
        sys.argv.append("random")
        with pytest.raises(ValueError):
            sac2asdf.__main__()
    finally:
        # Restore to not mess with any of pytests logic.
        sys.argv = sys_argv_backup

    with io.open(output_file, "wt") as fh:
        fh.write("1")

    sys_argv_backup = copy.copy(sys.argv)
    try:
        sys.argv = sys.argv[:1]
        sys.argv.append(tmpdir)
        # File already exists.
        sys.argv.append(output_file)
        sys.argv.append("random")
        with pytest.raises(ValueError):
            sac2asdf.__main__()
    finally:
        # Restore to not mess with any of pytests logic.
        sys.argv = sys_argv_backup


def test_sac2asdf_script(tmpdir, capsys):
    tmpdir = tmpdir.strpath

    # Create some test data
    data_1 = np.arange(10, dtype=np.float32)
    header = {
        "kstnm": "ANMO",
        "knetwk": "IU",
        "kcmpnm": "BHZ",
        "stla": 40.5,
        "stlo": -108.23,
        "stel": 100.0,
        "stdp": 3.4,
        "evla": -15.123,
        "evlo": 123,
        "evdp": 50,
        "nzyear": 2012,
        "nzjday": 123,
        "nzhour": 13,
        "nzmin": 43,
        "nzsec": 17,
        "nzmsec": 100,
        "delta": 1.0 / 40,
        "o": -10.0,
    }
    sac = SACTrace(data=data_1, **header)
    sac.write(os.path.join(tmpdir, "a.sac"))

    data_2 = 2.0 * np.arange(10, dtype=np.float32)
    header = {
        "kstnm": "BBBB",
        "knetwk": "AA",
        "kcmpnm": "CCC",
        "stla": 40.5,
        "stlo": -108.23,
        "stel": 200.0,
        "stdp": 2.4,
        "evla": -14.123,
        "evlo": 125,
        "evdp": 30,
        "nzyear": 2013,
        "nzjday": 123,
        "nzhour": 13,
        "nzmin": 43,
        "nzsec": 17,
        "nzmsec": 100,
        "delta": 1.0 / 40,
        "o": 10.0,
    }
    sac = SACTrace(data=data_2, **header)
    sac.write(os.path.join(tmpdir, "b.sac"))

    output_file = os.path.join(tmpdir, "out.h5")
    assert not os.path.exists(output_file)

    sys_argv_backup = copy.copy(sys.argv)
    try:
        sys.argv = sys.argv[:1]
        sys.argv.append(tmpdir)
        sys.argv.append(output_file)
        sys.argv.append("random")
        sac2asdf.__main__()
    finally:
        # Restore to not mess with any of pytests logic.
        sys.argv = sys_argv_backup

    non_verbose_out, non_verbose_err = capsys.readouterr()
    assert not non_verbose_err

    assert os.path.exists(output_file)
    with pyasdf.ASDFDataSet(output_file, mode="r") as ds:
        # 2 Events.
        assert len(ds.events) == 2
        # 2 Stations.
        assert len(ds.waveforms) == 2

        events = ds.events  # NOQA

        # Data should actually be fully identical
        np.testing.assert_equal(data_1, ds.waveforms.IU_ANMO.random[0].data)
        np.testing.assert_equal(data_2, ds.waveforms.AA_BBBB.random[0].data)

        assert ds.waveforms.IU_ANMO.random[0].id == "IU.ANMO..BHZ"
        assert ds.waveforms.AA_BBBB.random[0].id == "AA.BBBB..CCC"

        c = ds.waveforms.IU_ANMO.coordinates
        np.testing.assert_allclose(
            [c["latitude"], c["longitude"], c["elevation_in_m"]],
            [40.5, -108.23, 100.0],
        )
        c = ds.waveforms.AA_BBBB.coordinates
        np.testing.assert_allclose(
            [c["latitude"], c["longitude"], c["elevation_in_m"]],
            [40.5, -108.23, 200.0],
        )

        c = ds.waveforms.IU_ANMO.channel_coordinates["IU.ANMO..BHZ"][0]
        np.testing.assert_allclose(
            [
                c["latitude"],
                c["longitude"],
                c["elevation_in_m"],
                c["local_depth_in_m"],
            ],
            [40.5, -108.23, 100.0, 3.4],
        )
        c = ds.waveforms.AA_BBBB.channel_coordinates["AA.BBBB..CCC"][0]
        np.testing.assert_allclose(
            [
                c["latitude"],
                c["longitude"],
                c["elevation_in_m"],
                c["local_depth_in_m"],
            ],
            [40.5, -108.23, 200.0, 2.4],
        )

        # Events
        origin = (
            ds.waveforms.IU_ANMO.random[0]
            .stats.asdf.event_ids[0]
            .get_referred_object()
            .origins[0]
        )
        np.testing.assert_allclose(
            [origin.latitude, origin.longitude, origin.depth],
            [-15.123, 123.0, 50.0],
        )
        assert (
            origin.time
            == obspy.UTCDateTime(
                year=2012,
                julday=123,
                hour=13,
                minute=43,
                second=17,
                microsecond=100000,
            )
            - 10.0
        )

        origin = (
            ds.waveforms.AA_BBBB.random[0]
            .stats.asdf.event_ids[0]
            .get_referred_object()
            .origins[0]
        )
        np.testing.assert_allclose(
            [origin.latitude, origin.longitude, origin.depth],
            [-14.123, 125.0, 30.0],
        )
        assert (
            origin.time
            == obspy.UTCDateTime(
                year=2013,
                julday=123,
                hour=13,
                minute=43,
                second=17,
                microsecond=100000,
            )
            + 10.0
        )

    # Run once again in verbose mode but just test that the output is
    # actually more.
    os.remove(output_file)
    sys_argv_backup = copy.copy(sys.argv)
    try:
        sys.argv = sys.argv[:1]
        sys.argv.append("--verbose")
        sys.argv.append(tmpdir)
        sys.argv.append(output_file)
        sys.argv.append("random")
        sac2asdf.__main__()
    finally:
        # Restore to not mess with any of pytests logic.
        sys.argv = sys_argv_backup
    verbose_out, verbose_err = capsys.readouterr()
    assert not verbose_err
    assert len(verbose_out) > len(non_verbose_out)
