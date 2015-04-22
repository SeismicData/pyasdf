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

import copy
import inspect
import os

import obspy

from ..inventory_utils import isolate_and_merge_station, merge_inventories


data_dir = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


def test_merging_stations():
    """
    Tests reading a StationXML file with a couple of networks and duplicate
    stations and merging it.
    """
    inv = obspy.read_inventory(os.path.join(data_dir, "big_station.xml"),
                               format="stationxml")
    original_inv = copy.deepcopy(inv)

    assert len(inv.networks) == 2
    assert len(inv.select(network="BW")[0].stations) == 3

    new_inv = isolate_and_merge_station(inv, network_id="BW",
                                        station_id="RJOB")

    # The inventory object should also not be touched.
    assert inv == original_inv

    assert len(new_inv.networks) == 1
    assert len(new_inv[0].stations) == 1
    assert new_inv[0].code == "BW"
    assert new_inv[0][0].code == "RJOB"

    # Make sure the station dates have been set correctly.
    assert new_inv[0][0].start_date == \
       obspy.UTCDateTime("2001-05-15T00:00:00.000000Z")
    assert new_inv[0][0].end_date is None

    # The 9 channels should remain.
    assert len(new_inv[0][0].channels) == 9


def test_merge_inventories():
    """
    Silly test, merging the same inventory twice should result in the same
    as the test_merging_stations() test.
    """
    inv = obspy.read_inventory(os.path.join(data_dir, "big_station.xml"),
                               format="stationxml")
    original_inv = copy.deepcopy(inv)
    inv_2 = obspy.read_inventory(os.path.join(data_dir, "big_station.xml"),
                                 format="stationxml")

    assert len(inv.networks) == 2
    assert len(inv.select(network="BW")[0].stations) == 3

    new_inv = merge_inventories(inv, inv_2, network_id="BW", station_id="RJOB")

    # The inventory object should also not be touched.
    assert inv == original_inv

    assert len(new_inv.networks) == 1
    assert len(new_inv[0].stations) == 1
    assert new_inv[0].code == "BW"
    assert new_inv[0][0].code == "RJOB"

    # Make sure the station dates have been set correctly.
    assert new_inv[0][0].start_date == \
           obspy.UTCDateTime("2001-05-15T00:00:00.000000Z")
    assert new_inv[0][0].end_date is None

    # The 9 channels should remain.
    assert len(new_inv[0][0].channels) == 9
