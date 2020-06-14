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

import copy
import inspect
import os

import obspy
from obspy import UTCDateTime

from ..inventory_utils import (
    isolate_and_merge_station,
    merge_inventories,
    get_coordinates,
)


data_dir = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
    "data",
)


def test_merging_stations():
    """
    Tests reading a StationXML file with a couple of networks and duplicate
    stations and merging it.
    """
    inv = obspy.read_inventory(
        os.path.join(data_dir, "big_station.xml"), format="stationxml"
    )
    original_inv = copy.deepcopy(inv)

    assert len(inv.networks) == 2
    assert len(inv.select(network="BW")[0].stations) == 3

    new_inv = isolate_and_merge_station(
        inv, network_id="BW", station_id="RJOB"
    )

    # The inventory object should also not be touched.
    assert inv == original_inv

    assert len(new_inv.networks) == 1
    # Single station because the three identical stations have been merged.
    assert len(new_inv[0].stations) == 1
    assert new_inv[0].code == "BW"
    assert new_inv[0][0].code == "RJOB"

    # Make sure the station dates have been set correctly.
    assert new_inv[0][0].start_date == obspy.UTCDateTime(
        "2001-05-15T00:00:00.000000Z"
    )
    assert new_inv[0][0].end_date is None

    # The 9 channels should remain.
    assert len(new_inv[0][0].channels) == 9


def test_merge_inventories():
    """
    Silly test, merging the same inventory twice should result in the same
    as the test_merging_stations() test.
    """
    inv = obspy.read_inventory(
        os.path.join(data_dir, "big_station.xml"), format="stationxml"
    )
    original_inv = copy.deepcopy(inv)
    inv_2 = obspy.read_inventory(
        os.path.join(data_dir, "big_station.xml"), format="stationxml"
    )

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
    assert new_inv[0][0].start_date == obspy.UTCDateTime(
        "2001-05-15T00:00:00.000000Z"
    )
    assert new_inv[0][0].end_date is None

    # The 9 channels should remain.
    assert len(new_inv[0][0].channels) == 9


def test_merge_inventories_multiple_times():
    """
    Just merge it a bunch.
    """
    inv = obspy.read_inventory(
        os.path.join(data_dir, "big_station.xml"), format="stationxml"
    )

    assert len(inv.networks) == 2
    assert len(inv.select(network="BW")[0].stations) == 3

    merged_inv = merge_inventories(
        inv.copy(), inv.copy(), network_id="BW", station_id="RJOB"
    )

    for _ in range(5):
        merged_inv = merge_inventories(
            merged_inv, inv.copy(), network_id="BW", station_id="RJOB"
        )

    # Same assertions as in the previous test.
    assert len(merged_inv.networks) == 1
    assert len(merged_inv[0].stations) == 1
    assert merged_inv[0].code == "BW"
    assert merged_inv[0][0].code == "RJOB"

    # Make sure the station dates have been set correctly.
    assert merged_inv[0][0].start_date == obspy.UTCDateTime(
        "2001-05-15T00:00:00.000000Z"
    )
    assert merged_inv[0][0].end_date is None

    # The 9 channels should remain.
    assert len(merged_inv[0][0].channels) == 9


def test_quick_coordinate_extraction():
    """
    Tests the quick coordinate extraction.
    """
    filename = os.path.join(data_dir, "big_station.xml")

    # Test at the station level.
    with open(filename, "rb") as fh:
        coords = get_coordinates(fh, level="station")

    assert coords == {
        "BW.RJOB": {
            "elevation_in_m": 860.0,
            "latitude": 47.737167,
            "longitude": 12.795714,
        },
        "GR.FUR": {
            "elevation_in_m": 565.0,
            "latitude": 48.162899,
            "longitude": 11.2752,
        },
        "GR.WET": {
            "elevation_in_m": 613.0,
            "latitude": 49.144001,
            "longitude": 12.8782,
        },
    }

    # Test at the channel level. These are then time dependent.
    with open(filename, "rb") as fh:
        coords = get_coordinates(fh, level="channel")

    # Assert everything has been found.
    assert sorted(coords.keys()) == sorted(
        [
            "BW.RJOB..EHZ",
            "GR.WET..LHZ",
            "GR.FUR..HHN",
            "GR.FUR..BHZ",
            "GR.WET..BHE",
            "GR.FUR..HHE",
            "GR.FUR..VHZ",
            "GR.WET..HHE",
            "GR.FUR..VHE",
            "GR.FUR..HHZ",
            "BW.RJOB..EHN",
            "GR.FUR..BHE",
            "GR.WET..LHE",
            "GR.FUR..VHN",
            "GR.WET..LHN",
            "GR.FUR..BHN",
            "BW.RJOB..EHE",
            "GR.FUR..LHN",
            "GR.WET..HHN",
            "GR.FUR..LHE",
            "GR.WET..HHZ",
            "GR.WET..BHZ",
            "GR.FUR..LHZ",
            "GR.WET..BHN",
        ]
    )

    # Check one in detail.
    assert coords["BW.RJOB..EHE"] == [
        {
            "elevation_in_m": 860.0,
            "endtime": UTCDateTime(2006, 12, 12, 0, 0),
            "latitude": 47.737167,
            "local_depth_in_m": 0.0,
            "longitude": 12.795714,
            "starttime": UTCDateTime(2001, 5, 15, 0, 0),
        },
        {
            "elevation_in_m": 860.0,
            "endtime": UTCDateTime(2007, 12, 17, 0, 0),
            "latitude": 47.737167,
            "local_depth_in_m": 0.0,
            "longitude": 12.795714,
            "starttime": UTCDateTime(2006, 12, 13, 0, 0),
        },
        {
            "elevation_in_m": 860.0,
            "endtime": None,
            "latitude": 47.737167,
            "local_depth_in_m": 0.0,
            "longitude": 12.795714,
            "starttime": UTCDateTime(2007, 12, 17, 0, 0),
        },
    ]


def test_isolate_and_merge_with_station_level_information():
    """
    Tests isolate and merge with a station level station files which used to
    cause an error.
    """
    inv = obspy.read_inventory()
    for net in inv:
        for sta in net:
            sta.channels = []

    inv = isolate_and_merge_station(inv, network_id="GR", station_id="FUR")
    assert len(inv.networks) == 1
    assert len(inv[0].stations) == 1

    assert inv[0].code == "GR"
    assert inv[0][0].code == "FUR"


def test_information_from_other_namespaces_is_retained():
    """
    """
    extra = obspy.core.AttribDict(
        {
            "my_tag": {
                "value": True,
                "namespace": "http://some-page.de/xmlns/1.0",
                "attrib": {
                    "{http://some-page.de/xmlns/1.0}my_attrib1": "123.4",
                    "{http://some-page.de/xmlns/1.0}my_attrib2": "567",
                },
            },
            "my_tag_2": {
                "value": "True",
                "namespace": "http://some-page.de/xmlns/1.0",
            },
        }
    )

    inv = obspy.Inventory(
        [
            obspy.core.inventory.Network(
                "XX",
                stations=[
                    obspy.core.inventory.Station(
                        code="YY", latitude=1.0, longitude=2.0, elevation=3.0
                    )
                ],
            )
        ],
        "XX",
    )
    inv[0].extra = extra

    # Should survive the round trip.
    assert inv[0].extra == extra
    inv2 = isolate_and_merge_station(inv, network_id="XX", station_id="YY")
    assert inv2[0].extra == extra


def test_dont_merge_station_epochs():
    """
    Stations might have epochs with different information - don't merge these
    then.
    """
    filename = os.path.join(data_dir, "multi_station_epoch.xml")
    inv = obspy.read_inventory(filename)
    assert len(inv.get_contents()["stations"]) == 3
    assert len(inv.get_contents()["channels"]) == 9
    # Also contains some information from another namespace.
    assert inv[0][0].extra == {
        "something": {"namespace": "https://example.com", "value": "test"}
    }

    # All of this should survive the merge and isolate operation.
    inv2 = isolate_and_merge_station(inv, network_id="XX", station_id="YYY")
    assert len(inv2.get_contents()["stations"]) == 3
    assert len(inv2.get_contents()["channels"]) == 9
    # Also contains some information from another namespace.
    assert inv2[0][0].extra == {
        "something": {"namespace": "https://example.com", "value": "test"}
    }

    # In this case nothing actually changed.
    assert inv == inv2
