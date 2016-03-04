#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for dealing with inventory objects.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import copy
from lxml import etree
from obspy import UTCDateTime


def merge_inventories(inv_a, inv_b, network_id, station_id):
    """
    Takes two inventories, merges the contents of both and isolates the
    contents of a certain network and station id.

    Returns the processed inventory object. The original one will not be
    changed.

    :param inv_a: Inventory A. Contents of that inventory will be prioritized.
    :type inv_a: :class:`~obspy.core.inventory.inventory.Inventory`
    :param inv_b: Inventory B.
    :type inv_b: :class:`~obspy.core.inventory.inventory.Inventory`
    :param network_id: The network id.
    :type network_id: str
    :param station_id: The station id.
    :type station_id: str
    """
    inv = copy.deepcopy(inv_a)
    inv.networks.extend(copy.deepcopy(inv_b.networks))
    return isolate_and_merge_station(inv, network_id=network_id,
                                     station_id=station_id)


def isolate_and_merge_station(inv, network_id, station_id):
    """
    Takes an inventory object, isolates the given station and merged them.

    Merging is sometimes necessary as many files have the same station
    multiple times.

    Returns the processed inventory object. The original one will not be
    changed.

    :param inv: The inventory.
    :type inv: :class:`~obspy.core.inventory.inventory.Inventory`
    :param network_id: The network id.
    :type network_id: str
    :param station_id: The station id.
    :type station_id: str
    """
    inv = copy.deepcopy(inv.select(network=network_id, station=station_id,
                                   keep_empty=True))

    # Merge networks if necessary.
    if len(inv.networks) != 1:
        network = inv.networks[0]
        for other_network in inv.networks[1:]:
            # Merge the stations.
            network.stations.extend(other_network.stations)
            # Update the times if necessary.
            if other_network.start_date is not None:
                if network.start_date is None or \
                        network.start_date > other_network.start_date:
                    network.start_date = other_network.start_date
            # None is the "biggest" end_date.
            if network.end_date is not None and other_network.end_date is \
                    not None:
                if other_network.end_date > network.end_date:
                    network.end_date = other_network.end_date
            elif other_network.end_date is None:
                network.end_date = None
            # Update comments.
            network.comments = list(
                set(network.comments).union(set(other_network.comments)))
            # Update the number of stations.
            if other_network.total_number_of_stations:
                if network.total_number_of_stations or \
                        network.total_number_of_stations < \
                        other_network.total_number_of_stations:
                    network.total_number_of_stations = \
                        other_network.total_number_of_stations
            # Update the other elements
            network.alternate_code = (network.alternate_code or
                                      other_network.alternate_code) or None
            network.description = (network.description or
                                   other_network.description) or None
            network.historical_code = (network.historical_code or
                                       other_network.historical_code) or None
            network.restricted_status = network.restricted_status or \
                other_network.restricted_status
        inv.networks = [network]

    # Merge stations if necessary.
    if len(inv.networks[0].stations) != 1:
        station = inv.networks[0].stations[0]
        for other_station in inv.networks[0].stations[1:]:
            # Merge the channels.
            station.channels.extend(other_station.channels)
            # Update the times if necessary.
            if other_station.start_date is not None:
                if station.start_date is None or \
                        station.start_date > other_station.start_date:
                    station.start_date = other_station.start_date
            # None is the "biggest" end_date.
            if station.end_date is not None and other_station.end_date is \
                    not None:
                if other_station.end_date > station.end_date:
                    station.end_date = other_station.end_date
            elif other_station.end_date is None:
                station.end_date = None
            # Update comments.
            station.comments = list(
                set(station.comments).union(set(other_station.comments)))
            # Update the number of channels.
            if other_station.total_number_of_channels:
                if station.total_number_of_channels or \
                        station.total_number_of_channels < \
                        other_station.total_number_of_channels:
                    station.total_number_of_channels = \
                        other_station.total_number_of_channels
            # Update the other elements
            station.alternate_code = (station.alternate_code or
                                      other_station.alternate_code) or None
            station.description = (station.description or
                                   other_station.description) or None
            station.historical_code = (station.historical_code or
                                       other_station.historical_code) or None
            station.restricted_status = station.restricted_status or \
                other_station.restricted_status
        inv.networks[0].stations = [station]

    # Last but not least, remove duplicate channels. This is done on the
    # location and channel id, and the times, nothing else.
    unique_channels = []
    available_channel_hashes = []
    for channel in inv[0][0]:
        c_hash = hash((str(channel.start_date), str(channel.end_date),
                       channel.code, channel.location_code))
        if c_hash in available_channel_hashes:
            continue
        else:
            unique_channels.append(channel)
            available_channel_hashes.append(c_hash)
    inv[0][0].channels = unique_channels

    # Update the selected number of stations and channels.
    inv[0].selected_number_of_stations = 1
    inv[0][0].selected_number_of_channels = len(inv[0][0].channels)

    return inv


def get_coordinates(data, level="station"):
    """
    Very quick way to get coordinates from a StationXML file.

    Can extract coordinates at the station and at the channel level.
    """
    ns = "http://www.fdsn.org/xml/station/1"
    network_tag = "{%s}Network" % ns
    station_tag = "{%s}Station" % ns
    channel_tag = "{%s}Channel" % ns
    latitude_tag = "{%s}Latitude" % ns
    longitude_tag = "{%s}Longitude" % ns
    elevation_tag = "{%s}Elevation" % ns
    depth_tag = "{%s}Depth" % ns

    # Return station coordinates.
    if level == "station":
        coordinates = {}

        # Small state machine.
        net_state = None

        tags = (network_tag, station_tag)
        context = etree.iterparse(data, events=("start", ), tag=tags)

        for _, elem in context:
            if elem.tag == station_tag:
                station_coordinates = {}
                for child in elem.getchildren():
                    if child.tag == latitude_tag:
                        station_coordinates["latitude"] = float(child.text)
                    elif child.tag == longitude_tag:
                        station_coordinates["longitude"] = float(child.text)
                    elif child.tag == elevation_tag:
                        station_coordinates["elevation_in_m"] = float(
                            child.text)
                coordinates["%s.%s" % (net_state, elem.get("code"))] = \
                    station_coordinates
            elif elem.tag == network_tag:
                net_state = elem.get('code')
        return coordinates

    # Return channel coordinates.
    elif level == "channel":
        coordinates = collections.defaultdict(list)

        # Small state machine.
        net_state, sta_state = (None, None)

        tags = (network_tag, station_tag, channel_tag)
        context = etree.iterparse(data, events=("start", ), tag=tags)

        for _, elem in context:
            if elem.tag == channel_tag:
                # Get basics.
                channel = elem.get('code')
                location = elem.get('locationCode').strip()
                starttime = UTCDateTime(elem.get('startDate'))
                endtime = elem.get('endDate')
                if endtime:
                    endtime = UTCDateTime(endtime)

                tag = "%s.%s.%s.%s" % (net_state, sta_state, location, channel)
                channel_coordinates = {"starttime": starttime,
                                       "endtime": endtime}
                coordinates[tag].append(channel_coordinates)

                for child in elem.getchildren():
                    if child.tag == latitude_tag:
                        channel_coordinates["latitude"] = float(child.text)
                    elif child.tag == longitude_tag:
                        channel_coordinates["longitude"] = float(child.text)
                    elif child.tag == elevation_tag:
                        channel_coordinates["elevation_in_m"] = float(
                            child.text)
                    elif child.tag == depth_tag:
                        channel_coordinates["local_depth_in_m"] = float(
                            child.text)
            elif elem.tag == station_tag:
                sta_state = elem.get('code')
            elif elem.tag == network_tag:
                net_state = elem.get('code')
        return dict(coordinates)
    else:
        raise ValueError("Level must be either 'station' or 'channel'.")
