from __future__ import print_function

import argparse
import collections
import os
import re
import sys

import obspy
from obspy.io.sac.util import get_sac_reftime

import pyasdf


class _EventContainer(object):
    """
    Helper class to correctly add the event and related associations.
    """

    def __init__(self):
        self._events = {}

    def get_resource_identifier(self, latitude, longitude, depth, origin_time):
        key = (latitude, longitude, depth, origin_time.timestamp)
        if key not in self._events:
            self._events[key] = obspy.core.event.ResourceIdentifier()
        return self._events[key]

    def add_events_to_asdf_file(self, f):
        cat = obspy.core.event.Catalog()
        for key, res_id in self._events.items():
            event = obspy.core.event.Event(
                resource_id=res_id,
                origins=[
                    obspy.core.event.Origin(
                        time=obspy.UTCDateTime(key[3]),
                        longitude=key[1],
                        latitude=key[0],
                        depth=key[2],
                    )
                ],
            )
            cat.append(event)
        f.add_quakeml(cat)


def add_to_adsf_file(filename, folder, tag, verbose=False):
    files = [os.path.join(folder, _i) for _i in os.listdir(folder)]
    if verbose:
        print("Found %i potential files to add." % len(files))
    print("Opening '%s' ..." % filename)
    with pyasdf.ASDFDataSet(filename) as f:
        _add_to_adsf_file(f=f, files=files, tag=tag, verbose=verbose)


def _add_to_adsf_file(f, files, tag, verbose=False):
    count = 0

    event_handler = _EventContainer()
    channel_information = {}
    min_starttime = None
    max_endtime = None

    for filename in files:
        if not verbose:
            print(".", end="")
            sys.stdout.flush()
        else:
            print("Attempting to add '%s'." % filename)
        try:
            tr = obspy.read(filename, format="SAC")[0]
            if verbose:
                print("Success.")
        except Exception:
            print("\nFailed to read '%s' as a SAC file." % filename)
            continue

        # Get to coordinates if possible:
        try:
            coords = (
                tr.stats.sac.stlo,
                tr.stats.sac.stla,
                tr.stats.sac.stel,
                tr.stats.sac.stdp,
            )
            channel_information[tr.id] = coords
        except AttributeError:
            pass

        s, e = tr.stats.starttime, tr.stats.endtime
        if min_starttime is None or s < min_starttime:
            min_starttime = s
        if max_endtime is None or e < max_endtime:
            max_endtime = e

        try:
            ev = (
                tr.stats.sac.evlo,
                tr.stats.sac.evla,
                tr.stats.sac.evdp,
                tr.stats.sac.o,
            )
        except AttributeError:
            ev = None

        if ev:
            event_id = event_handler.get_resource_identifier(
                latitude=ev[1],
                longitude=ev[0],
                depth=ev[2],
                origin_time=get_sac_reftime(tr.stats.sac) + ev[3],
            )
        else:
            event_id = None

        f.add_waveforms(tr, tag=tag, event_id=event_id)
        count += 1

    # Add all events.
    event_handler.add_events_to_asdf_file(f)

    # Write all StationXML files at once.
    write_stationxmls(
        f,
        channel_information,
        starttime=min_starttime - 3600,
        endtime=max_endtime + 3600,
    )

    print("\n\nWritten %i SAC files to '%s'." % (count, f.filename))


def write_stationxmls(f, channels, starttime, endtime):
    # First sort by stations.
    stations = collections.defaultdict(set)
    for key, value in channels.items():
        key = key.split(".")
        key.extend(value)
        stations[tuple(key[:2])].add(tuple(key[2:]))

    for netsta, values in stations.items():
        values = list(values)
        station_coords = list(set([_i[2:5] for _i in values]))
        if len(station_coords) > 1:
            msg = (
                "Multiple station level coordinates for '%s.%s'. Will pick "
                "one at random for the StationXML file. Channel level "
                "coordinates will still be correct for all channels."
            )
            print(msg)
        sta_lo, sta_la, sta_el = station_coords[0]

        # Start building the inventory object.
        inv = obspy.core.inventory.Inventory(
            networks=[
                obspy.core.inventory.Network(
                    code=netsta[0],
                    stations=[
                        obspy.core.inventory.Station(
                            code=netsta[1],
                            latitude=sta_la,
                            longitude=sta_lo,
                            start_date=starttime,
                            end_date=endtime,
                            creation_date=obspy.UTCDateTime(),
                            site=obspy.core.inventory.Site(name=""),
                            elevation=sta_el,
                            channels=[
                                obspy.core.inventory.Channel(
                                    code=_i[1],
                                    location_code=_i[0],
                                    latitude=_i[3],
                                    longitude=_i[2],
                                    elevation=_i[4],
                                    depth=_i[5],
                                    start_date=starttime,
                                    end_date=endtime,
                                )
                                for _i in values
                            ],
                        )
                    ],
                )
            ],
            source="pyasdf sac2asdf converter",
        )
        f.add_stationxml(inv)


def __main__():
    parser = argparse.ArgumentParser(
        description="Convert a folder of SAC files to a single ASDF file."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print a lot more information."
    )
    parser.add_argument("folder", type=str, help="Folder with SAC files.")
    parser.add_argument("output_file", type=str, help="Output ASDF file.")
    parser.add_argument("tag", type=str, help="Tag of all seismograms.")

    args = parser.parse_args()
    if os.path.exists(args.output_file):
        raise ValueError("ASDF file already exists.")

    if not os.path.isdir(args.folder):
        raise ValueError("Folder must be an existing folder.")

    # Assert the tag.
    if not re.match(pyasdf.asdf_data_set.TAG_REGEX, args.tag):
        raise ValueError(
            "Invalid tag: '%s' - Must satisfy the regex "
            "'%s'." % (args.tag, pyasdf.asdf_data_set.TAG_REGEX.pattern)
        )

    add_to_adsf_file(
        filename=args.output_file,
        folder=args.folder,
        verbose=args.verbose,
        tag=args.tag,
    )


if __name__ == "__main__":  # pragma: no cover
    __main__()
