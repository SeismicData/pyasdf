Working With Station Data
=========================

The ASDF format stores station information in the StationXML format. Simply
accessing the ``StationXML`` attribute will parse the data to an
:class:`obspy.station.station.Inventory` object if the station has one. Please
see its documentation for more details of how to work with it.

>>> import pyasdf
>>> ds = pyasdf.ASDFDataSet("example_file.h5)
>>> ds.waveforms.IU_FURI.StationXML
Inventory created at 2014-12-09T19:43:16.000000Z
    Created by: IRIS WEB SERVICE: fdsnws-station | version: 1.1.9
            http://service.iris.edu/fdsnws/station/1/query?network=IU&level=res...
    Sending institution: IRIS-DMC (IRIS-DMC)
    Contains:
        Networks (1):
            IU
        Stations (1):
            IU.FURI (Mt. Furi, Ethiopia)
        Channels (3):
            IU.FURI.00.BHE, IU.FURI.00.BHN, IU.FURI.00.BHZ
>>> type(ds.waveforms.IU_FURI.StationXML)
obspy.station.inventory.Inventory


In some cases one only needs access to the coordinates and not the full
station information including the instrument response. As parsing the full
file is a fairly slow operation ``pyasdf`` offers functionality to very
quickly extract the coordinates for any station. Either at the station level

>>> ds.waveforms.IU_FURI.coordinates
{'elevation_in_m': 2570.0, 'latitude': 8.8952, 'longitude': 38.6798}

or at the channel level (now the information is time-dependent!):

>>> ds.waveforms.IU_FURI.channel_coordinates
{'IU.FURI.00.BHE': [{'elevation_in_m': 2565.0,
   'endtime': 2011-09-03T08:00:00.000000Z,
   'latitude': 8.8952,
   'local_depth_in_m': 5.0,
   'longitude': 38.6798,
   'starttime': 2009-10-04T00:00:00.000000Z}],
 'IU.FURI.00.BHN': [{'elevation_in_m': 2565.0,
   'endtime': 2012-10-26T00:00:00.000000Z,
   'latitude': 8.8952,
   'local_depth_in_m': 5.0,
   'longitude': 38.6798,
   'starttime': 2009-10-04T00:00:00.000000Z}],
 'IU.FURI.00.BHZ': [{'elevation_in_m': 2565.0,
   'endtime': 2011-09-03T08:00:00.000000Z,
   'latitude': 8.8952,
   'local_depth_in_m': 5.0,
   'longitude': 38.6798,
   'starttime': 2009-10-04T00:00:00.000000Z}]}


The ``pyasdf`` library will parse the StationXML files on demand. While this
will still read the whole StationXML file (and is thus I/O bound), it is much
faster then parsing all the information:

>>> len(ds.waveforms)
196
>>> %timeit for station in ds.waveforms: station.coordinates
1 loops, best of 3: 512 ms per loop
>>> %timeit for station in ds.waveforms: station.StationXML
1 loops, best of 3: 3.18 s per loop


A convenience function to extract the coordinates for all stations also exists:

>>> ds.get_all_coordinates()
{'AF.CVNA': {'elevation_in_m': 1050.0,
             'latitude': -31.482,
             'longitude': 19.762},
 'AF.DODT': {...},
 ...}
