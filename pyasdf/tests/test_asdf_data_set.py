#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2014
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import glob
import inspect
import io
import json
import shutil
import os

import h5py
import numpy as np
import obspy
from obspy import UTCDateTime
import prov
import pytest

from pyasdf import ASDFDataSet
from pyasdf.exceptions import WaveformNotInFileException, ASDFValueError
from pyasdf.header import FORMAT_VERSION, FORMAT_NAME


data_dir = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


class Namespace(object):
    """
    Simple helper class offering a namespace.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def example_data_set(tmpdir):
    """
    Fixture creating a small example file.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")

    data_set = ASDFDataSet(asdf_filename)

    for filename in glob.glob(os.path.join(data_path, "*.xml")):
        if "quake.xml" in filename:
            data_set.add_quakeml(filename)
        else:
            data_set.add_stationxml(filename)

    for filename in glob.glob(os.path.join(data_path, "*.mseed")):
        data_set.add_waveforms(filename, tag="raw_recording",
                               event_id=data_set.events[0])

    # Flush and finish writing.
    del data_set

    # Return filename and path to tempdir, no need to always create a
    # new one.
    return Namespace(filename=asdf_filename, tmpdir=tmpdir.strpath)


def test_data_set_creation(tmpdir):
    """
    Test data set creation with a small test dataset.

    It tests that the the stuff that goes in is correctly saved and
    can be retrieved again.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")

    data_set = ASDFDataSet(asdf_filename)

    for filename in glob.glob(os.path.join(data_path, "*.mseed")):
        data_set.add_waveforms(filename, tag="raw_recording")

    for filename in glob.glob(os.path.join(data_path, "*.xml")):
        if "quake.xml" in filename:
            data_set.add_quakeml(filename)
        else:
            data_set.add_stationxml(filename)

    # Flush and finish writing.
    del data_set

    # Open once again
    data_set = ASDFDataSet(asdf_filename)

    # ObsPy is tested enough to make this comparison meaningful.
    for station in (("AE", "113A"), ("TA", "POKR")):
        # Test the waveforms
        stream_asdf = \
            getattr(data_set.waveforms, "%s_%s" % station).raw_recording
        stream_file = obspy.read(os.path.join(
            data_path, "%s.%s.*.mseed" % station))
        # Delete the file format specific stats attributes. These are
        # meaningless inside ASDF data sets.
        for trace in stream_file:
            del trace.stats.mseed
            del trace.stats._format
        for trace in stream_asdf:
            del trace.stats.asdf
            del trace.stats._format
        assert stream_asdf == stream_file

        # Test the inventory data.
        inv_asdf = \
            getattr(data_set.waveforms, "%s_%s" % station).StationXML
        inv_file = obspy.read_inventory(
            os.path.join(data_path, "%s.%s..BH*.xml" % station))
        assert inv_file == inv_asdf
    # Test the event.
    cat_file = obspy.readEvents(os.path.join(data_path, "quake.xml"))
    cat_asdf = data_set.events
    # from IPython.core.debugger import Tracer; Tracer(colors="Linux")()
    assert cat_file == cat_asdf


def test_equality_checks(example_data_set):
    """
    Tests the equality operations.
    """
    filename_1 = example_data_set.filename
    filename_2 = os.path.join(example_data_set.tmpdir, "new.h5")
    shutil.copyfile(filename_1, filename_2)

    data_set_1 = ASDFDataSet(filename_1)
    data_set_2 = ASDFDataSet(filename_2)

    assert data_set_1 == data_set_2
    assert not (data_set_1 != data_set_2)

    # A tiny change at an arbitrary place should trigger an inequality.
    for tag, data_array in data_set_2._waveform_group["AE.113A"].items():
        if tag != "StationXML":
            break
    data_array[1] += 2.0
    assert not (data_set_1 == data_set_2)
    assert data_set_1 != data_set_2

    # Reverting should also work. Floating point math inaccuracies should
    # not matter at is only tests almost equality. This is not a proper test
    # for this behaviour though.
    data_array[1] -= 2.0
    assert data_set_1 == data_set_2
    assert not (data_set_1 != data_set_2)

    # Also check the StationXML.
    data_array = data_set_2._waveform_group["AE.113A"]["StationXML"]
    data_array[1] += 2.0
    assert not (data_set_1 == data_set_2)
    assert data_set_1 != data_set_2
    data_array[1] -= 2.0
    assert data_set_1 == data_set_2
    assert not (data_set_1 != data_set_2)

    # Test change of keys.
    del data_set_1._waveform_group["AE.113A"]
    assert data_set_1 != data_set_2


def test_adding_same_event_twice_raises(tmpdir):
    """
    Adding the same event twice raises.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")

    data_set = ASDFDataSet(asdf_filename)

    # Add once, all good.
    data_set.add_quakeml(os.path.join(data_path, "quake.xml"))
    assert len(data_set.events) == 1

    # Adding again should raise an error.
    with pytest.raises(ValueError):
        data_set.add_quakeml(os.path.join(data_path, "quake.xml"))


def test_adding_event_in_various_manners(tmpdir):
    """
    Events can be added either as filenames, open files, BytesIOs, or ObsPy
    objects. In any case, the result should be the same.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")
    event_filename = os.path.join(data_path, "quake.xml")

    ref_cat = obspy.readEvents(event_filename)

    # Add as filename
    data_set = ASDFDataSet(asdf_filename)
    assert len(data_set.events) == 0
    data_set.add_quakeml(event_filename)
    assert len(data_set.events) == 1
    assert data_set.events == ref_cat
    del data_set
    os.remove(asdf_filename)

    # Add as open file.
    data_set = ASDFDataSet(asdf_filename)
    assert len(data_set.events) == 0
    with open(event_filename, "rb") as fh:
        data_set.add_quakeml(fh)
    assert len(data_set.events) == 1
    assert data_set.events == ref_cat
    del data_set
    os.remove(asdf_filename)

    # Add as BytesIO.
    data_set = ASDFDataSet(asdf_filename)
    assert len(data_set.events) == 0
    with open(event_filename, "rb") as fh:
        temp = io.BytesIO(fh.read())
    temp.seek(0, 0)
    data_set.add_quakeml(temp)
    assert len(data_set.events) == 1
    assert data_set.events == ref_cat
    del data_set
    os.remove(asdf_filename)

    # Add as ObsPy Catalog.
    data_set = ASDFDataSet(asdf_filename)
    assert len(data_set.events) == 0
    data_set.add_quakeml(ref_cat.copy())
    assert len(data_set.events) == 1
    assert data_set.events == ref_cat
    del data_set
    os.remove(asdf_filename)

    # Add as an ObsPy event.
    data_set = ASDFDataSet(asdf_filename)
    assert len(data_set.events) == 0
    data_set.add_quakeml(ref_cat.copy()[0])
    assert len(data_set.events) == 1
    assert data_set.events == ref_cat
    del data_set
    os.remove(asdf_filename)


def test_assert_format_and_version_number_are_written(tmpdir):
    """
    Check that the version number and file format name are correctly written.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    # Create empty data set.
    data_set = ASDFDataSet(asdf_filename)
    # Flush and write.
    del data_set

    # Open again and assert name and version number.
    with h5py.File(asdf_filename, "r") as hdf5_file:
        assert hdf5_file.attrs["file_format_version"].decode() \
           == FORMAT_VERSION
        assert hdf5_file.attrs["file_format"].decode() == FORMAT_NAME


def test_dot_accessors(example_data_set):
    """
    Tests the dot accessors for waveforms and stations.
    """
    data_path = os.path.join(data_dir, "small_sample_data_set")
    data_set = ASDFDataSet(example_data_set.filename)

    # Get the contents, this also asserts that tab completions works.
    assert sorted(dir(data_set.waveforms)) == ["AE_113A", "TA_POKR"]
    assert sorted(dir(data_set.waveforms.AE_113A)) == \
        sorted(["StationXML", "_station_name", "raw_recording",
                "coordinates", "channel_coordinates"])
    assert sorted(dir(data_set.waveforms.TA_POKR)) == \
        sorted(["StationXML", "_station_name", "raw_recording",
                "coordinates", "channel_coordinates"])

    # Actually check the contents.
    waveform = data_set.waveforms.AE_113A.raw_recording.sort()
    waveform_file = obspy.read(os.path.join(data_path, "AE.*.mseed")).sort()
    for trace in waveform_file:
        del trace.stats.mseed
        del trace.stats._format
    for trace in waveform:
        del trace.stats.asdf
        del trace.stats._format
    assert waveform == waveform_file

    waveform = data_set.waveforms.TA_POKR.raw_recording.sort()
    waveform_file = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    for trace in waveform_file:
        del trace.stats.mseed
        del trace.stats._format
    for trace in waveform:
        del trace.stats.asdf
        del trace.stats._format
    assert waveform == waveform_file

    assert data_set.waveforms.AE_113A.StationXML == \
        obspy.read_inventory(os.path.join(data_path, "AE.113A..BH*.xml"))
    assert data_set.waveforms.TA_POKR.StationXML == \
        obspy.read_inventory(os.path.join(data_path, "TA.POKR..BH*.xml"))


def test_stationxml_is_invalid_tag_name(tmpdir):
    """
    StationXML is an invalid waveform path.
    """
    filename = os.path.join(tmpdir.strpath, "example.h5")

    data_set = ASDFDataSet(filename)
    st = obspy.read()

    with pytest.raises(ValueError):
        data_set.add_waveforms(st, tag="StationXML")
    with pytest.raises(ValueError):
        data_set.add_waveforms(st, tag="stationxml")

    # Adding with a proper path works just fine.
    data_set.add_waveforms(st, tag="random_waveform")


def test_saving_event_id(tmpdir):
    """
    Tests that the event_id can be saved and retrieved automatically.
    """
    data_path = os.path.join(data_dir, "small_sample_data_set")
    filename = os.path.join(tmpdir.strpath, "example.h5")
    event = obspy.readEvents(os.path.join(data_path, "quake.xml"))[0]

    # Add the event object, and associate the waveform with it.
    data_set = ASDFDataSet(filename)
    data_set.add_quakeml(event)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(waveform, "raw_recording", event_id=event)
    st = data_set.waveforms.TA_POKR.raw_recording
    for tr in st:
        assert tr.stats.asdf.event_id.getReferredObject() == event
    del data_set
    os.remove(filename)

    # Add as a string.
    data_set = ASDFDataSet(filename)
    data_set.add_quakeml(event)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(waveform, "raw_recording",
                           event_id=str(event.resource_id.id))
    st = data_set.waveforms.TA_POKR.raw_recording
    for tr in st:
        assert tr.stats.asdf.event_id.getReferredObject() == event
    del data_set
    os.remove(filename)

    # Add as a resource identifier object.
    data_set = ASDFDataSet(filename)
    data_set.add_quakeml(event)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(waveform, "raw_recording",
                           event_id=event.resource_id)
    st = data_set.waveforms.TA_POKR.raw_recording
    for tr in st:
        assert tr.stats.asdf.event_id.getReferredObject() == event
    del data_set
    os.remove(filename)


def test_event_association_is_persistent_through_processing(example_data_set):
    """
    Processing a file with an associated event and storing it again should
    keep the association.
    """
    data_set = ASDFDataSet(example_data_set.filename)
    st = data_set.waveforms.TA_POKR.raw_recording
    event_id = st[0].stats.asdf.event_id

    st.taper(max_percentage=0.05, type="cosine")

    data_set.add_waveforms(st, tag="processed")
    processed_st = data_set.waveforms.TA_POKR.processed
    assert event_id == processed_st[0].stats.asdf.event_id


def test_detailed_event_association_is_persistent_through_processing(
        example_data_set):
    """
    Processing a file with an associated event and storing it again should
    keep the association for all the possible event tags..
    """
    data_set = ASDFDataSet(example_data_set.filename)
    # Store a new waveform.
    event = data_set.events[0]
    origin = event.origins[0]
    magnitude = event.magnitudes[0]
    focmec = event.focal_mechanisms[0]

    tr = obspy.read()[0]
    tr.stats.network = "BW"
    tr.stats.station = "RJOB"

    data_set.add_waveforms(tr, tag="random", event_id=event,
                           origin_id=origin, focal_mechanism_id=focmec,
                           magnitude_id=magnitude)

    new_st = data_set.waveforms.BW_RJOB.random
    new_st.taper(max_percentage=0.05, type="cosine")

    data_set.add_waveforms(new_st, tag="processed")
    processed_st = data_set.waveforms.BW_RJOB.processed
    assert event.resource_id == processed_st[0].stats.asdf.event_id
    assert origin.resource_id == processed_st[0].stats.asdf.origin_id
    assert magnitude.resource_id == processed_st[0].stats.asdf.magnitude_id
    assert focmec.resource_id == processed_st[0].stats.asdf.focal_mechanism_id


def test_tag_iterator(example_data_set):
    """
    Tests the iteration over tags with the ifilter() method.
    """
    ds = ASDFDataSet(example_data_set.filename)

    expected_ids = ["AE.113A..BHE", "AE.113A..BHN", "AE.113A..BHZ",
                    "TA.POKR..BHE", "TA.POKR..BHN", "TA.POKR..BHZ"]

    for station in ds.ifilter(ds.q.tag == "raw_recording"):
        inv = station.StationXML
        for tr in station.raw_recording:
            assert tr.id in expected_ids
            expected_ids.remove(tr.id)
            assert bool(inv.select(
                network=tr.stats.network, station=tr.stats.station,
                channel=tr.stats.channel, location=tr.stats.location).networks)

    assert expected_ids == []

    # It will only return matching tags.
    count = 0
    for _ in ds.ifilter(ds.q.tag == "random"):
        count += 1
    assert count == 0


def test_processing_multiprocessing(example_data_set):
    """
    Tests the processing using multiprocessing.
    """
    def null_processing(st, inv):
        return st

    data_set = ASDFDataSet(example_data_set.filename)
    output_filename = os.path.join(example_data_set.tmpdir, "output.h5")
    # Do not actually do anything. Apply an empty function.
    data_set.process(null_processing, output_filename,
                     {"raw_recording": "raw_recording"})

    del data_set
    data_set = ASDFDataSet(example_data_set.filename)
    out_data_set = ASDFDataSet(output_filename)

    assert data_set == out_data_set


def test_format_version_decorator(example_data_set):
    """
    Tests the format version decorator.

    Also more or less tests that the format version is correctly written and
    read.
    """
    data_set = ASDFDataSet(example_data_set.filename)
    assert data_set.asdf_format_version == FORMAT_VERSION


def test_reading_and_writing_auxiliary_data(tmpdir):
    """
    Tests reading and writing auxiliary data.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    # Define some auxiliary data and add it.
    data = np.random.random(100)
    data_type = "RandomArrays"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(data=data, data_type=data_type, path=path,
                                parameters=parameters)
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)
    aux_data = new_data_set.auxiliary_data.RandomArrays.test_data
    np.testing.assert_equal(data, aux_data.data)
    aux_data.data_type == data_type
    aux_data.path == path
    aux_data.parameters == parameters

    # Test the same thing, but with nested data.
    data = np.random.random(100)
    data_type = "RandomArrays"
    path = "some/nested/path/test_data"
    parameters = {"a": 2, "b": 3.0, "e": "hallo_again"}

    new_data_set.add_auxiliary_data(data=data, data_type=data_type, path=path,
                                    parameters=parameters)
    del new_data_set

    newer_data_set = ASDFDataSet(asdf_filename)
    aux_data = newer_data_set.auxiliary_data.RandomArrays.some.nested\
        .path.test_data
    np.testing.assert_equal(data, aux_data.data)
    aux_data.data_type == data_type
    aux_data.path == path
    aux_data.parameters == parameters

    del newer_data_set


def test_looping_over_stations(example_data_set):
    """
    Tests that iterating over the stations of a dataset works as expected.
    """
    data_set = ASDFDataSet(example_data_set.filename)

    stations = ["AE.113A", "TA.POKR"]
    assert sorted([_i._station_name for _i in data_set.waveforms]) == stations


def test_accessing_non_existent_tag_raises(example_data_set):
    """
    Accessing a non-existing station should raise.
    """
    data_set = ASDFDataSet(example_data_set.filename)

    try:
        with pytest.raises(WaveformNotInFileException) as excinfo:
            data_set.waveforms.AE_113A.asdfasdf

        assert excinfo.value.args[0] == ("Tag 'asdfasdf' not part of the data "
                                         "set for station 'AE.113A'.")
    finally:
        data_set.__del__()


def test_waveform_accessor_printing(example_data_set):
    """
    Pretty printing of the waveform accessor proxy objects.
    """
    data_set = ASDFDataSet(example_data_set.filename)

    assert data_set.waveforms.AE_113A.__str__() == (
        "Contents of the data set for station AE.113A:\n"
        "    - Has a StationXML file\n"
        "    - 1 Waveform Tag(s):\n"
        "        raw_recording")

    data_set.__del__()
    del data_set


def test_coordinate_extraction(example_data_set):
    """
    Tests the quick coordinate extraction.
    """
    data_set = ASDFDataSet(example_data_set.filename)

    assert data_set.waveforms.AE_113A.coordinates == {
        'latitude': 32.7683,
        'longitude': -113.7667,
        'elevation_in_m': 118.0}

    assert data_set.waveforms.TA_POKR.coordinates == {
        'latitude': 65.1171,
        'longitude': -147.4335,
        'elevation_in_m': 501.0}


def test_coordinate_extraction_channel_level(example_data_set):
    """
    Tests the quick coordinate extraction at the channel level.
    """
    data_set = ASDFDataSet(example_data_set.filename)

    assert data_set.waveforms.AE_113A.channel_coordinates == {
        'AE.113A..BHE': [{
            'elevation_in_m': 118.0,
            'endtime': UTCDateTime(2599, 12, 31, 23, 59, 59),
            'latitude': 32.7683,
            'local_depth_in_m': 0.0,
            'longitude': -113.7667,
            'starttime': UTCDateTime(2011, 12, 1, 0, 0)}],
        'AE.113A..BHN': [{
            'elevation_in_m': 118.0,
            'endtime': UTCDateTime(2599, 12, 31, 23, 59, 59),
            'latitude': 32.7683,
            'local_depth_in_m': 0.0,
            'longitude': -113.7667,
            'starttime': UTCDateTime(2011, 12, 1, 0, 0)}],
        'AE.113A..BHZ': [{
            'elevation_in_m': 118.0,
            'endtime': UTCDateTime(2599, 12, 31, 23, 59, 59),
            'latitude': 32.7683,
            'local_depth_in_m': 0.0,
            'longitude': -113.7667,
            'starttime': UTCDateTime(2011, 12, 1, 0, 0)}]}

    assert sorted(data_set.waveforms.TA_POKR.channel_coordinates.keys()) == \
        sorted(['TA.POKR.01.BHZ', 'TA.POKR..BHE', 'TA.POKR..BHZ',
                'TA.POKR..BHN', 'TA.POKR.01.BHN', 'TA.POKR.01.BHE'])


def test_extract_all_coordinates(example_data_set):
    """
    Tests the extraction of all coordinates.
    """
    data_set = ASDFDataSet(example_data_set.filename)

    assert data_set.get_all_coordinates() == {
       "AE.113A": {
           "latitude": 32.7683,
           "longitude": -113.7667,
           "elevation_in_m": 118.0},

       "TA.POKR": {
        "latitude": 65.1171,
        "longitude": -147.4335,
        "elevation_in_m": 501.0}}


def test_trying_to_add_provenance_record_with_invalid_name_fails(tmpdir):
    """
    The name must be valid according to a particular regular expression.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    filename = os.path.join(data_dir, "example_schematic_processing_chain.xml")

    # First try adding it as a prov document.
    doc = prov.read(filename, format="xml")
    with pytest.raises(ASDFValueError) as err:
        data_set.add_provenance_document(doc, name="a-b-c")

    assert err.value.args[0] == (
        "Name 'a-b-c' is invalid. It must validate against the regular "
        "expression '^[0-9a-z][0-9a-z_]*[0-9a-z]$'.")

    # Must sometimes be called to get around some bugs.
    data_set.__del__()


def test_adding_a_provenance_record(tmpdir):
    """
    Tests adding a provenance record.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    filename = os.path.join(data_dir, "example_schematic_processing_chain.xml")

    # Add it as a document.
    doc = prov.read(filename, format="xml")
    data_set.add_provenance_document(doc, name="test_provenance")
    del data_set

    # Read it again.
    data_set = ASDFDataSet(asdf_filename)
    assert data_set.provenance.test_provenance == doc


def test_str_method_provenance_documents(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    filename = os.path.join(data_dir, "example_schematic_processing_chain.xml")
    data_set.add_provenance_document(filename, name="test_provenance")

    assert str(data_set.provenance) == (
        "1 Provenance Document(s):\n\ttest_provenance"
    )


def test_reading_and_writing_n_dimensional_auxiliary_data(tmpdir):
    """
    Tests reading and writing n-dimensional auxiliary data.
    """
    # 2D.
    asdf_filename = os.path.join(tmpdir.strpath, "test_2D.h5")
    data_set = ASDFDataSet(asdf_filename)

    data = np.random.random((10, 10))
    data_type = "RandomArrays"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(data=data, data_type=data_type, path=path,
                                parameters=parameters)
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)
    aux_data = new_data_set.auxiliary_data.RandomArrays.test_data
    np.testing.assert_equal(data, aux_data.data)
    aux_data.data_type == data_type
    aux_data.path == path
    aux_data.parameters == parameters

    del new_data_set

    # 3D.
    asdf_filename = os.path.join(tmpdir.strpath, "test_3D.h5")
    data_set = ASDFDataSet(asdf_filename)

    data = np.random.random((5, 5, 5))
    data_type = "RandomArrays"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(data=data, data_type=data_type, path=path,
                                parameters=parameters)
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)
    aux_data = new_data_set.auxiliary_data.RandomArrays.test_data
    np.testing.assert_equal(data, aux_data.data)
    aux_data.data_type == data_type
    aux_data.path == path
    aux_data.parameters == parameters

    del new_data_set

    # 4D.
    asdf_filename = os.path.join(tmpdir.strpath, "test_4D.h5")
    data_set = ASDFDataSet(asdf_filename)

    data = np.random.random((2, 3, 4, 5))
    data_type = "RandomArrays"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(data=data, data_type=data_type, path=path,
                                parameters=parameters)
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)
    aux_data = new_data_set.auxiliary_data.RandomArrays.test_data
    np.testing.assert_equal(data, aux_data.data)
    aux_data.data_type == data_type
    aux_data.path == path
    aux_data.parameters == parameters

    del new_data_set


def test_adding_auxiliary_data_with_invalid_data_type_name_raises(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    data = np.random.random((10, 10))
    # The data must NOT start with a number.
    data_type = "2DRandomArray"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    try:
        with pytest.raises(ASDFValueError) as err:
            data_set.add_auxiliary_data(data=data, data_type=data_type,
                                        path=path, parameters=parameters)

        assert err.value.args[0] == (
            "Data type name '2DRandomArray' is invalid. It must validate "
            "against the regular expression '^[A-Z][A-Za-z0-9]*$'.")
    finally:
        data_set.__del__()


def test_reading_and_writing_auxiliary_data_with_provenance_id(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    data = np.random.random((10, 10))
    # The data must NOT start with a number.
    data_type = "RandomArray"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}
    provenance_id = "{http://example.org}test"

    data_set.add_auxiliary_data(data=data, data_type=data_type,
                                path=path, parameters=parameters,
                                provenance_id=provenance_id)
    data_set.__del__()
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)
    assert new_data_set.auxiliary_data.RandomArray.test_data.provenance_id \
        == provenance_id


def test_str_method_of_aux_data(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    # With provenance id.
    data = np.random.random((10, 10))
    # The data must NOT start with a number.
    data_type = "RandomArray"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}
    provenance_id = "{http://example.org}test"

    data_set.add_auxiliary_data(data=data, data_type=data_type,
                                path=path, parameters=parameters,
                                provenance_id=provenance_id)
    assert \
        str(data_set.auxiliary_data.RandomArray.test_data) == (
            "Auxiliary Data of Type 'RandomArray'\n"
            "\tPath: 'test_data'\n"
            "\tProvenance ID: '{http://example.org}test'\n"
            "\tData shape: '(10, 10)', dtype: 'float64'\n"
            "\tParameters:\n"
            "\t\ta: 1\n"
            "\t\tb: 2.0\n"
            "\t\te: hallo")

    # Without.
    data = np.random.random((10, 10))
    # The data must NOT start with a number.
    data_type = "RandomArray"
    path = "test_data_2"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(data=data, data_type=data_type,
                                path=path, parameters=parameters)
    assert \
        str(data_set.auxiliary_data.RandomArray.test_data_2) == (
            "Auxiliary Data of Type 'RandomArray'\n"
            "\tPath: 'test_data_2'\n"
            "\tData shape: '(10, 10)', dtype: 'float64'\n"
            "\tParameters:\n"
            "\t\ta: 1\n"
            "\t\tb: 2.0\n"
            "\t\te: hallo")

    # Nested structure.
    data_set.add_auxiliary_data(data=data, data_type=data_type,
                                path="some/deeper/path/test_data",
                                parameters=parameters)

    assert str(
        data_set.auxiliary_data.RandomArray.some.deeper.path.test_data) == (
            "Auxiliary Data of Type 'RandomArray'\n"
            "\tPath: 'some/deeper/path/test_data'\n"
            "\tData shape: '(10, 10)', dtype: 'float64'\n"
            "\tParameters:\n"
            "\t\ta: 1\n"
            "\t\tb: 2.0\n"
            "\t\te: hallo")


def test_adding_waveforms_with_provenance_id(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")

    data_set = ASDFDataSet(asdf_filename)
    for filename in glob.glob(os.path.join(data_path, "*.mseed")):
        data_set.add_waveforms(filename, tag="raw_recording",
                               provenance_id="{http://example.org}test")

    data_set.__del__()
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)

    st = new_data_set.waveforms.TA_POKR.raw_recording
    for tr in st:
        assert tr.stats.asdf.provenance_id == "{http://example.org}test"

    new_data_set.__del__()
    del new_data_set


def test_coordinate_extraction_but_no_stationxml(tmpdir):
    """
    Tests what happens if no stationxml is defined for a station.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")

    data_set = ASDFDataSet(asdf_filename)
    for filename in glob.glob(os.path.join(data_path, "*.mseed")):
        data_set.add_waveforms(filename, tag="raw_recording")

    # If not stationxml exists it should just return an empty dictionary.
    assert data_set.get_all_coordinates() == {}


def test_adding_auxiliary_data_with_wrong_tag_name_raises(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    # With provenance id.
    data = np.random.random((10, 10))
    # The data must NOT start with a number.
    data_type = "RandomArray"
    path = "A.B.C"

    with pytest.raises(ASDFValueError) as err:
        data_set.add_auxiliary_data(
            data=data, data_type=data_type,
            path=path, parameters={})

    assert err.value.args[0] == (
        "Tag name 'A.B.C' is invalid. It must validate "
        "against the regular expression "
        "'^[a-zA-Z0-9][a-zA-Z0-9_]*[a-zA-Z0-9]$'.")

    data_set.__del__()


def test_adding_arbitrary_files(tmpdir):
    """
    Tests that adding arbitrary files works.
    """
    test_filename = os.path.join(tmpdir.strpath, "temp.json")
    test_dict = {"a": 1, "b": 2}
    with open(test_filename, "wt") as fh:
        json.dump(test_dict, fh, sort_keys=True)

    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    data_set.add_auxiliary_data_file(
        test_filename, path="test_file", parameters={"1": 1})

    data_set.__del__()
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)
    # Extraction works the same as always, but now has a special attribute,
    # that returns the data as a BytesIO.
    aux_data = new_data_set.auxiliary_data.File.test_file
    assert aux_data.parameters == {"1": 1}
    assert aux_data.path == "test_file"

    new_test_dict = json.loads(aux_data.file.read().decode())
    assert test_dict == new_test_dict

    aux_data.file.seek(0, 0)

    with open(test_filename, "rb") as fh:
        assert fh.read() == aux_data.file.read()


def test_provenance_list_command(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    filename = os.path.join(data_dir,
                            "example_schematic_processing_chain.xml")

    # Add it as a document.
    doc = prov.read(filename, format="xml")
    data_set.add_provenance_document(doc, name="test_provenance")

    assert data_set.provenance.list() == ["test_provenance"]


def test_provenance_dicionary_behaviour(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    filename = os.path.join(data_dir,
                            "example_schematic_processing_chain.xml")

    # Add it as a document.
    doc = prov.read(filename, format="xml")
    # Setting via setitem.
    data_set.provenance["test_provenance"] = doc

    data_set.__del__()
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)
    assert new_data_set.provenance.list() == ["test_provenance"]

    assert new_data_set.provenance["test_provenance"] == doc
    assert getattr(new_data_set.provenance, "test_provenance") == doc

    assert list(new_data_set.provenance.keys()) == ["test_provenance"]
    assert list(new_data_set.provenance.values()) == [doc]
    assert list(new_data_set.provenance.items()) == [("test_provenance", doc)]


def test_str_of_auxiliary_data_accessor(tmpdir):
    """
    Test the various __str__ method of auxiliary data types.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    assert str(data_set.auxiliary_data) == (
        "Data set contains no auxiliary data.")

    data = np.random.random((10, 10))
    data_type = "RandomArray"
    path = "test_data_1"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(data=data, data_type=data_type,
                                path=path, parameters=parameters)

    data = np.random.random((10, 10))
    data_type = "RandomArray"
    path = "test_data_2"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(data=data, data_type=data_type,
                                path=path, parameters=parameters)

    data = np.random.random((10, 10))
    data_type = "SomethingElse"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(data=data, data_type=data_type,
                                path=path, parameters=parameters)

    # Add a nested one.
    data_set.add_auxiliary_data(data=data, data_type=data_type,
                                path="some/deep/path",
                                parameters=parameters)

    assert str(data_set.auxiliary_data) == (
        "Data set contains the following auxiliary data types:\n"
        "\tRandomArray (2 item(s))\n"
        "\tSomethingElse (2 item(s))")

    assert str(data_set.auxiliary_data.RandomArray) == (
        "2 auxiliary data item(s) of type 'RandomArray' available:\n"
        "\ttest_data_1\n"
        "\ttest_data_2")

    assert str(data_set.auxiliary_data.SomethingElse) == (
        "1 auxiliary data sub group(s) of type 'SomethingElse' available:\n"
        "\tsome\n"
        "1 auxiliary data item(s) of type 'SomethingElse' available:\n"
        "\ttest_data")

    assert str(data_set.auxiliary_data.SomethingElse.some) == (
        "1 auxiliary data sub group(s) of type 'SomethingElse/some' "
        "available:\n"
        "\tdeep")

    assert str(data_set.auxiliary_data.SomethingElse.some.deep) == (
        "1 auxiliary data item(s) of type 'SomethingElse/some/deep' "
        "available:\n"
        "\tpath")


def test_item_access_of_auxiliary_data(tmpdir):
    """
    Make sure all auxiliary data types, and the data itsself can be accessed
    via dictionary like accesses.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    assert str(data_set.auxiliary_data) == (
        "Data set contains no auxiliary data.")

    data = np.random.random((10, 10))
    data_type = "RandomArray"
    path = "test_data_1"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(data=data, data_type=data_type,
                                path=path, parameters=parameters)

    assert data_set.auxiliary_data["RandomArray"]["test_data_1"].path == \
        data_set.auxiliary_data.RandomArray.test_data_1.path


def test_item_access_of_waveforms(example_data_set):
    """
    Tests that waveforms and stations can be accessed with item access.
    """
    data_set = ASDFDataSet(example_data_set.filename)

    assert data_set.waveforms["AE_113A"]["raw_recording"] == \
        data_set.waveforms.AE_113A.raw_recording == \
        data_set.waveforms["AE.113A"].raw_recording == \
        data_set.waveforms.AE_113A["raw_recording"]

    assert data_set.waveforms["AE_113A"]["StationXML"] == \
        data_set.waveforms.AE_113A.StationXML == \
        data_set.waveforms["AE.113A"].StationXML == \
        data_set.waveforms.AE_113A["StationXML"]


def test_list_method_of_waveform_accessor(example_data_set):
    data_set = ASDFDataSet(example_data_set.filename)

    assert data_set.waveforms.list() == ["AE.113A", "TA.POKR"]


def test_detailed_waveform_access(example_data_set):
    data_set = ASDFDataSet(example_data_set.filename)
    st = data_set.waveforms.AE_113A

    assert st.get_waveform_tags() == ["raw_recording"]
    assert st.list() == [
        'AE.113A..BHE__2013-05-24T05:40:00__2013-05-24T06:50:00'
        '__raw_recording',
        'AE.113A..BHN__2013-05-24T05:40:00__2013-05-24T06:50:00'
        '__raw_recording',
        'AE.113A..BHZ__2013-05-24T05:40:00__2013-05-24T06:50:00'
        '__raw_recording',
        'StationXML']

    assert st['AE.113A..BHZ__2013-05-24T05:40:00__2013-05-24T06:50:00'
              '__raw_recording'][0] == \
        st.raw_recording.select(channel='BHZ')[0]


def test_get_provenance_document_for_id(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    filename = os.path.join(data_dir,
                            "example_schematic_processing_chain.xml")

    doc = prov.read(filename)
    data_set.provenance["test_provenance"] = doc

    assert data_set.provenance.get_provenance_document_for_id(
            '{http://seisprov.org/seis_prov/0.1/#}sp002_dt_f87sf7sf78') == \
        {"name": "test_provenance", "document": doc}

    assert data_set.provenance.get_provenance_document_for_id(
            '{http://seisprov.org/seis_prov/0.1/#}sp004_lp_f87sf7sf78') == \
        {"name": "test_provenance", "document": doc}

    # Id not found.
    with pytest.raises(ASDFValueError) as err:
        data_set.provenance.get_provenance_document_for_id(
            '{http://seisprov.org/seis_prov/0.1/#}bogus_id')

    assert err.value.args[0] == (
        "Document containing id "
        "'{http://seisprov.org/seis_prov/0.1/#}bogus_id'"
        " not found in the data set.")

    # Not a qualified id.
    with pytest.raises(ASDFValueError) as err:
        data_set.provenance.get_provenance_document_for_id("bla")

    assert err.value.args[0] == ("Not a valid qualified name.")

    data_set.__del__()


def test_empty_asdf_file_has_no_quakeml_dataset(tmpdir):
    """
    There is no reason an empty ASDF file should have a QuakeML group.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)
    data_set.__del__()

    f = h5py.File(asdf_filename)
    assert "QuakeML" not in f

    # It should still return an empty catalog object if the events are
    # requested.
    new_data_set = ASDFDataSet(asdf_filename)
    assert len(new_data_set.events) == 0
    new_data_set.__del__()


def test_event_iteration(example_data_set):
    """
    Tests the iteration over a dataset by event attributes.
    """
    ds = ASDFDataSet(example_data_set.filename)
    # Store a new waveform.
    event_id = "quakeml:random/event"
    origin_id = "quakeml:random/origin"
    magnitude_id = "quakeml:random/magnitude"
    focmec_id = "quakeml:random/focmec"

    tr = obspy.read()[0]

    # Has all four.
    tr.stats.network = "AA"
    tr.stats.station = "AA"
    ds.add_waveforms(tr, tag="random_a", event_id=event_id,
                     origin_id=origin_id, focal_mechanism_id=focmec_id,
                     magnitude_id=magnitude_id)
    # Only event.
    tr.stats.network = "BB"
    tr.stats.station = "BB"
    ds.add_waveforms(tr, tag="random_b", event_id=event_id)

    # Only origin.
    tr.stats.network = "CC"
    tr.stats.station = "CC"
    ds.add_waveforms(tr, tag="random_c", origin_id=origin_id)

    # Only magnitude.
    tr.stats.network = "DD"
    tr.stats.station = "DD"
    ds.add_waveforms(tr, tag="random_d", magnitude_id=magnitude_id)

    # Only focal mechanism.
    tr.stats.network = "EE"
    tr.stats.station = "EE"
    ds.add_waveforms(tr, tag="random_e", focal_mechanism_id=focmec_id)

    # Nothing.
    tr.stats.network = "FF"
    tr.stats.station = "FF"
    ds.add_waveforms(tr, tag="random_f")

    # Test with random ids..should all return nothing.
    random_ids = [
        "test", "random",
        obspy.core.event.ResourceIdentifier(
            "smi:service.iris.edu/fdsnws/event/1/query?random_things")]
    for r_id in random_ids:
        assert list(ds.ifilter(ds.q.event == r_id)) == []
        assert list(ds.ifilter(ds.q.magnitude == r_id)) == []
        assert list(ds.ifilter(ds.q.origin == r_id)) == []
        assert list(ds.ifilter(ds.q.focal_mechanism == r_id)) == []

    # Event as a resource identifier and as a string, and with others equal to
    # None.
    result = [_i._station_name for _i in ds.ifilter(ds.q.event == event_id)]
    assert result == ["AA.AA", "BB.BB"]
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.event == str(event_id))]
    assert result == ["AA.AA", "BB.BB"]
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.event == str(event_id),
                                   ds.q.magnitude == None,
                                   ds.q.focal_mechanism == None)]  # NOQA
    assert result == ["BB.BB"]

    # Origin as a resource identifier and as a string, and with others equal to
    # None.
    result = [_i._station_name for _i in ds.ifilter(ds.q.origin == origin_id)]
    assert result == ["AA.AA", "CC.CC"]
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.origin == str(origin_id))]
    assert result == ["AA.AA", "CC.CC"]
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.origin == str(origin_id),
                                   ds.q.event == None)]  # NOQA
    assert result == ["CC.CC"]

    # Magnitude as a resource identifier and as a string, and with others equal
    # to None.
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.magnitude == magnitude_id)]
    assert result == ["AA.AA", "DD.DD"]
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.magnitude == str(magnitude_id))]
    assert result == ["AA.AA", "DD.DD"]
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.magnitude == str(magnitude_id),
                                   ds.q.origin == None)]  # NOQA
    assert result == ["DD.DD"]

    # focmec as a resource identifier and as a string, and with others equal to
    # None.
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.focal_mechanism == focmec_id)]
    assert result == ["AA.AA", "EE.EE"]
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.focal_mechanism == str(focmec_id))]
    assert result == ["AA.AA", "EE.EE"]
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.focal_mechanism == str(focmec_id),
                                   ds.q.origin == None)]  # NOQA
    assert result == ["EE.EE"]

    # No existing ids are treated like empty ids.
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.event == None,
                                   ds.q.magnitude == None,
                                   ds.q.origin == None,
                                   ds.q.focal_mechanism == None)]  # NOQA
    assert result == ["FF.FF"]

    result = [_i._station_name
              for _i in ds.ifilter(ds.q.event != None,
                                   ds.q.magnitude != None,
                                   ds.q.origin != None,
                                   ds.q.focal_mechanism != None)]  # NOQA
    assert result == ["AA.AA"]


def test_more_queries(example_data_set):
    """
    Test more queries.

    Example data set contains (with path "raw_recording"):

    AE.113A..BHE | 2013-05-24T05:40:00-2013-05-24T06:50:00.000000Z |
        40.0 Hz, 168001 samples
    AE.113A..BHN | 2013-05-24T05:40:00-2013-05-24T06:50:00.000000Z |
        40.0 Hz, 168001 samples
    AE.113A..BHZ | 2013-05-24T05:40:00-2013-05-24T06:50:00.000000Z |
        40.0 Hz, 168001 samples
    TA.POKR..BHE | 2013-05-24T05:40:00-2013-05-24T06:50:00.000001Z |
        40.0 Hz, 168001 samples
    TA.POKR..BHN | 2013-05-24T05:40:00-2013-05-24T06:50:00.000000Z |
        40.0 Hz, 168001 samples
    TA.POKR..BHZ | 2013-05-24T05:40:00-2013-05-24T06:50:00.000001Z |
        40.0 Hz, 168001 samples

    complete with StationXML information:

    >>> ds.waveforms.AE_113A.coordinates
    {'elevation_in_m': 118.0, 'latitude': 32.7683, 'longitude': -113.7667}
    >>> ds.waveforms.TA_POKR.coordinates
    {'elevation_in_m': 501.0, 'latitude': 65.1171, 'longitude': -147.4335}

    We'll add some more with:

    BW.RJOB..EHZ | 2009-08-24T00:20:03-2009-08-24T00:20:32.990000Z |
        100.0 Hz, 3000 samples
    BW.RJOB..EHN | 2009-08-24T00:20:03-2009-08-24T00:20:32.990000Z |
        100.0 Hz, 3000 samples
    BW.RJOB..EHE | 2009-08-24T00:20:03-2009-08-24T00:20:32.990000Z |
        100.0 Hz, 3000 samples

    with no station information.
    """
    ds = ASDFDataSet(example_data_set.filename)
    ds.add_waveforms(obspy.read(), tag="random")

    # Helper function.
    def collect_ids(it):
        collection = set()
        for _i in it:
            for tag in _i.get_waveform_tags():
                st = _i[tag]
                for tr in st:
                    collection.add(tr.id)
        return collection

    # Get a single trace.
    assert collect_ids(ds.ifilter(ds.q.network == "TA",
                                  ds.q.station == "POKR",
                                  ds.q.location == "",
                                  ds.q.channel == "BHZ")) == {
        "TA.POKR..BHZ"
    }

    # Get the three 100 Hz traces.
    assert collect_ids(ds.ifilter(ds.q.sampling_rate >= 100.0)) == {
        "BW.RJOB..EHE", "BW.RJOB..EHN", "BW.RJOB..EHZ"}

    # Get the "random" tagged traces in different ways.
    assert collect_ids(ds.ifilter(ds.q.tag == "random")) == {
        "BW.RJOB..EHE", "BW.RJOB..EHN", "BW.RJOB..EHZ"}
    assert collect_ids(ds.ifilter(ds.q.tag == ["random"])) == {
        "BW.RJOB..EHE", "BW.RJOB..EHN", "BW.RJOB..EHZ"}
    assert collect_ids(ds.ifilter(ds.q.tag == ["dummy", "r*m"])) == {
        "BW.RJOB..EHE", "BW.RJOB..EHN", "BW.RJOB..EHZ"}

    # Geographic constraints. Will never return the BW channels as they have
    # no coordinate information.
    assert collect_ids(ds.ifilter(ds.q.latitude >= 30.0,
                                  ds.q.latitude <= 40.0)) == {
        "AE.113A..BHE", "AE.113A..BHN", "AE.113A..BHZ"}
    assert collect_ids(ds.ifilter(ds.q.longitude >= -120.0,
                                  ds.q.longitude <= -110.0)) == {
               "AE.113A..BHE", "AE.113A..BHN", "AE.113A..BHZ"}
    assert collect_ids(ds.ifilter(ds.q.elevation_in_m < 200.0)) == {
               "AE.113A..BHE", "AE.113A..BHN", "AE.113A..BHZ"}

    # Make sure coordinates exist.
    assert collect_ids(ds.ifilter(ds.q.latitude != None)) == {  # NOQA
        "AE.113A..BHE", "AE.113A..BHN", "AE.113A..BHZ", "TA.POKR..BHE",
        "TA.POKR..BHZ", "TA.POKR..BHN"}
    # Opposite query
    assert collect_ids(ds.ifilter(ds.q.latitude == None)) == {  # NOQA
        "BW.RJOB..EHE", "BW.RJOB..EHN", "BW.RJOB..EHZ"}

    # Temporal constraints.
    assert collect_ids(ds.ifilter(ds.q.starttime <= "2010-01-01")) == {
        "BW.RJOB..EHE", "BW.RJOB..EHN", "BW.RJOB..EHZ"}

    # Exact endtime
    assert collect_ids(ds.ifilter(
            ds.q.endtime <=
            obspy.UTCDateTime("2009-08-24T00:20:32.990000Z"))) == {
        "BW.RJOB..EHE", "BW.RJOB..EHN", "BW.RJOB..EHZ"}
    assert collect_ids(ds.ifilter(
        ds.q.endtime <=
        obspy.UTCDateTime("2009-08-24T00:20:32.990000Z") - 1)) == set()
    assert collect_ids(ds.ifilter(
        ds.q.endtime <
        obspy.UTCDateTime("2009-08-24T00:20:32.990000Z"))) == set()

    # Number of samples.
    assert collect_ids(ds.ifilter(ds.q.npts > 1000, ds.q.npts < 5000)) == {
        "BW.RJOB..EHE", "BW.RJOB..EHN", "BW.RJOB..EHZ"}

    # All vertical channels.
    assert collect_ids(ds.ifilter(ds.q.channel == "*Z")) == {
        "BW.RJOB..EHZ", "TA.POKR..BHZ", "AE.113A..BHZ"}

    # Many keys cannot be None, as their value must always be given.
    for key in ["network", "station", "location", "channel", "tag",
                "starttime", "endtime", "sampling_rate", "npts"]:
        with pytest.raises(TypeError):
            ds.ifilter(getattr(ds.q, key) == None)


def test_saving_trace_labels(tmpdir):
    """
    Tests that the labels can be saved and retrieved automatically.
    """
    data_path = os.path.join(data_dir, "small_sample_data_set")
    filename = os.path.join(tmpdir.strpath, "example.h5")

    data_set = ASDFDataSet(filename)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(
        waveform, "raw_recording",
        labels=["hello", "what", "is", "going", "on?"])

    # Close and reopen.
    del data_set
    data_set = ASDFDataSet(filename)

    st = data_set.waveforms.TA_POKR.raw_recording
    for tr in st:
        assert tr.stats.asdf.labels == ["hello", "what", "is", "going", "on?"]
    del data_set
    os.remove(filename)

    # Try again but this time with unicode.
    data_set = ASDFDataSet(filename)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    labels = [u"?⸘‽", u"^§#⁇❦"]
    data_set.add_waveforms(waveform, "raw_recording", labels=labels)

    # Close and reopen.
    del data_set
    data_set = ASDFDataSet(filename)

    st = data_set.waveforms.TA_POKR.raw_recording
    for tr in st:
        assert tr.stats.asdf.labels == labels
    del data_set
    os.remove(filename)


def test_labels_are_persistent_through_processing(tmpdir):
    """
    Processing a file with an associated label and storing it again should
    keep the association.
    """
    data_path = os.path.join(data_dir, "small_sample_data_set")
    filename = os.path.join(tmpdir.strpath, "example.h5")

    data_set = ASDFDataSet(filename)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(
        waveform, "raw_recording",
        labels=["hello", "what", "is", "going", "on?"])

    # Close an reopen.
    del data_set
    data_set = ASDFDataSet(filename)

    st = data_set.waveforms.TA_POKR.raw_recording
    labels = st[0].stats.asdf.labels

    # Process and store.
    st.taper(max_percentage=0.05, type="cosine")
    data_set.add_waveforms(st, tag="processed")

    # Close an reopen.
    del data_set
    data_set = ASDFDataSet(filename)

    processed_st = data_set.waveforms.TA_POKR.processed
    assert labels == processed_st[0].stats.asdf.labels

    del data_set
    os.remove(filename)

    # Same again but for unicode.
    data_set = ASDFDataSet(filename)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(
        waveform, "raw_recording",
        labels=[u"?⸘‽", u"^§#⁇❦"])

    # Close an reopen.
    del data_set
    data_set = ASDFDataSet(filename)

    st = data_set.waveforms.TA_POKR.raw_recording
    labels = st[0].stats.asdf.labels

    # Process and store.
    st.taper(max_percentage=0.05, type="cosine")
    data_set.add_waveforms(st, tag="processed")

    # Close an reopen.
    del data_set
    data_set = ASDFDataSet(filename)

    processed_st = data_set.waveforms.TA_POKR.processed
    assert labels == processed_st[0].stats.asdf.labels

    del data_set
    os.remove(filename)


def test_queries_for_labels(tmpdir):
    """
    Tests the iteration over a dataset by labels.
    """
    filename = os.path.join(tmpdir.strpath, "example.h5")
    ds = ASDFDataSet(filename)

    # Store a new waveform.
    labels_b = ["what", "is", "happening"]
    labels_c = [u"?⸘‽", u"^§#⁇❦"]
    labels_d = ["single_label"]

    tr = obspy.read()[0]

    # Has no label.
    tr.stats.network = "AA"
    tr.stats.station = "AA"
    ds.add_waveforms(tr, tag="random_a")

    tr.stats.network = "BB"
    tr.stats.station = "BB"
    ds.add_waveforms(tr, tag="random_b", labels=labels_b)

    tr.stats.network = "CC"
    tr.stats.station = "CC"
    ds.add_waveforms(tr, tag="random_c", labels=labels_c)

    tr.stats.network = "DD"
    tr.stats.station = "DD"
    ds.add_waveforms(tr, tag="random_d", labels=labels_d)

    # Test with random labels...should all return nothing.
    assert list(ds.ifilter(ds.q.labels == ["hello", "hello2"])) == []
    assert list(ds.ifilter(ds.q.labels == ["random"])) == []

    # Once of each.
    result = [_i._station_name for _i in ds.ifilter(
        ds.q.labels == ["what", u"?⸘‽", "single_label"])]
    assert result == ["BB.BB", "CC.CC", "DD.DD"]

    # No labels.
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.labels == None)]  # NOQA
    assert result == ["AA.AA"]

    # Any label.
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.labels != None)]  # NOQA
    assert result == ["BB.BB", "CC.CC", "DD.DD"]

    # Unicode wildcard.
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.labels == u"^§#⁇*")]
    assert result == ["CC.CC"]

    # BB and DD.
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.labels == ["wha?", "sin*"])]
    assert result == ["BB.BB", "DD.DD"]

    # CC
    result = [_i._station_name
              for _i in ds.ifilter(ds.q.labels == u"^§#⁇*")]
    assert result == ["CC.CC"]


def test_waveform_accessor_attribute_access_error_handling(example_data_set):
    """
    Tests the error handling of the waveform accessors.
    """
    ds = ASDFDataSet(example_data_set.filename)

    # Add new data.
    ds.add_waveforms(obspy.read(), tag="random")

    # The existing station has a stationxml file.
    assert "StationXML" in ds.waveforms.AE_113A
    assert hasattr(ds.waveforms.AE_113A, "StationXML")

    # The new one has no StationXML file.
    assert "StationXML" not in ds.waveforms.BW_RJOB
    assert hasattr(ds.waveforms.BW_RJOB, "StationXML") is False

    with pytest.raises(AttributeError) as e:
        ds.waveforms.BW_RJOB.StationXML
    assert e.value.args[0] == \
        "'WaveformAccessor' object has no attribute 'StationXML'"

    with pytest.raises(KeyError) as e:
        ds.waveforms.BW_RJOB["StationXML"]
    assert e.value.args[0] == \
        "'WaveformAccessor' object has no attribute 'StationXML'"

    del ds


def test_reading_and_writing_auxiliary_nested_auxiliary_data(tmpdir):
    """
    Tests reading and writing auxiliary nested data.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    # Define some auxiliary data and add it.
    data = np.random.random(100)
    data_type = "RandomArrays"
    # The path can be a path. At that point it will be a nested structure.
    path = "some/deeper/path/test_data"

    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(data=data, data_type=data_type, path=path,
                                parameters=parameters)
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)

    aux_data = \
        new_data_set.auxiliary_data.RandomArrays.some.deeper.path.test_data
    np.testing.assert_equal(data, aux_data.data)
    aux_data.data_type == data_type
    aux_data.path == path
    aux_data.parameters == parameters
