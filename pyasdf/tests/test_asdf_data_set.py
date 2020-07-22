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

import glob
import inspect
import io
import json
import shutil
import os
import random
import sys
import warnings

import h5py
import numpy as np
import obspy
from obspy import UTCDateTime
import prov
import pytest

from pyasdf import ASDFDataSet
from pyasdf.exceptions import (
    WaveformNotInFileException,
    ASDFValueError,
    ASDFAttributeError,
    ASDFWarning,
)
from pyasdf.header import FORMAT_NAME


data_dir = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
    "data",
)


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
        data_set.add_waveforms(
            filename, tag="raw_recording", event_id=data_set.events[0]
        )

    # Flush and finish writing.
    del data_set

    # Return filename and path to tempdir, no need to always create a
    # new one.
    return Namespace(filename=asdf_filename, tmpdir=tmpdir.strpath)


def test_waveform_tags_attribute(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")

    data_set = ASDFDataSet(asdf_filename)

    itag = 1
    for filename in glob.glob(os.path.join(data_path, "*.mseed")):
        data_set.add_waveforms(filename, tag="tag%d" % itag)
        itag += 1

    expected = set(["tag1", "tag2", "tag3", "tag4", "tag5", "tag6"])
    assert data_set.waveform_tags == expected


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
        stream_asdf = getattr(
            data_set.waveforms, "%s_%s" % station
        ).raw_recording
        stream_file = obspy.read(
            os.path.join(data_path, "%s.%s.*.mseed" % station)
        )
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
        inv_asdf = getattr(data_set.waveforms, "%s_%s" % station).StationXML
        inv_file = obspy.read_inventory(
            os.path.join(data_path, "%s.%s..BH_.xml" % station)
        )
        assert inv_file == inv_asdf
    # Test the event.
    cat_file = obspy.read_events(os.path.join(data_path, "quake.xml"))
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

    ref_cat = obspy.read_events(event_filename)

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
        assert hdf5_file.attrs["file_format_version"].decode() == "1.0.3"
        assert hdf5_file.attrs["file_format"].decode() == FORMAT_NAME


def test_dot_accessors(example_data_set):
    """
    Tests the dot accessors for waveforms and stations.
    """
    data_path = os.path.join(data_dir, "small_sample_data_set")
    data_set = ASDFDataSet(example_data_set.filename)

    # Get the contents, this also asserts that tab completions works.
    assert sorted(dir(data_set.waveforms)) == ["AE_113A", "TA_POKR"]
    assert "raw_recording" in dir(data_set.waveforms.AE_113A)
    assert "raw_recording" in dir(data_set.waveforms.TA_POKR)

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

    assert data_set.waveforms.AE_113A.StationXML == obspy.read_inventory(
        os.path.join(data_path, "AE.113A..BH_.xml")
    )
    assert data_set.waveforms.TA_POKR.StationXML == obspy.read_inventory(
        os.path.join(data_path, "TA.POKR..BH_.xml")
    )


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
    event = obspy.read_events(os.path.join(data_path, "quake.xml"))[0]

    # Add the event object, and associate the waveform with it.
    data_set = ASDFDataSet(filename)
    data_set.add_quakeml(event)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(waveform, "raw_recording", event_id=event)
    st = data_set.waveforms.TA_POKR.raw_recording
    for tr in st:
        assert tr.stats.asdf.event_ids[0].get_referred_object() == event
    del data_set
    os.remove(filename)

    # Add as a string.
    data_set = ASDFDataSet(filename)
    data_set.add_quakeml(event)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(
        waveform, "raw_recording", event_id=str(event.resource_id.id)
    )
    st = data_set.waveforms.TA_POKR.raw_recording
    for tr in st:
        assert tr.stats.asdf.event_ids[0].get_referred_object() == event
    del data_set
    os.remove(filename)

    # Add as a resource identifier object.
    data_set = ASDFDataSet(filename)
    data_set.add_quakeml(event)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(
        waveform, "raw_recording", event_id=event.resource_id
    )
    st = data_set.waveforms.TA_POKR.raw_recording
    for tr in st:
        assert tr.stats.asdf.event_ids[0].get_referred_object() == event
    del data_set
    os.remove(filename)


def test_event_association_is_persistent_through_processing(example_data_set):
    """
    Processing a file with an associated event and storing it again should
    keep the association.
    """
    data_set = ASDFDataSet(example_data_set.filename)
    st = data_set.waveforms.TA_POKR.raw_recording
    event_id = st[0].stats.asdf.event_ids[0]

    st.taper(max_percentage=0.05, type="cosine")

    data_set.add_waveforms(st, tag="processed")
    processed_st = data_set.waveforms.TA_POKR.processed
    assert event_id == processed_st[0].stats.asdf.event_ids[0]


def test_detailed_event_association_is_persistent_through_processing(
    example_data_set,
):
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

    data_set.add_waveforms(
        tr,
        tag="random",
        event_id=event,
        origin_id=origin,
        focal_mechanism_id=focmec,
        magnitude_id=magnitude,
    )

    new_st = data_set.waveforms.BW_RJOB.random
    new_st.taper(max_percentage=0.05, type="cosine")

    data_set.add_waveforms(new_st, tag="processed")
    processed_st = data_set.waveforms.BW_RJOB.processed
    assert event.resource_id == processed_st[0].stats.asdf.event_ids[0]
    assert origin.resource_id == processed_st[0].stats.asdf.origin_ids[0]
    assert magnitude.resource_id == processed_st[0].stats.asdf.magnitude_ids[0]
    assert (
        focmec.resource_id == processed_st[0].stats.asdf.focal_mechanism_ids[0]
    )


def test_tag_iterator(example_data_set):
    """
    Tests the iteration over tags with the ifilter() method.
    """
    ds = ASDFDataSet(example_data_set.filename)

    expected_ids = [
        "AE.113A..BHE",
        "AE.113A..BHN",
        "AE.113A..BHZ",
        "TA.POKR..BHE",
        "TA.POKR..BHN",
        "TA.POKR..BHZ",
    ]

    for station in ds.ifilter(ds.q.tag == "raw_recording"):
        inv = station.StationXML
        for tr in station.raw_recording:
            assert tr.id in expected_ids
            expected_ids.remove(tr.id)
            assert bool(
                inv.select(
                    network=tr.stats.network,
                    station=tr.stats.station,
                    channel=tr.stats.channel,
                    location=tr.stats.location,
                ).networks
            )

        # Cheap test for the str() method.
        assert str(station).startswith("Filtered contents")

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
    data_set.process(
        null_processing, output_filename, {"raw_recording": "raw_recording"}
    )

    del data_set

    data_set = ASDFDataSet(example_data_set.filename)
    out_data_set = ASDFDataSet(output_filename)

    assert data_set == out_data_set


def test_processing_multiprocessing_without_compression(example_data_set):
    """
    Tests the processing using multiprocessing on a ASDF file without
    compression.
    """

    def null_processing(st, inv):
        return st

    data_set = ASDFDataSet(example_data_set.filename, compression=None)
    output_filename = os.path.join(example_data_set.tmpdir, "output.h5")
    # Do not actually do anything. Apply an empty function.
    data_set.process(
        null_processing, output_filename, {"raw_recording": "raw_recording"}
    )

    del data_set
    data_set = ASDFDataSet(example_data_set.filename)
    out_data_set = ASDFDataSet(output_filename)

    assert data_set == out_data_set


def test_format_version_handling(tmpdir):
    """
    Tests how pyasdf deals with different ASDF versions.

    Also more or less tests that the format version is correctly written and
    read.
    """
    filename = os.path.join(tmpdir.strpath, "test.h5")
    # There are two attributes to the data set object:
    #
    # * The ASDF version in the file.
    # * The used ASDF version.
    #
    # In most cases these should be identical.

    # Create.
    with ASDFDataSet(filename) as ds:
        assert ds.asdf_format_version_in_file == "1.0.3"
        assert ds.asdf_format_version == "1.0.3"
    # Open again.
    with ASDFDataSet(filename) as ds:
        assert ds.asdf_format_version_in_file == "1.0.3"
        assert ds.asdf_format_version == "1.0.3"

    os.remove(filename)

    # Directly specify it.
    with ASDFDataSet(filename, format_version="1.0.3") as ds:
        assert ds.asdf_format_version_in_file == "1.0.3"
        assert ds.asdf_format_version == "1.0.3"
    with ASDFDataSet(filename) as ds:
        assert ds.asdf_format_version_in_file == "1.0.3"
        assert ds.asdf_format_version == "1.0.3"

    os.remove(filename)

    with ASDFDataSet(filename, format_version="1.0.2") as ds:
        assert ds.asdf_format_version_in_file == "1.0.2"
        assert ds.asdf_format_version == "1.0.2"
    with ASDFDataSet(filename) as ds:
        assert ds.asdf_format_version_in_file == "1.0.2"
        assert ds.asdf_format_version == "1.0.2"

    os.remove(filename)

    with ASDFDataSet(filename, format_version="1.0.1") as ds:
        assert ds.asdf_format_version_in_file == "1.0.1"
        assert ds.asdf_format_version == "1.0.1"
    with ASDFDataSet(filename) as ds:
        assert ds.asdf_format_version_in_file == "1.0.1"
        assert ds.asdf_format_version == "1.0.1"

    os.remove(filename)

    # Also version 1.0.0
    with ASDFDataSet(filename, format_version="1.0.0") as ds:
        assert ds.asdf_format_version_in_file == "1.0.0"
        assert ds.asdf_format_version == "1.0.0"
    with ASDFDataSet(filename) as ds:
        assert ds.asdf_format_version_in_file == "1.0.0"
        assert ds.asdf_format_version == "1.0.0"

    os.remove(filename)
    # Both can also differ.
    with ASDFDataSet(filename, format_version="1.0.0") as ds:
        assert ds.asdf_format_version_in_file == "1.0.0"
        assert ds.asdf_format_version == "1.0.0"
    # Read again, but force an ASDF version that differs from the one in the
    # file. A warning will be raised in this case.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with ASDFDataSet(filename, format_version="1.0.1") as ds:
            assert ds.asdf_format_version_in_file == "1.0.0"
            assert ds.asdf_format_version == "1.0.1"

    assert w[0].message.args[0] == (
        "You are forcing ASDF version 1.0.1 but the version of the file is "
        "1.0.0. Please proceed with caution as other tools might not be able "
        "to read the file again."
    )

    # Once again but with some random version.
    os.remove(filename)
    # Both can also differ.
    with ASDFDataSet(filename) as ds:
        ds._ASDFDataSet__file.attrs[
            "file_format_version"
        ] = ds._zeropad_ascii_string("x.x.x")
        assert ds.asdf_format_version_in_file == "x.x.x"
        assert ds.asdf_format_version == "1.0.3"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with ASDFDataSet(filename) as ds:
            assert ds.asdf_format_version_in_file == "x.x.x"
            assert ds.asdf_format_version == "1.0.3"
    assert w[0].message.args[0] == (
        "The file claims an ASDF version of x.x.x. This version of pyasdf "
        "only supports versions: 1.0.0, 1.0.1, 1.0.2, 1.0.3. All following "
        "write operations will use version 1.0.3 - other tools might not be "
        "able to read the files again - proceed with caution."
    )
    # Again but force version.
    os.remove(filename)
    # Both can also differ.
    with ASDFDataSet(filename) as ds:
        ds._ASDFDataSet__file.attrs[
            "file_format_version"
        ] = ds._zeropad_ascii_string("x.x.x")
        assert ds.asdf_format_version_in_file == "x.x.x"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with ASDFDataSet(filename, format_version="1.0.0") as ds:
            assert ds.asdf_format_version_in_file == "x.x.x"
            assert ds.asdf_format_version == "1.0.0"
    assert w[0].message.args[0] == (
        "The file claims an ASDF version of x.x.x. This version of pyasdf "
        "only supports versions: 1.0.0, 1.0.1, 1.0.2, 1.0.3. All following "
        "write operations will use version 1.0.0 - other tools might not be "
        "able to read the files again - proceed with caution."
    )

    # Unsupported version number.
    os.remove(filename)
    with pytest.raises(ASDFValueError) as err:
        ASDFDataSet(filename, format_version="x.x.x")
    assert err.value.args[0] == (
        "ASDF version 'x.x.x' is not supported. Supported versions: 1.0.0, "
        "1.0.1, 1.0.2, 1.0.3"
    )
    # No file should be created.
    assert not os.path.exists(filename)

    # Create a file with a format name but no version.
    with ASDFDataSet(filename) as ds:
        del ds._ASDFDataSet__file.attrs["file_format_version"]
    # Has to raise a warning and will write the file format version to the
    # file.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with ASDFDataSet(filename) as ds:
            assert ds.asdf_format_version_in_file == "1.0.3"
            assert ds.asdf_format_version == "1.0.3"
    assert w[0].message.args[0] == (
        "No file format version given in file '%s'. The program will "
        "continue but the result is undefined." % os.path.abspath(filename)
    )


def test_asdf_version_handling_during_writing(tmpdir):
    filename = os.path.join(tmpdir.strpath, "test.h5")
    # Current there are only two support ASDF versions: 1.0.0 and 1.0.1 -
    # the only difference between both is that 1.0.1 supports 16 bit integers.

    tr = obspy.Trace(
        data=np.zeros(10, dtype=np.int16),
        header={
            "network": "XX",
            "station": "YYY",
            "location": "",
            "channel": "BHX",
        },
    )

    # Fail with version 1.0.0.
    with ASDFDataSet(filename, format_version="1.0.0") as ds:
        with pytest.raises(TypeError) as err:
            ds.add_waveforms(tr, tag="test")
        assert err.value.args[0] == (
            "The trace's dtype ('int16') is not allowed inside ASDF 1.0.0. "
            "Allowed are little and big endian 4 and 8 byte signed integers "
            "and floating point numbers."
        )
        assert len(ds.waveforms) == 0

    # Works fine with version 1.0.1.
    os.remove(filename)
    with ASDFDataSet(filename, format_version="1.0.1") as ds:
        ds.add_waveforms(tr, tag="test")
        assert len(ds.waveforms) == 1
        tr_new = ds.waveforms.XX_YYY.test[0]
        assert tr_new.data.dtype == np.int16

    # Some dtypes should not work at all.
    tr = obspy.Trace(
        data=np.zeros(10, dtype=np.int8),
        header={
            "network": "XX",
            "station": "YYY",
            "location": "",
            "channnel": "BHX",
        },
    )
    os.remove(filename)
    with ASDFDataSet(filename, format_version="1.0.0") as ds:
        with pytest.raises(TypeError) as err:
            ds.add_waveforms(tr, tag="test")
        assert err.value.args[0] == (
            "The trace's dtype ('int8') is not allowed inside ASDF 1.0.0. "
            "Allowed are little and big endian 4 and 8 byte signed integers "
            "and floating point numbers."
        )
        assert len(ds.waveforms) == 0
    os.remove(filename)
    with ASDFDataSet(filename, format_version="1.0.1") as ds:
        with pytest.raises(TypeError) as err:
            ds.add_waveforms(tr, tag="test")
        assert err.value.args[0] == (
            "The trace's dtype ('int8') is not allowed inside ASDF 1.0.1. "
            "Allowed are little and big endian 2, 4, and 8 byte signed "
            "integers and 4 and 8 byte floating point numbers."
        )
        assert len(ds.waveforms) == 0


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

    data_set.add_auxiliary_data(
        data=data, data_type=data_type, path=path, parameters=parameters
    )
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)

    assert len(new_data_set.auxiliary_data) == 1
    assert sorted(dir(new_data_set.auxiliary_data)), sorted(
        ["list", "RandomArrays"]
    )

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

    new_data_set.add_auxiliary_data(
        data=data, data_type=data_type, path=path, parameters=parameters
    )
    del new_data_set

    newer_data_set = ASDFDataSet(asdf_filename)
    aux_data = (
        newer_data_set.auxiliary_data.RandomArrays.some.nested.path.test_data
    )
    np.testing.assert_equal(data, aux_data.data)
    aux_data.data_type == data_type
    aux_data.path == path
    aux_data.parameters == parameters

    del newer_data_set


def test_reading_and_writing_auxiliary_data_with_extended_path_names(tmpdir):
    """
    ASDF 1.0.3 allows more characters in the auxiliary data tag names - test
    that here.
    """
    # Version 1.0.2 does not allow funky chars.
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    with ASDFDataSet(asdf_filename, format_version="1.0.2") as ds:
        data = np.random.random(100)
        data_type = "RandomArrays"
        path = "test_data/hallo/A.B.C/$1/A"
        parameters = {"a": 1, "b": 2.0, "e": "hallo"}
        with pytest.raises(ValueError) as err:
            ds.add_auxiliary_data(
                data=data,
                data_type=data_type,
                path=path,
                parameters=parameters,
            )
    assert err.value.args[0] == (
        "Path part name 'A.B.C' is invalid. It must validate against the "
        "regular expression '^[a-zA-Z0-9][a-zA-Z0-9_]*[a-zA-Z0-9]$' in ASDF "
        "version '1.0.2'."
    )

    # But version 1.0.3 does.
    asdf_filename_2 = os.path.join(tmpdir.strpath, "test_2.h5")
    data = np.random.random(100)
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}
    with ASDFDataSet(asdf_filename_2) as ds:
        data_type = "RandomArrays"
        path = "test_data/hallo/A.B.C/$1/A"
        ds.add_auxiliary_data(
            data=data, data_type=data_type, path=path, parameters=parameters
        )

    # Open again and check.
    with ASDFDataSet(asdf_filename_2) as ds:
        aux = ds.auxiliary_data.RandomArrays.test_data.hallo["A.B.C"]["$1"].A
        np.testing.assert_equal(aux.data[:], data)
        assert aux.parameters == parameters


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

        assert excinfo.value.args[0] == (
            "Tag 'asdfasdf' not part of the data " "set for station 'AE.113A'."
        )
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
        "        raw_recording"
    )

    data_set.__del__()
    del data_set


def test_coordinate_extraction(example_data_set):
    """
    Tests the quick coordinate extraction.
    """
    data_set = ASDFDataSet(example_data_set.filename)

    assert data_set.waveforms.AE_113A.coordinates == {
        "latitude": 32.7683,
        "longitude": -113.7667,
        "elevation_in_m": 118.0,
    }

    assert data_set.waveforms.TA_POKR.coordinates == {
        "latitude": 65.1171,
        "longitude": -147.4335,
        "elevation_in_m": 501.0,
    }


def test_coordinate_extraction_with_many_station_comments(tmpdir):
    """
    Regression test to guard against some strange/faulty behaviour of
    etree.iterparse() when there are a larger number of child elements.
    """
    ds = ASDFDataSet(os.path.join(tmpdir.strpath, "test.h5"))
    ds.add_stationxml(os.path.join(data_dir, "II.ABKT.xml"))
    assert ds.waveforms.II_ABKT.coordinates == {
        "latitude": 37.9304,
        "longitude": 58.1189,
        "elevation_in_m": 678.0,
    }
    assert ds.waveforms.II_ABKT.channel_coordinates == {
        "II.ABKT.00.BHE": [
            {
                "latitude": 37.9304,
                "local_depth_in_m": 7.0,
                "starttime": UTCDateTime(2010, 7, 14, 12, 0),
                "endtime": UTCDateTime(2013, 12, 30, 23, 59, 59),
                "longitude": 58.1189,
                "elevation_in_m": 678.0,
            }
        ],
        "II.ABKT.00.BHN": [
            {
                "latitude": 37.9304,
                "local_depth_in_m": 7.0,
                "starttime": UTCDateTime(2010, 7, 14, 12, 0),
                "endtime": UTCDateTime(2013, 12, 30, 23, 59, 59),
                "longitude": 58.1189,
                "elevation_in_m": 678.0,
            }
        ],
        "II.ABKT.00.BHZ": [
            {
                "latitude": 37.9304,
                "local_depth_in_m": 7.0,
                "starttime": UTCDateTime(2010, 7, 14, 12, 0),
                "endtime": UTCDateTime(2013, 12, 30, 23, 59, 59),
                "longitude": 58.1189,
                "elevation_in_m": 678.0,
            }
        ],
    }


def test_coordinate_extraction_channel_level(example_data_set):
    """
    Tests the quick coordinate extraction at the channel level.
    """
    data_set = ASDFDataSet(example_data_set.filename)

    assert data_set.waveforms.AE_113A.channel_coordinates == {
        "AE.113A..BHE": [
            {
                "elevation_in_m": 118.0,
                "endtime": UTCDateTime(2599, 12, 31, 23, 59, 59),
                "latitude": 32.7683,
                "local_depth_in_m": 0.0,
                "longitude": -113.7667,
                "starttime": UTCDateTime(2011, 12, 1, 0, 0),
            }
        ],
        "AE.113A..BHN": [
            {
                "elevation_in_m": 118.0,
                "endtime": UTCDateTime(2599, 12, 31, 23, 59, 59),
                "latitude": 32.7683,
                "local_depth_in_m": 0.0,
                "longitude": -113.7667,
                "starttime": UTCDateTime(2011, 12, 1, 0, 0),
            }
        ],
        "AE.113A..BHZ": [
            {
                "elevation_in_m": 118.0,
                "endtime": UTCDateTime(2599, 12, 31, 23, 59, 59),
                "latitude": 32.7683,
                "local_depth_in_m": 0.0,
                "longitude": -113.7667,
                "starttime": UTCDateTime(2011, 12, 1, 0, 0),
            }
        ],
    }

    assert sorted(
        data_set.waveforms.TA_POKR.channel_coordinates.keys()
    ) == sorted(
        [
            "TA.POKR.01.BHZ",
            "TA.POKR..BHE",
            "TA.POKR..BHZ",
            "TA.POKR..BHN",
            "TA.POKR.01.BHN",
            "TA.POKR.01.BHE",
        ]
    )

    # Add an inventory with no channel level => thus no channel level
    # coordinates.
    inv = obspy.read_inventory()
    for net in inv:
        for sta in net:
            sta.channels = []
    data_set.add_stationxml(inv)

    with pytest.raises(ASDFValueError):
        data_set.waveforms.BW_RJOB.channel_coordinates


def test_extract_all_coordinates(example_data_set):
    """
    Tests the extraction of all coordinates.
    """
    data_set = ASDFDataSet(example_data_set.filename)

    assert data_set.get_all_coordinates() == {
        "AE.113A": {
            "latitude": 32.7683,
            "longitude": -113.7667,
            "elevation_in_m": 118.0,
        },
        "TA.POKR": {
            "latitude": 65.1171,
            "longitude": -147.4335,
            "elevation_in_m": 501.0,
        },
    }


def test_trying_to_add_provenance_record_with_invalid_name_fails(tmpdir):
    """
    The name must be valid according to a particular regular expression.
    """
    filename = os.path.join(data_dir, "example_schematic_processing_chain.xml")
    doc = prov.read(filename, format="xml")

    # Invalid name in 1.0.2.
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    with ASDFDataSet(asdf_filename, format_version="1.0.2") as ds:
        with pytest.raises(ASDFValueError) as err:
            ds.add_provenance_document(doc, name="A.b-c")

    assert err.value.args[0] == (
        "Name 'A.b-c' is invalid. It must validate against the regular "
        "expression '^[0-9a-z][0-9a-z_]*[0-9a-z]$' in ASDF version '1.0.2'."
    )

    # Same name is valid in >= 1.0.3
    asdf_filename_2 = os.path.join(tmpdir.strpath, "test_2.h5")
    with ASDFDataSet(asdf_filename_2) as ds:
        ds.add_provenance_document(doc, name="A.b-c")
    # Can of course also be read again.
    with ASDFDataSet(asdf_filename_2) as ds:
        assert ds.provenance["A.b-c"] == doc

    # Non-ASCII chars are still invalid in >= 1.0.3
    asdf_filename_3 = os.path.join(tmpdir.strpath, "test_3.h5")
    with ASDFDataSet(asdf_filename_3) as ds:
        with pytest.raises(ASDFValueError) as err:
            ds.add_provenance_document(doc, name="A.b-cäöü")

    assert err.value.args[0] == (
        "Name 'A.b-cäöü' is invalid. It must validate against the regular "
        r"expression '^[ -~]+$' in ASDF version '1.0.3'."
    )


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

    data_set.add_auxiliary_data(
        data=data, data_type=data_type, path=path, parameters=parameters
    )
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

    data_set.add_auxiliary_data(
        data=data, data_type=data_type, path=path, parameters=parameters
    )
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

    data_set.add_auxiliary_data(
        data=data, data_type=data_type, path=path, parameters=parameters
    )
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
    # Cannot start with a slash.
    data_type = "/2DRandomArray"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    try:
        with pytest.raises(ASDFValueError) as err:
            data_set.add_auxiliary_data(
                data=data,
                data_type=data_type,
                path=path,
                parameters=parameters,
            )

        assert err.value.args[0] == (
            "Data type name '/2DRandomArray' is invalid. It must validate "
            "against the regular expression "
            r"'^[a-zA-Z0-9-_\.!#$%&*+,:;<=>\?@\^~]+$' in ASDF version '1.0.3'."
        )
    finally:
        data_set.__del__()


def test_more_lenient_auxiliary_data_type_names_in_1_0_3(tmpdir):
    """
    ASDF v1.0.3 allows more flexibility in the data types names. Check that
    here.
    """
    data = np.random.random((10, 10))
    # Dots are not valid in version <= 1.0.2
    data_type = "2D.RandomArray"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    with ASDFDataSet(asdf_filename, format_version="1.0.2") as ds:
        with pytest.raises(ASDFValueError) as err:
            ds.add_auxiliary_data(
                data=data,
                data_type=data_type,
                path=path,
                parameters=parameters,
            )

        assert err.value.args[0] == (
            "Data type name '2D.RandomArray' is invalid. It must validate "
            "against the regular expression "
            r"'^[a-zA-Z0-9][a-zA-Z0-9_]*[a-zA-Z0-9]$' in ASDF version "
            "'1.0.2'."
        )

    # But works fine with ASDF >= 1.0.3
    asdf_filename_2 = os.path.join(tmpdir.strpath, "test_2.h5")
    with ASDFDataSet(asdf_filename_2) as ds:
        ds.add_auxiliary_data(
            data=data, data_type=data_type, path=path, parameters=parameters
        )
    with ASDFDataSet(asdf_filename_2) as ds:
        np.testing.assert_equal(
            ds.auxiliary_data["2D.RandomArray"].test_data.data[:], data
        )
        assert (
            ds.auxiliary_data["2D.RandomArray"].test_data.parameters
            == parameters
        )


def test_reading_and_writing_auxiliary_data_with_provenance_id(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    data = np.random.random((10, 10))
    # The data must NOT start with a number.
    data_type = "RandomArrays"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}
    provenance_id = "{http://example.org}test"

    data_set.add_auxiliary_data(
        data=data,
        data_type=data_type,
        path=path,
        parameters=parameters,
        provenance_id=provenance_id,
    )
    data_set.__del__()
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)
    assert (
        new_data_set.auxiliary_data.RandomArrays.test_data.provenance_id
        == provenance_id
    )


def test_str_method_of_aux_data(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    # With provenance id.
    data = np.random.random((10, 10))
    # The data must NOT start with a number.
    data_type = "RandomArrays"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}
    provenance_id = "{http://example.org}test"

    data_set.add_auxiliary_data(
        data=data,
        data_type=data_type,
        path=path,
        parameters=parameters,
        provenance_id=provenance_id,
    )
    assert str(data_set.auxiliary_data.RandomArrays.test_data) == (
        "Auxiliary Data of Type 'RandomArrays'\n"
        "\tPath: 'test_data'\n"
        "\tProvenance ID: '{http://example.org}test'\n"
        "\tData shape: '(10, 10)', dtype: 'float64'\n"
        "\tParameters:\n"
        "\t\ta: 1\n"
        "\t\tb: 2.0\n"
        "\t\te: hallo"
    )

    # Without.
    data = np.random.random((10, 10))
    # The data must NOT start with a number.
    data_type = "RandomArrays"
    path = "test_data_2"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(
        data=data, data_type=data_type, path=path, parameters=parameters
    )
    assert str(data_set.auxiliary_data.RandomArrays.test_data_2) == (
        "Auxiliary Data of Type 'RandomArrays'\n"
        "\tPath: 'test_data_2'\n"
        "\tData shape: '(10, 10)', dtype: 'float64'\n"
        "\tParameters:\n"
        "\t\ta: 1\n"
        "\t\tb: 2.0\n"
        "\t\te: hallo"
    )

    # Nested structure.
    data_set.add_auxiliary_data(
        data=data,
        data_type=data_type,
        path="some/deeper/path/test_data",
        parameters=parameters,
    )

    assert str(
        data_set.auxiliary_data.RandomArrays.some.deeper.path.test_data
    ) == (
        "Auxiliary Data of Type 'RandomArrays'\n"
        "\tPath: 'some/deeper/path/test_data'\n"
        "\tData shape: '(10, 10)', dtype: 'float64'\n"
        "\tParameters:\n"
        "\t\ta: 1\n"
        "\t\tb: 2.0\n"
        "\t\te: hallo"
    )


def test_adding_waveforms_with_provenance_id(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")

    data_set = ASDFDataSet(asdf_filename)
    for filename in glob.glob(os.path.join(data_path, "*.mseed")):
        data_set.add_waveforms(
            filename,
            tag="raw_recording",
            provenance_id="{http://example.org}test",
        )

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
    data_type = "RandomArrays"
    path = "(ABC)"

    with pytest.raises(ASDFValueError) as err:
        data_set.add_auxiliary_data(
            data=data, data_type=data_type, path=path, parameters={}
        )

    assert err.value.args[0] == (
        "Path part name '(ABC)' is invalid. It must validate against the "
        r"regular expression '^[a-zA-Z0-9-_\.!#$%&*+,:;<=>\?@\^~]+$' in ASDF "
        "version '1.0.3'."
    )

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
        test_filename, path="test_file", parameters={"1": 1}
    )

    data_set.__del__()
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)
    # Extraction works the same as always, but now has a special attribute,
    # that returns the data as a BytesIO.
    aux_data = new_data_set.auxiliary_data.Files.test_file
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

    filename = os.path.join(data_dir, "example_schematic_processing_chain.xml")

    # Add it as a document.
    doc = prov.read(filename, format="xml")
    data_set.add_provenance_document(doc, name="test_provenance")

    assert data_set.provenance.list() == ["test_provenance"]


def test_provenance_dicionary_behaviour(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    filename = os.path.join(data_dir, "example_schematic_processing_chain.xml")

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
        "Data set contains no auxiliary data."
    )

    data = np.random.random((10, 10))
    data_type = "RandomArrays"
    path = "test_data_1"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(
        data=data, data_type=data_type, path=path, parameters=parameters
    )

    data = np.random.random((10, 10))
    data_type = "RandomArrays"
    path = "test_data_2"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(
        data=data, data_type=data_type, path=path, parameters=parameters
    )

    data = np.random.random((10, 10))
    data_type = "SomethingElse"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(
        data=data, data_type=data_type, path=path, parameters=parameters
    )

    # Add a nested one.
    data_set.add_auxiliary_data(
        data=data,
        data_type=data_type,
        path="some/deep/path",
        parameters=parameters,
    )

    assert str(data_set.auxiliary_data) == (
        "Data set contains the following auxiliary data types:\n"
        "\tRandomArrays (2 item(s))\n"
        "\tSomethingElse (2 item(s))"
    )

    assert str(data_set.auxiliary_data.RandomArrays) == (
        "2 auxiliary data item(s) of type 'RandomArrays' available:\n"
        "\ttest_data_1\n"
        "\ttest_data_2"
    )

    assert str(data_set.auxiliary_data.SomethingElse) == (
        "1 auxiliary data sub group(s) of type 'SomethingElse' available:\n"
        "\tsome\n"
        "1 auxiliary data item(s) of type 'SomethingElse' available:\n"
        "\ttest_data"
    )

    assert str(data_set.auxiliary_data.SomethingElse.some) == (
        "1 auxiliary data sub group(s) of type 'SomethingElse/some' "
        "available:\n"
        "\tdeep"
    )

    assert str(data_set.auxiliary_data.SomethingElse.some.deep) == (
        "1 auxiliary data item(s) of type 'SomethingElse/some/deep' "
        "available:\n"
        "\tpath"
    )


def test_item_access_of_auxiliary_data(tmpdir):
    """
    Make sure all auxiliary data types, and the data itsself can be accessed
    via dictionary like accesses.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    assert str(data_set.auxiliary_data) == (
        "Data set contains no auxiliary data."
    )

    data = np.random.random((10, 10))
    data_type = "RandomArrays"
    path = "test_data_1"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(
        data=data, data_type=data_type, path=path, parameters=parameters
    )

    assert (
        data_set.auxiliary_data["RandomArrays"]["test_data_1"].path
        == data_set.auxiliary_data.RandomArrays.test_data_1.path
    )

    # Test __contains__.
    assert "RandomArrays" in data_set.auxiliary_data
    assert "SomethingElse" not in data_set.auxiliary_data

    # Test iteration over an auxiliary data group.
    expected = [("RandomArrays", "test_data_1")]
    actual = []
    for item in data_set.auxiliary_data.RandomArrays:
        actual.append((item.data_type, item.path))

    assert expected == actual


def test_item_access_of_waveforms(example_data_set):
    """
    Tests that waveforms and stations can be accessed with item access.
    """
    data_set = ASDFDataSet(example_data_set.filename)

    assert (
        data_set.waveforms["AE_113A"]["raw_recording"]
        == data_set.waveforms.AE_113A.raw_recording
        == data_set.waveforms["AE.113A"].raw_recording
        == data_set.waveforms.AE_113A["raw_recording"]
    )

    assert (
        data_set.waveforms["AE_113A"]["StationXML"]
        == data_set.waveforms.AE_113A.StationXML
        == data_set.waveforms["AE.113A"].StationXML
        == data_set.waveforms.AE_113A["StationXML"]
    )


def test_list_method_of_waveform_accessor(example_data_set):
    data_set = ASDFDataSet(example_data_set.filename)

    assert data_set.waveforms.list() == ["AE.113A", "TA.POKR"]


def test_detailed_waveform_access(example_data_set):
    data_set = ASDFDataSet(example_data_set.filename)
    st = data_set.waveforms.AE_113A

    assert st.get_waveform_tags() == ["raw_recording"]
    assert st.list() == [
        "AE.113A..BHE__2013-05-24T05:40:00__2013-05-24T06:50:00"
        "__raw_recording",
        "AE.113A..BHN__2013-05-24T05:40:00__2013-05-24T06:50:00"
        "__raw_recording",
        "AE.113A..BHZ__2013-05-24T05:40:00__2013-05-24T06:50:00"
        "__raw_recording",
        "StationXML",
    ]

    assert (
        st[
            "AE.113A..BHZ__2013-05-24T05:40:00__2013-05-24T06:50:00"
            "__raw_recording"
        ][0]
        == st.raw_recording.select(channel="BHZ")[0]
    )


def test_get_provenance_document_for_id(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    filename = os.path.join(data_dir, "example_schematic_processing_chain.xml")

    doc = prov.read(filename)
    data_set.provenance["test_provenance"] = doc

    assert data_set.provenance.get_provenance_document_for_id(
        "{http://seisprov.org/seis_prov/0.1/#}sp002_dt_f87sf7sf78"
    ) == {"name": "test_provenance", "document": doc}

    assert data_set.provenance.get_provenance_document_for_id(
        "{http://seisprov.org/seis_prov/0.1/#}sp004_lp_f87sf7sf78"
    ) == {"name": "test_provenance", "document": doc}

    # Id not found.
    with pytest.raises(ASDFValueError) as err:
        data_set.provenance.get_provenance_document_for_id(
            "{http://seisprov.org/seis_prov/0.1/#}bogus_id"
        )

    assert err.value.args[0] == (
        "Document containing id "
        "'{http://seisprov.org/seis_prov/0.1/#}bogus_id'"
        " not found in the data set."
    )

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

    f = h5py.File(asdf_filename, mode="a")
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
    ds.add_waveforms(
        tr,
        tag="random_a",
        event_id=event_id,
        origin_id=origin_id,
        focal_mechanism_id=focmec_id,
        magnitude_id=magnitude_id,
    )
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
        "test",
        "random",
        obspy.core.event.ResourceIdentifier(
            "smi:service.iris.edu/fdsnws/event/1/query?random_things"
        ),
    ]
    for r_id in random_ids:
        assert list(ds.ifilter(ds.q.event == r_id)) == []
        assert list(ds.ifilter(ds.q.magnitude == r_id)) == []
        assert list(ds.ifilter(ds.q.origin == r_id)) == []
        assert list(ds.ifilter(ds.q.focal_mechanism == r_id)) == []

    # Event as a resource identifier and as a string, and with others equal to
    # None.
    result = [_i._station_name for _i in ds.ifilter(ds.q.event == event_id)]
    assert result == ["AA.AA", "BB.BB"]
    result = [
        _i._station_name for _i in ds.ifilter(ds.q.event == str(event_id))
    ]
    assert result == ["AA.AA", "BB.BB"]
    result = [
        _i._station_name
        for _i in ds.ifilter(
            ds.q.event == str(event_id),
            ds.q.magnitude == None,
            ds.q.focal_mechanism == None,
        )
    ]  # NOQA
    assert result == ["BB.BB"]

    # Origin as a resource identifier and as a string, and with others equal to
    # None.
    result = [_i._station_name for _i in ds.ifilter(ds.q.origin == origin_id)]
    assert result == ["AA.AA", "CC.CC"]
    result = [
        _i._station_name for _i in ds.ifilter(ds.q.origin == str(origin_id))
    ]
    assert result == ["AA.AA", "CC.CC"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.origin == str(origin_id), ds.q.event == None)
    ]  # NOQA
    assert result == ["CC.CC"]

    # Magnitude as a resource identifier and as a string, and with others equal
    # to None.
    result = [
        _i._station_name for _i in ds.ifilter(ds.q.magnitude == magnitude_id)
    ]
    assert result == ["AA.AA", "DD.DD"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.magnitude == str(magnitude_id))
    ]
    assert result == ["AA.AA", "DD.DD"]
    result = [
        _i._station_name
        for _i in ds.ifilter(
            ds.q.magnitude == str(magnitude_id), ds.q.origin == None
        )
    ]  # NOQA
    assert result == ["DD.DD"]

    # focmec as a resource identifier and as a string, and with others equal to
    # None.
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.focal_mechanism == focmec_id)
    ]
    assert result == ["AA.AA", "EE.EE"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.focal_mechanism == str(focmec_id))
    ]
    assert result == ["AA.AA", "EE.EE"]
    result = [
        _i._station_name
        for _i in ds.ifilter(
            ds.q.focal_mechanism == str(focmec_id), ds.q.origin == None
        )
    ]  # NOQA
    assert result == ["EE.EE"]

    # No existing ids are treated like empty ids.
    result = [
        _i._station_name
        for _i in ds.ifilter(
            ds.q.event == None,
            ds.q.magnitude == None,
            ds.q.origin == None,
            ds.q.focal_mechanism == None,
        )
    ]  # NOQA
    assert result == ["FF.FF"]

    result = [
        _i._station_name
        for _i in ds.ifilter(
            ds.q.event != None,
            ds.q.magnitude != None,
            ds.q.origin != None,
            ds.q.focal_mechanism != None,
        )
    ]  # NOQA
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
    assert collect_ids(
        ds.ifilter(
            ds.q.network == "TA",
            ds.q.station == "POKR",
            ds.q.location == "",
            ds.q.channel == "BHZ",
        )
    ) == {"TA.POKR..BHZ"}

    # Get nothing with a not existing location code.
    assert not collect_ids(ds.ifilter(ds.q.location == "99"))

    # Get the three 100 Hz traces.
    assert collect_ids(ds.ifilter(ds.q.sampling_rate >= 100.0)) == {
        "BW.RJOB..EHE",
        "BW.RJOB..EHN",
        "BW.RJOB..EHZ",
    }

    # Get the "random" tagged traces in different ways.
    assert collect_ids(ds.ifilter(ds.q.tag == "random")) == {
        "BW.RJOB..EHE",
        "BW.RJOB..EHN",
        "BW.RJOB..EHZ",
    }
    assert collect_ids(ds.ifilter(ds.q.tag == ["random"])) == {
        "BW.RJOB..EHE",
        "BW.RJOB..EHN",
        "BW.RJOB..EHZ",
    }
    assert collect_ids(ds.ifilter(ds.q.tag == ["dummy", "r*m"])) == {
        "BW.RJOB..EHE",
        "BW.RJOB..EHN",
        "BW.RJOB..EHZ",
    }

    # Geographic constraints. Will never return the BW channels as they have
    # no coordinate information.
    assert collect_ids(
        ds.ifilter(ds.q.latitude >= 30.0, ds.q.latitude <= 40.0)
    ) == {"AE.113A..BHE", "AE.113A..BHN", "AE.113A..BHZ"}
    assert collect_ids(
        ds.ifilter(ds.q.longitude >= -120.0, ds.q.longitude <= -110.0)
    ) == {"AE.113A..BHE", "AE.113A..BHN", "AE.113A..BHZ"}
    assert collect_ids(ds.ifilter(ds.q.elevation_in_m < 200.0)) == {
        "AE.113A..BHE",
        "AE.113A..BHN",
        "AE.113A..BHZ",
    }

    # Make sure coordinates exist.
    assert collect_ids(ds.ifilter(ds.q.latitude != None)) == {  # NOQA
        "AE.113A..BHE",
        "AE.113A..BHN",
        "AE.113A..BHZ",
        "TA.POKR..BHE",
        "TA.POKR..BHZ",
        "TA.POKR..BHN",
    }
    # Opposite query
    assert collect_ids(ds.ifilter(ds.q.latitude == None)) == {  # NOQA
        "BW.RJOB..EHE",
        "BW.RJOB..EHN",
        "BW.RJOB..EHZ",
    }

    # Temporal constraints.
    assert collect_ids(ds.ifilter(ds.q.starttime <= "2010-01-01")) == {
        "BW.RJOB..EHE",
        "BW.RJOB..EHN",
        "BW.RJOB..EHZ",
    }

    # Exact endtime
    assert collect_ids(
        ds.ifilter(
            ds.q.endtime <= obspy.UTCDateTime("2009-08-24T00:20:32.990000Z")
        )
    ) == {"BW.RJOB..EHE", "BW.RJOB..EHN", "BW.RJOB..EHZ"}
    assert (
        collect_ids(
            ds.ifilter(
                ds.q.endtime
                <= obspy.UTCDateTime("2009-08-24T00:20:32.990000Z") - 1
            )
        )
        == set()
    )
    assert (
        collect_ids(
            ds.ifilter(
                ds.q.endtime < obspy.UTCDateTime("2009-08-24T00:20:32.990000Z")
            )
        )
        == set()
    )

    # Number of samples.
    assert collect_ids(ds.ifilter(ds.q.npts > 1000, ds.q.npts < 5000)) == {
        "BW.RJOB..EHE",
        "BW.RJOB..EHN",
        "BW.RJOB..EHZ",
    }

    # All vertical channels.
    assert collect_ids(ds.ifilter(ds.q.channel == "*Z")) == {
        "BW.RJOB..EHZ",
        "TA.POKR..BHZ",
        "AE.113A..BHZ",
    }

    # Many keys cannot be None, as their value must always be given.
    for key in [
        "network",
        "station",
        "location",
        "channel",
        "tag",
        "starttime",
        "endtime",
        "sampling_rate",
        "npts",
    ]:
        with pytest.raises(TypeError):
            ds.ifilter(getattr(ds.q, key) == None)  # NOQA


def test_saving_trace_labels(tmpdir):
    """
    Tests that the labels can be saved and retrieved automatically.
    """
    data_path = os.path.join(data_dir, "small_sample_data_set")
    filename = os.path.join(tmpdir.strpath, "example.h5")

    data_set = ASDFDataSet(filename)
    waveform = obspy.read(os.path.join(data_path, "TA.*.mseed")).sort()
    data_set.add_waveforms(
        waveform,
        "raw_recording",
        labels=["hello", "what", "is", "going", "on?"],
    )

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
    labels = ["?⸘‽", "^§#⁇❦"]
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
        waveform,
        "raw_recording",
        labels=["hello", "what", "is", "going", "on?"],
    )

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
    data_set.add_waveforms(waveform, "raw_recording", labels=["?⸘‽", "^§#⁇❦"])

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
    labels_c = ["?⸘‽", "^§#⁇❦"]
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
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.labels == ["what", "?⸘‽", "single_label"])
    ]
    assert result == ["BB.BB", "CC.CC", "DD.DD"]

    # No labels.
    result = [
        _i._station_name for _i in ds.ifilter(ds.q.labels == None)
    ]  # NOQA
    assert result == ["AA.AA"]

    # Any label.
    result = [
        _i._station_name for _i in ds.ifilter(ds.q.labels != None)
    ]  # NOQA
    assert result == ["BB.BB", "CC.CC", "DD.DD"]

    # Unicode wildcard.
    result = [_i._station_name for _i in ds.ifilter(ds.q.labels == "^§#⁇*")]
    assert result == ["CC.CC"]

    # BB and DD.
    result = [
        _i._station_name for _i in ds.ifilter(ds.q.labels == ["wha?", "sin*"])
    ]
    assert result == ["BB.BB", "DD.DD"]

    # CC
    result = [_i._station_name for _i in ds.ifilter(ds.q.labels == "^§#⁇*")]
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
    assert (
        e.value.args[0]
        == "'WaveformAccessor' object has no attribute 'StationXML'"
    )

    with pytest.raises(KeyError) as e:
        ds.waveforms.BW_RJOB["StationXML"]
    assert (
        e.value.args[0]
        == "'WaveformAccessor' object has no attribute 'StationXML'"
    )

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

    data_set.add_auxiliary_data(
        data=data, data_type=data_type, path=path, parameters=parameters
    )
    del data_set

    new_data_set = ASDFDataSet(asdf_filename)

    aux_data = (
        new_data_set.auxiliary_data.RandomArrays.some.deeper.path.test_data
    )
    np.testing.assert_equal(data, aux_data.data)
    aux_data.data_type == data_type
    aux_data.path == path
    aux_data.parameters == parameters


def test_usage_as_context_manager(tmpdir):
    """
    Tests the usage of a pyasdf as a context manager.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")

    with ASDFDataSet(asdf_filename) as ds:
        ds.add_waveforms(obspy.read(), tag="hello")

    with ASDFDataSet(asdf_filename) as ds:
        assert ds.waveforms.list() == ["BW.RJOB"]
        assert ds.waveforms["BW.RJOB"]["hello"]

    # Writing does not work anymore as the file has been closed.
    with pytest.raises(Exception):
        # Different tag.
        ds.add_waveforms(obspy.read(), tag="random")


def test_validate_function(tmpdir, capsys):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")

    # Add only a waveform
    with ASDFDataSet(asdf_filename) as ds:
        ds.add_waveforms(obspy.read(), tag="random")

        ds.validate()

    out, _ = capsys.readouterr()

    assert out == (
        "No station information available for station 'BW.RJOB'\n"
        "\n"
        "Checked 1 station(s):\n"
        "\t1 station(s) have no available station information\n"
        "\t0 station(s) with no waveforms\n"
        "\t0 good station(s)\n"
    )

    os.remove(asdf_filename)

    # Add only a StationXML file containing information about 3 stations.
    with ASDFDataSet(asdf_filename) as ds:
        ds.add_stationxml(obspy.read_inventory())

        ds.validate()

    out, _ = capsys.readouterr()

    assert out == (
        "Station with no waveforms: 'BW.RJOB'\n"
        "Station with no waveforms: 'GR.FUR'\n"
        "Station with no waveforms: 'GR.WET'\n"
        "\n"
        "Checked 3 station(s):\n"
        "\t0 station(s) have no available station information\n"
        "\t3 station(s) with no waveforms\n"
        "\t0 good station(s)\n"
    )

    os.remove(asdf_filename)

    # Add both.
    with ASDFDataSet(asdf_filename) as ds:
        ds.add_waveforms(obspy.read(), tag="random")
        ds.add_stationxml(obspy.read_inventory())

        ds.validate()

    out, _ = capsys.readouterr()

    assert out == (
        "Station with no waveforms: 'GR.FUR'\n"
        "Station with no waveforms: 'GR.WET'\n"
        "\n"
        "Checked 3 station(s):\n"
        "\t0 station(s) have no available station information\n"
        "\t2 station(s) with no waveforms\n"
        "\t1 good station(s)\n"
    )


def test_deletions_of_dataset_object(tmpdir):
    """
    Only the events on the dataset object can be deleted.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_path = os.path.join(data_dir, "small_sample_data_set")

    ds = ASDFDataSet(asdf_filename)

    # Add event.
    ds.add_quakeml(os.path.join(data_path, "quake.xml"))
    assert len(ds.events) == 1

    # Delete
    del ds.events
    assert len(ds.events) == 0

    # The same can be added again.
    ds.add_quakeml(os.path.join(data_path, "quake.xml"))
    assert len(ds.events) == 1

    # Non-existing keys raise AttributeError
    with pytest.raises(AttributeError) as excinfo:
        del ds.random

    assert excinfo.value.args[0] == (
        "'ASDFDataSet' object has no attribute 'random'"
    )
    del excinfo

    # Existing but other attributes cannot be deleted.
    with pytest.raises(AttributeError) as excinfo:
        del ds.waveforms
    assert excinfo.value.args[0] == (
        "Attribute 'waveforms' cannot be " "deleted."
    )
    del excinfo


def test_deletion_of_whole_waveform_groups(tmpdir):
    """
    Tests the deletion of whole waveform groups.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")

    ds = ASDFDataSet(asdf_filename)
    ds.add_waveforms(obspy.read(), tag="random")

    assert [_i._station_name for _i in ds.waveforms] == ["BW.RJOB"]

    # Deletion over attribute access.
    del ds.waveforms.BW_RJOB
    assert [_i._station_name for _i in ds.waveforms] == []

    # An over item access.
    ds.add_waveforms(obspy.read(), tag="random")
    assert [_i._station_name for _i in ds.waveforms] == ["BW.RJOB"]
    del ds.waveforms["BW.RJOB"]
    assert [_i._station_name for _i in ds.waveforms] == []

    # Non-existing keys raise AttributeError
    with pytest.raises(AttributeError) as excinfo:
        del ds.waveforms.random
    assert excinfo.value.args[0] == ("Attribute 'random' not found.")
    del excinfo

    # Existing but other attributes cannot be deleted.
    with pytest.raises(AttributeError) as excinfo:
        del ds.waveforms.__init__
    assert excinfo.value.args[0] == (
        "Attribute '__init__' cannot be " "deleted."
    )
    del excinfo

    # Same thing, but this time with item access.
    with pytest.raises(KeyError) as excinfo:
        del ds.waveforms["random"]
    assert excinfo.value.args[0] == ("Attribute 'random' not found.")
    del excinfo


def test_deletion_of_single_waveforms(tmpdir):
    """
    Tests the deletion of single waveform files.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")

    ds = ASDFDataSet(asdf_filename)

    # Deleting all waveforms with a certain tag over key access.
    ds.add_waveforms(obspy.read(), tag="random")
    assert ds.waveforms["BW.RJOB"]["random"]
    del ds.waveforms["BW.RJOB"]["random"]
    with pytest.raises(KeyError):
        ds.waveforms["BW.RJOB"]["random"]

    # Deleting all waveforms with a certain tag over attribute access.
    ds.add_waveforms(obspy.read(), tag="random")
    assert ds.waveforms["BW.RJOB"].random
    del ds.waveforms["BW.RJOB"]["random"]
    with pytest.raises(WaveformNotInFileException):
        ds.waveforms["BW.RJOB"].random

    # Deleting StationXML over key access.
    ds.add_stationxml(obspy.read_inventory())
    assert ds.waveforms["BW.RJOB"]["StationXML"]
    del ds.waveforms["BW.RJOB"]["StationXML"]
    with pytest.raises(KeyError):
        ds.waveforms["BW.RJOB"]["StationXML"]

    # Deleting StationXML over attribute access.
    ds.add_stationxml(obspy.read_inventory())
    assert ds.waveforms["BW.RJOB"].StationXML
    del ds.waveforms["BW.RJOB"].StationXML
    with pytest.raises(AttributeError):
        ds.waveforms["BW.RJOB"].StationXML

    # Deleting a single waveform trace.
    ds.add_waveforms(obspy.read(), tag="random")
    assert len(ds.waveforms["BW.RJOB"]["random"]) == 3
    assert ds.waveforms["BW.RJOB"][
        "BW.RJOB..EHE__2009-08-24T00:20:03__2009-08-24T00:20:32__random"
    ]
    del ds.waveforms["BW.RJOB"][
        "BW.RJOB..EHE__2009-08-24T00:20:03__2009-08-24T00:20:32__random"
    ]
    with pytest.raises(KeyError):
        ds.waveforms["BW.RJOB"][
            "BW.RJOB..EHE__2009-08-24T00:20:03__2009-08-24T00:20:32__random"
        ]

    # Other waveforms with the same tag are still around. Just one less than
    # before.
    assert len(ds.waveforms["BW.RJOB"]["random"]) == 2


def test_deleting_provenance_records(tmpdir):
    """
    Tests the deletion of provenance records.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    ds = ASDFDataSet(asdf_filename)

    filename = os.path.join(data_dir, "example_schematic_processing_chain.xml")

    doc = prov.read(filename, format="xml")

    # Delete via attribute access.
    ds.add_provenance_document(doc, name="test_provenance")
    assert ds.provenance.test_provenance
    del ds.provenance.test_provenance
    with pytest.raises(AttributeError):
        ds.provenance.test_provenance

    # Delete via key access.
    ds.add_provenance_document(doc, name="test_provenance")
    assert ds.provenance["test_provenance"]
    del ds.provenance["test_provenance"]
    with pytest.raises(KeyError):
        ds.provenance["test_provenance"]


def test_deleting_auxiliary_data(tmpdir):
    """
    Tests deleting auxiliary data.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    ds = ASDFDataSet(asdf_filename)

    def _add_aux_data(data_type, path):
        # Define some auxiliary data and add it.
        data = np.random.random(100)
        parameters = {"a": 1, "b": 2.0, "e": "hallo"}
        ds.add_auxiliary_data(
            data=data, data_type=data_type, path=path, parameters=parameters
        )

    # Delete the whole group using attribute access.
    _add_aux_data("RandomArrays", "test_data")
    assert ds.auxiliary_data.RandomArrays
    del ds.auxiliary_data.RandomArrays
    with pytest.raises(AttributeError):
        ds.auxiliary_data.RandomArrays

    # Same with key access.
    _add_aux_data("RandomArrays", "test_data")
    assert ds.auxiliary_data["RandomArrays"]
    del ds.auxiliary_data["RandomArrays"]
    with pytest.raises(KeyError):
        ds.auxiliary_data["RandomArrays"]

    # Now try with a single piece of auxiliary data.
    _add_aux_data("RandomArrays", "test_data")
    assert ds.auxiliary_data.RandomArrays
    assert ds.auxiliary_data.RandomArrays.test_data
    del ds.auxiliary_data.RandomArrays.test_data
    # This still works.
    assert len(ds.auxiliary_data.RandomArrays) == 0
    # But the actual data set does not longer exist.
    with pytest.raises(AttributeError):
        ds.auxiliary_data.RandomArrays.test_data

    # Same with key access.
    _add_aux_data("RandomArrays", "test_data")
    assert ds.auxiliary_data["RandomArrays"]
    assert ds.auxiliary_data["RandomArrays"]["test_data"]
    del ds.auxiliary_data["RandomArrays"]["test_data"]
    assert len(ds.auxiliary_data["RandomArrays"]) == 0
    with pytest.raises(KeyError):
        ds.auxiliary_data["RandomArrays"]["test_data"]

    # All previous tests again but with nested data.
    _add_aux_data("RandomArrays", "some/nested/path")
    assert ds.auxiliary_data.RandomArrays
    del ds.auxiliary_data.RandomArrays
    with pytest.raises(AttributeError):
        ds.auxiliary_data.RandomArrays

    _add_aux_data("RandomArrays", "some/nested/path")
    assert ds.auxiliary_data["RandomArrays"]
    del ds.auxiliary_data["RandomArrays"]
    with pytest.raises(KeyError):
        ds.auxiliary_data["RandomArrays"]

    _add_aux_data("RandomArrays", "some/nested/path")
    assert ds.auxiliary_data.RandomArrays
    assert ds.auxiliary_data.RandomArrays.some.nested.path
    del ds.auxiliary_data.RandomArrays.some.nested.path
    # Don't delete the groups.
    assert len(ds.auxiliary_data.RandomArrays.some.nested) == 0
    with pytest.raises(AttributeError):
        ds.auxiliary_data.RandomArrays.some.nested.path

    # Try deleting something that does not exist.
    with pytest.raises(AttributeError):
        del ds.auxiliary_data.i_do_not_exist
    with pytest.raises(KeyError):
        del ds.auxiliary_data["i_do_not_exist"]

    _add_aux_data("RandomArrays", "some/nested/path")
    assert ds.auxiliary_data["RandomArrays"]
    assert ds.auxiliary_data["RandomArrays"]["some"]["nested"]["path"]
    del ds.auxiliary_data["RandomArrays"]["some"]["nested"]["path"]
    assert len(ds.auxiliary_data["RandomArrays"]["some"]["nested"]) == 0
    with pytest.raises(KeyError):
        ds.auxiliary_data["RandomArrays"]["some"]["nested"]["path"]

    # Try to delete a "deeper" group.
    assert len(ds.auxiliary_data["RandomArrays"]["some"]["nested"]) == 0
    assert len(ds.auxiliary_data["RandomArrays"]["some"]) == 1
    del ds.auxiliary_data["RandomArrays"]["some"]["nested"]
    assert len(ds.auxiliary_data["RandomArrays"]["some"]) == 0
    with pytest.raises(KeyError):
        ds.auxiliary_data["RandomArrays"]["some"]["nested"]

    # Again but with attribute access.
    _add_aux_data("RandomArrays", "some/nested/path")
    assert ds.auxiliary_data.RandomArrays
    assert ds.auxiliary_data.RandomArrays.some.nested.path
    del ds.auxiliary_data.RandomArrays.some.nested.path

    assert len(ds.auxiliary_data.RandomArrays.some.nested) == 0
    assert len(ds.auxiliary_data.RandomArrays.some) == 1
    del ds.auxiliary_data.RandomArrays.some.nested
    assert len(ds.auxiliary_data.RandomArrays.some) == 0
    with pytest.raises(AttributeError):
        ds.auxiliary_data.RandomArrays.some.nested


def test_using_invalid_tag_names(tmpdir):
    """
    Tags must only contain lower case letters, numbers, or underscores.
    """
    filename = os.path.join(tmpdir.strpath, "example.h5")

    data_set = ASDFDataSet(filename)
    st = obspy.read()

    # Empty tag
    with pytest.raises(ValueError):
        data_set.add_waveforms(st, tag="")

    # Uppercase letters
    with pytest.raises(ValueError):
        data_set.add_waveforms(st, tag="HELLO")
    with pytest.raises(ValueError):
        data_set.add_waveforms(st, tag="ellO")
    with pytest.raises(ValueError):
        data_set.add_waveforms(st, tag="Hello")

    # Other symbols.
    with pytest.raises(ValueError) as e:
        data_set.add_waveforms(st, tag="_$$$Hello")

    assert e.value.args[0] == (
        "Invalid tag: '_$$$Hello' - Must satisfy the " "regex '^[a-z_0-9]+$'."
    )

    del e
    del data_set


def test_associating_multiple_events_origin_and_other_thingsG(tmpdir):
    """
    Each trace should be able to refer to multiple events, origins,
    magnitudes, and focal mechanisms.
    """
    filename = os.path.join(tmpdir.strpath, "example.h5")

    ds = ASDFDataSet(filename)

    cat = obspy.read_events()
    # Add random focal mechanisms.
    cat[0].focal_mechanisms.append(obspy.core.event.FocalMechanism())
    cat[1].focal_mechanisms.append(obspy.core.event.FocalMechanism())
    cat[2].focal_mechanisms.append(obspy.core.event.FocalMechanism())

    ds.add_quakeml(cat)

    event_1 = ds.events[0]
    origin_1 = event_1.origins[0]
    magnitude_1 = event_1.magnitudes[0]
    focmec_1 = event_1.focal_mechanisms[0]

    event_2 = ds.events[1]
    origin_2 = event_2.origins[0]
    magnitude_2 = event_2.magnitudes[0]
    focmec_2 = event_2.focal_mechanisms[0]

    event_3 = ds.events[2]
    origin_3 = event_3.origins[0]
    magnitude_3 = event_3.magnitudes[0]
    focmec_3 = event_3.focal_mechanisms[0]

    tr = obspy.read()[0]
    tr.stats.network = "BW"
    tr.stats.station = "RJOB"

    # Just a single one.
    ds.add_waveforms(
        tr,
        tag="random",
        event_id=event_1,
        origin_id=origin_1,
        focal_mechanism_id=focmec_1,
        magnitude_id=magnitude_1,
    )

    st = ds.waveforms["BW.RJOB"]["random"]
    assert [event_1.resource_id] == st[0].stats.asdf.event_ids
    assert [origin_1.resource_id] == st[0].stats.asdf.origin_ids
    assert [magnitude_1.resource_id] == st[0].stats.asdf.magnitude_ids
    assert [focmec_1.resource_id] == st[0].stats.asdf.focal_mechanism_ids

    # Again just a single one but passed as a list.
    ds.add_waveforms(
        tr,
        tag="random_2",
        event_id=[event_1],
        origin_id=[origin_1],
        focal_mechanism_id=[focmec_1],
        magnitude_id=[magnitude_1],
    )

    st = ds.waveforms["BW.RJOB"]["random_2"]
    assert [event_1.resource_id] == st[0].stats.asdf.event_ids
    assert [origin_1.resource_id] == st[0].stats.asdf.origin_ids
    assert [magnitude_1.resource_id] == st[0].stats.asdf.magnitude_ids
    assert [focmec_1.resource_id] == st[0].stats.asdf.focal_mechanism_ids

    # Actually doing multiple ones.
    ds.add_waveforms(
        tr,
        tag="random_3",
        event_id=[event_1, event_2, event_3],
        origin_id=[origin_1, origin_2, origin_3],
        focal_mechanism_id=[focmec_1, focmec_2, focmec_3],
        magnitude_id=[magnitude_1, magnitude_2, magnitude_3],
    )

    st = ds.waveforms["BW.RJOB"]["random_3"]
    assert [
        event_1.resource_id,
        event_2.resource_id,
        event_3.resource_id,
    ] == st[0].stats.asdf.event_ids
    assert [
        origin_1.resource_id,
        origin_2.resource_id,
        origin_3.resource_id,
    ] == st[0].stats.asdf.origin_ids
    assert [
        magnitude_1.resource_id,
        magnitude_2.resource_id,
        magnitude_3.resource_id,
    ] == st[0].stats.asdf.magnitude_ids
    assert [
        focmec_1.resource_id,
        focmec_2.resource_id,
        focmec_3.resource_id,
    ] == st[0].stats.asdf.focal_mechanism_ids


def test_multiple_event_associations_are_persistent_through_processing(tmpdir):
    """
    Processing a file with an associated event and storing it again should
    keep the association for all the possible event tags..
    """
    filename = os.path.join(tmpdir.strpath, "example.h5")

    ds = ASDFDataSet(filename)

    cat = obspy.read_events()
    # Add random focal mechanisms.
    cat[0].focal_mechanisms.append(obspy.core.event.FocalMechanism())
    cat[1].focal_mechanisms.append(obspy.core.event.FocalMechanism())
    cat[2].focal_mechanisms.append(obspy.core.event.FocalMechanism())

    ds.add_quakeml(cat)

    event_1 = ds.events[0]
    origin_1 = event_1.origins[0]
    magnitude_1 = event_1.magnitudes[0]
    focmec_1 = event_1.focal_mechanisms[0]

    event_2 = ds.events[1]
    origin_2 = event_2.origins[0]
    magnitude_2 = event_2.magnitudes[0]
    focmec_2 = event_2.focal_mechanisms[0]

    event_3 = ds.events[2]
    origin_3 = event_3.origins[0]
    magnitude_3 = event_3.magnitudes[0]
    focmec_3 = event_3.focal_mechanisms[0]

    tr = obspy.read()[0]
    tr.stats.network = "BW"
    tr.stats.station = "RJOB"

    # Actually doing multiple ones.
    ds.add_waveforms(
        tr,
        tag="random",
        event_id=[event_1, event_2, event_3],
        origin_id=[origin_1, origin_2, origin_3],
        focal_mechanism_id=[focmec_1, focmec_2, focmec_3],
        magnitude_id=[magnitude_1, magnitude_2, magnitude_3],
    )

    new_st = ds.waveforms.BW_RJOB.random
    new_st.taper(max_percentage=0.05, type="cosine")

    ds.add_waveforms(new_st, tag="processed")

    # Close and reopen.
    del ds
    ds = ASDFDataSet(filename)

    st = ds.waveforms["BW.RJOB"]["processed"]
    assert [
        event_1.resource_id,
        event_2.resource_id,
        event_3.resource_id,
    ] == st[0].stats.asdf.event_ids
    assert [
        origin_1.resource_id,
        origin_2.resource_id,
        origin_3.resource_id,
    ] == st[0].stats.asdf.origin_ids
    assert [
        magnitude_1.resource_id,
        magnitude_2.resource_id,
        magnitude_3.resource_id,
    ] == st[0].stats.asdf.magnitude_ids
    assert [
        focmec_1.resource_id,
        focmec_2.resource_id,
        focmec_3.resource_id,
    ] == st[0].stats.asdf.focal_mechanism_ids


def test_event_iteration_with_multiple_events(tmpdir):
    """
    Tests the iteration over a dataset by event attributes.
    """
    filename = os.path.join(tmpdir.strpath, "example.h5")

    ds = ASDFDataSet(filename)

    cat = obspy.read_events()
    # Add random focal mechanisms.
    cat[0].focal_mechanisms.append(obspy.core.event.FocalMechanism())
    cat[1].focal_mechanisms.append(obspy.core.event.FocalMechanism())

    ds.add_quakeml(cat)

    event_1 = ds.events[0]
    origin_1 = event_1.origins[0]
    magnitude_1 = event_1.magnitudes[0]
    focmec_1 = event_1.focal_mechanisms[0]

    event_2 = ds.events[1]
    origin_2 = event_2.origins[0]
    magnitude_2 = event_2.magnitudes[0]
    focmec_2 = event_2.focal_mechanisms[0]

    tr = obspy.read()[0]

    # Has everything, two of each.
    tr.stats.network = "AA"
    tr.stats.station = "AA"
    ds.add_waveforms(
        tr,
        tag="random_a",
        event_id=[event_1, event_2],
        origin_id=[origin_1, origin_2],
        focal_mechanism_id=[focmec_1, focmec_2],
        magnitude_id=[magnitude_1, magnitude_2],
    )
    # Has only the events.
    tr.stats.network = "BB"
    tr.stats.station = "BB"
    ds.add_waveforms(tr, tag="random_b", event_id=[event_1, event_2])

    # Only the origins.
    tr.stats.network = "CC"
    tr.stats.station = "CC"
    ds.add_waveforms(tr, tag="random_c", origin_id=[origin_1, origin_2])

    # Only the magnitude.
    tr.stats.network = "DD"
    tr.stats.station = "DD"
    ds.add_waveforms(
        tr, tag="random_d", magnitude_id=[magnitude_1, magnitude_2]
    )

    # Only the focal mechanisms.
    tr.stats.network = "EE"
    tr.stats.station = "EE"
    ds.add_waveforms(
        tr, tag="random_e", focal_mechanism_id=[focmec_1, focmec_2]
    )

    # Nothing.
    tr.stats.network = "FF"
    tr.stats.station = "FF"
    ds.add_waveforms(tr, tag="random_f")

    # Test with random ids..should all return nothing.
    random_ids = [
        "test",
        "random",
        obspy.core.event.ResourceIdentifier(
            "smi:service.iris.edu/fdsnws/event/1/query?random_things"
        ),
    ]
    for r_id in random_ids:
        assert list(ds.ifilter(ds.q.event == r_id)) == []
        assert list(ds.ifilter(ds.q.magnitude == r_id)) == []
        assert list(ds.ifilter(ds.q.origin == r_id)) == []
        assert list(ds.ifilter(ds.q.focal_mechanism == r_id)) == []

    # Both events in various realizations.
    result = [_i._station_name for _i in ds.ifilter(ds.q.event == event_1)]
    assert result == ["AA.AA", "BB.BB"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.event == event_1.resource_id)
    ]
    assert result == ["AA.AA", "BB.BB"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.event == str(event_1.resource_id))
    ]
    assert result == ["AA.AA", "BB.BB"]
    result = [
        _i._station_name
        for _i in ds.ifilter(
            ds.q.event == event_1,
            ds.q.magnitude == None,
            ds.q.focal_mechanism == None,
        )
    ]  # NOQA
    assert result == ["BB.BB"]
    result = [_i._station_name for _i in ds.ifilter(ds.q.event == event_2)]
    assert result == ["AA.AA", "BB.BB"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.event == event_2.resource_id)
    ]
    assert result == ["AA.AA", "BB.BB"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.event == str(event_2.resource_id))
    ]
    assert result == ["AA.AA", "BB.BB"]
    result = [
        _i._station_name
        for _i in ds.ifilter(
            ds.q.event == event_2,
            ds.q.magnitude == None,
            ds.q.focal_mechanism == None,
        )
    ]  # NOQA
    assert result == ["BB.BB"]

    # Origin as a resource identifier and as a string, and with others equal to
    # None.
    result = [_i._station_name for _i in ds.ifilter(ds.q.origin == origin_1)]
    assert result == ["AA.AA", "CC.CC"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.origin == origin_1.resource_id)
    ]
    assert result == ["AA.AA", "CC.CC"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.origin == str(origin_1.resource_id))
    ]
    assert result == ["AA.AA", "CC.CC"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.origin == origin_1, ds.q.event == None)
    ]  # NOQA
    assert result == ["CC.CC"]
    result = [_i._station_name for _i in ds.ifilter(ds.q.origin == origin_2)]
    assert result == ["AA.AA", "CC.CC"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.origin == origin_2.resource_id)
    ]
    assert result == ["AA.AA", "CC.CC"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.origin == str(origin_2.resource_id))
    ]
    assert result == ["AA.AA", "CC.CC"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.origin == origin_2, ds.q.event == None)
    ]  # NOQA
    assert result == ["CC.CC"]

    # Magnitude as a resource identifier and as a string, and with others
    # equal to None.
    result = [
        _i._station_name for _i in ds.ifilter(ds.q.magnitude == magnitude_1)
    ]
    assert result == ["AA.AA", "DD.DD"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.magnitude == magnitude_1.resource_id)
    ]
    assert result == ["AA.AA", "DD.DD"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.magnitude == str(magnitude_1.resource_id))
    ]
    assert result == ["AA.AA", "DD.DD"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.magnitude == magnitude_1, ds.q.event == None)
    ]  # NOQA
    assert result == ["DD.DD"]
    result = [
        _i._station_name for _i in ds.ifilter(ds.q.magnitude == magnitude_2)
    ]
    assert result == ["AA.AA", "DD.DD"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.magnitude == magnitude_2.resource_id)
    ]
    assert result == ["AA.AA", "DD.DD"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.magnitude == str(magnitude_2.resource_id))
    ]
    assert result == ["AA.AA", "DD.DD"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.magnitude == magnitude_2, ds.q.event == None)
    ]  # NOQA
    assert result == ["DD.DD"]

    # Focal mechanisms as a resource identifier and as a string, and with
    # others equal to None.
    result = [
        _i._station_name for _i in ds.ifilter(ds.q.focal_mechanism == focmec_1)
    ]
    assert result == ["AA.AA", "EE.EE"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.focal_mechanism == focmec_1.resource_id)
    ]
    assert result == ["AA.AA", "EE.EE"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.focal_mechanism == str(focmec_1.resource_id))
    ]
    assert result == ["AA.AA", "EE.EE"]
    result = [
        _i._station_name
        for _i in ds.ifilter(
            ds.q.focal_mechanism == focmec_1, ds.q.event == None
        )
    ]  # NOQA
    assert result == ["EE.EE"]
    result = [
        _i._station_name for _i in ds.ifilter(ds.q.focal_mechanism == focmec_2)
    ]
    assert result == ["AA.AA", "EE.EE"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.focal_mechanism == focmec_2.resource_id)
    ]
    assert result == ["AA.AA", "EE.EE"]
    result = [
        _i._station_name
        for _i in ds.ifilter(ds.q.focal_mechanism == str(focmec_2.resource_id))
    ]
    assert result == ["AA.AA", "EE.EE"]
    result = [
        _i._station_name
        for _i in ds.ifilter(
            ds.q.focal_mechanism == focmec_2, ds.q.event == None
        )
    ]  # NOQA
    assert result == ["EE.EE"]

    # No existing ids are treated like empty ids.
    result = [
        _i._station_name
        for _i in ds.ifilter(
            ds.q.event == None,
            ds.q.magnitude == None,
            ds.q.origin == None,
            ds.q.focal_mechanism == None,
        )
    ]  # NOQA
    assert result == ["FF.FF"]

    result = [
        _i._station_name
        for _i in ds.ifilter(
            ds.q.event != None,
            ds.q.magnitude != None,
            ds.q.origin != None,
            ds.q.focal_mechanism != None,
        )
    ]  # NOQA
    assert result == ["AA.AA"]


def test_provenance_accessor_missing_lines(tmpdir):
    """
    Tests some missing lines in the provenance accessor.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    assert str(data_set.provenance) == "No provenance document in file."

    filename = os.path.join(data_dir, "example_schematic_processing_chain.xml")

    # Add it as a document.
    doc = prov.read(filename, format="xml")
    data_set.add_provenance_document(doc, name="test_provenance")
    del data_set

    # Read it again.
    data_set = ASDFDataSet(asdf_filename)

    with pytest.raises(AttributeError):
        del data_set.provenance.random

    assert sorted(dir(data_set.provenance)) == sorted(
        [
            "test_provenance",
            "list",
            "keys",
            "values",
            "items",
            "get_provenance_document_for_id",
        ]
    )


def test_auxiliary_data_container_missing_lines(tmpdir):
    """
    Improving test coverage for the auxiliary data container.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    # Define some auxiliary data and add it.
    data = np.random.random(100)
    data_type = "RandomArrays"
    path = "test_data"
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(
        data=data, data_type=data_type, path=path, parameters=parameters
    )
    data_set.add_auxiliary_data(
        data=data,
        data_type=data_type,
        path="test_data_2",
        parameters=parameters,
    )

    container = data_set.auxiliary_data.RandomArrays.test_data

    # Test comparison methods.
    assert container != 1
    assert container != "A"

    container_2 = data_set.auxiliary_data.RandomArrays.test_data

    assert container == container_2

    container_2.data = container_2.data[:] * -1.0

    assert container != container_2

    container_3 = data_set.auxiliary_data.RandomArrays.test_data_2

    assert container != container_3

    # File accessor does not work for non file items.
    with pytest.raises(ASDFAttributeError):
        container.file


def test_auxiliary_data_accessor_missing_lines(tmpdir):
    """
    Improving test coverage for the auxiliary data accessor.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    # Define some auxiliary data and add it.
    data = np.random.random(100)
    parameters = {"a": 1, "b": 2.0, "e": "hallo"}

    data_set.add_auxiliary_data(
        data=data, data_type="AA", path="random", parameters=parameters
    )
    data_set.add_auxiliary_data(
        data=data, data_type="BB", path="random", parameters=parameters
    )

    accessor_1 = data_set.auxiliary_data.AA
    accessor_2 = data_set.auxiliary_data.BB

    assert accessor_1.auxiliary_data_type == "AA"
    assert accessor_2.auxiliary_data_type == "BB"

    assert sorted(dir(accessor_1)) == sorted(
        ["random", "auxiliary_data_type", "list"]
    )

    # Test comparision methods.
    assert accessor_1 == data_set.auxiliary_data.AA
    assert accessor_1 != accessor_2
    assert accessor_1 != 1
    assert accessor_1 != "random string."

    asdf_filename_2 = os.path.join(tmpdir.strpath, "test_2.h5")
    data_set_2 = ASDFDataSet(asdf_filename_2)
    # This is exactly the same as accessor_1 but not with the same data set,
    # thus it should not compare equal.
    data_set_2.add_auxiliary_data(
        data=data, data_type="AA", path="random", parameters=parameters
    )
    assert data_set.auxiliary_data.AA != data_set_2.auxiliary_data.AA

    # Try deleting not existing data set.
    with pytest.raises(AttributeError):
        del accessor_1.random_thingy

    # Empty accessor.
    del data_set_2.auxiliary_data.AA.random
    assert "Empty auxiliary data group." in str(data_set_2.auxiliary_data.AA)


def test_waveform_data_accessor_missing_lines(example_data_set, tmpdir):
    """
    Improving test coverage for the waveform data accessor.
    """
    ds = ASDFDataSet(example_data_set.filename)

    # Test comparison methods.
    assert ds.waveforms.AE_113A != 1
    assert ds.waveforms.AE_113A != "Hello"
    assert ds.waveforms.AE_113A == ds.waveforms.AE_113A
    assert ds.waveforms.AE_113A != ds.waveforms.TA_POKR

    ds2 = ASDFDataSet(filename=os.path.join(tmpdir.strpath, "blub.h5"))
    ds2.add_waveforms(ds.waveforms.AE_113A.raw_recording, tag="raw_recording")

    assert ds.waveforms.AE_113A != ds2.waveforms.AE_113A


def test_only_some_dtypes_are_allowed(tmpdir):
    """
    Waveform data should only be one of a couple of dtypes.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    data_set = ASDFDataSet(asdf_filename)

    valid_dtypes = [
        np.dtype("i4"),
        np.dtype("=i4"),
        np.dtype("<i4"),
        np.dtype(">i4"),
        np.dtype("i8"),
        np.dtype("=i8"),
        np.dtype("<i8"),
        np.dtype(">i8"),
        np.dtype("f4"),
        np.dtype("=f4"),
        np.dtype("<f4"),
        np.dtype(">f4"),
        np.dtype("f8"),
        np.dtype("=f8"),
        np.dtype("<f8"),
        np.dtype(">f8"),
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ]

    invalid_dtypes = [
        np.complex64,
        np.complex128,
        np.dtype("|S1"),
        np.dtype("|S2"),
        np.dtype("|S4"),
        np.dtype("b"),
        np.dtype(">H"),
        np.dtype("<H"),
        np.dtype("<u4"),
        np.dtype(">u4"),
        np.dtype("<u8"),
        np.dtype(">u8"),
    ]

    random.seed(12345)
    for dtype in valid_dtypes:
        tr = obspy.Trace(data=np.zeros(10, dtype=dtype))
        data_set.add_waveforms(tr, tag=str(random.randint(0, 1e6)))

    for dtype in invalid_dtypes:
        tr = obspy.Trace(data=np.zeros(10, dtype=dtype))
        with pytest.raises(TypeError):
            data_set.add_waveforms(tr, tag=str(random.randint(0, 1e6)))


def test_waveform_appending(tmpdir):
    """
    Tests the appending of waveforms.
    """
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")

    traces = [
        obspy.Trace(
            data=np.ones(10), header={"starttime": obspy.UTCDateTime(0)}
        ),
        obspy.Trace(
            data=np.ones(10), header={"starttime": obspy.UTCDateTime(10)}
        ),
        obspy.Trace(
            data=np.ones(10), header={"starttime": obspy.UTCDateTime(20)}
        ),
        obspy.Trace(
            data=np.ones(10), header={"starttime": obspy.UTCDateTime(30)}
        ),
        obspy.Trace(
            data=np.ones(10), header={"starttime": obspy.UTCDateTime(40)}
        ),
    ]

    for tr in traces:
        tr.stats.update({"network": "XX", "station": "YY", "channel": "EHZ"})

    # These can all be seamlessly merged.
    ds = ASDFDataSet(asdf_filename)
    for tr in traces:
        ds.append_waveforms(tr, tag="random")

    assert ds.waveforms.XX_YY.list() == [
        "XX.YY..EHZ__1970-01-01T00:00:00__1970-01-01T00:00:49__random"
    ]

    del ds
    os.remove(asdf_filename)

    # Slightly more complicated merging - it will only append to the back.
    ds = ASDFDataSet(asdf_filename)
    ds.append_waveforms(traces[0], tag="random")
    ds.append_waveforms(traces[2], tag="random")
    ds.append_waveforms(traces[4], tag="random")
    ds.append_waveforms(traces[1], tag="random")
    ds.append_waveforms(traces[3], tag="random")

    assert sorted(ds.waveforms.XX_YY.list()) == [
        "XX.YY..EHZ__1970-01-01T00:00:00__1970-01-01T00:00:19__random",
        "XX.YY..EHZ__1970-01-01T00:00:20__1970-01-01T00:00:39__random",
        "XX.YY..EHZ__1970-01-01T00:00:40__1970-01-01T00:00:49__random",
    ]


def test_dataset_accessing_limit(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")

    # This is exactly half a megabyte.
    tr = obspy.Trace(data=np.ones(131072, dtype=np.float32))
    tr.stats.network = "XX"
    tr.stats.station = "YY"
    tr.stats.channel = "BHZ"

    ds = ASDFDataSet(asdf_filename, compression=None)
    ds.add_waveforms(tr, tag="random")

    # The default limit (1GB) is plenty to read that.
    st = ds.waveforms.XX_YY.random
    assert len(st) == 1
    assert st[0].stats.network == "XX"
    assert st[0].stats.station == "YY"
    assert st[0].stats.npts == 131072

    # Setting it to exactly 0.5 MB should still be fine.
    ds.single_item_read_limit_in_mb = 0.5
    st = ds.waveforms.XX_YY.random
    assert len(st) == 1
    assert st[0].stats.network == "XX"
    assert st[0].stats.station == "YY"
    assert st[0].stats.npts == 131072

    # Any smaller and it has a problem. There is some leverage here as it
    # checks the file of the size before doing the expensive computation
    # regarding the size estimation.
    ds.single_item_read_limit_in_mb = 0.2
    with pytest.raises(ASDFValueError) as e:
        ds.waveforms.XX_YY.random

    assert e.value.args[0] == (
        "All waveforms for station 'XX.YY' and item 'random' would require "
        "'0.50 MB of memory. The current limit is 0.20 MB. Adjust by setting "
        "'ASDFDataSet.single_item_read_limit_in_mb' or use a different "
        "method to read the waveform data."
    )
    # hdf5 garbage collection messing with Python's...
    del e

    # Slightly different error message for direct access.
    with pytest.raises(ASDFValueError) as e:
        ds._get_waveform(
            "XX.YY..BHZ__1970-01-01T00:00:00__1970-01-02T12:24:31__random"
        )
    assert e.value.args[0] == (
        "The current selection would read 0.50 MB into memory from "
        "'XX.YY..BHZ__1970-01-01T00:00:00__"
        "1970-01-02T12:24:31__random'. The current limit is 0.20 MB. "
        "Adjust by setting "
        "'ASDFDataSet.single_item_read_limit_in_mb' or use a different "
        "method to read the waveform data."
    )
    del e


def test_get_waveforms_method(tmpdir):
    """
    The file wide get_waveforms() method is useful to for example work with
    continuous datasets.
    """
    # Create some data that can be tested with.
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    traces = [
        obspy.Trace(
            data=np.ones(10) * 0.0, header={"starttime": obspy.UTCDateTime(0)}
        ),
        obspy.Trace(
            data=np.ones(10) * 1.0, header={"starttime": obspy.UTCDateTime(10)}
        ),
        obspy.Trace(
            data=np.ones(10) * 2.0, header={"starttime": obspy.UTCDateTime(20)}
        ),
        obspy.Trace(
            data=np.ones(10) * 3.0, header={"starttime": obspy.UTCDateTime(30)}
        ),
        obspy.Trace(
            data=np.ones(10) * 4.0, header={"starttime": obspy.UTCDateTime(40)}
        ),
    ]
    for tr in traces:
        tr.stats.update({"network": "XX", "station": "YY", "channel": "EHZ"})
    ds = ASDFDataSet(asdf_filename)
    for tr in traces:
        ds.add_waveforms(tr, tag="random")

    # Get everything.
    st = ds.get_waveforms(
        network="XX",
        station="YY",
        location="",
        channel="EHZ",
        tag="random",
        starttime=obspy.UTCDateTime(0),
        endtime=obspy.UTCDateTime(49),
    )
    assert len(st) == 1
    assert st[0].stats.starttime == obspy.UTCDateTime(0)
    assert st[0].stats.endtime == obspy.UTCDateTime(49)

    # Get everything, but don't merge.
    st = ds.get_waveforms(
        network="XX",
        station="YY",
        location="",
        channel="EHZ",
        tag="random",
        automerge=False,
        starttime=obspy.UTCDateTime(0),
        endtime=obspy.UTCDateTime(49),
    )
    assert len(st) == 5
    assert st[0].stats.starttime == obspy.UTCDateTime(0)
    assert st[-1].stats.endtime == obspy.UTCDateTime(49)

    # Only access part of a the data.
    st = ds.get_waveforms(
        network="XX",
        station="YY",
        location="",
        channel="EHZ",
        tag="random",
        automerge=True,
        starttime=obspy.UTCDateTime(5),
        endtime=obspy.UTCDateTime(15),
    )
    assert len(st) == 1
    assert st[0].stats.starttime == obspy.UTCDateTime(5)
    assert st[0].stats.endtime == obspy.UTCDateTime(15)
    np.testing.assert_allclose(st[0].data, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


@pytest.mark.skipif(sys.version_info.major == 2, reason="Only run on Python 3")
def test_warning_that_data_exists_shows_up_every_time(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")
    ds = ASDFDataSet(asdf_filename)

    tr = obspy.read()[0]
    # Make sure hash is unique.
    tr.stats.starttime += 12345.789

    # No warning for first time.
    with warnings.catch_warnings(record=True) as w:
        ds.add_waveforms(tr, tag="a")
    assert len(w) == 0

    # Warning for all subsequent times.
    for _i in range(10):
        _i += 1
        with pytest.warns(ASDFWarning, match="already exists in file"):
            ds.add_waveforms(tr, tag="a")


def test_get_waveform_attributes(example_data_set):
    with ASDFDataSet(example_data_set.filename) as ds:
        assert ds.waveforms.AE_113A.get_waveform_attributes() == {
            "AE.113A..BHE__2013-05-24T05:40:00__"
            "2013-05-24T06:50:00__raw_recording": {
                "event_ids": [
                    "smi:service.iris.edu/fdsnws/event/1/query?"
                    "eventid=4218658"
                ],
                "sampling_rate": 40.0,
                "starttime": 1369374000000000000,
            },
            "AE.113A..BHN__2013-05-24T05:40:00__"
            "2013-05-24T06:50:00__raw_recording": {
                "event_ids": [
                    "smi:service.iris.edu/fdsnws/event/1/query?"
                    "eventid=4218658"
                ],
                "sampling_rate": 40.0,
                "starttime": 1369374000000000000,
            },
            "AE.113A..BHZ__2013-05-24T05:40:00__"
            "2013-05-24T06:50:00__raw_recording": {
                "event_ids": [
                    "smi:service.iris.edu/fdsnws/event/1/query?"
                    "eventid=4218658"
                ],
                "sampling_rate": 40.0,
                "starttime": 1369374000000000000,
            },
        }


def test_datesets_with_less_then_1_second_length(tmpdir):
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")

    tr = obspy.Trace(
        np.linspace(0, 1, 777),
        header={
            "network": "AA",
            "station": "BB",
            "location": "",
            "channel": "000",
            "starttime": obspy.UTCDateTime(38978345.3445843),
        },
    )

    # Don't use nano-seconds if longer than one second.
    with ASDFDataSet(asdf_filename) as ds:
        tr.stats.sampling_rate = 1.0
        ds.add_waveforms(tr, tag="test")
        dataset_name = ds.waveforms["AA.BB"].list()[0]
    os.remove(asdf_filename)

    assert (
        dataset_name
        == "AA.BB..000__1971-03-28T03:19:05__1971-03-28T03:32:01__test"
    )

    # Do, if shorter than one second.
    with ASDFDataSet(asdf_filename) as ds:
        tr.stats.sampling_rate = 474505737
        ds.add_waveforms(tr, tag="test")
        dataset_name = ds.waveforms["AA.BB"].list()[0]
    os.remove(asdf_filename)

    assert dataset_name == (
        "AA.BB..000__1971-03-28T03:19:05.344584304__"
        "1971-03-28T03:19:05.344585939__test"
    )

    # Don't do it for older versions to not write invalid files.
    with ASDFDataSet(asdf_filename, format_version="1.0.1") as ds:
        tr.stats.sampling_rate = 474505737
        ds.add_waveforms(tr, tag="test")
        dataset_name = ds.waveforms["AA.BB"].list()[0]
    os.remove(asdf_filename)

    assert dataset_name == (
        "AA.BB..000__1971-03-28T03:19:05__1971-03-28T03:19:05__test"
    )

    # Check that leading nulls are also written.
    with ASDFDataSet(asdf_filename) as ds:
        tr.stats.starttime = obspy.UTCDateTime(0)
        ds.add_waveforms(tr, tag="test")
        dataset_name = ds.waveforms["AA.BB"].list()[0]
    os.remove(asdf_filename)

    assert dataset_name == (
        "AA.BB..000__1970-01-01T00:00:00.000000000__"
        "1970-01-01T00:00:00.000001635__test"
    )


def test_information_in_other_namespaces_stationxml_is_retained(tmpdir):
    filename = os.path.join(data_dir, "multi_station_epoch.xml")
    asdf_filename = os.path.join(tmpdir.strpath, "test.h5")

    inv = obspy.read_inventory(filename)
    # We want to retain this.
    assert inv[0][0].extra == {
        "something": {"namespace": "https://example.com", "value": "test"}
    }
    assert inv[0][1].extra == {}
    assert inv[0][2].extra == {}

    # Write and close.
    with ASDFDataSet(asdf_filename) as ds:
        ds.add_stationxml(filename)

    # Open again and test.
    with ASDFDataSet(asdf_filename) as ds:
        inv2 = ds.waveforms.XX_YYY.StationXML
    assert inv2[0][0].extra == {
        "something": {"namespace": "https://example.com", "value": "test"}
    }
    assert inv2[0][1].extra == {}
    assert inv2[0][2].extra == {}
