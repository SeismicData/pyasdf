# Pyasdf

Python module for the Adaptable Seismic Data Format (ASDF).

[![PyPI Version](https://img.shields.io/pypi/v/pyasdf.svg)](https://pypi.python.org/pypi/pyasdf) |
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyasdf.svg)](https://anaconda.org/conda-forge/pyasdf) |
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/pyasdf.svg)](https://anaconda.org/conda-forge/pyasdf) |
[![License](https://img.shields.io/pypi/l/pyasdf.svg)](https://pypi.python.org/pypi/pyasdf/) |
![Build Status](https://github.com/SeismicData/pyasdf/actions/workflows/pyasdf.yml/badge.svg)

----

**Python library to read and write ASDF files.**

----

Landing page for all things **ASDF**: https://seismic-data.org

> This is the **A**daptable **S**eismic **D**ata **F**ormat - if you are looking for the **A**dvanced **S**cientific **D**ata **F**ormat, go here: https://asdf.readthedocs.io/en/latest/

[Documentation](http://seismicdata.github.io/pyasdf/)

![Logo](/doc/logo/pyasdf_logo.png)

### Changelog

#### Version 0.8.0 (August 3, 2023)
* Drop support for Python <= 3.8.
* Support for the latest versions of all dependencies including Python 3.11.
* Internal code modernization.

#### Version 0.7.5 (April 16, 2021)
* Fix an issue when appending waveforms of less than one second in length.

#### Version 0.7.4 (August 21, 2020)
* Workaround for an issue with HDF5 on PPC64.

#### Version 0.7.3 (July 22, 2020)
* Use a different way to copy the StationXML data in the process() function
    so it no longer gets stuck for large runs with MPI. See #62.

#### Version 0.7.2 (July 22, 2020)
* Replaced some numpy functionality that will be deprecated in future numpy
    versions.
* Warnings will now raise when running the tests.

#### Version 0.7.1 (June 15, 2020)
* Write namespace abbreviations so that ObsPy can read the written
    StationXML files again with the other namespaces.

#### Version 0.7.0 (June 14, 2020)
* More conservative merging of stations. Stations are now only merged if
    everything except start and end date and the channels are the same.
    Otherwise they are not merged and will be retained. See #61. This might
    result in some slightly larger StationXML files in the ASDF files but
    should make no difference for downstream tools so it should affect
    users.

#### Version 0.6.1 (April 9, 2020)
* `pytest` is no longer a runtime dependency.

#### Version 0.6.0 (March 19, 2020)
* Drop support for Python <= 3.6.
* Fix a few deprecations warnings in Python >= 3.7.
* `flake8` and `pytest` no longer are runtime dependencies.
* Get rid of the special warnings handler as it interferes with other
    packages.

#### Version 0.5.1 (September 24, 2019)
* Restore the ability to run tests with `python -m pyasdf.tests`.

#### Version 0.5.0 (September 24, 2019)
* Implement ASDF version 1.0.3 which allows a bit more flexibility
    regarding names of auxiliary data sets as well as provenance files.
* New .waveform_tags property for the dataset object return a set of all
    available waveform tags (see #46, #47)

#### Version 0.4.0 (March 12, 2018)
* Support for ASDF version 1.0.2. Allows writing traces that are less than
    one second short (see #44, #45).
* New get_waveform_attributes() method to quickly get all attributes
    for the waveforms of a stations (see #38, #39).

#### Version 0.3.0 (October 19, 2017)
* Support for ASDF 1.0.1 (the only difference to 1.0.0 is support for
    16 bit integer waveform data).

#### Version 0.2.1 (September 21, 2017)
* Don't attempt to write ASDF header info to files in read-only mode.
* get_coordinates() now works for StationXML files with a very large number
    of comments.
* Station.__getattr__() now works with underscores or dots as the network
    and station separator.
* __contains__() implemented for the station accessor object.

#### Version 0.2.0 (April 7, 2017)
* New script to convert a folder of SAC files to ASDF.

#### Version 0.1.4 (March 9, 2017)
* More visible warnings on Python 2 if necessary.

#### Version 0.1.3 (March 8, 2017)
* Now also works with Python 3 under windows.

#### Version 0.1.2 (March 7, 2017)
* Also shipping license file.

#### Version 0.1.1 (March 7, 2017)
* Stable, tagged version.