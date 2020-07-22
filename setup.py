#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python module for the Adaptable Seismic Data Format (ASDF).

For further information and contact information please see these two web sites:

* Landing page of the ASDF data format: http://seismic-data.org
* Github repository of pyasdf: http://www.github.com/SeismicData/pyasdf


Changelog
---------

::

    Version 0.7.2 (July 22, 2020)
    ----------------------------------
    * Replaced some numpy functionality that will be deprecated in future numpy
      versions.
    * Warnings will now raise when running the tests.

    Version 0.7.1 (June 15, 2020)
    ----------------------------------
    * Write namespace abbreviations so that ObsPy can read the written
      StationXML files again with the other namespaces.

    Version 0.7.0 (June 14, 2020)
    ----------------------------------
    * More conservative merging of stations. Stations are now only merged if
      everything except start and end date and the channels are the same.
      Otherwise they are not merged and will be retained. See #61. This might
      result in some slightly larger StationXML files in the ASDF files but
      should make no difference for downstream tools so it should affect
      users.

    Version 0.6.1 (April 9, 2020)
    ----------------------------------
    * `pytest` is no longer a runtime dependency.

    Version 0.6.0 (March 19, 2020)
    ----------------------------------
    * Drop support for Python <= 3.6.
    * Fix a few deprecations warnings in Python >= 3.7.
    * `flake8` and `pytest` no longer are runtime dependencies.
    * Get rid of the special warnings handler as it interferes with other
      packages.

    Version 0.5.1 (September 24, 2019)
    ----------------------------------
    * Restore the ability to run tests with `python -m pyasdf.tests`.

    Version 0.5.0 (September 24, 2019)
    ----------------------------------
    * Implement ASDF version 1.0.3 which allows a bit more flexibility
      regarding names of auxiliary data sets as well as provenance files.
    * New .waveform_tags property for the dataset object return a set of all
      available waveform tags (see #46, #47)

    Version 0.4.0 (March 12, 2018)
    -------------------
    * Support for ASDF version 1.0.2. Allows writing traces that are less than
      one second short (see #44, #45).
    * New get_waveform_attributes() method to quickly get all attributes
      for the waveforms of a stations (see #38, #39).

    Version 0.3.0 (October 19, 2017)
    ----------------------------------
    * Support for ASDF 1.0.1 (the only difference to 1.0.0 is support for
      16 bit integer waveform data).

    Version 0.2.1 (September 21, 2017)
    ----------------------------------
    * Don't attempt to write ASDF header info to files in read-only mode.
    * get_coordinates() now works for StationXML files with a very large number
      of comments.
    * Station.__getattr__() now works with underscores or dots as the network
      and station separator.
    * __contains__() implemented for the station accessor object.

    Version 0.2.0 (April 7, 2017)
    ---------------------------
    * New script to convert a folder of SAC files to ASDF.

    Version 0.1.4 (March 9, 2017)
    ---------------------------
    * More visible warnings on Python 2 if necessary.

    Version 0.1.3 (March 8, 2017)
    ---------------------------
    * Now also works with Python 3 under windows.

    Version 0.1.2 (March 7, 2017)
    ---------------------------
    * Also shipping license file.

    Version 0.1.1 (March 7, 2017)
    ---------------------------
    * Stable, tagged version.


Licensing Information
---------------------

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2013-2020
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import inspect
import os

from setuptools import setup, find_packages


DOCSTRING = __doc__.strip().split("\n")


def get_package_data():
    """
    Returns a list of all files needed for the installation relative to the
    'pyasdf' subfolder.
    """
    filenames = []
    # The lasif root dir.
    root_dir = os.path.join(
        os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        ),
        "pyasdf",
    )
    # Recursively include all files in these folders:
    folders = [os.path.join(root_dir, "tests", "data")]
    for folder in folders:
        for directory, _, files in os.walk(folder):
            for filename in files:
                # Exclude hidden files.
                if filename.startswith("."):
                    continue
                filenames.append(
                    os.path.relpath(
                        os.path.join(directory, filename), root_dir
                    )
                )
    return filenames


setup_config = dict(
    name="pyasdf",
    version="0.7.2",
    description=DOCSTRING[0],
    long_description="\n".join(DOCSTRING),
    author="Lion Krischer",
    author_email="lion.krischer@gmail.com",
    url="https://github.com/SeismicData/pyasdf",
    packages=find_packages(),
    license="BSD",
    platforms="OS Independent",
    install_requires=[
        "numpy",
        "obspy>=1.2.1",
        "h5py",
        "colorama",
        "prov",
        "dill",
    ],
    extras_require={"mpi": ["mpi4py"], "tests": ["pytest"]},
    package_data={"pyasdf": get_package_data()},
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)


if __name__ == "__main__":
    setup(**setup_config)
