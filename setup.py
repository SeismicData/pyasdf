#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for the pyasdf module.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import inspect
import os

from setuptools import setup, find_packages


def get_package_data():
    """
    Returns a list of all files needed for the installation relative to the
    'pyasdf' subfolder.
    """
    filenames = []
    # The lasif root dir.
    root_dir = os.path.join(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe()))), "pyasdf")
    # Recursively include all files in these folders:
    folders = [os.path.join(root_dir, "tests", "data")]
    for folder in folders:
        for directory, _, files in os.walk(folder):
            for filename in files:
                # Exclude hidden files.
                if filename.startswith("."):
                    continue
                filenames.append(os.path.relpath(
                    os.path.join(directory, filename),
                    root_dir))
    return filenames


setup_config = dict(
    name="pyasdf",
    version="0.0.1a",
    description="Module for creating and processing ASDF files.",
    author="Lion Krischer",
    author_email="krischer@geophysik.uni-muenchen.de",
    url="http: //github.com/SeismicData/pyasdf",
    packages=find_packages(),
    license="BSD",
    platforms="OS Independent",
    install_requires=["obspy>=0.10.1", "h5py", "colorama", "pytest",
                      "flake8", "prov"],
    extras_require={"mpi": ["mpi4py"]},
    package_data={
        "pyasdf": get_package_data()},
)


if __name__ == "__main__":
    setup(**setup_config)
