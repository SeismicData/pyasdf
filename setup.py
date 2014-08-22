#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for obspy_asdf module.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import os

from setuptools import setup, find_packages


def get_package_data():
    """
    Returns a list of all files needed for the installation relativ to the
    "obspy_asdf" subfolder.
    """
    filenames = []
    # The lasif root dir.
    root_dir = os.path.join(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe()))), "obspy_asdf")
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
    name="obspy_asdf",
    version="0.0.1a",
    description="Module for creating and processing ASDF files.",
    author="Lion Krischer",
    author_email="krischer@geophysik.uni-muenchen.de",
    url="http: //github.com/krischer/ASDF",
    packages=find_packages(),
    license="GNU Lesser General Public License, version 3 (LGPLv3)",
    platforms="OS Independent",
    install_requires=["obspy >=0.9.2", "h5py", "colorama"],
    extras_require={
        "tests": ["pytest", "flake8"]},
    package_data={
        "obspy_asdf": get_package_data()},
)


if __name__ == "__main__":
    setup(**setup_config)
