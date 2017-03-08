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
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2017
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
    version="0.1.3",
    description=DOCSTRING[0],
    long_description="\n".join(DOCSTRING),
    author="Lion Krischer",
    author_email="krischer@geophysik.uni-muenchen.de",
    url="https://github.com/SeismicData/pyasdf",
    packages=find_packages(),
    license="BSD",
    platforms="OS Independent",
    install_requires=["numpy", "obspy>=1.0.0", "h5py", "colorama", "pytest",
                      "flake8", "prov", "dill"],
    extras_require={"mpi": ["mpi4py"]},
    package_data={
        "pyasdf": get_package_data()},
)


if __name__ == "__main__":
    setup(**setup_config)
