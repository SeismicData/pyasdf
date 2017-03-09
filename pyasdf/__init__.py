#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .exceptions import ASDFException, ASDFWarning, WaveformNotInFileException
from .asdf_data_set import ASDFDataSet


__all__ = ["__version__", "ASDFDataSet", "ASDFException", "ASDFWarning",
           "WaveformNotInFileException", "print_sys_info", "get_sys_info"]

__version__ = "0.1.4"


def print_sys_info():
    """
    Prints information about the system, HDF5 and h5py version.

    Very useful to judge the installation.
    """
    from .watermark import get_watermark  # NOQA
    wm = get_watermark()

    info = (
        "pyasdf version {pyasdf_version}\n"
        "{div}\n"
        "{python} {py_version}, compiler: {py_compiler}\n"
        "{platform} {platform_release} {architecture}\n"
        "Machine: {machine}, Processor: {processor} with {count} cores\n"
        "{div}\n"
        "HDF5 version {hdf5_version}, h5py version: {h5py_version}\n"
        "MPI: {mpi_vendor}, version: {mpi_vendor_version}, "
        "mpi4py version: {mpi4py_version}\n"
        "Parallel I/O support: {is_parallel}\n"
        "Problematic multiprocessing: {problematic_mp}\n"
        "{div}\n"
        "Other_modules:\n\t{other_modules}"
    )

    print(info.format(
        pyasdf_version=__version__,
        div="=" * 79,
        python=wm["python_implementation"],
        py_version=wm["python_version"],
        py_compiler=wm["python_compiler"],
        platform=wm["platform_system"],
        platform_release=wm["platform_release"],
        architecture=wm["platform_architecture"],
        machine=wm["platform_machine"],
        processor=wm["platform_processor"],
        count=wm["platform_processor_count"],
        hdf5_version=wm["hdf5_version"],
        h5py_version=wm["module_versions"]["h5py"],
        mpi_vendor=wm["mpi_vendor"],
        mpi_vendor_version=wm["mpi_vendor_version"],
        mpi4py_version=wm["module_versions"]["mpi4py"],
        is_parallel=wm["parallel_h5py"],
        problematic_mp=wm["problematic_multiprocessing"],
        other_modules="\n\t".join(
            "%s: %s" % (key, value) for key, value in
            sorted(wm["module_versions"].items(), key=lambda x: x[0])
            if key != "mpi4py" and key != "h5py")

    ))
