#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Watermark of the current system to increase reproducibility and provenance
and ease debugging.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from multiprocessing import cpu_count
from pkg_resources import get_distribution
import platform
from socket import gethostname
from time import strftime

import h5py
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from .utils import is_multiprocessing_problematic

# Dependencies.
modules = ["numpy", "scipy", "obspy", "lxml", "h5py", "prov", "dill"]
if MPI:
    modules.append("mpi4py")


def get_watermark():
    """
    Return information about the current system relevant for pyasdf.
    """
    vendor = MPI.get_vendor() if MPI else None

    c = h5py.get_config()
    if not hasattr(c, "mpi") or not c.mpi:
        is_parallel = False
    else:
        is_parallel = True

    watermark = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "platform_processor_count": cpu_count(),
        "platform_architecture": platform.architecture()[0],
        "platform_hostname": gethostname(),
        "date": strftime('%d/%m/%Y'),
        "time": strftime('%H:%M:%S'),
        "timezone": strftime('%Z'),
        "hdf5_version": h5py.version.hdf5_version,
        "parallel_h5py": is_parallel,
        "mpi_vendor": vendor[0] if vendor else None,
        "mpi_vendor_version": ".".join(map(str, vendor[1]))
        if vendor else None,
        "problematic_multiprocessing": is_multiprocessing_problematic()
        }

    watermark["module_versions"] = {
        module: get_distribution(module).version for module in modules}
    if MPI is None:
        watermark["module_versions"]["mpi4py"] = None

    return watermark
