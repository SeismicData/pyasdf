"""
Regression tests for system information without setuptools installed.
"""

import sys

import h5py
import numpy as np

from pyasdf import __version__, print_sys_info


def test_system_info_without_pkg_resources(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pkg_resources", None)

    from pyasdf.watermark import get_watermark

    watermark = get_watermark()
    assert watermark["module_versions"]["numpy"] == np.__version__
    assert watermark["module_versions"]["h5py"] == h5py.__version__
    assert watermark["hdf5_version"] == h5py.version.hdf5_version

    print_sys_info()
    output = capsys.readouterr().out
    assert "pyasdf version %s" % __version__ in output
    assert "numpy: %s" % np.__version__ in output
    assert "HDF5 version %s" % h5py.version.hdf5_version in output
