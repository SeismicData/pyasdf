#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests all Python files of the project with flake8. This ensure PEP8 conformance
and some other sanity checks as well.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import flake8
import flake8.engine
import flake8.main
import inspect
import os
import re

import pytest

import pyasdf

_pattern = re.compile(r"^\d+\.\d+\.\d+$")
CLEAN_VERSION_NUMBER = bool(_pattern.match(pyasdf.__version__))


@pytest.mark.skipif(CLEAN_VERSION_NUMBER,
                    reason="Don't test code formatting for release versions.")
def test_flake8():
    test_dir = os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe())))
    pyasdf_dir = os.path.dirname(test_dir)

    # Possibility to ignore some files and paths.
    ignore_paths = [
        os.path.join(pyasdf_dir, "doc"),
        os.path.join(pyasdf_dir, ".git")]
    files = []
    for dirpath, _, filenames in os.walk(pyasdf_dir):
        ignore = False
        for path in ignore_paths:
            if dirpath.startswith(path):
                ignore = True
                break
        if ignore:
            continue
        filenames = [_i for _i in filenames if
                     os.path.splitext(_i)[-1] == os.path.extsep + "py"]
        if not filenames:
            continue
        for py_file in filenames:
            full_path = os.path.join(dirpath, py_file)
            files.append(full_path)

    # Get the style checker with the default style.
    flake8_style = flake8.engine.get_style_guide(
        parse_argv=False, config_file=flake8.main.DEFAULT_CONFIG)

    report = flake8_style.check_files(files)

    # Make sure at least 3 files are tested.
    assert report.counters["files"] > 3
    # And no errors occured.
    assert report.get_count() == 0
