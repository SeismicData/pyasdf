#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The only purpose of this file is to be able to run the pyasdf test suite with

python -m pyasdf.tests

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
if __name__ == "__main__":
    import inspect
    import os
    import pytest
    import sys
    PATH = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))))

    sys.exit(pytest.main(PATH))
