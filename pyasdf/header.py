#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function)

import re

import numpy as np

# List all compression options.
COMPRESSIONS = {
    None: (None, None),
    "lzf": ("lzf", None),
    "gzip-0": ("gzip", 0),
    "gzip-1": ("gzip", 1),
    "gzip-2": ("gzip", 2),
    "gzip-3": ("gzip", 3),
    "gzip-4": ("gzip", 4),
    "gzip-5": ("gzip", 5),
    "gzip-6": ("gzip", 6),
    "gzip-7": ("gzip", 7),
    "gzip-8": ("gzip", 8),
    "gzip-9": ("gzip", 9),
    "szip-ec-8": ("szip", ("ec", 8)),
    "szip-ec-10": ("szip", ("ec", 10)),
    "szip-nn-8": ("szip", ("nn", 8)),
    "szip-nn-10": ("szip", ("nn", 10))
}


# The inversion mapping also works.
for key, value in list(COMPRESSIONS.items()):
    COMPRESSIONS[value] = key


FORMAT_NAME = "ASDF"
FORMAT_VERSION = "1.0.0"


# Regular expression for allowed filenames within the provenance group.
PROV_FILENAME_REGEX = re.compile(r"^[0-9a-z][0-9a-z_]*[0-9a-z]$")

# Regular expression for allowed tag names.
TAG_REGEX = re.compile(r"^[a-z_0-9]+$")

# 4 and 8 bytes signed integers and floating points.
VALID_SEISMOGRAM_DTYPES = (
    np.dtype("<i4"), np.dtype(">i4"),
    np.dtype("<i8"), np.dtype(">i8"),
    np.dtype("<f4"), np.dtype(">f4"),
    np.dtype("<f8"), np.dtype(">f8")
)

# MPI message tags used for communication.
MSG_TAGS = [
    # This message is sent from the master to all workers when metadata
    # should be synchronized and the data should be written.
    "MASTER_FORCES_WRITE",
    # Message sent from Master to one of the workers containing a new job.
    "MASTER_SENDS_ITEM",
    # Sent from worker to master to request a new job.
    "WORKER_REQUESTS_ITEM",
    # Information message from worker to master to indicate a job has fully
    # completed.
    "WORKER_DONE_WITH_ITEM",
    # Buffer of worker is full and it would like to write. Master will
    # initialize a metadata synchronization once a certain number of workers
    # reached a full buffer.
    "WORKER_REQUESTS_WRITE",
    # Send from a worker to indicate that it received the poison pill and is
    # now either waiting for a final write or done with everything.
    "POISON_PILL_RECEIVED",
    # Message send by the master to indicate everything has been processed.
    # Otherwise all workers will keep looping to be able to synchronize
    # metadata.
    "ALL_DONE",
    ]

# Convert to two-way dict as MPI only knows integer tags.
MSG_TAGS = {msg: i for i, msg in enumerate(MSG_TAGS)}
MSG_TAGS.update({value: key for key, value in MSG_TAGS.items()})

# Poison pill sent from master to workers to indicate that no more work is
# available.
POISON_PILL = "POISON_PILL"


MAX_MEMORY_PER_WORKER_IN_MB = 256
