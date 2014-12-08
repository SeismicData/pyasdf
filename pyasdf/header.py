class ASDFException(Exception):
    """
    Generic exception for the Python ASDF implementation.
    """
    pass


class ASDFWarnings(UserWarning):
    """
    Generic ASDF warning.
    """
    pass


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
    "gzip8-": ("gzip", 8),
    "gzip-9": ("gzip", 9),
    "szip-ec-8": ("szip", ("ec", 8)),
    "szip-ec-10": ("szip", ("ec", 10)),
    "szip-nn-8": ("szip", ("nn", 8)),
    "szip-nn-10": ("szip", ("nn", 10))
}


FORMAT_NAME = "ASDF"
FORMAT_VERSION = "0.0.1b"

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
    # Message send by the master to indicate everything has been processed.
    # Otherwise all workers will keep looping to be able to synchronize
    # metadata.
    "ALL_DONE",
    ]
# Convert to two-way dict as MPI only knows integer tags.
MSG_TAGS = {msg: i  for i, msg in enumerate(MSG_TAGS)}
MSG_TAGS.update({value: key for key, value in MSG_TAGS.items()})

# Poison pill sent from master to workers to indicate that no more work is
# available.
POISON_PILL = "POISON_PILL"
#
MAX_MEMORY_PER_WORKER_IN_MB = 2

