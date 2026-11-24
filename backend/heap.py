"""Hand large freed buffers back to the OS (glibc only)."""

from __future__ import annotations

import ctypes
import sys

_M_MMAP_THRESHOLD = -3
# Image copies and CNN batches are far above this, so each gets its own
# mapping, unmapped when freed. glibc otherwise raises the threshold after the
# first large free and serves later ones from the heap, where fragmentation
# kept the web app ~170 MB above its idle size after a run of photo uploads.
# It costs a few percent of a reading's time.
MMAP_THRESHOLD = 128 * 1024


def return_large_frees() -> None:
    """Fix glibc's mmap threshold at MMAP_THRESHOLD; a no-op elsewhere."""
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:  # not glibc
        return
    libc.mallopt(_M_MMAP_THRESHOLD, MMAP_THRESHOLD)
