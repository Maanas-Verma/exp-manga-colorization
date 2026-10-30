# SPDX-License-Identifier: GPL-3.0-only
"""Process memory accounting; macOS GPU allocations can be absent from RSS."""
import ctypes
from functools import lru_cache
import os
import platform
import resource


class _MacRusageV4(ctypes.Structure):
    # Stable rusage_info_v4 ABI: UUID followed by 35 uint64 counters.
    # Apple xnu/bsd/sys/resource.h: resident=6, footprint=7, lifetime peak=28.
    _fields_=[('uuid',ctypes.c_uint8*16),('counters',ctypes.c_uint64*35)]


@lru_cache(maxsize=1)
def _reader():
    library=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
    call=library.proc_pid_rusage
    call.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];call.restype=ctypes.c_int
    return library,call


def memory_snapshot(pid=None):
    pid=os.getpid() if pid is None else pid
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if pid==os.getpid() else None
    if peak is not None and platform.system()!='Darwin':peak*=1024
    result={'rss_bytes':None,'physical_footprint_bytes':None,
            'peak_physical_footprint_bytes':None,'peak_rss_bytes':int(peak) if peak is not None else None}
    if platform.system()=='Darwin':
        try:
            _,read=_reader();buffer=ctypes.create_string_buffer(4096)
            if read(pid,4,buffer)==0:
                counters=_MacRusageV4.from_buffer(buffer).counters
                result.update(rss_bytes=int(counters[6]),physical_footprint_bytes=int(counters[7]),
                              peak_physical_footprint_bytes=int(counters[28]))
        except (OSError,AttributeError):pass
    return result


def recycle_gpu_worker(backend,memory,limit_gib,completed):
    if backend!='coreml-gpu':return False
    footprint=(memory or {}).get('physical_footprint_bytes')
    if footprint is not None:return footprint>=limit_gib*1024**3
    return completed>=8  # Bounded fallback when physical accounting is unavailable.
