"""Idempotent, owned-process suspend/resume using documented Windows thread counts.

SuspendThread returns the preceding count; a balanced ResumeThread restores a
query's temporary increment. Process wait-state labels do not expose these counts.
Only threads with counts zero or one are accepted; nested external suspension is
never silently removed. Used for external experiment control, not application locks.
"""
import ctypes
from ctypes import wintypes
import time
import psutil

kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenThread.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
kernel.GetProcessIdOfThread.argtypes = [wintypes.HANDLE]
kernel.GetProcessIdOfThread.restype = wintypes.DWORD
for name in ('SuspendThread', 'ResumeThread'):
    function = getattr(kernel, name)
    function.argtypes = [wintypes.HANDLE]
    function.restype = wintypes.DWORD


def thread_counts(process, mode='query'):
    if mode not in ('query', 'park', 'resume'):
        raise ValueError('Invalid thread transition')
    rows = []
    for thread in process.threads():
        handle = kernel.OpenThread(0x0802, False, thread.id)
        if not handle:
            # A background thread may exit between enumeration and OpenThread.
            if thread.id not in {t.id for t in process.threads()}:
                continue
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if kernel.GetProcessIdOfThread(handle) != process.pid:
                raise RuntimeError('Thread ID was reused by another process')
            before = kernel.SuspendThread(handle)
            if before == 0xffffffff:
                if thread.id not in {t.id for t in process.threads()}:
                    continue
                raise ctypes.WinError(ctypes.get_last_error())
            # Keeping our increment is appropriate only for a zero-count thread
            # being parked. Otherwise undo the probe before any further action.
            if mode != 'park' or before != 0:
                restored = kernel.ResumeThread(handle)
                if restored != before+1:
                    raise RuntimeError('Concurrent thread suspend-count change')
            if before > 1:
                raise RuntimeError('Nested suspension is not owned by this lease')
            after = 1 if mode == 'park' else before
            if mode == 'resume' and before == 1:
                if kernel.ResumeThread(handle) != 1:
                    raise RuntimeError('Concurrent thread resume-count change')
                after = 0
            rows.append({'tid': thread.id, 'before': before, 'after': after})
        finally:
            kernel.CloseHandle(handle)
    return rows


def park(process):
    previous = None
    for attempt in range(20):
        rows = thread_counts(process, 'park')
        current = {row['tid'] for row in rows}
        if current == previous and all(row['before'] == 1 for row in rows):
            return rows
        previous = current
        time.sleep(.01)
    raise RuntimeError('Native thread set did not settle while parking')


def resume(process):
    # Newly created threads already have count zero and need no decrement.
    return thread_counts(process, 'resume')


def primary_parked(process, tid):
    rows = thread_counts(process)
    return any(row['tid'] == tid and row['before'] == 1 for row in rows)


def all_running(process):
    return all(row['before'] == 0 for row in thread_counts(process))
