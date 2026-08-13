from __future__ import annotations

import ctypes
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_count: int
    total_ram_gb: float
    available_ram_gb: float
    ram_used_percent: float
    free_disk_gb: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _windows_memory() -> tuple[int, int]:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.ullTotalPhys, status.ullAvailPhys


def snapshot(path: Path) -> ResourceSnapshot:
    if os.name != "nt":
        raise RuntimeError("This lightweight preflight currently supports Windows only")
    total, available = _windows_memory()
    disk = shutil.disk_usage(path.resolve())
    gib = 1024**3
    return ResourceSnapshot(
        cpu_count=os.cpu_count() or 1,
        total_ram_gb=round(total / gib, 2),
        available_ram_gb=round(available / gib, 2),
        ram_used_percent=round((1 - available / total) * 100, 1),
        free_disk_gb=round(disk.free / gib, 2),
    )


def assert_safe(
    path: Path,
    *,
    min_available_ram_gb: float = 2.0,
    min_free_disk_gb: float = 20.0,
) -> ResourceSnapshot:
    state = snapshot(path)
    problems: list[str] = []
    if state.available_ram_gb < min_available_ram_gb:
        problems.append(
            f"available RAM {state.available_ram_gb:.2f} GB is below "
            f"the {min_available_ram_gb:.2f} GB safety floor"
        )
    if state.free_disk_gb < min_free_disk_gb:
        problems.append(
            f"free disk {state.free_disk_gb:.2f} GB is below "
            f"the {min_free_disk_gb:.2f} GB safety floor"
        )
    if problems:
        raise RuntimeError("Resource preflight refused the run: " + "; ".join(problems))
    return state
