"""Resources this PROCESS may use, not what the machine happens to have.

Inside a container ``os.cpu_count()`` and ``psutil.virtual_memory()``
report the HOST. On a 47 GB / 12-core machine, ``docker run --memory=8g
--cpus=4`` still looks like 47 GB and 12 cores, so the planner sizes the
prediction batch, the producer threads and a ten-worker vector pool for
memory the container will never get, and the kernel OOM-kills it.

That matters more here than in most pipelines: the whole of issue #43 was
memory pressure, and the vector pool is sized directly from total RAM
(classes/ExecutionPlan._vector_memory_cap).

Everything degrades to the host value when no limit is set, so this is
safe on bare metal and in a VM as well as under Docker/Podman/k8s.
"""
import os

#: cgroup v1 writes a huge sentinel instead of "unlimited". Anything at or
#: above this is "no limit", not a real ceiling.
_NO_LIMIT = 1 << 60


def _read_first(*paths):
    for p in paths:
        try:
            with open(p) as fh:
                return fh.read().strip()
        except OSError:
            continue
    return None


def cpu_quota() -> float | None:
    """CPUs this process may use from a cgroup quota, or None.

    Covers `--cpus=N` (a quota/period pair). `--cpuset-cpus` is handled by
    sched_getaffinity in cpu_count() instead.
    """
    # cgroup v2: "max 100000" or "400000 100000"
    v2 = _read_first("/sys/fs/cgroup/cpu.max")
    if v2:
        parts = v2.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = float(parts[0]), float(parts[1])
                if quota > 0 and period > 0:
                    return quota / period
            except ValueError:
                pass
    # cgroup v1: -1 means unlimited
    q = _read_first("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    p = _read_first("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        if q and p and float(q) > 0 and float(p) > 0:
            return float(q) / float(p)
    except ValueError:
        pass
    return None


def cpu_count() -> int:
    """Usable CPUs: affinity mask and cgroup quota, whichever is tighter."""
    try:
        affinity = len(os.sched_getaffinity(0))     # honours --cpuset-cpus
    except (AttributeError, OSError):
        affinity = os.cpu_count() or 1
    quota = cpu_quota()                             # honours --cpus
    if quota:
        return max(1, min(affinity, int(quota) or 1))
    return max(1, affinity)


def memory_limit_bytes() -> int | None:
    """Memory ceiling from a cgroup, or None when unlimited."""
    for path in ("/sys/fs/cgroup/memory.max",                    # v2
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # v1
        raw = _read_first(path)
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if 0 < value < _NO_LIMIT:
            return value
    return None


def total_memory_bytes() -> int:
    """Memory this process may use: the cgroup limit, else the host's."""
    limit = memory_limit_bytes()
    if limit:
        return limit
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 2 * (1024 ** 3)          # last resort, deliberately small


def available_memory_bytes() -> int:
    """Free memory, clamped to the cgroup limit when one is set.

    Without the clamp a limited container reads the host's free memory and
    over-provisions exactly as badly as reading the host's total.
    """
    limit = memory_limit_bytes()
    try:
        import psutil
        free = int(psutil.virtual_memory().available)
    except Exception:
        free = limit or total_memory_bytes()
    if limit:
        used = 0
        raw = _read_first("/sys/fs/cgroup/memory.current",
                          "/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if raw:
            try:
                used = int(raw)
            except ValueError:
                used = 0
        return max(0, min(free, limit - used)) or limit
    return free


def describe() -> str:
    limit = memory_limit_bytes()
    where = "cgroup limit" if limit else "host"
    return (f"usable: {cpu_count()} CPU(s), "
            f"{total_memory_bytes() / (1024 ** 3):.1f} GB RAM ({where})")
