"""Reader threads that decode tiles ahead of inference.

Before this, each GPU worker read its own tiles serially between
inferences: ~27 ms of CPU decode per tile against ~4 ms of GPU time, so
an 8xH100 box sat ~89% idle. A spike on carrot (2026-09-15, R13) showed
reader THREADS scale 7.5x at 16 -- rasterio releases the GIL around
GDALRasterIO and, with resize in-graph, nothing GIL-bound is left per
tile -- and that 16 threads saturate one H100. So: threads, not
processes, one handle each, a bounded queue for backpressure.

The pool knows nothing about tiles. ``read_fn(src, batch_jobs)`` is
injected by the caller; this module only owns the threads, the queue and
the failure protocol.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, List

import rasterio


class ReaderDied(RuntimeError):
    """A reader thread is gone without having reported completion."""


class _Done:
    __slots__ = ()


class _Err:
    __slots__ = ('exc',)

    def __init__(self, exc):
        self.exc = exc


class ReaderPool:
    def __init__(
        self,
        input_raster: str,
        batches: List[list],
        read_fn: Callable,
        n_readers: int,
        queue_depth: int,
        open_fn: Callable = rasterio.open,
    ):
        self._path = input_raster
        self._read_fn = read_fn
        self._open_fn = open_fn
        self._n = max(1, int(n_readers))
        # Round-robin split keeps every reader busy to the end of the shard.
        self._slices = [batches[i::self._n] for i in range(self._n)]
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, int(queue_depth)))
        self._threads: List[threading.Thread] = []
        self._finished = 0
        self._stop = threading.Event()

    def start(self) -> None:
        for idx, my_batches in enumerate(self._slices):
            t = threading.Thread(
                target=self._run, args=(my_batches,),
                name=f"winmol-reader-{idx}", daemon=True)
            t.start()
            self._threads.append(t)

    def _run(self, my_batches) -> None:
        # One handle PER THREAD. Measured identical to a shared handle,
        # and a shared-handle non-boundless read across threads terminated
        # abnormally in the spike; own handles remove that class of bug.
        try:
            with self._open_fn(self._path) as src:
                for batch_jobs in my_batches:
                    if self._stop.is_set():
                        return
                    tiles, masks, stats = self._read_fn(src, batch_jobs)
                    self._q.put((batch_jobs, tiles, masks, stats))
        except BaseException as exc:  # noqa: BLE001 -- must reach consumer
            self._q.put(_Err(exc))
            return
        self._q.put(_Done())

    def get(self, timeout: float = 30.0):
        """Next decoded batch, or None once every reader has finished.

        Raises the reader's own exception if one failed, and ReaderDied if
        a thread is dead without having put its sentinel. A run must fail
        loudly; it must never wait forever on a queue nobody will fill.
        """
        while True:
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                alive = sum(t.is_alive() for t in self._threads)
                if alive + self._finished < self._n:
                    raise ReaderDied(
                        f"{self._n - alive - self._finished} reader thread(s) "
                        "exited without finishing")
                continue
            if isinstance(item, _Err):
                raise item.exc
            if isinstance(item, _Done):
                self._finished += 1
                if self._finished == self._n:
                    return None
                continue
            return item

    def close(self) -> None:
        import time
        self._stop.set()
        deadline = time.time() + 10.0
        while any(t.is_alive() for t in self._threads):
            try:
                self._q.get(timeout=0.1)
            except queue.Empty:
                pass
            if time.time() > deadline:
                break
        for t in self._threads:
            t.join(timeout=0)
