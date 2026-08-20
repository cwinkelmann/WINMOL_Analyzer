"""Turn winmol_run.py's stdout into a run progress percentage.

Parses the ``done/total`` counters the pipeline already prints (see the
regexes below) and maps them onto phase bands, instead of counting lines --
a plain line-count is structurally wrong because setup logging dwarfs the
first inference tiles.

Phase bands (a documented judgement call, not an exact model, derived from
``docs/benchmark-full-ortho.md``): ``Stems`` is setup 0-2, prediction 2-99.
``Trees``/``Nodes`` are setup 0-2, prediction 2-57, vector 57-95, merge
95-99. 100 is emitted only by :meth:`RunProgress.finish` on success.

Contract with the producers -- ``tests/test_run_progress.py`` pins these
exact prefixes so a format change fails CI instead of silently freezing
the bar:

* ``utils/Prediction.py``          ``Written tile {done}/{total} | ...``
* ``utils/PredictWorkers.py``      ``Multi-GPU prediction {done}/{total} | ...``
* ``utils/VectorTilePipeline.py``  ``Vector tiles {done}/{total} | ...``
* ``winmol_run.py``                ``Prepared {n}/{m} vector tiles ...``
* ``utils/IO.py``                  ``MERGE TILE READ | tile {id} | ...``

Pure stdlib, no Qt and no QGIS imports, so it is unit-testable off QGIS.
"""

import re

#: Percent reserved for setup (model load + the one-time batch autotune).
SETUP_END = 2

#: Band ends per process type: (prediction, vector, merge).
BANDS = {
    "Stems": (99, 99, 99),
    "Trees": (57, 95, 99),
    "Nodes": (57, 95, 99),
}

_RE_LOADING = re.compile(r"^\s*Loading Model\.\.\.\s*$")
_RE_AUTOTUNE = re.compile(
    r"^\s*\S.*autotune candidate (\d+)/(\d+)\b")
_RE_PREDICT = re.compile(r"^\s*Written tile (\d+)/(\d+) \|")
_RE_PREDICT_MGPU = re.compile(r"^\s*Multi-GPU prediction (\d+)/(\d+) \|")
_RE_VECTOR = re.compile(r"^\s*Vector tiles (\d+)/(\d+) \|")
_RE_PREPARED = re.compile(r"^\s*Prepared (\d+)/(\d+) vector tiles")
_RE_MERGE = re.compile(r"^\s*MERGE TILE READ \| tile ")


def _band(lo, hi, done, total):
    """Linear position inside ``[lo, hi]`` for ``done`` of ``total``."""
    if total <= 0:
        return lo
    frac = min(1.0, max(0.0, done / float(total)))
    return int(lo + (hi - lo) * frac)


class RunProgress:
    """Incremental parser: feed it stdout lines, get a percentage.

    ``feed(line)`` returns the new percent only when it changed, else
    ``None``. The value is clamped monotonically non-decreasing and never
    exceeds 99 until :meth:`finish` is called.
    """

    def __init__(self, process_type="Trees"):
        self.process_type = process_type
        pred_end, vec_end, merge_end = BANDS.get(
            process_type, BANDS["Trees"])
        self._pred_end = pred_end
        self._vec_end = vec_end
        self._merge_end = merge_end
        self.percent = 0
        #: True once any structured counter line has been recognised.
        self.matched = False
        self._merge_total = 0
        self._merge_done = 0

    # -- internals ---------------------------------------------------------

    def _set(self, value):
        value = max(0, min(99, int(value)))
        if value <= self.percent:
            return None                     # monotonic: never go backwards
        self.percent = value
        return value

    # -- public API --------------------------------------------------------

    def feed(self, line):
        """Consume one stdout line; return the new percent or ``None``."""
        if not line:
            return None

        match = _RE_PREDICT.match(line) or _RE_PREDICT_MGPU.match(line)
        if match:
            self.matched = True
            return self._set(_band(SETUP_END, self._pred_end,
                                   int(match.group(1)), int(match.group(2))))

        match = _RE_VECTOR.match(line)
        if match:
            self.matched = True
            # The vector phase only starts once prediction is complete.
            self.percent = max(self.percent, self._pred_end)
            return self._set(_band(self._pred_end, self._vec_end,
                                   int(match.group(1)), int(match.group(2))))

        match = _RE_PREPARED.match(line)
        if match:
            # Number of tiles that carry foreground == the number the merge
            # stage will read back. Captures the merge denominator only.
            self._merge_total = int(match.group(1))
            return None

        if _RE_MERGE.match(line):
            self.matched = True
            self._merge_done += 1
            self.percent = max(self.percent, self._vec_end)
            return self._set(_band(self._vec_end, self._merge_end,
                                   self._merge_done, self._merge_total))

        match = _RE_AUTOTUNE.match(line)
        if match:
            # The one-time batch-size autotune runs before the first tile and
            # can take a minute; give the bar a heartbeat inside the setup
            # band so it does not look like a hang.
            self.matched = True
            return self._set(_band(0, SETUP_END,
                                   int(match.group(1)), int(match.group(2))))

        if _RE_LOADING.match(line):
            # Explicitly the zero point of the run: everything before it is
            # argument echo, config dump and planning, which cost nothing.
            return None

        return None

    def finish(self, ok=True):
        """Return the terminal percent: 100 on success, else unchanged.

        On failure the bar is left where it stood; the log says what went
        wrong.
        """
        if not ok:
            return self.percent
        self.percent = 100
        return 100
