"""Turn winmol_run.py's stdout into a run progress percentage.

Why this exists
---------------
The dialog's run progress bar used to be a LINE COUNTER: ``tasks_threads``
divided the number of stdout lines seen so far by a hardcoded constant
(``{"Stems": 34, "Trees": 118, "Nodes": 125}``). ``winmol_run.py`` is very
chatty during setup — ``Config.display()`` alone prints ~58 lines and the
execution plan another 16 — so **91 lines are printed before the first
inference**. 91/118 = 77 %: the bar sat at ~78 % before any real work had
happened, which is exactly what the user reported.

Rescaling the constant cannot fix it. The prediction phase logs one line per
``progress_interval_s`` (60 s on GPU) plus the first and last tile, so on the
documented 580-tile benchmark the ENTIRE inference phase is worth ~7 lines
against ~91 lines of setup. The mapping from lines to work is structurally
wrong, not merely miscalibrated.

So this module ignores line counts and parses the ``done/total`` counters the
pipeline **already** prints, mapping them onto phase bands.

Phase weights
-------------
Taken from ``docs/benchmark-full-ortho.md`` ("Wall time per phase", a
10528x7252 ortho, ``Trees``, defaults) for the two shipped U-Net models:

===========  ==========  ==========  =======
model        prediction  vector      merge
===========  ==========  ==========  =======
U-Net #1     308.0 s     182.2 s     5.3 s
             (62 %)      (37 %)      (1 %)
U-Net #2     220.7 s     177.9 s     6.3 s
             (55 %)      (44 %)      (1.5 %)
===========  ==========  ==========  =======

Chosen bands (a documented judgement call, not an exact model):

* ``Stems``          setup 0-2, prediction 2-99.
* ``Trees``/``Nodes`` setup 0-2, prediction 2-57, vector 57-95, merge 95-99.

100 is emitted only by :meth:`RunProgress.finish` on a successful exit.

The third model in that benchmark (a DeepLabV3+ where prediction is only 23 %
of wall time) is an outlier: with a fast detector the bar advances faster than
linear through the prediction band and then crawls. That is accepted — every
number the bar shows still corresponds to real tiles completed, it is still
monotonic, and it is never an interpolation over time.

Contract with the producers
---------------------------
The parsed formats live in four modules; ``tests/test_run_progress.py``
asserts the exact prefixes at their production sites so a format change fails
CI instead of silently freezing the bar:

* ``utils/Prediction.py``          ``Written tile {done}/{total} | ...``
* ``utils/PredictWorkers.py``      ``Multi-GPU prediction {done}/{total} | ...``
* ``utils/VectorTilePipeline.py``  ``Vector tiles {done}/{total} | ...``
* ``winmol_run.py``                ``Prepared {n}/{m} vector tiles ...``
* ``utils/IO.py``                  ``MERGE TILE READ | tile {id} | ...``

Everything up to and including the first ``|`` is the contract; the payload
after it is free. The unit labels the producers now append ("prediction
tile", "vector tile ~4144x4144 px") and the standalone ``PREDICTION PHASE``
/ ``VECTOR PHASE`` headers live entirely in that free part, and
``tests/test_run_progress.py`` reconstructs each producer's line from its
own f-string and feeds it through :class:`RunProgress` — a source-substring
pin alone would not catch text inserted *inside* the counter.

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

    ``feed(line)`` returns the new percent **only when it changed**, else
    ``None`` — so the caller emits one Qt signal per actual step instead of
    one per log line. The value is clamped monotonically non-decreasing and
    never exceeds 99 until :meth:`finish` is called.
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

        On failure the bar is deliberately left where it stood rather than
        snapped anywhere — the log says what went wrong.
        """
        if not ok:
            return self.percent
        self.percent = 100
        return 100
