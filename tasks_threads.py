"""GUI-thread-off workers for the WINMOL plugin: running the compute
child process, and (first run only) building the compute environment.
"""
import subprocess
import sys

from PyQt5.QtCore import (
    QObject,
    pyqtSignal,
)

from .plugin_utils.childenv import child_env, safe_child_cwd
from .plugin_utils.installer import setup_environment, venv_location
from .plugin_utils.model_registry import ensure_model


class Worker(QObject):
    """Runs winmol_run.py in a subprocess on a QThread, streaming its
    output to the dialog log. Emits `succeeded` on exit code 0 and
    `error` otherwise (stderr is merged into the stream, so failures
    are visible in the log)."""

    finished = pyqtSignal()          # always, for thread teardown
    succeeded = pyqtSignal()         # exit code 0
    error = pyqtSignal(str)          # nonzero exit / launch failure
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(self, command, env_extra=None):
        super().__init__()
        self.command = command
        self.env_extra = dict(env_extra or {})
        self._popen = None
        self._cancelled = False

    def run_process(self):
        self.progress_signal.emit(0)
        try:
            startupinfo = None
            if sys.platform.startswith("win"):
                # Hide the console window on Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            # child_env() strips the PYTHONHOME/PYTHONPATH and GDAL_DATA/
            # PROJ_LIB that QGIS exports. winmol_run.py runs a DIFFERENT
            # interpreter with its own vendored GDAL, so inheriting QGIS's
            # would either stop it starting or point it at the wrong
            # proj.db. On Windows PATH also doubles as the DLL search
            # path, so the child's own venv is kept via python_exe.
            self._popen = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # surface errors in the log
                text=True,
                bufsize=1,
                startupinfo=startupinfo,
                env=child_env(self.env_extra or None,
                              python_exe=self.command[0]),
                cwd=safe_child_cwd(self.command[0]),
            )
            for line in iter(self._popen.stdout.readline, ""):
                self.update_signal.emit(line.rstrip("\n"))

            self._popen.stdout.close()
            return_code = self._popen.wait()
        except Exception as exc:            # launch failure (bad exe, etc.)
            self.error.emit(f"Failed to start analysis: {exc}")
            self.finished.emit()
            return

        # Raw line-count-based progress heuristic dropped; a real
        # progress bar driven by the pipeline's own printed counters is
        # plugin-gui scope. 100 here just clears whatever the bar showed.
        self.progress_signal.emit(100)
        if self._cancelled:
            self.error.emit("Analysis cancelled.")
        elif return_code == 0:
            self.succeeded.emit()
        else:
            self.error.emit(
                f"Analysis failed (exit code {return_code}). "
                "See the log above for details.")
        self.finished.emit()

    def cancel(self):
        """Terminate the running child so Cancel actually stops the work."""
        self._cancelled = True
        popen = self._popen
        if popen and popen.poll() is None:
            popen.terminate()
            try:
                popen.wait(timeout=5)
            except Exception:
                popen.kill()


class EnvSetupWorker(QObject):
    """Builds the WINMOL compute environment off the GUI thread: creates
    the managed venv and pip-installs requirements/cpu.txt (idempotent
    via the sentinel, see plugin_utils/installer.py). Keeps QGIS
    responsive during a multi-minute first-run install."""

    log = pyqtSignal(str)
    done = pyqtSignal(str)      # interpreter path on success
    failed = pyqtSignal(str)    # error message

    def __init__(self, plugin_dir):
        super().__init__()
        self.plugin_dir = plugin_dir

    def run(self):
        try:
            info = setup_environment(
                venv_location(self.plugin_dir), progress=self.log.emit)
            self.done.emit(info["python"])
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelEnsureWorker(QObject):
    """Resolves one registry model entry to a verified local file off the
    GUI thread, downloading it if missing/stale (model_registry.
    ensure_model, which is itself idempotent -- a verified-existing file
    short-circuits, so calling this on every run is cheap)."""

    log = pyqtSignal(str)
    done = pyqtSignal(str)      # local model file path on success
    failed = pyqtSignal(str)    # error message

    def __init__(self, entry, model_dir):
        super().__init__()
        self.entry = entry
        self.model_dir = model_dir

    def run(self):
        last_pct = [-1]

        def progress(done, total, entry):
            if not total:
                return
            pct = done * 100 // total
            if pct != last_pct[0] and pct % 10 == 0:
                last_pct[0] = pct
                self.log.emit(f"Downloading {entry.label}: {pct}%")

        try:
            path = ensure_model(self.entry, self.model_dir, progress=progress)
            self.done.emit(path)
        except Exception as exc:
            self.failed.emit(str(exc))
