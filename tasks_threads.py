import subprocess
import sys

from qgis.PyQt.QtCore import QObject, pyqtSignal

from .plugin_utils.childenv import child_env


class Worker(QObject):
    """Runs winmol_run.py in a subprocess on a QThread, streaming its output
    to the dialog log. Emits `succeeded` on exit code 0 and `error` otherwise
    (stderr is merged into the stream, so failures are visible in the log)."""

    finished = pyqtSignal()          # always, for thread teardown
    succeeded = pyqtSignal()         # exit code 0
    error = pyqtSignal(str)          # nonzero exit / launch failure
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self._popen = None
        self._cancelled = False

    def run_process(self):
        total_expected = self.get_total_lines()
        total_lines = 0
        self.progress_signal.emit(0)
        try:
            startupinfo = None
            if sys.platform.startswith("win"):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            # child_env() strips the PYTHONHOME/PYTHONPATH and GDAL_DATA/
            # PROJ_LIB that QGIS exports. winmol_run.py runs a DIFFERENT
            # interpreter with its own vendored GDAL, so inheriting QGIS's
            # would either stop it starting or point it at the wrong proj.db.
            # WINMOL_* overrides (e.g. WINMOL_CONFIG_OVERRIDES_JSON) are
            # deliberately preserved.
            self._popen = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # surface errors in the log
                universal_newlines=True,
                bufsize=1,
                startupinfo=startupinfo,
                env=child_env(),
            )
            for line in iter(self._popen.stdout.readline, ""):
                self.update_signal.emit(line.rstrip("\n"))
                total_lines += 1
                self.progress_signal.emit(
                    min(99, int(total_lines / max(total_expected, 1) * 100)))

            self._popen.stdout.close()
            return_code = self._popen.wait()
        except Exception as exc:            # launch failure (bad exe, etc.)
            self.error.emit(f"Failed to start analysis: {exc}")
            self.finished.emit()
            return

        if self._cancelled:
            self.error.emit("Analysis cancelled.")
        elif return_code == 0:
            self.progress_signal.emit(100)
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

    def get_total_lines(self):
        return {"Stems": 34, "Trees": 118, "Nodes": 125}.get(
            self.command[-1], 100)


class EnvSetupWorker(QObject):
    """Builds the WINMOL compute environment off the GUI thread: resolves /
    downloads Python 3.11, creates the venv, pip-installs the deps. Emits log
    lines for progress and done/failed at the end. Keeps QGIS responsive during
    a multi-minute first-run setup."""

    log = pyqtSignal(str)
    done = pyqtSignal(str)      # interpreter path on success ('' if unknown)
    failed = pyqtSignal(str)    # error message

    def __init__(self, plugin_dir):
        super().__init__()
        self.plugin_dir = plugin_dir
        self._cancelled = False

    def cancel(self):
        """Best-effort cancel flag. The heavy steps (download, pip) aren't
        interruptible mid-call, but network reads are timeout-bounded, so the
        worker returns within a bounded time and teardown can wait it out."""
        self._cancelled = True

    def run(self):
        try:
            from .plugin_utils import installer
            venv = installer.venv_location(self.plugin_dir)
            info = installer.setup_environment(
                venv, plugin_dir=self.plugin_dir,
                progress=lambda m: self.log.emit(str(m)))
            self.done.emit(info.get("python") or "")
        except Exception as exc:
            self.failed.emit(str(exc))
