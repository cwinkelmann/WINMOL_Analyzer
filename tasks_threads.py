import subprocess
import sys

from qgis.PyQt.QtCore import QObject, pyqtSignal

from .plugin_utils.childenv import child_env
from .plugin_utils.run_progress import RunProgress


class Worker(QObject):
    """Runs winmol_run.py in a subprocess on a QThread, streaming its output
    to the dialog log. Emits `succeeded` on exit code 0 and `error` otherwise
    (stderr is merged into the stream, so failures are visible in the log)."""

    finished = pyqtSignal()          # always, for thread teardown
    succeeded = pyqtSignal()         # exit code 0
    error = pyqtSignal(str)          # nonzero exit / launch failure
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(self, command, env_extra=None):
        super().__init__()
        self.command = command
        # Extra environment for the compute child on top of child_env()'s
        # sanitised copy — currently WINMOL_AUTOTUNE_CACHE, which points the
        # child at the plugin's managed cache instead of a HOME-derived one.
        self.env_extra = dict(env_extra or {})
        self._popen = None
        self._cancelled = False

    def run_process(self):
        # Progress comes from the done/total counters winmol_run.py already
        # prints, mapped onto phase bands — NOT from a line count. See
        # plugin_utils/run_progress.py for why (the old heuristic sat at
        # ~78 % before the first inference).
        progress = RunProgress(self.command[-1])
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
                # command[0] is the compute interpreter: child_env() uses it
                # to put that venv's CUDA/cuDNN wheel directories on the
                # loader path, without which onnxruntime-gpu runs on the CPU.
                env=child_env(self.env_extra or None,
                              python_exe=self.command[0]),
            )
            for line in iter(self._popen.stdout.readline, ""):
                text = line.rstrip("\n")
                self.update_signal.emit(text)
                percent = progress.feed(text)
                if percent is not None:
                    self.progress_signal.emit(percent)

            self._popen.stdout.close()
            return_code = self._popen.wait()
        except Exception as exc:            # launch failure (bad exe, etc.)
            self.error.emit(f"Failed to start analysis: {exc}")
            self.finished.emit()
            return

        if self._cancelled:
            self.error.emit("Analysis cancelled.")
        elif return_code == 0:
            self.progress_signal.emit(progress.finish(ok=True))
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
    """Builds the WINMOL compute environment off the GUI thread: resolves /
    downloads Python 3.11, creates the venv, pip-installs the deps. Emits log
    lines for progress and done/failed at the end. Keeps QGIS responsive during
    a multi-minute first-run setup."""

    log = pyqtSignal(str)
    done = pyqtSignal(str)      # interpreter path on success ('' if unknown)
    failed = pyqtSignal(str)    # error message

    def __init__(self, plugin_dir, target_exe=None, gpu=False):
        super().__init__()
        self.plugin_dir = plugin_dir
        # When set, install the deps INTO this existing interpreter instead of
        # building the managed venv (the "Choose interpreter…" path). Running
        # it here rather than inline keeps pip off the GUI thread, which is
        # what used to freeze QGIS for the whole install.
        self.target_exe = target_exe
        # requirements/gpu.txt (onnxruntime-gpu, ~2.4 GB) instead of
        # cpu.txt. Decided and CONFIRMED by the dialog before we get here —
        # a worker must never start a multi-gigabyte download on its own
        # initiative.
        self.gpu = bool(gpu)
        self._cancelled = False

    def cancel(self):
        """Best-effort cancel flag. The heavy steps (download, pip) aren't
        interruptible mid-call, but network reads are timeout-bounded, so the
        worker returns within a bounded time and teardown can wait it out."""
        self._cancelled = True

    def _emit(self, message):
        self.log.emit(str(message))

    def run(self):
        try:
            from .plugin_utils import installer
            if self.target_exe:
                installer.install_requirements_into(self.target_exe,
                                                    progress=self._emit,
                                                    gpu=self.gpu)
                if not installer._has_compute_deps(self.target_exe):
                    self.failed.emit(
                        f"{self.target_exe} still lacks the WINMOL "
                        "dependencies after the install.")
                    return
                self._verify_gpu(self.target_exe)
                self.done.emit(self.target_exe)
                return
            venv = installer.venv_location(self.plugin_dir)
            # download=False: the Setup tab owns every byte of model
            # traffic. Bundling a model fetch into the environment build
            # is what used to end a first-ever setup with a download
            # failure the user could neither retry nor understand.
            info = installer.setup_environment(
                venv, plugin_dir=self.plugin_dir, download=False,
                progress=self._emit, gpu=self.gpu)
            python = info.get("python") or ""
            self._verify_gpu(python)
            self.done.emit(python)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _verify_gpu(self, python_exe):
        """Prove the GPU runtime is a GPU runtime, in the CHILD.

        A GPU install that silently produces a CPU-only runtime is the
        exact bug this feature fixes, so the claim is never made on the
        strength of "pip exited 0". It is NOT promoted to ``failed``: the
        environment still works, just slowly, and failing would strand the
        user with no environment at all. The message says which it is.
        """
        if not self.gpu or not python_exe:
            return
        try:
            from .plugin_utils import installer
            installer.verify_gpu_runtime(python_exe,
                                         plugin_dir=self.plugin_dir,
                                         progress=self._emit)
        except Exception as exc:                   # pragma: no cover - env
            self._emit(f"GPU runtime verification could not run: {exc}")


class ModelDownloadWorker(QObject):
    """Fetches ONE registry model off the GUI thread via
    plugin_utils.model_registry.ensure_model (streaming download to a
    .part file, checksum verification, atomic rename). Emits progress
    percent for the dialog's progress bar, log lines, and done(path) /
    failed(message) at the end. Modeled on EnvSetupWorker; the dialog
    must apply the same quit/wait/park teardown discipline."""

    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(str)      # verified local model path
    failed = pyqtSignal(str)    # error message

    def __init__(self, entry, models_dir):
        super().__init__()
        self.entry = entry
        self.models_dir = models_dir
        self._cancelled = False

    def cancel(self):
        """Best-effort flag. Network reads are timeout-bounded (30 s per
        socket op in model_registry), so the worker returns within a
        bounded time and teardown can wait it out."""
        self._cancelled = True

    def run(self):
        try:
            from .plugin_utils import model_registry

            def _cb(done_bytes, total, _entry):
                if total:
                    self.progress.emit(
                        min(99, int(done_bytes * 100 / total)))

            self.log.emit(
                f"Downloading {self.entry.label} "
                f"({self.entry.file}) …")
            path = model_registry.ensure_model(
                self.entry, self.models_dir, progress=_cb)
            self.progress.emit(100)
            self.log.emit(f"Model ready: {path}")
            self.done.emit(path)
        except Exception as exc:
            self.failed.emit(str(exc))


class EnvRemoveWorker(QObject):
    """Deletes the managed environment off the GUI thread.

    An rmtree over a 2 GB venv on a network home directory takes seconds
    to minutes, and on Windows it may block on a file another process
    still holds — exactly the shape of freeze this whole change exists to
    remove. Signals mirror EnvSetupWorker so the dialog's existing
    quit/wait/park teardown covers it unchanged; ``done`` carries
    installer.remove_environment's result dict, NOT an interpreter path,
    which is why the dialog connects terminal slots per worker instance
    rather than per thread pair.
    """

    log = pyqtSignal(str)
    status = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(object)   # the remove_environment result dict
    failed = pyqtSignal(str)

    def __init__(self, plugin_dir, remove_venv=True, remove_runtime=False,
                 remove_models=False, configured_exe=None, dry_run=False):
        super().__init__()
        self.plugin_dir = plugin_dir
        self.remove_venv = remove_venv
        self.remove_runtime = remove_runtime
        self.remove_models = remove_models
        self.configured_exe = configured_exe
        # Pricing a deletion is a full directory walk over a 2 GB tree —
        # seconds on a network home directory. The confirmation dialog's
        # "frees N GB" figure therefore comes back through this worker
        # too, not from a walk on the GUI thread.
        self.dry_run = dry_run
        self._cancelled = False

    def cancel(self):
        """Best-effort flag. A single rmtree is not interruptible, but it
        is bounded, so teardown can wait it out."""
        self._cancelled = True

    def _emit(self, message):
        self.log.emit(str(message))

    def run(self):
        try:
            from .plugin_utils import installer
            result = installer.remove_environment(
                self.plugin_dir,
                remove_venv=self.remove_venv,
                remove_runtime=self.remove_runtime,
                remove_models=self.remove_models,
                configured_exe=self.configured_exe,
                dry_run=self.dry_run,
                progress=self._emit)
            result["dry_run"] = self.dry_run
            self.done.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class EnvProbeWorker(QObject):
    """Measures the compute environment off the GUI thread.

    ``setup_state.env_info`` spawns interpreters (a version probe and an
    ``import onnxruntime, rasterio, geopandas`` that costs over a second
    warm, and up to 60 s cold) and ``installer.directory_size`` walks a
    2 GB tree. Doing that inline is what re-froze the dialog on open and
    on every Rescan, so the whole measurement lives here and the dialog
    paints a cheap ``setup_state.env_seed`` until the result arrives.

    Read-only by construction: it creates, deletes and writes nothing, so
    unlike the other workers it does NOT take part in the one-job-at-a-
    time interlock and never disables a button. ``token`` is echoed back
    so the dialog can drop a result that a later change superseded.
    """

    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(object)   # {'token','info','usage','accel'}
    failed = pyqtSignal(str)

    def __init__(self, plugin_dir, configured_exe=None, token=0):
        super().__init__()
        self.plugin_dir = plugin_dir
        self.configured_exe = configured_exe
        self.token = token
        self._cancelled = False

    def cancel(self):
        """Best-effort flag. Every step is timeout-bounded (30 s / 60 s
        subprocess timeouts, a bounded walk), so teardown can wait it
        out."""
        self._cancelled = True

    def run(self):
        try:
            from .plugin_utils import installer, setup_state
            info = setup_state.env_info(self.plugin_dir, self.configured_exe)
            usage = {
                "venv": installer.directory_size(info.venv_path),
                "runtime": installer.directory_size(info.runtime_path),
            }
            # The accelerator verdict belongs here for the same reason
            # everything else does: it shells out to nvidia-smi and spawns
            # the child interpreter to list its execution providers.
            accel = setup_state.accelerator_from_machine(
                info.exe if info.exists else None,
                plugin_dir=self.plugin_dir)
            self.done.emit({"token": self.token, "info": info,
                            "usage": usage, "accel": accel})
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelMaintenanceWorker(QObject):
    """Verifies or deletes ONE model file off the GUI thread.

    Hashing a 374 MB file is seconds of solid CPU, and an unlink on an
    SMB home directory or a Windows-locked file blocks; neither belongs
    on the thread that paints the dialog. Rides the same thread pair as
    ModelDownloadWorker, so only one model operation runs at a time.
    """

    log = pyqtSignal(str)
    status = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(object)   # {'action','entry_id','ok','freed'}
    failed = pyqtSignal(str)

    def __init__(self, action, entry, models_dir):
        super().__init__()
        self.action = action        # "verify" | "delete"
        self.entry = entry
        self.models_dir = models_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from .plugin_utils import model_registry
            if self.action == "verify":
                self.done.emit(self._verify(model_registry))
            elif self.action == "delete":
                self.done.emit(self._delete(model_registry))
            else:
                self.failed.emit(f"Unknown action {self.action!r}.")
        except Exception as exc:
            self.failed.emit(str(exc))

    def _verify(self, model_registry):
        self.log.emit(f"Verifying {self.entry.file} …")

        def _cb(done_bytes, total):
            if total:
                self.progress.emit(min(99, int(done_bytes * 100 / total)))

        ok = model_registry.verify_entry(self.entry, self.models_dir,
                                         progress=_cb)
        self.progress.emit(100)
        self.log.emit(
            f"{self.entry.file}: checksum {'OK' if ok else 'MISMATCH'}.")
        return {"action": "verify", "entry_id": self.entry.id,
                "ok": ok, "freed": 0}

    def _delete(self, model_registry):
        self.log.emit(f"Deleting {self.entry.file} …")
        freed = model_registry.remove_model(self.entry, self.models_dir)
        self.progress.emit(100)
        self.log.emit(f"Deleted {self.entry.file} ({freed} bytes freed).")
        return {"action": "delete", "entry_id": self.entry.id,
                "ok": True, "freed": freed}
