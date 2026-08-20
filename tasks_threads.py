"""GUI-thread-off workers for the WINMOL plugin: running the compute
child process, building/removing the compute environment, and bulk
model maintenance.
"""
import os
import subprocess
import sys

from PyQt5.QtCore import (
    QObject,
    pyqtSignal,
)

from .plugin_utils.childenv import child_env, safe_child_cwd
from .plugin_utils.installer import (
    _run_streamed,
    gpu_requested,
    plugin_requirements_path,
    remove_environment,
    remove_model_files,
    setup_environment,
    uninstall_conflicting_runtime,
    venv_location,
    verify_gpu_providers,
)
from .plugin_utils.model_registry import (
    ensure_model,
    load_registry,
    local_path,
    verify_file,
)


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
                env=child_env(self.env_extra,
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
    """Builds the WINMOL compute environment off the GUI thread:
    creates the managed venv and pip-installs requirements/cpu.txt —
    or gpu.txt for the CUDA runtime (idempotent via the sentinel, see
    plugin_utils/installer.py). Keeps QGIS responsive during a
    multi-minute first-run install.

    ``gpu`` selects the runtime variant: ``None`` (default) honors the
    WINMOL_GPU env var (``installer.gpu_requested``), the pre-Setup-tab
    opt-in; ``True``/``False`` is the Setup tab's explicit choice
    (Install GPU runtime / repair-preserving-variant).

    ``target_exe`` redirects the install into a bring-your-own
    interpreter instead of the managed venv: no venv is created, no
    sentinel written — just the requirements pip-installed into THAT
    python, so the runtime lands in the same environment the detection
    actually runs in. ``done`` then carries ``target_exe`` back."""

    log = pyqtSignal(str)
    done = pyqtSignal(str)      # interpreter path on success
    failed = pyqtSignal(str)    # error message

    def __init__(self, plugin_dir, gpu=None, target_exe=None):
        super().__init__()
        self.plugin_dir = plugin_dir
        self.gpu = gpu
        self.target_exe = target_exe

    def run(self):
        gpu = gpu_requested() if self.gpu is None else bool(self.gpu)
        try:
            if self.target_exe:
                _install_into_interpreter(
                    self.target_exe, progress=self.log.emit, gpu=gpu)
                self.done.emit(self.target_exe)
                return
            info = setup_environment(
                venv_location(self.plugin_dir), progress=self.log.emit,
                gpu=gpu)
            self.done.emit(info["python"])
        except Exception as exc:
            self.failed.emit(str(exc))


def _install_into_interpreter(python_exe, progress, gpu=False):
    """pip-install requirements/cpu.txt (or gpu.txt) into a
    user-picked interpreter — ``installer.install_requirements``'s
    exact flow with the interpreter given directly instead of derived
    from the managed venv. Same flags for the same reasons:
    ``--no-input`` prevents a hidden prompt, ``--progress-bar off``
    stops the \\r spam a QPlainTextEdit cannot render, and the
    conflicting onnxruntime distribution is removed FIRST — a swap,
    never an addition (both distributions provide the ``onnxruntime``
    module, pip will never resolve that itself)."""
    uninstall_conflicting_runtime(python_exe, gpu, progress=progress)
    req = str(plugin_requirements_path(gpu=gpu))
    progress(f"Installing packages from {os.path.basename(req)} into "
             f"{python_exe} — this downloads a few hundred MB and can "
             "take several minutes …")
    _run_streamed(
        [python_exe, "-u", "-m", "pip", "install", "--upgrade",
         "--no-input", "--progress-bar", "off", "-r", req],
        progress=progress,
        label=f"pip install -r {os.path.basename(req)}", timeout=3600)
    if gpu:
        # The install-time provider verdict, same as the managed GPU
        # build: a CUDA-generation mismatch installs fine and only
        # fails at runtime, so say what the interpreter actually got.
        progress(verify_gpu_providers(python_exe))


class ModelEnsureWorker(QObject):
    """Resolves one registry model entry to a verified local file off the
    GUI thread, downloading it if missing/stale (model_registry.
    ensure_model, which is itself idempotent -- a verified-existing file
    short-circuits, so calling this on every run is cheap)."""

    log = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100 download percent (only with a total)
    done = pyqtSignal(str)      # local model file path on success
    failed = pyqtSignal(str)    # error message

    def __init__(self, entry, model_dir):
        super().__init__()
        self.entry = entry
        self.model_dir = model_dir

    def run(self):
        last_pct = [-1]

        def report(done, total, entry):
            # ensure_model -> download_model's callback contract:
            # (bytes_done, bytes_total, entry); total is 0/None when the
            # server sent no Content-Length — then there is no percent
            # and the bar stays indeterminate.
            if not total:
                return
            pct = min(100, done * 100 // total)
            if pct == last_pct[0]:
                return
            last_pct[0] = pct
            self.progress.emit(int(pct))
            if pct % 10 == 0:
                self.log.emit(f"Downloading {entry.label}: {pct}%")

        try:
            path = ensure_model(self.entry, self.model_dir, progress=report)
            self.done.emit(path)
        except Exception as exc:
            self.failed.emit(str(exc))


class EnvRemoveWorker(QObject):
    """Prices and removes the managed environment off the GUI thread
    (installer.remove_environment, which refuses anything outside the
    managed tree). Always emits ``priced`` first — the SAME code path
    that deletes produces the "frees N GB" figure — then, unless
    constructed with ``dry_run=True``, performs the removal."""

    log = pyqtSignal(str)
    priced = pyqtSignal(dict)   # remove_environment(dry_run=True) result
    done = pyqtSignal(dict)     # remove_environment() result
    failed = pyqtSignal(str)    # unexpected error message

    def __init__(self, plugin_dir, remove_venv=True, remove_runtime=False,
                 remove_models=False, configured_exe=None, dry_run=False):
        super().__init__()
        self.plugin_dir = plugin_dir
        self.dry_run = dry_run
        self.kwargs = dict(remove_venv=remove_venv,
                           remove_runtime=remove_runtime,
                           remove_models=remove_models,
                           configured_exe=configured_exe)

    def run(self):
        try:
            plan = remove_environment(self.plugin_dir, dry_run=True,
                                      **self.kwargs)
            self.priced.emit(plan)
            if self.dry_run:
                return
            result = remove_environment(self.plugin_dir,
                                        progress=self.log.emit,
                                        **self.kwargs)
            self.done.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelMaintenanceWorker(QObject):
    """Bulk model maintenance off the GUI thread. ``action`` is one of
    ``download-all-recommended`` (ensure_model for the registry's
    recommended list), ``verify-all`` (checksum pinned files on disk —
    all of them, or only ``entry_ids`` when given) or ``delete``
    (remove ``entry_ids``' files under models_dir).
    Emits ``done({'action', 'ok': [id], 'failed': [(id, msg)]})``."""

    log = pyqtSignal(str)
    done = pyqtSignal(dict)     # summary
    failed = pyqtSignal(str)    # error message

    ACTIONS = ("download-all-recommended", "verify-all", "delete")

    def __init__(self, action, config_path, models_dir, entry_ids=None,
                 device="auto"):
        super().__init__()
        self.action = action
        self.config_path = config_path
        self.models_dir = models_dir
        self.entry_ids = list(entry_ids or [])
        self.device = device

    def run(self):
        try:
            if self.action not in self.ACTIONS:
                raise ValueError(f"unknown action: {self.action}")
            registry = load_registry(self.config_path)
            handler = {"download-all-recommended": self._download_all,
                       "verify-all": self._verify_all,
                       "delete": self._delete}[self.action]
            self.done.emit(handler(registry))
        except Exception as exc:
            self.failed.emit(str(exc))

    def _summary(self, ok, failed):
        return {"action": self.action, "ok": ok, "failed": failed}

    def _recommended(self, registry):
        """Device-resolved recommended entries, deduplicated."""
        ids = registry.recommended or []
        entries, seen = [], set()
        for mid in ids:
            entry = registry.resolve(mid, device=self.device)
            if entry.id not in seen:
                seen.add(entry.id)
                entries.append(entry)
        return entries or [registry.default_entry(self.device)]

    def _download_all(self, registry):
        ok, failed = [], []
        for entry in self._recommended(registry):
            self.log.emit(f"Fetching {entry.label} …")
            try:
                ensure_model(entry, self.models_dir)
                ok.append(entry.id)
            except Exception as exc:
                failed.append((entry.id, str(exc)))
        return self._summary(ok, failed)

    def _verify_all(self, registry):
        """Checksum pinned files on disk — every entry, or only
        ``entry_ids`` when given (the Verify button's per-selected-row
        scope)."""
        ok, failed = [], []
        entries = ([registry.get(eid) for eid in self.entry_ids]
                   if self.entry_ids else list(registry.entries.values()))
        for entry in entries:
            path = local_path(entry, self.models_dir)
            if not os.path.exists(path) or not entry.sha256:
                continue
            self.log.emit(f"Verifying {entry.file} …")
            if verify_file(path, entry.sha256):
                ok.append(entry.id)
            else:
                failed.append((entry.id, "checksum mismatch"))
        return self._summary(ok, failed)

    def _delete(self, registry):
        # installer.remove_model_files owns the deletion, including the
        # containment refusal for a registry ``file`` that escapes the
        # models directory — one owner, shared with remove_environment.
        ok, failed = [], []
        for eid in self.entry_ids:
            try:
                result = remove_model_files(
                    registry.get(eid), self.models_dir)
                if result["failed"]:
                    failed.append((eid, result["failed"][0][1]))
                else:
                    ok.append(eid)
            except Exception as exc:
                failed.append((eid, str(exc)))
        return self._summary(ok, failed)
