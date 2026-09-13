"""Run prediction in a child process that EXITS before vectorisation.

Why a separate process and not a call
--------------------------------------
A CUDA onnxruntime session maps ~10 GB of virtual address space, and the
driver keeps that reservation for the life of the process: measured on
R13, VmSize 10.3 GB after one inference, still 9.3 GB after close() and
cudaDeviceReset(). RSS is small, but the kernel's overcommit accounting
counts committed virtual, and CommitLimit is only RAM/2 + swap. On the
46 GB dev box a single session lifted Committed_AS from 19.4 to 24.0 GB
against a 25.4 GB limit -- 95% -- before the vector phase allocated a
byte. The tile-split loop that follows was then killed nine times in a
row, in a process state it could never leave, while the same loop in a
fresh process wrote 78 GB and survived.

On smaller machines it is worse, not merely tight: CommitLimit on a 16 GB
host is ~10 GB, which a 10 GB session exceeds on its own.

Exiting the process is the only thing that returns the commit. So
prediction runs here, in a spawned child, and the parent -- which never
imports the CUDA runtime -- vectorises after the child is gone. The stem
map crosses the boundary as a file, exactly as it always did, so the
output is bit-identical by construction.

`spawn`, not fork: a forked child would inherit the parent's mappings and
the CUDA context is not fork-safe anyway (PredictWorkers has the same
rule for the same reason).
"""
import multiprocessing as mp
import traceback


def _child(model_path, uav_path, stem_path, config_dict, result_q):
    try:
        from classes.Config import Config
        from utils import IO
        from utils import Prediction as Pred
        from utils.onnx_runtime import last_active_report

        # Rehydrate exactly as PredictWorkers._config_from_dict does: a
        # key that will not set (a property, a derived field) is skipped,
        # not fatal.
        config = Config()
        for key, value in config_dict.items():
            try:
                setattr(config, key, value)
            except Exception:
                pass

        print("\nLoading Model...", flush=True)
        model = IO.load_model_from_path(model_path, config)
        report = last_active_report()
        if report:
            print(
                f"Execution providers (active): "
                f"{report['active_providers']} "
                f"(device: {report['accelerator_label']})", flush=True)
        print("\nPerforming Prediction with Resampling in stream mode...",
              flush=True)
        profile = Pred.predict_stream_to_raster(
            uav_path, stem_path, model, config)
        result_q.put(('ok', dict(profile)))
    except BaseException:                            # surfaced by the parent
        result_q.put(('error', traceback.format_exc()))


def predict_in_child(model_path, uav_path, stem_path, config):
    """Run the single-GPU streamed prediction in a spawned child and
    return the output profile once the child has exited.

    Raises RuntimeError carrying the child's traceback on failure, so a
    prediction error ends the run the same way it did in-process.
    """
    ctx = mp.get_context('spawn')
    result_q = ctx.Queue(maxsize=1)
    cfg_dict = dict(getattr(config, 'to_dict', lambda: {})())
    child = ctx.Process(
        target=_child,
        args=(model_path, uav_path, stem_path, cfg_dict, result_q),
        name='winmol-prediction',
    )
    child.start()
    # Read BEFORE join: a large profile could otherwise fill the pipe and
    # deadlock against a child blocked on put().
    try:
        status, payload = result_q.get()
    finally:
        child.join()
    if status != 'ok':
        raise RuntimeError(
            "prediction child process failed:\n" + str(payload))
    if child.exitcode not in (0, None):
        raise RuntimeError(
            f"prediction child exited with code {child.exitcode}")
    return payload
