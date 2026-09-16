"""run_prediction_phase expands gpu_ids by workers_per_gpu -- two workers
per device on a single-GPU machine, unchanged behaviour everywhere else."""
import types
import winmol_run


def _capture(monkeypatch):
    calls = []

    def fake_run(*a, **k):
        calls.append(k['gpu_ids'])
        return {}
    monkeypatch.setattr('utils.PredictWorkers.run_multi_gpu_prediction',
                        fake_run)
    monkeypatch.setattr(winmol_run, '_release_prediction_memory',
                        lambda *a, **k: None)
    return calls


def _proc():
    p = winmol_run.ImageProcessing.__new__(winmol_run.ImageProcessing)
    p.model_path = 'm.onnx'
    p.uav_path = 'in.tif'
    p.stem_path = 'out.tif'
    p.config = types.SimpleNamespace(prediction_backend='auto')
    return p


def _plan(mode, gpu_workers, workers_per_gpu):
    return types.SimpleNamespace(
        prediction_mode=mode, gpu_workers=gpu_workers,
        workers_per_gpu=workers_per_gpu, producer_workers=3)


def test_single_gpu_two_workers_expands_to_0_0(monkeypatch):
    calls = _capture(monkeypatch)
    _proc().run_prediction_phase(_plan('multi_gpu', 1, 2))
    assert calls == [[0, 0]]


def test_multi_gpu_one_worker_each_is_unchanged(monkeypatch):
    calls = _capture(monkeypatch)
    _proc().run_prediction_phase(_plan('multi_gpu', 3, 1))
    assert calls == [[0, 1, 2]]


def test_multi_gpu_two_workers_each_interleaves_by_device(monkeypatch):
    calls = _capture(monkeypatch)
    _proc().run_prediction_phase(_plan('multi_gpu', 2, 2))
    assert calls == [[0, 0, 1, 1]]


def test_cpu_stream_stays_a_single_none_worker(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr('utils.onnx_runtime.selected_providers',
                        lambda: ['CPUExecutionProvider'])
    _proc().run_prediction_phase(_plan('cpu_stream', 0, 2))
    assert calls == [[None]]
