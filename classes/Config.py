class Config(object):
    # execution / backend selection
    prediction_backend = "auto"          # auto | cpu | single_gpu | multi_gpu

    # tiled stream production pipeline
    tile_inner_px = 4096
    tile_overlap_m = 12.0
    prediction_prefetch = 2
    producer_queue_batches = 4
    prediction_producer_workers_cpu = 1
    prediction_producer_workers_gpu = 6
    prediction_producer_workers_multi_gpu = 6

    # resources
    max_gpu_workers = 8
    max_cpu_workers = 32
    gpu_memory_fraction = 0.9

    # runtime worker / batching knobs
    prediction_batch_cpu = 1
    prediction_batch_gpu = 4
    prediction_batch_max_gpu = 16
    prediction_batch_multi_gpu = 12     # local per-worker batch
    # Off by default: the autotune times ~6 batch sizes x 5 repeats before the
    # first tile, and on Apple/CoreML each distinct batch size forces a model
    # recompile -> a long SILENT stall for ~1% throughput. Enable via
    # WINMOL_CONFIG_OVERRIDES_JSON on CUDA if you want it.
    prediction_batch_autotune = False
    prediction_batch_autotune_patience = 4
    prediction_batch_autotune_repeats = 5
    prediction_batch_autotune_min_improve = 0.005
    prediction_batch_autotune_stop_on_oom = True
    prediction_batch_autotune_quiet = True
    progress_interval_s_cpu = 45.0
    progress_interval_s_gpu = 60.0
    progress_interval_s_multi_gpu = 20.0
    single_gpu_cpu_workers = 24
    multi_gpu_cpu_workers = 48

    # runtime state populated by planner
    cpu_workers = None
    gpu_workers = None
    vector_mode = 'none'
    vector_tile_workers = 1
    max_vector_tile_workers = 4
    prediction_batch_size = None
    prediction_producer_workers = None
    progress_interval_s = 30.0

    # behavior
    stream_prediction = True
    keep_temp_tiles = False
    compress_output = True

    # logging / diagnostics
    # log_level drives utils/Log.py: 'quiet' (warnings/errors only),
    # 'normal' (phases, progress, summaries, results) or 'debug' (adds the
    # per-tile MERGE/VECTOR diagnostics). Overridable per run through
    # WINMOL_LOG_LEVEL / WINMOL_VERBOSE=1, which also survive the spawned
    # vector-tile pool. vector_debug=True still implies 'debug'.
    log_level = "normal"                 # quiet | normal | debug
    prediction_tile_log = True
    vector_debug = False
    vector_summary_log = True

    # semantic segmentation
    tile_size = 15
    img_width = 512
    img_height = 512
    img_bit = 8
    n_channels = 3
    num_classes = 1
    overlap_pred = 8

    # binary stem-map prediction
    stem_map_binary = True
    stem_binary_threshold = 0.5

    # stem vectorization
    min_length = 2.0
    max_distance = 8
    max_tree_height = 32
    tolerance_angle = 7

    # measuring points / diameter estimation
    measuring_point_spacing_m = 0.5
    diameter_method = "contour"         # contour | edt
    diameter_vector_half_length_m = 1.0
    edt_clip_max_m = None                # optional clip for extreme EDT radii

    def __init__(self):
        pass

    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}

    def display(self):
        print("\nConfigurations:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        print("\n")
