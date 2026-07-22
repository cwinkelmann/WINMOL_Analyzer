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
    # Separate ceiling for a cpu_stream plan. It used to share
    # prediction_batch_max_gpu, so a 4-core CPU box swept b1..b16 -- 16
    # candidates before the first tile was written.
    prediction_batch_max_cpu = 8
    prediction_batch_multi_gpu = 12     # local per-worker batch
    # Manual pin, in tiles. None (or 0) = let the planner size the batch and
    # let the autotune refine it. Any value >= 1 is used verbatim by
    # build_execution_plan and skips the autotune completely -- the escape
    # hatch for "just use this and start working". Exposed in the QGIS plugin
    # as Detection -> Prediction batch size, and settable from the CLI with
    # WINMOL_CONFIG_OVERRIDES_JSON='{"prediction_batch_override": 4}'.
    # NOTE: prediction_batch_size below is planner-owned and overwritten on
    # every run, so it can never serve as the pin.
    prediction_batch_override = None
    # Tri-state, resolved by plugin_utils.autotune_cache.resolve_mode():
    #   "auto"  -- use the persisted result if one matches this device+model,
    #              otherwise tune ONCE and save it (the default).
    #   False   -- never tune. Pin this for benchmarks and determinism runs.
    #   True    -- tune on every run and refresh the cache (force a re-tune).
    # Tuning costs real time before the first tile is written, and on
    # Apple/CoreML each distinct batch size forces a model recompile. Worth
    # paying once per device+model, never worth paying every run -- hence the
    # cache. $WINMOL_BATCH_AUTOTUNE (off|auto|force) overrides this without
    # touching config.
    prediction_batch_autotune = "auto"
    # --- how the sweep is bounded (safety first) ---
    # The sweep is capped by FREE memory before anything is timed: never ask
    # for more than this share of what is actually free right now. A user's
    # Linux box was taken down by the old unbounded sweep marching towards
    # b16 while six producer threads held prefetched tiles -- host RAM
    # exhaustion raises no exception, it just swaps until the OOM killer
    # fires. 0.6 leaves headroom for exactly those neighbours.
    prediction_batch_autotune_memory_fraction = 0.6
    # Device bytes per tile ~= img_h * img_w * (channels + classes) * 4 bytes
    # * this factor. Measured with the Spruce ONNX model on the CPU EP:
    # ~113 MB of working set per tile at b8 against a 4.19 MB raw tensor,
    # i.e. ~28x. 32 is that, rounded up.
    prediction_batch_autotune_activation_factor = 32
    # Hard cap on how many candidates may be timed, whatever the memory
    # ceiling allows.
    prediction_batch_autotune_max_candidates = 6
    # --- when the sweep stops ---
    prediction_batch_autotune_patience = 2
    prediction_batch_autotune_repeats = 3
    # A candidate counts as an improvement only if it is faster by BOTH a
    # relative margin and an absolute one. The absolute bar is what stops the
    # sweep chasing 3 ms differences: 0.337 vs 0.340 s/tile is noise, not a
    # reason to time another batch size.
    prediction_batch_autotune_min_improve = 0.005      # fraction
    prediction_batch_autotune_min_improve_s = 0.2      # seconds per tile
    # Abort immediately (regardless of patience) once a candidate is this
    # much slower than the best seen. Past the memory cliff the times explode
    # -- b7=0.471 then b9=1.135 against a best of 0.340 -- and marching on is
    # exactly what preceded the crash. 1.25 sits above run-to-run jitter
    # (~3 %) and well below the first real cliff (1.4x).
    prediction_batch_autotune_degrade_factor = 1.25
    prediction_batch_autotune_stop_on_oom = True
    prediction_batch_autotune_quiet = True
    progress_interval_s_cpu = 45.0
    progress_interval_s_gpu = 60.0
    progress_interval_s_multi_gpu = 20.0
    single_gpu_cpu_workers = 24
    multi_gpu_cpu_workers = 48

    # runtime state populated by planner
    # The detected HardwareInfo, so downstream code (e.g. the autotune cache
    # key) can see the GPUs without re-probing nvidia-smi. Not a user knob.
    hardware = None
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
