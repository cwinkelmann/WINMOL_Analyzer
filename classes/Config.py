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
    # Tri-state, resolved by plugin_utils.autotune_cache.resolve_mode():
    # "auto" tunes ONCE per (hardware, model, execution provider, tile
    # geometry) and reuses the persisted answer from then on; True/"force"
    # always re-sweeps and refreshes the cache entry; False/"off" never
    # sweeps. $WINMOL_BATCH_AUTOTUNE overrides this attribute.
    # Precedence in utils.Prediction._autotune_batch_size:
    # prediction_batch_override > $WINMOL_BATCH_AUTOTUNE (or this attribute)
    # == "off" > a cache hit in range > sweep (then persist to the cache).
    prediction_batch_autotune = "auto"
    prediction_batch_autotune_patience = 4
    prediction_batch_autotune_repeats = 5
    prediction_batch_autotune_min_improve = 0.005
    # Absolute floor a candidate must beat (in addition to the relative
    # min_improve above) to count as progress -- kills jitter-chasing where
    # e.g. 0.337 vs 0.340 s/tile is noise, not a real win.
    prediction_batch_autotune_min_improve_s = 0.2
    prediction_batch_autotune_stop_on_oom = True
    prediction_batch_autotune_quiet = True
    # Share of FREE memory (host RAM, or free VRAM when CUDA is the active
    # provider) the sweep is allowed to spend; the rest is headroom for
    # everything else already resident (OS, raster reader, model runtime).
    prediction_batch_autotune_memory_fraction = 0.6
    # Fudge factor over the raw tensor bytes (H*W*(C+classes)*4) to account
    # for activations/workspace the runtime allocates per tile.
    prediction_batch_autotune_activation_factor = 32
    # Manual escape hatch: set to an int >= 1 to pin the prediction
    # micro-batch verbatim and skip the autotune sweep entirely (no probing,
    # no timing). None/0 = off (the sweep runs as usual). Env-overridable
    # via WINMOL_CONFIG_OVERRIDES_JSON, e.g.
    # WINMOL_CONFIG_OVERRIDES_JSON='{"prediction_batch_override": 4}'.
    prediction_batch_override = None
    progress_interval_s_cpu = 45.0
    progress_interval_s_gpu = 60.0
    progress_interval_s_multi_gpu = 20.0
    single_gpu_cpu_workers = 24
    multi_gpu_cpu_workers = 48

    # runtime state populated by planner
    # The detected HardwareInfo, so downstream code (e.g. the autotune
    # cache) can key on it without re-probing nvidia-smi.
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
