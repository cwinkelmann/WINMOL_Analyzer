## QGIS plugin

### env
the env setup isn't working, I want to install the plugin, then when opening the plugin I want menu where I can select an environment on disk (venv, or conda) or let one create

✅ FIXED — in-dialog "Environment…" picker (Choose interpreter… / Create for me),
now also a **permanent Environment… button** in a top bar showing the active
interpreter. Applies live via QgsSettings, no QGIS restart.

### plugin uninstall failed
Couldn't delete the plugin from the QGIS Plugin Manager.

✅ FIXED — cause: a half-built `winmol_venv` (with broken python symlinks) lived
INSIDE the plugin directory, so QGIS's uninstall (a recursive delete) tripped
over it. The auto-created venv now lives OUTSIDE the plugin dir, under the QGIS
profile (`<profile>/winmol/winmol_venv`), so uninstall never touches it. The
stale leftover was removed manually.

## input dialoge, 
currently it says "Input UAV" Which is weird, because that should be a geotiff, having a typo in a 2 year old plugin is not a good sign

✅ FIXED — label is now "Input GeoTiff".

The "Input GeoTiff" should allow to select a layer in QGIS too.

✅ FIXED — added a QgsMapLayerComboBox ("Loaded layer:") that lists loaded
raster layers; picking one fills the input path from its source.

## after run
for a while nothings happens, then I get a message in the logs "Starting the process...
Traceback (most recent call last):
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/winmol_run.py", line 15, in <module>
    from utils import IO
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/utils/IO.py", line 69, in <module>
    transform, compress: str | None = 'DEFLATE',
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
Analysis failed (exit code 1). See the log above for details."

✅ FIXED — the code uses PEP 604 unions (`str | None`) and needs Python ≥ 3.10.
The installer now requires 3.10+ and rejects the macOS system 3.9. Confirmed:
the next run came up under the conda Python 3.11 env instead.


## Bug
Starting the process...
Start timer
Initialization
Hardware detected: CPUs=8, RAM=24.0 GB, GPUs=1
Visible GPUs: ['Apple Silicon GPU (Metal)']
Traceback (most recent call last):
  File "rasterio/_base.pyx", line 311, in rasterio._base.DatasetBase.__init__
  File "rasterio/_base.pyx", line 222, in rasterio._base.open_dataset
  File "rasterio/_err.pyx", line 359, in rasterio._err.exc_wrap_pointer
rasterio._err.CPLE_OpenFailedError: : No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/winmol_run.py", line 373, in <module>
    plan = image_processor.build_plan(hardware)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/winmol_run.py", line 119, in build_plan
    raster_info = IO.get_raster_info(self.uav_path)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/utils/IO.py", line 39, in get_raster_info
    with rasterio.open(path) as src:
         ^^^^^^^^^^^^^^^^^^^
  File "/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/lib/python3.11/site-packages/rasterio/env.py", line 463, in wrapper
    return f(*args, **kwds)
           ^^^^^^^^^^^^^^^^
  File "/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/lib/python3.11/site-packages/rasterio/__init__.py", line 356, in open
    dataset = DatasetReader(path, driver=driver, sharing=sharing, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "rasterio/_base.pyx", line 313, in rasterio._base.DatasetBase.__init__
rasterio.errors.RasterioIOError: : No such file or directory
Analysis failed (exit code 1). See the log above for details.

✅ FIXED — the input path was EMPTY (`: No such file...` with nothing before the
colon). `self.uav_path` was only captured by the file-picker callback and never
re-read from the input field at run time, so a typed path (or one that got out
of sync) was ignored. Now the input path is re-read from the field on every run
and validated (clear warning if empty/missing) before the subprocess launches.


### inference fails
Starting the process...
Start timer
Initialization
Hardware detected: CPUs=8, RAM=24.0 GB, GPUs=1
Visible GPUs: ['Apple Silicon GPU (Metal)']
Execution plan:
  process_type     = Stems
  prediction_mode  = stream
  vector_mode      = none
  tile_inner_px    = 4096
  tile_overlap_m   = 12.0
  halo_px          = 582
  gpu_workers      = 1
  cpu_workers      = 7
  vector_tile_workers = 1
  vector_inner_workers = 7
  prediction_batch = 4
  queue_batches    = 4
  producer_workers = 2
  progress_interval_s = 60.0
  est_pred_tiles   = 182
Enabled memory growth for 1 GPU(s).
Check CUDA environment
No NVIDIA GPU available or drivers not installed.
CUDA is available: Unknown
cuDNN version: Unknown
Num GPUs for CUDA processing: 1
Tensorflow version: 2.16.2
Keras version: 3.15.0
Command-line arguments:
Model Path: /Volumes/storage/hnee/WINMOL/models/r_keras/singlestage_SpecDS_ready.hdf5
Image Path: /Users/christian/data/Winmol/Winmol Orthos/orthomosaics_storm_CW/20220212_Barnekow_4.tiff
Semantic Stem Map Path: 
Process type: Stems

Configurations:
compress_output                True
cpu_workers                    7
diameter_method                contour
diameter_vector_half_length_m  1.0
edt_clip_max_m                 None
gpu_memory_fraction            0.9
gpu_workers                    1
img_bit                        8
img_height                     512
img_width                      512
keep_temp_tiles                False
max_cpu_workers                32
max_distance                   8
max_gpu_workers                8
max_tree_height                32
max_vector_tile_workers        4
measuring_point_spacing_m      0.5
min_length                     2.0
multi_gpu_cpu_workers          48
n_channels                     3
num_classes                    1
overlap_pred                   8
prediction_backend             auto
prediction_batch_autotune      True
prediction_batch_autotune_min_improve 0.005
prediction_batch_autotune_patience 4
prediction_batch_autotune_quiet True
prediction_batch_autotune_repeats 5
prediction_batch_autotune_stop_on_oom True
prediction_batch_cpu           1
prediction_batch_gpu           4
prediction_batch_max_gpu       16
prediction_batch_multi_gpu     12
prediction_batch_size          4
prediction_prefetch            2
prediction_producer_workers    2
prediction_producer_workers_cpu 1
prediction_producer_workers_gpu 6
prediction_producer_workers_multi_gpu 6
prediction_tile_log            True
producer_queue_batches         4
progress_interval_s            60.0
progress_interval_s_cpu        45.0
progress_interval_s_gpu        60.0
progress_interval_s_multi_gpu  20.0
single_gpu_cpu_workers         24
stem_binary_threshold          0.5
stem_map_binary                True
stream_prediction              True
tile_inner_px                  4096
tile_overlap_m                 12.0
tile_size                      15
tolerance_angle                7
vector_debug                   False
vector_mode                    none
vector_summary_log             True
vector_tile_workers            1



Loading Model...
Trying to load model using open_model()
open_model() failed: Error when deserializing class 'Dropout' using config={'name': 'dropout', 'trainable': True, 'dtype': 'float32', 'rate': 0.1, 'noise_shape': None, 'seed': 1.0}.

Exception encountered: Argument `seed` must be an integer. Received: seed=1.0
Retrying with custom layers (Dropout, Conv2DTranspose)
2026-07-17 14:58:23.486154: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2
2026-07-17 14:58:23.486344: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 24.00 GB
2026-07-17 14:58:23.486357: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 8.00 GB
2026-07-17 14:58:23.486654: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2026-07-17 14:58:23.486672: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)

Performing Prediction with Resampling in stream mode...
#######################################################
Prediction of the semantic stem map
Resampling tiles while analyzing (stream mode)
Traceback (most recent call last):
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/winmol_run.py", line 383, in <module>
    image_processor.run_stem_pipeline(plan)
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/winmol_run.py", line 281, in run_stem_pipeline
    self.run_prediction_phase(plan)
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/winmol_run.py", line 179, in run_prediction_phase
    profile = Pred.predict_stream_to_raster(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/utils/Prediction.py", line 546, in predict_stream_to_raster
    tmp_path = IO.atomic_tmp_path(output_stem_map)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/utils/IO.py", line 59, in atomic_tmp_path
    return str(p.with_suffix(p.suffix + '.tmp'))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/lib/python3.11/pathlib.py", line 694, in with_suffix
    raise ValueError("%r has an empty name" % (self,))
ValueError: PosixPath('.') has an empty name
Analysis failed (exit code 1). See the log above for details.


and 

Starting the process...
Start timer
Initialization
Hardware detected: CPUs=8, RAM=24.0 GB, GPUs=1
Visible GPUs: ['Apple Silicon GPU (Metal)']
Execution plan:
  process_type     = Stems
  prediction_mode  = stream
  vector_mode      = none
  tile_inner_px    = 4096
  tile_overlap_m   = 12.0
  halo_px          = 582
  gpu_workers      = 1
  cpu_workers      = 7
  vector_tile_workers = 1
  vector_inner_workers = 7
  prediction_batch = 4
  queue_batches    = 4
  producer_workers = 2
  progress_interval_s = 60.0
  est_pred_tiles   = 182
Enabled memory growth for 1 GPU(s).
Check CUDA environment
No NVIDIA GPU available or drivers not installed.
CUDA is available: Unknown
cuDNN version: Unknown
Num GPUs for CUDA processing: 1
Tensorflow version: 2.16.2
Keras version: 3.15.0
Command-line arguments:
Model Path: /Volumes/storage/hnee/WINMOL/models/pytorch/singlestage_specds_only/deeplabv3plus.onnx
Image Path: /Users/christian/data/Winmol/Winmol Orthos/orthomosaics_storm_CW/20220212_Barnekow_4.tiff
Semantic Stem Map Path: 
Process type: Stems

Configurations:
compress_output                True
cpu_workers                    7
diameter_method                contour
diameter_vector_half_length_m  1.0
edt_clip_max_m                 None
gpu_memory_fraction            0.9
gpu_workers                    1
img_bit                        8
img_height                     512
img_width                      512
keep_temp_tiles                False
max_cpu_workers                32
max_distance                   8
max_gpu_workers                8
max_tree_height                32
max_vector_tile_workers        4
measuring_point_spacing_m      0.5
min_length                     2.0
multi_gpu_cpu_workers          48
n_channels                     3
num_classes                    1
overlap_pred                   8
prediction_backend             auto
prediction_batch_autotune      True
prediction_batch_autotune_min_improve 0.005
prediction_batch_autotune_patience 4
prediction_batch_autotune_quiet True
prediction_batch_autotune_repeats 5
prediction_batch_autotune_stop_on_oom True
prediction_batch_cpu           1
prediction_batch_gpu           4
prediction_batch_max_gpu       16
prediction_batch_multi_gpu     12
prediction_batch_size          4
prediction_prefetch            2
prediction_producer_workers    2
prediction_producer_workers_cpu 1
prediction_producer_workers_gpu 6
prediction_producer_workers_multi_gpu 6
prediction_tile_log            True
producer_queue_batches         4
progress_interval_s            60.0
progress_interval_s_cpu        45.0
progress_interval_s_gpu        60.0
progress_interval_s_multi_gpu  20.0
single_gpu_cpu_workers         24
stem_binary_threshold          0.5
stem_map_binary                True
stream_prediction              True
tile_inner_px                  4096
tile_overlap_m                 12.0
tile_size                      15
tolerance_angle                7
vector_debug                   False
vector_mode                    none
vector_summary_log             True
vector_tile_workers            1



Loading Model...
✅ FIXED (both runs above) — note the header "Semantic Stem Map Path:" is EMPTY.
The MODEL loaded fine both times (the Keras .hdf5 retried past the Dropout seed
quirk and set up Metal; the deeplabv3plus.onnx loaded via CoreML). The crash was
the empty OUTPUT stem-map path: it was only auto-filled when the input was chosen
via the file picker, so typing / using the layer selector left it empty, and ""
flowed into IO.atomic_tmp_path -> Path("").with_suffix() -> "PosixPath('.') has
an empty name". Now the output path is auto-derived from the input at run time
(and shown in the field), validated before launch, and IO.atomic_tmp_path raises
a clear "No output stem-map path" error for CLI callers.

Loading ONNX model via OnnxSegmenter: /Volumes/storage/hnee/WINMOL/models/pytorch/singlestage_specds_only/deeplabv3plus.onnx
2026-07-17 14:59:21.774 python3.11[60321:11175241] 2026-07-17 14:59:21.774863 [W:onnxruntime:, coreml_execution_provider.cc:137 GetCapability] CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 4 number of nodes in the graph: 126 number of nodes supported by CoreML: 119

Performing Prediction with Resampling in stream mode...
#######################################################
Prediction of the semantic stem map
Resampling tiles while analyzing (stream mode)
Traceback (most recent call last):
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/winmol_run.py", line 383, in <module>
    image_processor.run_stem_pipeline(plan)
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/winmol_run.py", line 281, in run_stem_pipeline
    self.run_prediction_phase(plan)
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/winmol_run.py", line 179, in run_prediction_phase
    profile = Pred.predict_stream_to_raster(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/utils/Prediction.py", line 546, in predict_stream_to_raster
    tmp_path = IO.atomic_tmp_path(output_stem_map)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/utils/IO.py", line 59, in atomic_tmp_path
    return str(p.with_suffix(p.suffix + '.tmp'))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/lib/python3.11/pathlib.py", line 694, in with_suffix
    raise ValueError("%r has an empty name" % (self,))
ValueError: PosixPath('.') has an empty name
Analysis failed (exit code 1). See the log above for details.


### Loading Mode from network storage
Trying to load model using open_model()
open_model() failed: Unable to synchronously open file (truncated file: eof = 113836032, sblock->base_addr = 0, stored_eof = 373951004)
Retrying with custom layers (Dropout, Conv2DTranspose)
Loading with custom layers also failed: Unable to synchronously open file (truncated file: eof = 113836032, sblock->base_addr = 0, stored_eof = 373951004)
Traceback (most recent call last):
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/winmol_run.py", line 383, in <module>
    image_processor.run_stem_pipeline(plan)
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/winmol_run.py", line 281, in run_stem_pipeline
    self.run_prediction_phase(plan)
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/winmol_run.py", line 177, in run_prediction_phase
    model = IO.load_model_from_path(self.model_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/utils/IO.py", line 234, in load_model_from_path
    raise RuntimeError("Failed to load model with all methods.")
RuntimeError: Failed to load model with all methods.
Analysis failed (exit code 1). See the log above for details.

⚠️ ENVIRONMENTAL (not a code bug) — "truncated file: eof=113836032, stored_eof=
373951004" means the .hdf5 on /Volumes network storage was only partially
read (~108 MB of a 356 MB file) — a network-mount read hiccup, not the loader.
The error surfacing IS improved: load_model_from_path now raises
"Failed to load Keras model <path>: <real error>" instead of the generic
"all methods" message. Best fix: copy the model to a LOCAL disk, or use the
bundled .onnx models (no network read). Not reproducible with a local file.

### loading an older model which is published

Loading Model...
Trying to load model using open_model()
open_model() failed: Error when deserializing class 'Dropout' using config={'name': 'dropout_18', 'trainable': True, 'dtype': 'float32', 'rate': 0.1, 'noise_shape': None, 'seed': 1.0}.

Exception encountered: Argument `seed` must be an integer. Received: seed=1.0
Retrying with custom layers (Dropout, Conv2DTranspose)
2026-07-17 15:21:13.275282: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2
2026-07-17 15:21:13.275337: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 24.00 GB
2026-07-17 15:21:13.275352: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 8.00 GB
2026-07-17 15:21:13.275376: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2026-07-17 15:21:13.275400: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)

Performing Prediction with Resampling in stream mode...
#######################################################
Prediction of the semantic stem map
Resampling tiles while analyzing (stream mode)

✅ FIXED — this is the Dropout(seed=1.0) Keras-2/Keras-3 issue. The loader now
registers the tolerant Dropout/Conv2DTranspose shims BEFORE the (single) load,
so it loads cleanly on the first try — no "open_model() failed" / retry noise.
(As the log shows, the retry already worked; the fix just removes the alarming
first-attempt failure.)

### High read IO at beginning of prediction
Performing Prediction with Resampling in stream mode...
#######################################################
Prediction of the semantic stem map
Resampling tiles while analyzing (stream mode)

after a while we have this:
Performing Prediction with Resampling in stream mode...
#######################################################
Prediction of the semantic stem map
Resampling tiles while analyzing (stream mode)
Prediction micro-batch autotune: b4=0.805s/tile, b5=0.798s/tile, b6=0.878s/tile, b7=1.374s/tile, b8=1.266s/tile, b9=1.207s/tile -> selected 5 (stopped after 4 non-improving step(s))
Written tile 5/182 | 2.7% | 1.1 tiles/min | ETA 2h 43m 44s | avg read 0.097s prep 0.691s infer 0.620s write 0.002s | batch 5 | queue 100% full | producers 2 | src 727x727 -> out 504x504
Written tile 65/182 | 35.7% | 11.4 tiles/min | ETA 10m 15s | avg read 0.033s prep 0.782s infer 0.307s write 0.000s | batch 5 | queue 100% full | producers 2 | src 727x727 -> out 504x504
Written tile 120/182 | 65.9% | 17.9 tiles/min | ETA 03m 28s | avg read 0.034s prep 0.798s infer 0.297s write 0.000s | batch 5 | queue 100% full | producers 2 | src 727x727 -> out 504x504


60 stem segments analyzed
0 stem segments appended to other stems
30 duplicates are removed
0 stem fragments with a length less than  2.0 m are filtered out
final number of stems 30
Elapsed time: 0.0326 seconds
#######################################################

MERGE EDGE CONNECT | inner 534 | connected_edge 30 | final 564

MERGE SUMMARY
Tiles processed:       13
Total stems written:   564
Total nodes written:   4998
Total vectors written: 4998
Layers written:        ['stems', 'nodes', 'vectors']
Output saved to: /Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/winmol/tmp/repr_20240719_FR17203_Abt-128_129_30_SELEKTION_1c527804_detected_stems.gpkg
Stop timer
Elapsed time: 697.8100 seconds
Outputs were written to a temp folder. Use 'Export…' to save them to a permanent location.

✅ FIXED — the long SILENT gap before tile 1 was the batch-size AUTOTUNE (it
timed ~6 batch sizes x 5 repeats, each forcing a CoreML model recompile, for
~1% gain: b4=0.805 vs selected b5=0.798). Now OFF by default
(prediction_batch_autotune=False) so prediction starts immediately. The high
read IO itself is normal: producer threads prefetch tiles ahead of the GPU. The
slow "prep 0.8 s/tile" is fixed by the GDAL-read resample (see below).

#### why on earth is there a tile resampling?
I have the feeling this resemble to some old trick in the segmentor model or the tiled inference. 
for now we do GDAL resampling. But that seems to be a bit of a hack. I would like to see the resampling done in the model itself, so that we can have a more robust inference or it should be trained in the first place with a range of resolutions. Doing this at inference level where the resampling takes 4 times the time of the inference is not a good idea.

✅ SPEED FIXED / ⚠️ DESIGN NOTED. Why it resamples: the U-Net has a FIXED input
GSD (~2.93 cm/px = tile_size 15 m / 512 px); your ortho is ~2.06 cm/px, so each
tile is rescaled to the training scale — a CNN is scale-sensitive (a stem must
appear at ~the trained pixel size). SPEED: the resample now happens inside the
GDAL read (rasterio out_shape + bilinear, in C), so prep dropped from ~0.8 s to
~0.05 s/tile — no longer 4x the inference. DESIGN (your preference): removing it
entirely is a MODEL change, not a pipeline change — train at your deployment GSD
(feed native tiles; loses cross-GSD generality) or with multi-scale
augmentation (robust to a GSD band). Resample-to-fixed-GSD is what lets ONE
model work on orthos of any resolution. -> training-side follow-up, not plugin.

### In the top right of the barnekow orthomosaic, 
stems where found by the unet but the vectorisations seems to have skipped it, Same with bottem left.

✅ FIXED (root cause confirmed by a 5-probe adversarial code review). Cause:
_raster_filter_geom (utils/IO.py) did `box(*bounds).buffer(-edge_buffer_m)`,
shrinking EVERY tile footprint inward on all four sides to dedup stems shared
with an overlapping neighbour. On INTERIOR seams the neighbour's inner region
recovers the stem (fine). But on the ortho's TRUE OUTER boundary there is no
neighbour, so stems within the 12 m band were dropped and never recovered; the
edge-reconnect pass runs AFTER the filter so it couldn't restore them. Corners
(top-right / bottom-left) touch the boundary on TWO sides, so the deletion band
wraps the corner — exactly the symptom. Fix: per-side buffering — only shrink a
tile side if it's an interior seam; keep the true extent where the side
coincides with the ortho boundary. Wired the (previously dead) stem_map_path
param through winmol_run -> merge_and_filter_tiled_results -> _process_tile ->
_raster_filter_geom to carry the ortho extent. Effect on the test crop: merged
stems 23 -> 60 (a small crop is almost all edge). Golden suite green.


### TIle Overloap
tile_overlap_m                 12.0
tile_size                      15
Not sure if this is a bug, but a tile size of 15 with an overlap of 12 sounds like an excessive overlap. I would expect a tile size of 15 with an overlap of 3 to 4 at most. 

⚠️ CLARIFIED — these are two DIFFERENT tile concepts (easy to conflate):
  * tile_size = 15 m  -> the MODEL prediction tile footprint (15 m -> 512 px).
  * tile_inner_px = 4096 + tile_overlap_m = 12 m -> the VECTOR tiling: large
    ~82 m tiles (4096 px @ ~2 cm/px) that overlap 12 m at their seams for stem
    dedup. So it is 12 m overlap on ~82 m tiles (~15%), NOT 12 m on 15 m.
So it is NOT excessive. The "long stem vs 12 m overlap" worry was investigated
and RULED OUT: the halo overlap equals the edge buffer, so an interior stem is
always fully covered by at least one tile's inner region and is never dropped
from all tiles simultaneously (the edge-reconnect stitches cross-seam stems).
The only real edge defect was the ortho OUTER boundary — fixed above. ✅ OK.

### Resizing to XYZ
how should be resized, I think nearest is used but that would create artifcats.

✅ CLARIFIED / FIXED — the IMAGE tiles are resampled with BILINEAR (smooth, no
nearest-neighbour blockiness); NEAREST is used ONLY for the binary validity
mask, which is correct (a 0/1 mask must not be interpolated). This is the new
GDAL-read path (rasterio out_shape + Resampling.bilinear / .nearest).


### Model download links
Downloading the onnx models should be optional. The released hdf5 models should be the source of truth, converting them locally could be better.


### Clean up Root dir. 
There are many scripts which seem weird. I.e. resources.py
Plugin_upload.py seems to be unused
there are many qgis related files which seem relevant to qgis only and could be in packaged away


### QGIS Docker Container
The current container is fixed on QGIS 3.28.2, which is a way too old (2022, no LTR(. I would suggest to use the latest LTR version of QGIS 3.44) and make it run on QGIS 4.2.0