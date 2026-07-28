## QGIS plugin

### env
the env setup isn't working, I want to install the plugin, then when opening the plugin I want menu where I can select an environment on disk (venv, or conda) or let one create


#### installation of venv progress bar.
The progressbar should end up with 100% when the venv setup is complete. Currently it states 0% so it is unclear the setup is finished. 

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


### Segmentation failure. 
With the barnekow tiff at the top of the image, the segmentation artifacts are created.
In a previous version quite huge area was left out when segmenting. which removed quite a lot of trees. Can this be fixed by adding a boarrder to the orthomosaic?  

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

⚠️ DECIDED 2026-07-21 — **ONNX download it is; no local conversion.** Converting
HDF5 → ONNX on the user's machine requires `requirements/convert.txt`
(`tensorflow==2.16.2` + `tf-keras` + `tf2onnx` + `onnx`, ~1 GB) inside the
plugin venv, which today is deliberately TensorFlow-free
(`requirements/plugin.txt` = base + psutil; see the docstring of
`plugin_utils/installer.py`). Shipping TF again would undo the whole point of
the ONNX migration (PR #2) and re-introduce the CUDA/TF-vs-QGIS isolation
problems the venv exists to avoid — for a conversion whose output we already
publish. So:

- **Runtime path:** the plugin downloads `.onnx` on demand, pinned to a GitHub
  release (`WINMOL_segmentor_pt` `models-v1`), verified by sha256.
- **HDF5 stays the provenance/source of truth**, referenced in `config.json`
  at Zenodo record **15907576** (DOI 10.5281/zenodo.15907576 — the record that
  holds *all four*; the older 10491124 is missing Spruce_Deadwood, which is why
  they were hard to find).
- `scripts/convert_models_to_onnx.py` remains the **dev-only** reproducibility
  tool (it verifies ONNX-vs-Keras parity on real tiles) — not a user path.

Implemented on branch `feat/model-zoo-registry`.


### Clean up Root dir. 
There are many scripts which seem weird. I.e. resources.py
Plugin_upload.py seems to be unused
there are many qgis related files which seem relevant to qgis only and could be in packaged away

✅ FIXED (deletions) / ⚠️ PROPOSED (the move) — `fix/rr-root-dir-cleanup`.
`plugin_upload.py` **was** dead: nothing imported it and its only caller was
the Makefile `upload` target, which is itself dead — deleted, along with
`pylintrc` (nothing runs pylint; CI is flake8-only) and the `upload` /
`transup` / `transcompile` / `transclean` targets that referenced nothing.
**`resources.py` is NOT weird and stays** — it is *generated* from
`resources.qrc` by `pyrcc5` (`make compile`) and is genuinely loaded at
runtime for the `:/plugins/...` icon paths; deleting it breaks the toolbar
icon. The "package the QGIS files away" idea is deliberately NOT done here:
QGIS loads a plugin from a fixed directory layout keyed on the folder name,
so moving `winmol_analyzer*.py` / `metadata.txt` risks breaking installs in a
way nothing in CI can catch. The evidence-backed migration proposal (what may
move, what must stay at the root, and why) is in **`docs/root-layout.md`** —
needs a live-QGIS validation pass before anyone acts on it.
New guard: `tests/test_plugin_package.py` pins the shipped zip manifest, so a
future deletion cannot silently drop a packaged file.


### QGIS Docker Container
The current container is fixed on QGIS 3.28.2, which is a way too old (2022, no LTR(. I would suggest to use the latest LTR version of QGIS 3.44) and make it run on QGIS 4.2.0

✅ FIXED (the bump) / 🐛 OPEN (QGIS 4) — `fix/rr-qgis-container-version`.
Tags verified against the Docker Hub API (441 tags, real digests, 2026-07-21)
rather than assumed — and **QGIS 4.2.0 does exist** (`4.2.0`, `4.2`,
`4.2.0-questing`, `4.2.0-trixie`, `stable`). Base bumped
`final-3_28_13` (2023-11-24, 2.7 GB) → **`3.44.12-noble`**, parameterised as
`ARG QGIS_TAG` before `FROM`, so trialling QGIS 4 is
`QGIS_TAG=4.2.0-trixie ./startDocker.sh` with no file edit; the image name now
embeds the tag so a 3.44 and a 4.2 image coexist instead of clobbering each
other. Two *latent build breakers* fixed on the way: `apt install
python3.10-venv` only resolves on jammy (any newer base would have failed with
"Unable to locate package"), and `ln -s` aborts if the base already ships
`/usr/bin/python` (now `ln -sf`). All qgis/qgis tags are amd64-only — noted in
`docs/CONTAINERS.md`.
**Still open — "make it run on QGIS 4.2.0" is NOT delivered.** 4.2.0 is not the
default and `metadata.txt` is untouched, because raising `qgisMaximumVersion`
ships an *unvalidated* Qt6-compatibility claim to end users; QGIS derives the
default max from `qgisMinimumVersion[0]` (verified in
`pyplugin_installer/installer_data.py`). A test goes red the moment the default
is bumped to 4.x without the metadata change. Validating Qt6 needs a live
QGIS 4 session — see `docs/CONTAINERS.md` for the one-command trial.

---

## Review round — 2026-07-21 · branch `fix/review-findings`

**What is under review:** `fix/review-findings`, cut from `feat/gpu-container`
@ `7c0628b` (the live line, post-restack).

**Do not review from tag `v0.6.0.2`.** That tag sits on the pre-restack
lineage (see `backup/feat-gpu-container-old-line`); it is not an ancestor of
`feat/gpu-container` and the two have diverged by 50 commits, so anything
fixed from that tag can never fast-forward back into the stack. When a
reviewable build is needed, tag *this* line instead (pushing a tag triggers
the release ZIP via `on-push-tags.yml`).

**Where fixes go:** commit them on this branch. It PRs into
`feat/gpu-container`, i.e. it sits on *top* of the PR stack — nothing below
(#1 → #2 → #3/#4 → #9 → #10 → #12) ever needs rebasing.

**Convention in this file:** `🐛 OPEN` · `✅ FIXED` (state the cause, not just
the fix) · `⚠️ CLARIFIED` / `ENVIRONMENTAL` (not a code bug). One `###` per
finding: what you did, what you expected, what happened, logs verbatim.

### (findings below)



## Release Bugs

### OPEN — 2026-07-21
I created a demo release https://github.com/cwinkelmann/WINMOL_Analyzer/releases/tag/v0.0.0-demo1
#### the zip installed, but creating and python environment looks stuck. Nothing happens for a while, making it more verbose would be nice. 

✅ FIXED — `fix/rr-installer-verbosity`. The log channel already ran end-to-end
(`installer.setup_environment(progress=)` → `EnvSetupWorker.log` →
`update_output_log` → the same widget prediction uses); the defect was
**granularity and buffering**, so nothing was plumbed, it was made to speak.
Now: timestamped phase lines (`[  12s] …`), pip run as `-u` with
`PYTHONUNBUFFERED=1` and **stderr merged into stdout** (pip writes failure
detail to *stdout*, and `capture_output` kept the streams apart — which is why
the previously-surfaced error tail was usually empty), line-by-line streaming,
a heartbeat (`… still working (Ns)`) so a long silent step still shows life,
per-model download progress, and a final `Environment ready in Xs`.
`progress=None` (batch/CLI/headless) is byte-for-byte unaffected.
**Second, worse bug found and fixed while in there:** `_pip_install_into` ran
pip *synchronously on the GUI thread*, so "Choose interpreter… → install deps"
genuinely froze QGIS (the reported path was merely mute). It now goes through
the same worker thread. 16 new tests in `tests/test_installer_progress.py`,
all with a stubbed `Popen` — plus one verification against a real subprocess,
since stubs cannot prove streaming actually streams.
*Still wants a GUI pass:* fresh machine, no `winmol_venv`, hit Environment and
confirm lines appear within a couple of seconds and keep coming.

#### The zip had no semantic versioning number.

✅ FIXED — `fix/rr-release-zip-version`. Confirmed at the source: the
`v0.0.0-demo1` asset was literally `WINMOL_Analyzer_QGIS_Plugin.zip`, with the
name hardcoded twice in `on-push-tags.yml`; `metadata.txt` was already stamped
correctly. New `scripts/plugin_version.py` derives the version (one leading
`v` stripped; `v0.6.0.2` and `v0.0.0-demo1` both valid; a non-numeric lead like
`models-onnx-v1` rejected) and the artifact is now
**`WINMOL_Analyzer-<version>.zip`**.
⚠️ **Judgement call worth your eye:** `documentation/index.html` links
`releases/latest/download/WINMOL_Analyzer_QGIS_Plugin.zip` in **five** places,
and that GitHub permalink requires a *fixed* asset name (no wildcard) — a plain
rename silently breaks every Download button on the public site. So the
workflow uploads the identical archive under **both** names: the versioned one
(primary) and the old name as a stable alias. Cost: one redundant ~1.8 MB asset
per release. The alternative was rewriting those five links on every release.
Note a suffixed version like `0.0.0-demo1` sorts *below* `0.6.0` in QGIS's
installer, so demo tags never present themselves as upgrades.

#### Per default all 3 products should be enabled in the plugin

✅ FIXED — `fix/rr-default-all-products`. Your premise was right, and here is
what they are: three `QCheckBox`es in `winmol_analyzer_dialog_base.ui` —
`output_checkBox_stem` (was checked), `output_checkBox_trees` (**was false**),
`output_checkBox_nodes` (**was false**) — so a default run produced the stem
map only. All three now default to checked.
Worth knowing: they collapse to ONE `process_type` via a strict ladder
(nodes → `Nodes`, elif trees → `Trees`, else `Stems`), and a single
`winmol_run.py` invocation with `Nodes` **already writes all three products**
(prediction writes the raster; `write_all_layers_to_gpkg` emits the
stems + vectors + nodes layers). So this was a UI-defaults bug, not a
missing-feature one — no extra subprocesses were added. Selection logic
extracted to `plugin_utils/output_selection.py` and covered by
`tests/test_plugin_output_defaults.py`.

The plugin should be capable of downloading more models in a dialogue. For now we should pin them against a github release




#### Missing models
Download missing models from the release page. The plugin should be able to download them automatically.
https://github.com/cwinkelmann/WINMOL_segmentor_pt/releases 
Spruce Deadwood as INT8 should be default, A SpecDS INT8 W05 would be second best

✅ CORRECTED 2026-07-21 (later the same day) — **the note below was WRONG and
you were right.** After the `models-v1` assets were renamed, the provenance is
explicit: **`model_UNet_SpecDS_Beech_512_pytorch_w05_int8.onnx`** — a SpecDS,
w05, int8 build. So "a SpecDS INT8 W05" *does* exist; I claimed it did not
because the old asset name (`unet_w05_int8_cpu.onnx`) hid the training set.
It is registered as `UNet_PT_int8` and is the runner-up default, as you asked.
Two related facts the rename also settled: the family's fp32 member
(`..._pytorch.onnx`, 124.1 MB) is the FULL-WIDTH reference, not a w05 build;
and w05 exists only in the `_pytorch` lineage, never in the converted-Keras
entries. Labels and `docs/CONFIG.md` corrected accordingly.

⚠️ NAMING (superseded by the correction above) — as read on 2026-07-21:
- 1st choice exists as asked: **`model_UNet_SpecDS_Spruce_Deadwood_512_int8.onnx`**
  (31.4 MB, CPU static int8, domain-calibrated; the int8 of the classic
  Spruce_Deadwood model).
- 2nd choice **"SpecDS INT8 W05" does not exist** — there is no SpecDS×W05
  build. `w05` belongs to the *PyTorch UNet* family, not the Keras SpecDS one.
  The closest real asset is **`unet_w05_int8_cpu.onnx`** (7.9 MB, ~10x faster
  on CPU via AVX-VNNI, TestDS **F1 0.760**, lossless vs its fp32).
  Worth noting it is both **4x smaller** and **higher-scoring** than the
  SpecDS Keras line (the R/Keras beech retrain is F1 0.738), so "second best"
  may understate it — it is arguably the better default for CPU users, with
  Spruce_Deadwood int8 the better *domain-specific* pick. Defaults implemented
  as requested (Spruce_Deadwood int8 first); say the word to flip.

### Add a setup tab to the plugin
There env setup, deletion and model download can be combined. Don't do that in that single tab "Detect stems from UAV images"
### Model download failed with HTTP 404 (release v0.0.0-demo3)

✅ FIXED — `280c362` + `657825c`. Reported from the GUI:
"download failed for model_UNet_SpecDS_Spruce_Deadwood_512_int8.onnx …
HTTP Error 404". Two independent causes, both invisible to the checks that
were run at the time:
1. The `models-v1` assets were **renamed** after `config.json` was written
   (to the timestamped scheme that fixes the "names are not helpful"
   complaint). **0 of 22** registry filenames still matched.
2. `WINMOL_segmentor_pt` was **private**, so the plugin's unauthenticated
   download could never succeed regardless of names. (Now public.)
**Why it shipped:** the assets were verified with `gh`, which is
*authenticated* — that masked both the privacy and the renames. A single
anonymous fetch would have caught it.
**Fix:** all 22 entries re-derived from the release `SHA256SUMS` **by
digest**, not by guessing names — every one mapped to exactly one current
asset and not a single checksum changed (identical bytes, renamed files).
**Guards added** (`tests/test_model_urls.py`, vendored
`tests/fixtures/models_v1_SHA256SUMS`): an offline test that `file` ==
basename(url) and that every digest matches the published sums, plus an
anonymous-reachability test over every URL, opt-in via `WINMOL_NET_TESTS=1`
so CI stays offline. Verified 26/26 reachable without auth.


### GUI bugs
But I have some bugs to report first. The progress bar at inferences reaches 78% before the first inference happens. I want to have 0% at "Loading Model"     
  Then there is too verbose output like "MERGE DISCOVERY | root /Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/winmol/tmp/winmol_tiles_qq6zt6ri | gpkg_files 3
MERGE INPUT | /Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/winmol/tmp/winmol_tiles_qq6zt6ri/raster_r00000_c00000.gpkg
MERGE INPUT | /Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/winmol/tmp/winmol_tiles_qq6zt6ri/raster_r00000_c00001.gpkg" I dont want to see that in the normal model but allow a verbose debug mode. At the beginning the output is fine. But         
  "Hardware detected: CPUs=8, RAM=24.0 GB, GPUs=0" on a mac is wrong, it should be 8 cpu core and a reference that we are talking about apple metal when the GPU check happens.                  
  When an autotune never happened before it should be run. The tensforflow error seems obsolete.


### The GUI window is too high/has not scroll bar
Currently I can press run because it is just outside my screen. I would suggest to make the window smaller and add a scroll bar if necessary.



### Button positions
Run should be at the top
Export at the bottom.
The tab setup at most left


## Code Review - fix later

> **Triage 2026-07-28** (branch `fix/review-findings`). All 11 items below were
> re-investigated against the current code. Outcome: **7 already resolved** by
> the rc round, **2 not-a-bug / by-design**, **1 cleanup done here**
> (messy-root, tier a), and **1 real bug fixed here** (reinstall after a manual
> env delete). Each item is annotated with its verdict inline.

### Functions in Winmol_batch
In winmol_batch are many model helpers which seem to be off. Those should checked if they are necessary. I would suggest to move them to a helper class or remove them if they are not used.
There is a tensorflow function remaining despite tensorflow is not used anymore. I would suggest to remove it.

> _**Resolved.** The heavy model-resolution logic was extracted to
> `plugin_utils/model_registry.py`; every function left in `winmol_batch.py`
> is reached and tested (`tests/test_batch_jobs.py`). There is no TensorFlow
> function in the file (there never was) — the only repo TF import is the
> dev-only `scripts/convert_models_to_onnx.py`, sanctioned by CLAUDE.md._


### Messy Project Root
metadata.txt, pb_tool.cfg, resources.qrc, resurces.py, tasks_threads.py, winmol_analyzer_dialog_base.ui etc. look they don't belong in there.

> _**Partly addressed here (tier a).** Full per-file audit in
> `docs/root-layout.md`: 4 of the 6 named files are load-bearing
> (`metadata.txt`, `tasks_threads.py`, `winmol_analyzer_dialog_base.ui` are
> required flat by QGIS; `resources.py` is dead but hard-required by the
> release packaging script). Done in this PR: `pb_tool.cfg` deleted (nothing
> ran pb_tool), and the untracked Dockerfile variants / release zips /
> `push_restack.sh` are gitignored. Deferred (own reviewed commit): removing
> the dead `resources.py`/`resources.qrc`, and the risky `qgis_plugin/`
> subpackage reorg (no CI loads QGIS)._


### is mps used?
CUrrently I am running this:
Loading Model...
Loading ONNX model via OnnxSegmenter: /Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/models/model_UNet_SpecDS_Spruce_Deadwood_512_2024-12-19_194758_fp16.onnx

Performing Prediction with Resampling in stream mode...
#######################################################
Prediction of the semantic stem map
Resampling tiles while analyzing (stream mode)
Written tile 1/182 | 0.5% | 4.5 tiles/min | ETA 40m 02s | avg read 2.314s prep 0.013s infer 10.897s write 0.006s | batch 1 | queue 75% full | producers 1 | src 727x727 -> out 504x504
Written tile 7/182 | 3.8% | 6.9 tiles/min | ETA 25m 28s | avg read 0.864s prep 0.014s infer 8.378s write 0.001s | batch 1 | queue 100% full | producers 1 | src 727x727 -> out 504x504
Written tile 15/182 | 8.2% | 8.4 tiles/min | ETA 19m 59s | avg read 0.403s prep 0.011s infer 7.012s write 0.001s | batch 1 | queue 100% full | producers 1 | src 727x727 -> out 504x504
Written tile 24/182 | 13.2% | 9.4 tiles/min | ETA 16m 49s | avg read 0.401s prep 0.010s infer 6.275s write 0.001s | batch 1 | queue 100% full | producers 1 | src 727x727 -> out 504x504
Written tile 34/182 | 18.7% | 10.0 tiles/min | ETA 14m 47s | avg read 0.388s prep 0.
It seems very slow. With previous bug fixes it should be clear that Apple MPS is used, or CUDA if not taht shoudl be fixed now.

> _**Resolved.** On Apple Silicon the runtime now selects CoreML and
> *verifies* the bound provider, surfacing a silent CPU fallback rather than
> hiding it (`utils/onnx_runtime.py` `_default_providers`/`_verify_providers`).
> The slow log above is the **fp16** model: on CoreML fp16 is ~14.6× slower
> than fp32, so the Apple device default was switched fp16→fp32
> (`plugin_utils/model_registry.py::_device_variant`, cpu→int8 / gpu→fp16 /
> coreml→fp32). Op-level proof of what ran on Metal/ANE is available via
> `WINMOL_ONNX_PROFILE=1`. (Separately: a CUDA→CPU provider demotion on a
> mis-matched driver is the true cause of the "zero GPU usage" reports; it is
> detected by `verify_session_providers` and is an environment issue, not an
> autotune one.)_

### plugin released twice
There are two on github
WINMOL_Analyzer-0.6.1-rc1.zip
sha256:1404190fe3ed315e096d1f9cb9d773fa8ef12b103773b77e7e058fb222c9184e
1
1.84 MB
2 hours ago
WINMOL_Analyzer_QGIS_Plugin.zip
sha256:1404190fe3ed315e096d1f9cb9d773fa8ef12b103773b77e7e058fb222c9184e
1.84 MB
2 hours ago

> _**Not a bug — by design.** One archive is built and `cp`'d to a stable
> alias: `WINMOL_Analyzer-<version>.zip` (identifiable on disk) and
> `WINMOL_Analyzer_QGIS_Plugin.zip` (the fixed name the `releases/latest/
> download/` permalinks in `documentation/index.html` require). Identical
> sha256 is the expected consequence of the same bytes under two names. See
> the comment in `.github/workflows/on-push-tags.yml` and CLAUDE.md._


### The Autotune is a bit pointless
It gets slower and slower
Prediction micro-batch autotune candidate 1/13: b4 = 0.340s/tile
Prediction micro-batch autotune candidate 2/13: b5 = 0.337s/tile
Prediction micro-batch autotune candidate 3/13: b6 = 0.330s/tile
Prediction micro-batch autotune candidate 4/13: b7 = 0.471s/tile
Prediction micro-batch autotune candidate 5/13: b8 = 0.561s/tile
But it is not stopping.
Prediction micro-batch autotune candidate 6/13: b9 = 1.135s/tile

It would be much quicker to just check for GPU memory usage and set the batch size accordingly. The autotune is a bit pointless and takes a lot of time. It should be possible to set the batch size manually in the GUI.
I tested it on GPU T14 with the RTX 4080. It starts with a batch size of 1. GPU usage up till 5 seems to be zero. There is some CPU usage, but it is way too slow. 

You need to set 

#### Autotune on GPU Linux lead to crash
THe Autotune started very slow with zero GPU usage and about 5s per tile, after a while it crashed my full machine. So There is quite some optimization potential in the autotune. It should be possible to set the batch size manually in the GUI.

#### Autotue on CPU
is surprisingly fast per tile. But the overall speed is terrible. CPU usage during that is just 40% on 4 Core CPU. at b6 performance got worse.

> _**Resolved.** The sweep was rebuilt (`utils/Prediction.py`): a dual
> improvement bar (relative + absolute s/tile) stops chasing jitter, a
> degradation guard (`degrade_factor=1.25`) aborts as soon as a candidate is
> ≥1.25× the best — so the report's `b7=1.43×` now halts there or earlier via
> patience (lowered 4→2), the candidate cap is 13→6, and a pre-emptive
> free-memory ceiling (bounded by host RAM when a CUDA run silently lands on
> the CPU EP) prevents the host-RAM exhaustion that crashed the machine.
> Manual control exists end-to-end: the GUI "Prediction batch size" spinbox →
> `WINMOL_CONFIG_OVERRIDES_JSON` → `Config.prediction_batch_override` skips the
> sweep entirely, plus a "Clear autotune cache" button. Default is now
> `"auto"` (tune once per hardware/model/EP/tile, then persist)._

### Download recommended models
On a old CPU only windows machine it seems to work BUT downloading a 124MB spruce model cannot be the recommended one. Since we know the int8 one performaned nearly as good, that should be the recommended one. The download of the int8 model is much quicker and it should be the default one in these CPU only cases.

> _**Resolved.** A CPU-only machine is now recommended and offered the 31.4 MB
> **int8** build, not the 124.6 MB fp32 one. The declared default is int8
> (`config.json` `gui_default="Spruce_Deadwood_int8"`), and the device-variant
> resolver maps cpu→int8. Pinned by a regression test against the shipped
> config: `tests/test_model_status.py::test_a_cpu_only_machine_is_recommended_the_int8_build`._


### Logs are unclear
The logs are ambiguous. For inference it says how quick the inference is 

Written tile 156/182 | 85.7% | 153.7 tiles/min | ETA 00m 10s | avg read 0.056s prep 0.004s infer 0.382s write 0.000s | batch 6 | queue 75% full | producers 2 | src 727x727 -> out 504x504


Vector tiles 3/3 | 100.0% | 4.7 tiles/min | ETA 0s | wrote 3 | empty 0 | no_output 0 | avg total 12.707s quant 1.008s connect 2.854s

It should be clear the vector tiles are much larger.

> _**Resolved.** Each phase now prints a header naming its unit and tile size
> (`PREDICTION PHASE | … src WxH -> out WxH` / `VECTOR PHASE | … ~PxP px each`)
> and every counter line carries a unit label + pixel dims (`… | prediction
> tile | …` vs `… | vector tile ~4144x4144 px | …`). The `run_progress.py`
> parser prefixes are preserved (new labels sit after the first `|`), guarded
> by `tests/test_run_progress.py`._

### uninstalling the plugin does not remove the folder
'/Users/christian/Library/Application Support/QGIS/QGIS3/profiles/default/winmol' stays untouched after deinstalling. 
Does it even make sense the installation is there? I would suggest to remove it on uninstall. If the user wants to keep it, he can copy it to another location before uninstalling.

> _**Resolved — leave-and-warn is deliberate.** QGIS has no uninstall hook:
> `uninstallPlugin` is `unloadPlugin()` + `removeDir(plugin_dir)`, and
> `unload()` is the SAME callback QGIS fires on disable, reload and app quit —
> so deleting the ~1–2 GB venv there would erase it on every QGIS close (locked
> out by `tests/test_dialog_lint.py::test_unload_never_deletes_the_environment`).
> Instead the env sits beside the plugin under `<profile>/winmol` so uninstall
> can't reach it, the Setup tab warns that uninstall leaves it, and an explicit
> itemised "Delete environment…" flow removes it on demand. See `docs/SETUP.md`._


### Delete environment doesen't exist
in the plugin I can't delete the environemt, but with open Folder, I did it manually.
Reinstalling the plugin didn't work on the GPU machine.

> _**Fixed here.** The "Delete environment…" action itself already exists
> (Setup tab, thread-safe, itemised). The real residual — and the cause of
> "Reinstalling … didn't work" after the manual `rm -rf` — was that the saved
> `winmol/python_executable` setting still pointed at the deleted managed venv,
> so `resolve_environment` returned a dead-end "Python 0.0" error instead of
> rebuilding. Fix: `resolve_environment` no longer honours a configured
> interpreter that lives inside the managed venv but no longer exists on disk —
> it ignores it and falls through to rebuild (the dialog rewrites the key on
> success). A genuine missing EXTERNAL interpreter still errors. Covered by
> `tests/test_plugin_installer.py::test_resolve_environment_ignores_stale_managed_interpreter`
> (+ the external-boundary test)._

### final Plugin testing
Windows CPU old X1 - fine with int8 model

Windows CPU new (T14)
Windows GPU RTX4080 new (T14)

MacOS MPS M2 - works fine

Linux CPU new (T14) - 
Linux GPU RTX4080 new (T14) - works fine

Linux GPU Blackwell H100 ( carrot )
Linux GPU A100 ( olive )

### remove the tensorflow dependency
there are still tensorflow junk files, at least 3 requirements files. I would suggest to remove them and make sure the plugin is tensorflow free. Only for the conversion it is fine.

> _**Resolved.** Requirements are the canonical 7 files; TensorFlow is isolated
> to `requirements/convert.txt` (self-declared as the only TF file). The runtime
> and plugin path are provably TF-free, enforced by
> `tests/test_plugin_compute_contract.py` (runs `winmol_run.py` with TF imports
> blocked) and `tests/test_load_model_onnx.py`. No `.hdf5/.h5/.keras` or stray
> `requirements.txt` is tracked._


### Installation location
In linux there is a problem with the installation location. The plugin is installed in ....QGIS/QGIS3/profiles/default/python/plugins/Winmol_Analyzer.
But I found the plugin to be installed in profiles/default/winmol

> _**Not a bug — by design.** Two directories exist on purpose: the plugin CODE
> (and re-downloadable models) live in `…/python/plugins/WINMOL_Analyzer`, while
> the plugin's self-built environment (venv, downloaded py311, autotune cache,
> tmp) lives in a SEPARATE `<profile>/winmol` so QGIS's recursive uninstall of
> the plugin dir never chokes on thousands of venv files — this split was itself
> the fix for an earlier uninstall-failure bug. Documented in `docs/SETUP.md`
> ("Where the plugin's environment lives", incl. the Linux path)._


