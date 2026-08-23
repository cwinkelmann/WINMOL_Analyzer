#!/usr/bin/env python

###############################################################################
"""Imports"""

import json
import os
import tempfile
import shutil
import errno
import numpy as np
import rasterio
import geopandas as gpd
import fiona
import pandas as pd
try:
    import pyogrio
    _HAVE_PYOGRIO = True
except Exception:
    pyogrio = None
    _HAVE_PYOGRIO = False
from rasterio.enums import Resampling
from shapely.geometry import LineString, Point, box
from collections.abc import Mapping
from pyproj import CRS
from pathlib import Path

import utils.Quantification as Quant
from classes.Stem import Stem


###############################################################################

"""Streaming and tiling operations"""


#: Below this, a run is too short for the read cost to matter and the
#: overview warning would just be noise.
_OVERVIEW_WARN_GB = 2.0


def _warn_if_no_overviews(src, estimated_input_gb):
    """Tell the user when an ortho will be read the slow way.

    Prediction resamples each tile to the model grid during the GDAL
    read. When the file HAS overviews, GDAL serves that from a decimated
    level -- measured 7.5 ms per tile on Tegel R13. Without them it must
    read every source pixel and shrink in RAM: 13.9 ms per tile, and
    ~4x the bytes through GDAL's global block cache (default 5% of RAM,
    shared by every producer thread). On a large ortho that is what makes
    throughput decay and then collapse (issue #43).

    Building overviews once is the fix, and `-ro` keeps the original
    file untouched by writing a .ovr sidecar.
    """
    try:
        overviews = src.overviews(1)
    except Exception:
        return
    if overviews or estimated_input_gb < _OVERVIEW_WARN_GB:
        return
    print(
        f"WARNING: {src.name} has NO overviews and is "
        f"{estimated_input_gb:.1f} GB. Every prediction tile will be read "
        f"at full resolution and downsampled in RAM -- roughly 2x the read "
        f"time and 4x the bytes through GDAL's shared cache, which makes "
        f"throughput decay on large orthos. Build them once with:\n"
        f"    gdaladdo -ro -r average {src.name} 2 4 8 16 32 64 128\n"
        f"(-ro writes a .ovr sidecar and leaves the original file "
        f"unchanged.)", flush=True)


def get_raster_info(path) -> dict:
    with rasterio.open(path) as src:
        dtype = src.dtypes[0] if src.dtypes else 'unknown'
        estimated_input_gb = (
            src.width * src.height * src.count * np.dtype(dtype).itemsize
        ) / (1024 ** 3)
        _warn_if_no_overviews(src, estimated_input_gb)
        return {
            'width': int(src.width),
            'height': int(src.height),
            'bands': int(src.count),
            'dtype': str(dtype),
            'pixel_size_x': float(abs(src.transform.a)),
            'pixel_size_y': float(abs(src.transform.e)),
            'estimated_input_gb': float(estimated_input_gb),
            'crs': src.crs,
            'transform': src.transform,
        }


def atomic_tmp_path(final_path: str) -> str:
    p = Path(final_path)
    return str(p.with_suffix(p.suffix + '.tmp'))


def finalize_raster(tmp_path: str, final_path: str) -> str:
    os.replace(tmp_path, final_path)
    return final_path


def build_safe_prediction_profile(
    src_profile, width: int, height: int,
    transform, compress: str | None = 'DEFLATE',
    dtype: str = 'float32'
):
    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'count': 1,
        'width': int(width),
        'height': int(height),
        'transform': transform,
        'crs': src_profile.get('crs', None),
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'BIGTIFF': 'YES',
        'interleave': 'BAND',
    }
    if compress is not None:
        c = str(compress).upper()
        if c in {'DEFLATE', 'LZW', 'ZSTD'}:
            profile['compress'] = c
            profile['predictor'] = 2
            if c == 'DEFLATE':
                profile['zlevel'] = 1
        elif c in {'NONE', 'OFF', 'FALSE', '0'}:
            pass
        else:
            profile['compress'] = 'DEFLATE'
            profile['predictor'] = 2
            profile['zlevel'] = 1
    return profile


def create_output_raster_like(
    src_path: str,
    dst_path: str,
    dtype: str = "float32",
    count: int = 1,
    width: int | None = None,
    height: int | None = None,
    transform=None,
    compress: str | None = "DEFLATE",
):
    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = build_safe_prediction_profile(
            src.profile,
            width=int(width or src.width),
            height=int(height or src.height),
            transform=transform or src.transform,
            compress=compress,
        )
        profile['dtype'] = dtype
        profile['count'] = count
    with rasterio.open(dst_path, 'w', **profile):
        pass
    return profile


def read_window(path: str, window, bands:
                list[int] | None = None, boundless: bool = True, fill_value=0
                ):
    with rasterio.open(path) as src:
        indexes = bands if bands is not None else list(range(1, src.count + 1))
        return src.read(
            indexes, window=window, boundless=boundless, fill_value=fill_value)


def write_window(dst_path: str, array, window, band: int = 1):
    with rasterio.open(dst_path, 'r+') as dst:
        if array.ndim == 2:
            dst.write(array, band, window=window)
        else:
            dst.write(array, window=window)


def write_tile_raster(pred_tile, tile_profile, output_path: str):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    prof = build_safe_prediction_profile(
        tile_profile,
        width=pred_tile.shape[1],
        height=pred_tile.shape[0],
        transform=tile_profile['transform'],
        compress=None,
    )
    with rasterio.open(output_path, 'w', **prof) as dst:
        dst.write(pred_tile.astype(np.uint8), 1)
    return output_path


def iter_prediction_tiles(src_path: str, config, tile_jobs):
    with rasterio.open(src_path) as src:
        for job in tile_jobs:
            window = job.halo_window if hasattr(job, 'halo_window') else job
            arr = src.read([1], window=window, boundless=True, fill_value=0)[0]
            profile = src.profile.copy()
            profile['width'] = int(window.width)
            profile['height'] = int(window.height)
            profile['transform'] = rasterio.windows.transform(
                window, src.transform)
            yield job, arr, profile


def load_raster_window_with_profile(path: str, window):
    with rasterio.open(path) as src:
        pred = src.read(1, window=window, boundless=True, fill_value=0)
        profile = src.profile.copy()
        profile['width'] = int(window.width)
        profile['height'] = int(window.height)
        profile['transform'] = rasterio.windows.transform(
            window, src.transform)
        return pred, profile


"""File operations"""


def load_model_from_path(model_path, config=None, wrap_preprocess=None):
    # ONNX models are architecture-agnostic: they are served by an
    # OnnxSegmenter adapter that duck-types the Keras model's
    # predict_on_batch(NHWC) interface, so no TensorFlow/Keras code is
    # needed here at all.
    #
    # wrap_preprocess: None resolves from the read-strategy flag (`graph`
    # wraps); False forces the raw model for callers that feed
    # pre-normalized float tiles themselves (PredictWorkers).
    if str(model_path).lower().endswith(".onnx"):
        return _load_onnx_model(model_path, config, wrap_preprocess)

    # The shipped runtime is TensorFlow-free: it loads only .onnx models
    # via onnxruntime. Legacy Keras/TensorFlow models (.hdf5/.h5/.keras)
    # are no longer loadable here -- convert them to ONNX first.
    raise RuntimeError(
        f"Unsupported model format for {model_path!r}: the WINMOL "
        "runtime loads only .onnx models (onnxruntime, no TensorFlow). "
        "Convert legacy Keras/TensorFlow models (.hdf5/.h5/.keras) to "
        "ONNX first with scripts/convert_models_to_onnx.py, then pass "
        "the resulting .onnx file.")


def _load_onnx_model(model_path, config=None, wrap_preprocess=None):
    """Load a .onnx segmenter via the vendored OnnxSegmenter.

    OnnxSegmenter exposes predict_on_batch(NHWC), so it is a drop-in for
    the old Keras model everywhere the analyzer runs inference. This
    import is lazy (deferred to call time) so a missing/broken
    onnxruntime install only breaks the .onnx path, not module import,
    and so tests can stub utils.onnx_runtime without onnxruntime present.
    """
    try:
        from utils.onnx_runtime import OnnxSegmenter
    except Exception as e:
        raise RuntimeError(
            f"Cannot load ONNX model {model_path!r}: onnxruntime is not "
            "available (" + str(e) + "). Install it with "
            "'pip install onnxruntime' (or 'onnxruntime-gpu' for CUDA) "
            "and try again.") from e
    antialias = False
    if wrap_preprocess is None:
        from utils.Prediction import (resolve_read_strategy,
                                      strategy_wraps_graph)
        strategy = resolve_read_strategy(config)
        wrap_preprocess = strategy_wraps_graph(strategy)
        antialias = strategy == "graph_aa"
    if wrap_preprocess:
        # Prepend normalize + bicubic resize to the graph so they run on
        # the session's device instead of the CPU, and GDAL goes back to
        # plain native reads. See utils/onnx_preprocess.
        from utils.onnx_preprocess import build_preprocessed_model
        target = (int(getattr(config, 'img_height', None) or 512),
                  int(getattr(config, 'img_width', None) or 512))
        wrapped = build_preprocessed_model(model_path, target,
                                           antialias=antialias)
        print(f"Loading ONNX model with IN-GRAPH preprocessing "
              f"(normalize + bicubic resize on device): {wrapped}")
        if antialias:
            # onnxruntime's CUDA EP mis-executes the opset-18 antialias
            # Resize (measured: 82 stems vs 478 on the CPU EP, same run).
            # Pin the CPU provider until that is fixed upstream; graph_aa
            # is a comparison mode, so correctness beats speed here.
            print("graph_aa: pinning CPUExecutionProvider (CUDA EP "
                  "computes antialias Resize incorrectly, ORT<=1.19)")
            return OnnxSegmenter(wrapped,
                                 providers=["CPUExecutionProvider"])
        return OnnxSegmenter(wrapped)
    print(f"Loading ONNX model via OnnxSegmenter: {model_path}")
    return OnnxSegmenter(model_path)


def load_orthomosaic(path, config):
    with rasterio.open(path) as src:
        img = src.read(list(range(1, config.n_channels + 1)))\
            .transpose(1, 2, 0)
        img = (img / 255).astype(np.float32)
        return img, src.profile


def load_orthomosaic_with_resampling(path, config):
    with rasterio.open(path) as src:
        scale_factor_x = src.res[0] / (config.tile_size / config.img_width)
        scale_factor_y = src.res[1] / (config.tile_size / config.img_width)
        img = src.read(
            list(range(1, config.n_channels + 1)),
            out_shape=(
                config.n_channels,
                int(src.height * scale_factor_y),
                int(src.width * scale_factor_x)
            ),
            # cubic for the same reason as the streamed read in
            # Prediction.py: bilinear thins the mask at scale.
            resampling=Resampling.cubic
        )
        img = img[0:3, :, :].transpose(1, 2, 0)
        transform = src.transform * src.transform.scale(
            (src.width / img.shape[-2]),
            (src.height / img.shape[-3])
        )
        img = (img / 255).astype(np.float32)
        profile = src.profile.copy()
        profile['transform'] = transform
    return img, profile


def load_stem_map(path):
    if path.endswith('.tif') or path.endswith('.tiff'):
        print("#######################################################")
        print("#######################################################")
        print("")
        print(path)
        print("")
        with rasterio.open(path) as src:
            pred = src.read(1)
            profile = src.profile
        return pred, profile
    raise ValueError(f'Unsupported stem map path: {path}')


def export_stem_map(pred, profile, pred_dir, pred_name, compress="DEFLATE"):
    final_path = os.path.join(pred_dir, f'{pred_name}.tiff')
    tmp_path = atomic_tmp_path(final_path)
    os.makedirs(pred_dir or '.', exist_ok=True)
    height, width = pred.shape
    safe_profile = build_safe_prediction_profile(
        src_profile=profile,
        width=width,
        height=height,
        transform=profile['transform'],
        compress=compress,
        dtype=str(pred.dtype),
    )
    with rasterio.open(tmp_path, 'w', **safe_profile) as dst:
        dst.write(pred.astype(pred.dtype, copy=False), 1)
    finalize_raster(tmp_path, final_path)


def get_bounds_from_profile(profile):
    left = profile['transform'][2]
    right = profile['transform'][2] \
        + profile['transform'][0] * profile['width']
    bot = profile['transform'][5] \
        + profile['transform'][4] * profile['height']
    top = profile['transform'][5]
    return rasterio.coords.BoundingBox(left, bot, right, top)


def _profile_get(profile, key, default=None):
    if profile is None:
        return default
    if hasattr(profile, "get"):
        return profile.get(key, default)
    if isinstance(profile, Mapping):
        return profile.get(key, default)
    return getattr(profile, key, default)


def _crs_from_profile(profile):
    crs_in = _profile_get(profile, "crs")
    if crs_in is None:
        return None

    try:
        crs = CRS.from_user_input(crs_in)
    except Exception:
        try:
            if hasattr(crs_in, "to_wkt"):
                crs = CRS.from_wkt(crs_in.to_wkt())
            else:
                return str(crs_in)
        except Exception:
            return str(crs_in)

    epsg = None
    try:
        epsg = crs.to_epsg()
    except Exception:
        pass
    return CRS.from_epsg(epsg) if epsg else crs


def _jsonify_list(x):
    try:
        return json.dumps([float(v) for v in x], ensure_ascii=False)
    except Exception:
        return json.dumps(x, ensure_ascii=False)


def stems_to_gdf(stems, profile):
    crs = _crs_from_profile(profile)

    rows = []
    geoms = []
    for i, s in enumerate(stems):
        try:
            sx, sy = list(s.start.coords)[0]
        except Exception:
            sx, sy = (None, None)
        try:
            ex, ey = list(s.stop.coords)[0]
        except Exception:
            ex, ey = (None, None)

        if hasattr(s.path, "geom_type"):
            geom = s.path
        else:
            geom = LineString(list(s.path.coords))
        geoms.append(geom)

        rows.append({
            "stem_id": i,
            "start_x": sx, "start_y": sy,
            "stop_x": ex, "stop_y": ey,
            "length": float(getattr(s, "length", 0.0)),
            "volume": float(getattr(s, "volume", 0.0)),
            "d_json": _jsonify_list(getattr(s, "segment_diameter_list", [])),
            "l_json": _jsonify_list(getattr(s, "segment_length_list", [])),
            "v_json": _jsonify_list(getattr(s, "segment_volume_list", [])),
        })

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def nodes_to_gdf(stems, profile):
    crs = _crs_from_profile(profile)

    rows = []
    geoms = []
    for i, s in enumerate(stems):
        coords = list(s.path.coords)
        for j, xy in enumerate(coords):
            geoms.append(Point(xy))
            d = None
            try:
                d = float(s.segment_diameter_list[j])
            except Exception:
                pass
            rows.append({
                "stem_id": i,
                "node": j,
                "d": d,
            })

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def vectors_to_gdf(stems, profile):
    crs = _crs_from_profile(profile)

    rows = []
    geoms = []

    for i in range(len(stems)):
        n_path = len(getattr(stems[i].path, "coords", []))
        vecs = getattr(stems[i], "vector", []) or []
        diams = getattr(stems[i], "segment_diameter_list", []) or []

        for j in range(n_path):
            if j >= len(vecs):
                continue

            v = vecs[j]
            try:
                raw_coords = list(v.coords[:])
            except Exception:
                continue

            coords = []
            bad_coords = False
            for xy in raw_coords:
                try:
                    x = float(xy[0])
                    y = float(xy[1])
                    if not np.isfinite(x) or not np.isfinite(y):
                        bad_coords = True
                        break
                    coords.append((x, y))
                except Exception:
                    bad_coords = True
                    break

            if bad_coords or len(coords) < 2:
                continue

            try:
                geom = LineString(coords)
            except Exception:
                continue

            try:
                if geom.is_empty or (not geom.is_valid) or geom.length <= 0:
                    continue
            except Exception:
                continue

            d = diams[j] if j < len(diams) else None
            try:
                d = float(d) if d is not None else None
            except Exception:
                d = None

            rows.append({
                "stem_id": int(i),
                "node": int(j),
                "d": d,
            })
            geoms.append(geom)

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def _safe_finalize_gpkg(tmp_path: str, final_path: str) -> str:
    tmp_path = str(tmp_path)
    final_path = str(final_path)

    tmp_dir = Path(tmp_path).parent
    dst = Path(final_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    def _cleanup():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def _copy_to(target: Path) -> str:
        shutil.copy2(tmp_path, str(target))
        _cleanup()
        return str(target)

    def _alt_new_paths(base: Path):
        yield base.with_name(base.stem + "_new" + base.suffix)
        yield base.with_name(base.stem + f"_new_{os.getpid()}" + base.suffix)

    try:
        os.replace(tmp_path, final_path)
        _cleanup()
        return final_path

    except PermissionError:
        for alt in _alt_new_paths(dst):
            try:
                return _copy_to(alt)
            except PermissionError:
                continue
        raise

    except OSError as e:
        exdev = getattr(errno, "EXDEV", 18)
        if e.errno in (18, exdev):
            try:
                return _copy_to(dst)
            except PermissionError:
                for alt in _alt_new_paths(dst):
                    try:
                        return _copy_to(alt)
                    except PermissionError:
                        continue
            raise
        raise


def _drop_bad_geoms(gdf):
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna()].copy()
    try:
        gdf = gdf[~gdf.geometry.is_empty].copy()
    except Exception:
        pass
    try:
        gdf = gdf[gdf.geometry.is_valid].copy()
    except Exception:
        pass
    try:
        geom_types = gdf.geometry.geom_type.astype(str)
        is_pointlike = geom_types.isin(["Point", "MultiPoint"])

        lengths = gdf.geometry.length
        keep = is_pointlike | ((~lengths.isna()) & (lengths > 0))

        gdf = gdf[keep].copy()
    except Exception:
        pass
    return gdf


def _normalize_dtypes(gdf):
    if gdf is None or gdf.empty:
        return gdf

    def _all_instance(values, types_):
        try:
            return all(isinstance(v, types_) for v in values)
        except Exception:
            return False

    for col in list(gdf.columns):
        if col == gdf.geometry.name:
            continue

        s = gdf[col]
        dt = str(s.dtype).lower()
        non_null = [v for v in s.tolist() if pd.notna(v)]

        if not non_null:
            gdf[col] = s.astype(object).where(~s.isna(), None)
            continue

        if "int" in dt and "interval" not in dt:
            gdf[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
            continue

        if "float" in dt:
            gdf[col] = pd.to_numeric(s, errors="coerce").astype("float64")
            continue

        if "bool" in dt:
            gdf[col] = s.astype(object).where(~s.isna(), None)
            continue

        if "string" in dt or _all_instance(non_null, str):
            gdf[col] = s.astype(object).where(~s.isna(), None)
            continue

        if _all_instance(non_null, (int, np.integer)):
            gdf[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
            continue

        if _all_instance(non_null, (int, float, np.integer, np.floating)):
            gdf[col] = pd.to_numeric(s, errors="coerce").astype("float64")
            continue

        gdf[col] = s.map(lambda v: None if pd.isna(v)
                         else str(v)).astype(object)

    return gdf


def _schema_type_for_series(series):
    non_null = [v for v in series.tolist() if pd.notna(v)]
    if not non_null:
        return "str"

    def _all_instance(values, types_):
        try:
            return all(isinstance(v, types_) for v in values)
        except Exception:
            return False

    dt = str(series.dtype).lower()
    if "int" in dt and "interval" not in dt:
        return "int"
    if "float" in dt:
        return "float"
    if _all_instance(non_null, (bool, np.bool_)):
        return "int"
    if _all_instance(non_null, (int, np.integer)):
        return "int"
    if _all_instance(non_null, (int, float, np.integer, np.floating)):
        return "float"
    return "str"


def _infer_geometry_type(gdf):
    try:
        geom_types = [g.geom_type for g in gdf.geometry
                      if g is not None and not g.is_empty]
    except Exception:
        geom_types = []

    if not geom_types:
        return "Unknown"

    unique = set(geom_types)
    if len(unique) == 1:
        return next(iter(unique))

    if unique <= {"LineString", "MultiLineString"}:
        return "MultiLineString"
    if unique <= {"Point", "MultiPoint"}:
        return "MultiPoint"
    if unique <= {"Polygon", "MultiPolygon"}:
        return "MultiPolygon"
    return "Unknown"


def _infer_fiona_schema(gdf):
    props = {}
    geom_col = gdf.geometry.name
    for col in gdf.columns:
        if col == geom_col:
            continue
        props[str(col)] = _schema_type_for_series(gdf[col])
    return {
        "geometry": _infer_geometry_type(gdf),
        "properties": props,
    }


def _feature_records(gdf, schema):
    geom_col = gdf.geometry.name
    prop_types = schema.get("properties", {})

    for _, row in gdf.iterrows():
        geom = row[geom_col]
        if geom is None:
            continue
        try:
            if geom.is_empty:
                continue
        except Exception:
            pass

        props = {}
        for col, typ in prop_types.items():
            val = row[col]
            if pd.isna(val):
                props[col] = None
            elif typ == "int":
                try:
                    props[col] = int(val)
                except Exception:
                    props[col] = None
            elif typ == "float":
                try:
                    props[col] = float(val)
                except Exception:
                    props[col] = None
            else:
                props[col] = str(val)

        yield {
            "geometry": geom.__geo_interface__,
            "properties": props,
        }


def _fiona_write_layer(path, layer_name, gdf, crs, append=False):
    schema = _infer_fiona_schema(gdf)

    layer_exists = False
    if os.path.exists(path):
        try:
            layer_exists = layer_name in set(fiona.listlayers(path))
        except Exception:
            layer_exists = False

    mode = "a" if append and layer_exists else "w"

    crs_wkt = None
    if crs is not None:
        try:
            crs_wkt = CRS.from_user_input(crs).to_wkt()
        except Exception:
            try:
                crs_wkt = crs.to_wkt()
            except Exception:
                crs_wkt = None

    print(
        f"Fiona schema for layer '{layer_name}': {schema} | mode {mode} | "
        f"layer_exists {layer_exists}",
    )

    kwargs = {
        "driver": "GPKG",
        "layer": layer_name,
        "schema": schema,
    }
    if crs_wkt is not None:
        kwargs["crs_wkt"] = crs_wkt

    with fiona.open(path, mode=mode, **kwargs) as dst:
        dst.writerecords(_feature_records(gdf, schema))


def _write_layers_to_temp_gpkg(  # noqa: C901
    layers, crs, final_path: str) \
        -> str:
    final_parent = Path(final_path).parent
    final_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="winmol_gpkg_", dir=str(final_parent))
    tmp_path = str(Path(tmp_dir) / (Path(final_path).stem + ".gpkg"))

    def _prep(name, gdf):
        if gdf is None:
            return None
        gdf = _drop_bad_geoms(_normalize_dtypes(gdf))
        if gdf is None or gdf.empty:
            return None

        if crs is not None:
            try:
                if (gdf.crs is None) or (gdf.crs != crs):
                    gdf = gdf.set_crs(crs, allow_override=True)
            except Exception:
                try:
                    gdf = gdf.set_crs(crs, allow_override=True)
                except Exception:
                    pass

        return gdf

    prepared = []
    for name, gdf in layers:
        pgdf = _prep(name, gdf)
        if pgdf is not None:
            prepared.append((name, pgdf))

    if not prepared:
        layer_name = layers[0][0] if layers else "layer"
        empty = gpd.GeoDataFrame(
            {"geometry": []}, geometry="geometry", crs=crs)
        empty.to_file(tmp_path, layer=layer_name, driver="GPKG", index=False)
        return tmp_path

    if _HAVE_PYOGRIO:
        try:
            first = True
            for name, gdf in prepared:
                pyogrio.write_dataframe(
                    gdf,
                    tmp_path,
                    layer=name,
                    driver="GPKG",
                    append=(not first),
                )
                first = False
            return tmp_path
        except Exception as e:
            print("pyogrio GeoPackage write failed, "
                  f"falling back to Fiona: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    first = True
    for name, gdf in prepared:
        print(f"Writing GPKG layer '{name}' with {len(gdf)} features")
        print(f"Layer '{name}' dtypes: {dict(gdf.dtypes.astype(str))}")
        try:
            if first:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                _fiona_write_layer(tmp_path, name, gdf, crs=crs, append=False)
                first = False
            else:
                _fiona_write_layer(tmp_path, name, gdf, crs=crs, append=True)
        except Exception as e:
            raise RuntimeError(f"Failed while writing GeoPackage layer '{name}'"
                               f" at'{tmp_path}'") from e
    return tmp_path


def write_stems_to_gpkg(stems, profile, path_prefix):
    final_path = str(Path(path_prefix).with_suffix(".gpkg"))
    crs = _crs_from_profile(profile)
    gdf_stems = stems_to_gdf(stems, profile)

    tmp_path = _write_layers_to_temp_gpkg(
        layers=[("stems", gdf_stems)],
        crs=crs,
        final_path=final_path,
    )
    return _safe_finalize_gpkg(tmp_path, final_path)


def write_all_layers_to_gpkg(stems, profile, path_prefix):
    final_path = str(Path(path_prefix).with_suffix(".gpkg"))
    crs = _crs_from_profile(profile)

    gdf_stems = stems_to_gdf(stems, profile)
    gdf_vectors = vectors_to_gdf(stems, profile)
    gdf_nodes = nodes_to_gdf(stems, profile)

    tmp_path = _write_layers_to_temp_gpkg(
        layers=[
            ("stems", gdf_stems),
            ("vectors", gdf_vectors),
            ("nodes", gdf_nodes),
        ],
        crs=crs,
        final_path=final_path,
    )
    print("Geopackage written to temporary file:", tmp_path)
    return _safe_finalize_gpkg(tmp_path, final_path)


def save_image(data, output_name, size=(15, 15), dpi=300):
    from matplotlib import pyplot as plt
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('hot')
    ax.imshow(data, aspect='equal')
    plt.savefig(output_name, dpi=dpi)


#############################################################################


"""Merge and filter tiled results"""


def _pick_id_col(gdf):
    for c in ("stem_id", "id", "ID", "StemID", "stemID"):
        if c in gdf.columns:
            return c
    return None


def _tile_id_from_prefix(prefix):
    if prefix.startswith("raster_"):
        return prefix[len("raster_"):]
    return prefix


def _read_gpkg_layer(gpkg_path, layer_names):
    errors = []
    for ln in layer_names:
        try:
            with fiona.open(gpkg_path, layer=ln) as src:
                feats = list(src)
                try:
                    crs = src.crs_wkt or src.crs
                except Exception:
                    crs = None

            if not feats:
                gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
            else:
                gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
            print(
                f"MERGE READ OK | file {gpkg_path} |"
                f" layer {ln} | rows {len(gdf)}",
                flush=True,
            )
            return gdf
        except Exception as exc:
            errors.append((ln, exc))

    if errors:
        tried = ", ".join(
            f"{ln}: {type(exc).__name__}: {exc}" for ln, exc in errors
        )
        print(
            f"MERGE READ FAIL | file {gpkg_path} | tried [{tried}]",
            flush=True,
        )
    else:
        print(
            f"MERGE READ FAIL | file {gpkg_path} | tried []",
            flush=True,
        )
    return gpd.GeoDataFrame(geometry=[])


def _read_tile_gpkg(gpkg_path):
    stems = _read_gpkg_layer(gpkg_path, ["stems", "stem", "trees", "tree"])
    nodes = _read_gpkg_layer(gpkg_path, ["nodes", "node"])
    vectors = _read_gpkg_layer(
        gpkg_path,
        ["vectors", "vector", "segments", "segment"],
    )
    return stems, nodes, vectors


def _ensure_crs(gdf, target_crs):
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame()
    if target_crs is None:
        return gdf
    if gdf.crs is None:
        gdf = gdf.copy()
        gdf.set_crs(target_crs, inplace=True)
        return gdf
    if gdf.crs != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


def _raster_filter_geom(raster_path, edge_buffer_m, ortho_bounds=None):
    """Keep-region for a tile's stems during the merge.

    Shrinking the tile footprint inward by edge_buffer_m dedups stems that
    also appear in the overlapping neighbour tile. But a tile side that
    coincides with the ortho's TRUE OUTER boundary has no neighbour, so
    shrinking there silently drops real stems (worst at the corners, where
    two sides meet). When ortho_bounds is known we buffer ONLY the
    interior-seam sides and leave boundary sides at the true extent.
    """
    if not raster_path:
        return None, None
    try:
        with rasterio.open(raster_path) as src:
            b = src.bounds
            eb = abs(edge_buffer_m)
            if ortho_bounds is not None:
                tol = eb * 1e-3
                ob = ortho_bounds

                def _side(v, o, sign):
                    # keep true extent on the ortho boundary; else shrink in
                    return v if abs(v - o) <= tol else v + sign * eb
                left = _side(b.left, ob.left, +1)
                bottom = _side(b.bottom, ob.bottom, +1)
                right = _side(b.right, ob.right, -1)
                top = _side(b.top, ob.top, -1)
                inner = box(left, bottom, right, top) if right > left \
                    and top > bottom else box(*b)
            else:
                inner = box(*b).buffer(-eb)
            if getattr(inner, 'is_empty', False):
                inner = box(*b)
            return inner, src.crs
    except Exception:
        return None, None


def _detect_tiles(work_dir, output_gpkg):
    rasters = [
        f
        for f in os.listdir(work_dir)
        if f.lower().endswith((".tif", ".tiff")) and "_roi_stem_map" in f
    ]

    tiles = []
    work_dir_path = Path(work_dir)

    if rasters:
        for rf in sorted(rasters):
            prefix = rf
            for ext in ("_roi_stem_map.tif", "_roi_stem_map.tiff"):
                if rf.endswith(ext):
                    prefix = rf.replace(ext, "")
                    break

            gpkg = sorted(
                str(p) for p in work_dir_path.rglob(f"{prefix}*.gpkg")
            )
            if not gpkg:
                continue

            if len(gpkg) > 1:
                raise RuntimeError(
                    f"Multiple GPKG candidates found "
                    f"for tile '{prefix}': {gpkg}"
                )

            tiles.append((prefix, gpkg[0], os.path.join(work_dir, rf)))

        return tiles

    for gpkg_path in sorted(work_dir_path.rglob("*.gpkg")):
        gpkg = str(gpkg_path)
        if output_gpkg:
            if os.path.abspath(gpkg) == os.path.abspath(output_gpkg):
                continue
        prefix = os.path.splitext(os.path.basename(gpkg))[0]
        tiles.append((prefix, gpkg, None))

    return tiles


def _default_output_gpkg(work_dir, output_gpkg):
    if output_gpkg is not None:
        return output_gpkg
    folder = os.path.basename(os.path.normpath(work_dir))
    return os.path.join(work_dir, f"{folder}_merged_data.gpkg")


def _remove_existing_output(path):
    if not os.path.exists(path):
        return
    try:
        os.remove(path)
    except PermissionError:
        print(
            f"MERGE OUTPUT LOCKED | keeping existing file and writing"
            f" fallback if needed: {path}",
            flush=True,
        )


def _globalize_stems(stems, tile_id, filter_geom):
    id_col = _pick_id_col(stems)
    if not id_col:
        print(f"MERGE FILTER | tile {tile_id} |"
              f" missing stem id column", flush=True)
        return gpd.GeoDataFrame(), set()

    stems = stems.copy()
    stems["_stem_id_local"] = stems[id_col].astype(str)
    stems["stem_id"] = tile_id + "_" + stems["_stem_id_local"]
    stems["tile_id"] = tile_id

    before = len(stems)
    if filter_geom is not None:
        kept_mask = stems.intersects(filter_geom)
        stems = stems[kept_mask].copy()
        print(
            f"MERGE FILTER | tile {tile_id} | before {before} |"
            f" after {len(stems)} "
            f"| filter_empty {getattr(filter_geom, 'is_empty', False)} "
            f"| filter_bounds {getattr(filter_geom, 'bounds', None)}",
            flush=True,
        )
    else:
        print(
            f"MERGE FILTER | tile {tile_id} | before {before} |"
            f" after {before} | filter none",
            flush=True,
        )

    kept_local = set(stems["_stem_id_local"].tolist())
    return stems, kept_local


def _select_child(gdf, tile_id, kept_local):
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame()

    id_col = _pick_id_col(gdf)
    if not id_col:
        return gpd.GeoDataFrame()

    out = gdf[gdf[id_col].astype(str).isin(kept_local)].copy()
    if out.empty:
        return gpd.GeoDataFrame()

    out["_stem_id_local"] = out[id_col].astype(str)
    out["stem_id"] = tile_id + "_" + out["_stem_id_local"]
    out["tile_id"] = tile_id
    return out


def _process_tile(prefix, gpkg_path, raster_path, edge_buffer_m, target_crs,
                  ortho_bounds=None):
    tile_id = _tile_id_from_prefix(prefix)

    filter_geom, raster_crs = _raster_filter_geom(
        raster_path, edge_buffer_m, ortho_bounds=ortho_bounds)

    stems, nodes, vectors = _read_tile_gpkg(gpkg_path)
    print(
        f"MERGE TILE READ | tile {tile_id} | file {gpkg_path} |"
        f" stems {0 if stems is None else len(stems)} "
        f"| nodes {0 if nodes is None else len(nodes)} |"
        f" vectors {0 if vectors is None else len(vectors)} "
        f"| raster {raster_path}",
        flush=True,
    )
    if stems is None or stems.empty:
        return None, target_crs

    if target_crs is None:
        target_crs = stems.crs or raster_crs

    stems = _ensure_crs(stems, target_crs)
    nodes = _ensure_crs(nodes, target_crs)
    vectors = _ensure_crs(vectors, target_crs)

    stems, kept_local = _globalize_stems(stems, tile_id, filter_geom)
    if stems.empty:
        return None, target_crs

    nodes_sel = _select_child(nodes, tile_id, kept_local)
    vectors_sel = _select_child(vectors, tile_id, kept_local)

    counts = (len(stems), len(nodes_sel), len(vectors_sel))
    return (stems, nodes_sel, vectors_sel, counts), target_crs


def _window_geom_from_profile(profile, window):
    bounds = rasterio.windows.bounds(window, profile['transform'])
    return box(*bounds)


def process_tile_gpkg(tile_job, gpkg_path, raster_profile, target_crs=None):
    stems, nodes, vectors = _read_tile_gpkg(gpkg_path)
    print(
        f"MERGE TILE READ | tile {tile_job.tile_id} | file {gpkg_path} |"
        f" stems {0 if stems is None else len(stems)} "
        f"| nodes {0 if nodes is None else len(nodes)} |"
        f" vectors {0 if vectors is None else len(vectors)}",
        flush=True,
    )
    if stems is None or stems.empty:
        return None, target_crs

    raster_crs = raster_profile.get('crs') if raster_profile else None
    if target_crs is None:
        target_crs = stems.crs or raster_crs

    stems = _ensure_crs(stems, target_crs)
    nodes = _ensure_crs(nodes, target_crs)
    vectors = _ensure_crs(vectors, target_crs)

    filter_geom = \
        _window_geom_from_profile(raster_profile, tile_job.inner_window)
    stems, kept_local = _globalize_stems(stems, tile_job.tile_id, filter_geom)
    if stems.empty:
        return None, target_crs

    nodes_sel = _select_child(nodes, tile_job.tile_id, kept_local)
    vectors_sel = _select_child(vectors, tile_job.tile_id, kept_local)
    counts = (len(stems), len(nodes_sel), len(vectors_sel))
    return (stems, nodes_sel, vectors_sel, counts), target_crs


def merge_selected_tile_results(
    tile_records,
    output_gpkg: str,
    raster_profile,
    keep_temp: bool = False,
):
    _remove_existing_output(output_gpkg)

    merged_stems = []
    merged_nodes = []
    merged_vectors = []
    target_crs = None
    total_stems = 0
    total_nodes = 0
    total_vectors = 0
    tile_count = 0

    tile_records = sorted(tile_records, key=lambda item: str(item[1]))
    gpkg_files = sorted(
        {str(Path(gpkg_path)) for _, gpkg_path in tile_records})
    if gpkg_files:
        merge_root = Path(
            os.path.commonpath([str(Path(p).parent) for p in gpkg_files]))
    else:
        merge_root = Path(output_gpkg).parent
    recursive_candidates = []
    if merge_root.exists():
        recursive_candidates = \
            sorted(str(p) for p in merge_root.rglob('*.gpkg'))
    print(
        f"MERGE DISCOVERY | root {merge_root} |"
        f" gpkg_files {len(recursive_candidates)}",
        flush=True,
    )
    for candidate in recursive_candidates[:5]:
        print(f'MERGE INPUT | {candidate}', flush=True)

    for tile_job, gpkg_path in tile_records:
        out, target_crs = process_tile_gpkg(
            tile_job, gpkg_path, raster_profile, target_crs=target_crs)
        if out is not None:
            stems, nodes, vectors, (n_s, n_n, n_v) = out
            merged_stems.append(stems)
            if not nodes.empty:
                merged_nodes.append(nodes)
            if not vectors.empty:
                merged_vectors.append(vectors)
            tile_count += 1
            total_stems += n_s
            total_nodes += n_n
            total_vectors += n_v
        if not keep_temp:
            try:
                os.remove(gpkg_path)
            except Exception:
                pass

    if merged_stems or merged_nodes or merged_vectors:
        written_gpkg = _write_merged(
            output_gpkg, merged_stems, merged_nodes, merged_vectors)
        print("")
        print("MERGE SUMMARY")
        print(f"Tiles processed:       {tile_count}")
        print(f"Total stems written:   {total_stems}")
        print(f"Total nodes written:   {total_nodes}")
        print(f"Total vectors written: {total_vectors}")
        print(f"Output saved to: {written_gpkg}")
    else:
        print("")
        print("MERGE SUMMARY")
        print("Tiles processed:       0")
        print("Total stems written:   0")
        print("Total nodes written:   0")
        print("Total vectors written: 0")
        print(f"No output GPKG created: {output_gpkg} (0 features written)")
    return written_gpkg if merged_stems or merged_nodes or merged_vectors\
        else output_gpkg


def _concat_merged_layer(gdfs):
    if not gdfs:
        return None

    non_empty = [gdf for gdf in gdfs if gdf is not None and not gdf.empty]
    if not non_empty:
        return None

    first = non_empty[0]
    geom_col = first.geometry.name
    crs = first.crs
    merged = pd.concat(non_empty, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry=geom_col, crs=crs)


def _write_merged(
    output_gpkg, merged_stems, merged_nodes, merged_vectors
):
    stems_gdf = _concat_merged_layer(merged_stems)
    nodes_gdf = _concat_merged_layer(merged_nodes)
    vectors_gdf = _concat_merged_layer(merged_vectors)

    crs = None
    for gdf in (stems_gdf, nodes_gdf, vectors_gdf):
        if gdf is not None and getattr(gdf, 'crs', None) is not None:
            crs = gdf.crs
            break

    tmp_path = _write_layers_to_temp_gpkg(
        layers=[
            ("stems", stems_gdf),
            ("nodes", nodes_gdf),
            ("vectors", vectors_gdf),
        ],
        crs=crs,
        final_path=output_gpkg,
    )
    return _safe_finalize_gpkg(tmp_path, output_gpkg)


def _json_list_or_empty(value):
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        out = json.loads(value)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def _stem_from_row(row):
    geom = getattr(row, 'geometry', None)
    if geom is None or getattr(geom, 'is_empty', True):
        return None
    try:
        coords = list(geom.coords)
    except Exception:
        return None
    if len(coords) < 2:
        return None
    sx = getattr(row, 'start_x', None)
    sy = getattr(row, 'start_y', None)
    ex = getattr(row, 'stop_x', None)
    ey = getattr(row, 'stop_y', None)
    start_xy = \
        coords[0] if sx is None or sy is None else (float(sx), float(sy))
    stop_xy = \
        coords[-1] if ex is None or ey is None else (float(ex), float(ey))
    return Stem(
        start=Point(start_xy),
        stop=Point(stop_xy),
        path=LineString(coords),
        vector=[],
        segment_diameter_list=[
            float(v) for v in _json_list_or_empty(getattr(row, 'd_json', []))],
        segment_length_list=[
            float(v) for v in _json_list_or_empty(getattr(row, 'l_json', []))],
        segment_volume_list=[
            float(v) for v in _json_list_or_empty(getattr(row, 'v_json', []))],
        crs=getattr(row, 'crs', None),
    )


def _stems_from_gdf(stems_gdf):
    stems = []
    if stems_gdf is None or stems_gdf.empty:
        return stems
    crs = getattr(stems_gdf, 'crs', None)
    for row in stems_gdf.itertuples(index=False):
        stem = _stem_from_row(row)
        if stem is None:
            continue
        stem.crs = crs
        stems.append(stem)
    return stems


def _restore_stem_measurement_vectors(stem, config=None):
    coords = list(getattr(getattr(stem, 'path', None), 'coords', []) or [])
    if len(coords) < 2:
        stem.vector = []
        return stem

    existing = list(getattr(stem, 'vector', []) or [])
    if len(existing) >= len(coords):
        return stem

    default_half = \
        float(getattr(config, 'diameter_vector_half_length_m', 1.0)) \
        if config is not None else 1.0
    diameter_method = \
        str(getattr(config, 'diameter_method', 'contour') or 'contour') \
        if config is not None else 'contour'
    diameters = list(getattr(stem, 'segment_diameter_list', []) or [])

    rebuilt = []
    for idx, xy in enumerate(coords):
        try:
            normal = Quant._local_normal(coords, idx)
        except Exception:
            normal = (0.0, 1.0)

        half_len = default_half
        if diameter_method == 'edt' and idx < len(diameters):
            try:
                half_len = max(default_half, float(diameters[idx]) / 2.0)
            except Exception:
                half_len = default_half

        try:
            rebuilt.append(Quant._measurement_vector(xy, normal, half_len))
        except Exception:
            continue

    stem.vector = rebuilt
    return stem


def _stems_to_layer_gdfs(stems, crs, config=None):
    profile = {'crs': crs}
    if stems:
        for stem in stems:
            _restore_stem_measurement_vectors(stem, config=config)
    stems_gdf = stems_to_gdf(stems, profile) if stems else None
    nodes_gdf = nodes_to_gdf(stems, profile) if stems else None
    vectors_gdf = vectors_to_gdf(stems, profile) if stems else None
    return stems_gdf, nodes_gdf, vectors_gdf


def _tile_edge_band_geom(raster_path, edge_buffer_m):
    if not raster_path:
        return None, None
    try:
        with rasterio.open(raster_path) as src:
            geom = box(*src.bounds)
            inner = geom.buffer(-abs(edge_buffer_m))
            if inner.is_empty:
                band = geom
            else:
                band = geom.difference(inner)
            return band, src.crs
    except Exception:
        return None, None


def _reconstruct_edge_stems_for_tiled_merge(
    merged_stems,
    tiles,
    target_crs,
    edge_buffer_m,
    config=None,
):
    """Reconnect edge candidates using the same connect_stems pipeline
    as tile-local merging.

    Important: we keep the direct output of Vec.connect_stems() and do
    NOT rebuild merged edge stems from broad contributor matching. That
    contributor-based reconstruction could bypass local gates and reintroduce
    repeated paths or jump segments.
    """
    stems_gdf = _concat_merged_layer(merged_stems)
    if stems_gdf is None or stems_gdf.empty:
        return stems_gdf, None, None

    if target_crs is not None:
        stems_gdf = _ensure_crs(stems_gdf, target_crs)

    edge_indices = set()
    for prefix, _gpkg_path, raster_path in tiles:
        tile_id = _tile_id_from_prefix(prefix)
        tile_rows = stems_gdf[stems_gdf.get('tile_id', '') == tile_id]
        if tile_rows.empty:
            continue
        edge_band, raster_crs = \
            _tile_edge_band_geom(raster_path, edge_buffer_m)
        if edge_band is None:
            continue
        tile_rows = _ensure_crs(tile_rows, target_crs or raster_crs)
        try:
            idx = tile_rows[tile_rows.intersects(edge_band)].index.tolist()
        except Exception:
            idx = []
        edge_indices.update(idx)

    print(
        f"MERGE EDGE CONNECT | candidates {len(edge_indices)} |"
        f" total {len(stems_gdf)} | edge_buffer_m {edge_buffer_m}",
        flush=True,
    )

    all_stems = _stems_from_gdf(stems_gdf)
    if not edge_indices:
        return _stems_to_layer_gdfs(all_stems, target_crs, config=config)

    edge_gdf = stems_gdf.loc[sorted(edge_indices)].copy()
    inner_gdf = stems_gdf.drop(index=sorted(edge_indices)).copy()

    inner_stems = _stems_from_gdf(inner_gdf)
    original_edge_stems = _stems_from_gdf(edge_gdf)

    if not original_edge_stems:
        return _stems_to_layer_gdfs(inner_stems, target_crs, config=config)

    import utils.Vectorization as Vec

    if config is None:
        from classes.Config import Config
        recon_cfg = Config()
    else:
        recon_cfg = config

    connected_edge_stems = \
        Vec.connect_stems(list(original_edge_stems), recon_cfg)

    # Quantify the direct connect_stems outputs so their lengths/volumes are
    # consistent with the merged geometry. connect_stems now merges the
    # parents' per-node diameter lists (Vectorization._merge_diameter_lists),
    # but guard anyway: a stem whose diameter list does not match its path
    # would crash quantify_stem with an IndexError at the very last step
    # (issue #41) -- keep its geometry and clear the measures instead of dying.
    quantified_edge_stems = []
    n_unmeasured = 0
    for stem in connected_edge_stems:
        if len(stem.segment_diameter_list) == len(stem.path.coords):
            quantified_edge_stems.append(Quant.quantify_stem(stem))
        else:
            stem.segment_length_list = []
            stem.segment_volume_list = []
            n_unmeasured += 1
            quantified_edge_stems.append(stem)
    if n_unmeasured:
        print(f"WARNING: {n_unmeasured} merged edge stems kept without "
              f"re-quantified measures (diameter/path mismatch)", flush=True)

    final_stems = inner_stems + quantified_edge_stems
    print(
        f"MERGE EDGE CONNECT | inner {len(inner_stems)} | "
        f"connected_edge {len(quantified_edge_stems)} | "
        f"final {len(final_stems)}",
        flush=True,
    )
    return _stems_to_layer_gdfs(final_stems, target_crs, config=recon_cfg)


def merge_and_filter_tiled_results(
    work_dir: str,
    output_gpkg: str | None = None,
    edge_buffer_m: float = 1.0,
    config=None,
    stem_map_path: str | None = None,
):
    work_dir = os.path.abspath(work_dir)
    output_gpkg = _default_output_gpkg(work_dir, output_gpkg)

    # Full-ortho extent: lets _raster_filter_geom keep stems on the ortho's
    # true outer boundary (only interior tile seams get the dedup shrink).
    ortho_bounds = None
    if stem_map_path:
        try:
            with rasterio.open(stem_map_path) as _s:
                ortho_bounds = _s.bounds
        except Exception:
            ortho_bounds = None

    _remove_existing_output(output_gpkg)

    tiles = _detect_tiles(work_dir, output_gpkg)
    recursive_candidates = \
        sorted(str(p) for p in Path(work_dir).rglob('*.gpkg'))
    print(
        f"MERGE DISCOVERY | root {work_dir} |"
        f" gpkg_files {len(recursive_candidates)}",
        flush=True,
    )
    for candidate in recursive_candidates[:5]:
        print(f'MERGE INPUT | {candidate}', flush=True)
    if not tiles:
        # No tiles wrote a GeoPackage. That is what an orthomosaic with no
        # detections looks like, and it used to raise here -- marking the
        # whole ortho FAILED for the crime of containing no trees. Fall
        # through to the zero-feature summary below instead; the MERGE
        # DISCOVERY line above still shows the root that came up empty, so
        # a genuinely wrong work_dir stays diagnosable.
        print(f"No .gpkg files found in: {work_dir} (0 detections)",
              flush=True)

    merged_stems = []
    merged_nodes = []
    merged_vectors = []

    target_crs = None
    tile_count = 0
    total_stems = 0
    total_nodes = 0
    total_vectors = 0

    for prefix, gpkg_path, raster_path in tiles:
        out, target_crs = _process_tile(
            prefix,
            gpkg_path,
            raster_path,
            edge_buffer_m,
            target_crs,
            ortho_bounds=ortho_bounds,
        )
        if out is None:
            continue

        stems, nodes, vectors, (n_s, n_n, n_v) = out
        merged_stems.append(stems)
        if not nodes.empty:
            merged_nodes.append(nodes)
        if not vectors.empty:
            merged_vectors.append(vectors)

        tile_count += 1
        total_stems += n_s
        total_nodes += n_n
        total_vectors += n_v

    if merged_stems or merged_nodes or merged_vectors:
        stems_gdf, nodes_gdf, vectors_gdf = \
            _reconstruct_edge_stems_for_tiled_merge(
                merged_stems,
                tiles,
                target_crs,
                edge_buffer_m,
                config=config,
            )
        final_stems = [stems_gdf] \
            if stems_gdf is not None and not stems_gdf.empty else []
        final_nodes = [nodes_gdf] \
            if nodes_gdf is not None and not nodes_gdf.empty else []
        final_vectors = [vectors_gdf] \
            if vectors_gdf is not None and not vectors_gdf.empty else []
        written_gpkg = _write_merged(
            output_gpkg,
            final_stems,
            final_nodes,
            final_vectors,
        )
        written_layers = []
        try:
            written_layers = list(fiona.listlayers(written_gpkg))
        except Exception as exc:
            print(
                f"MERGE VERIFY FAIL | file {written_gpkg} "
                f"| {type(exc).__name__}: {exc}",
                flush=True,
            )

        final_stem_count = 0 if stems_gdf is None else len(stems_gdf)
        final_node_count = 0 if nodes_gdf is None else len(nodes_gdf)
        final_vector_count = 0 if vectors_gdf is None else len(vectors_gdf)

        print("")
        print("MERGE SUMMARY")
        print(f"Tiles processed:       {tile_count}")
        print(f"Total stems written:   {final_stem_count}")
        print(f"Total nodes written:   {final_node_count}")
        print(f"Total vectors written: {final_vector_count}")
        print(f"Layers written:        {written_layers}")
        print(f"Output saved to: {written_gpkg}")
    else:
        print("")
        print("MERGE SUMMARY")
        print("Tiles processed:       0")
        print("Total stems written:   0")
        print("Total nodes written:   0")
        print("Total vectors written: 0")
        print(f"No output GPKG created: {output_gpkg} (0 features written)")
    return written_gpkg if merged_stems or merged_nodes or merged_vectors \
        else output_gpkg
