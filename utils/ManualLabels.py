"""Downstream metrics from manual labels instead of a model prediction.

The measuring code is the pipeline's own (`Quantification.quantify_stems`), so a number from
hand labels and a number from a prediction differ because of the input, not the method.
Hand-drawn centre lines are resampled to `measuring_point_spacing_m` first, the spacing the
vectoriser gives its nodes, because diameters and volumes depend on where nodes fall.
"""
import json

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union

from classes.Stem import Stem
from utils import IO
from utils import Quantification as Quant

RASTER_SUFFIXES = (".tif", ".tiff")


def read_layer(path, layer=None):
    # pyogrio is a core requirement; geopandas' default engine depends on the fiona version
    return gpd.read_file(path, layer=layer, engine="pyogrio")


def load_vector(path, layer=None, crs=None):
    gdf = read_layer(path, layer)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if crs is not None:
        if gdf.crs is None:
            raise ValueError(f"{path} has no CRS, so it cannot be aligned with {crs}")
        gdf = gdf.to_crs(crs)
    return gdf


def grid_from_reference(path):
    with rasterio.open(path) as src:
        return {"transform": src.transform, "width": src.width, "height": src.height,
                "crs": src.crs}


def grid_from_bounds(bounds, pixel_size, crs, pad_m=2.0):
    """A north-up grid around `bounds`, padded so measuring vectors stay inside it."""
    minx, miny, maxx, maxy = bounds
    minx, miny = minx - pad_m, miny - pad_m
    maxx, maxy = maxx + pad_m, maxy + pad_m
    return {"transform": from_origin(minx, maxy, pixel_size, pixel_size),
            "width": int(np.ceil((maxx - minx) / pixel_size)),
            "height": int(np.ceil((maxy - miny) / pixel_size)),
            "crs": crs}


def rasterize_polygons(polygons, grid):
    """Binary uint8 mask of the polygons on `grid`, by pixel centre as the stem map is."""
    shapes = [(g, 1) for g in polygons.to_crs(grid["crs"]).geometry]
    if not shapes:
        return np.zeros((grid["height"], grid["width"]), dtype=np.uint8)
    return rasterio.features.rasterize(
        shapes, out_shape=(grid["height"], grid["width"]), transform=grid["transform"],
        fill=0, dtype=np.uint8)


def mask_profile(mask, grid):
    return {"driver": "GTiff", "dtype": "uint8", "count": 1, "nodata": None,
            "width": mask.shape[1], "height": mask.shape[0],
            "crs": grid["crs"], "transform": grid["transform"], "compress": "deflate"}


def write_mask(mask, grid, path):
    with rasterio.open(path, "w", **mask_profile(mask, grid)) as dst:
        dst.write(mask, 1)
    return path


def load_mask(path, layer=None, reference=None, pixel_size=None, crs=None):
    """(mask, grid) from a raster, or from polygons rasterised onto the given grid."""
    if path.lower().endswith(RASTER_SUFFIXES):
        with rasterio.open(path) as src:
            mask = (src.read(1) > 0).astype(np.uint8)
            return mask, {"transform": src.transform, "width": src.width,
                          "height": src.height, "crs": src.crs}
    polygons = load_vector(path, layer)
    if reference:
        grid = grid_from_reference(reference)
    else:
        polygons = polygons.to_crs(crs) if crs else polygons
        grid = grid_from_bounds(polygons.total_bounds, pixel_size, polygons.crs)
    return rasterize_polygons(polygons, grid), grid


def resample_line(line, spacing):
    """Evenly spaced nodes along the line, always keeping both ends."""
    segments = max(int(round(line.length / spacing)), 1)
    at = np.linspace(0.0, line.length, segments + 1)
    return LineString([line.interpolate(float(d)) for d in at])


def _source_id(value, row):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"null-{row}"       # an unlabelled line is its own stem, never merged with others
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _line_parts(geoms):
    """Connected pieces of one stem's lines, with touching lines joined end to end."""
    joined = unary_union(geoms)
    merged = linemerge(joined) if hasattr(joined, "geoms") else joined
    parts = merged.geoms if hasattr(merged, "geoms") else [merged]
    return sorted((p for p in parts if p.geom_type == "LineString" and p.length > 0),
                  key=lambda p: tuple(p.coords[0]))


def stems_from_lines(lines, id_field, spacing, aoi=None):
    """One Stem per connected line. `source_id` keeps the hand ID, shared by a stem's parts."""
    if id_field not in lines.columns:
        raise ValueError(f"no field {id_field!r} in the centre lines; "
                         f"available: {[c for c in lines.columns if c != 'geometry']}")
    by_id = {}
    for row, (value, geom) in enumerate(zip(lines[id_field], lines.geometry)):
        by_id.setdefault(_source_id(value, row), []).append(geom)
    stems = []
    for source_id in sorted(by_id):
        geoms = by_id[source_id]
        if aoi is not None:
            geoms = [g.intersection(aoi) for g in geoms]
            geoms = [g for g in geoms if not g.is_empty]
            if not geoms:
                continue
        for part, line in enumerate(_line_parts(geoms)):
            path = resample_line(line, spacing)
            stem = Stem(start=Point(path.coords[0]), stop=Point(path.coords[-1]), path=path,
                        vector=[], segment_diameter_list=[], segment_length_list=[],
                        segment_volume_list=[])
            stem.source_id = source_id
            stem.part = part
            stems.append(stem)
    return stems


def measure_lines(stems, mask, grid, config):
    """The pipeline's diameter, length and volume along the given lines, in a stable order."""
    measured = Quant.quantify_stems(stems, mask, mask_profile(mask, grid), config=config)
    return sorted(measured, key=lambda s: (s.source_id, s.part))


def summarise(gpkg_path):
    """Plot totals from a trees GeoPackage, whichever route produced it."""
    stems = read_layer(gpkg_path, "stems")
    nodes = read_layer(gpkg_path, "nodes")
    diameters = nodes["d"].dropna().to_numpy(dtype=float)
    positive = diameters[diameters > 0]
    return {
        "stems": int(stems["source_id"].nunique()) if "source_id" in stems else len(stems),
        "lines": len(stems),
        "length_m": float(stems["length"].sum()),
        "volume_m3": float(stems["volume"].sum()),
        "nodes": len(diameters),
        "nodes_zero_diameter": int((diameters <= 0).sum()),
        "diameter_mean_m": float(positive.mean()) if len(positive) else None,
        "diameter_median_m": float(np.median(positive)) if len(positive) else None,
    }


def write_outputs(stems, grid, out_prefix):
    path = IO.write_all_layers_to_gpkg(stems, mask_profile(np.zeros((1, 1)), grid), out_prefix)
    return path


def write_summary(summary, path):
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    return path
