#!/usr/bin/env python
"""Run the WINMOL downstream stages on manual labels instead of a model prediction.

Mask only: the mask goes through the full vectoriser, so the result shows what the pipeline
produces from a perfect segmentation.

    python winmol_from_labels.py --mask stems.gpkg --out plot/trees

Mask and centre lines: skeletonising and stem connection are skipped and diameters, lengths and
volumes are measured along the given lines, which gives reference values for those metrics.

    python winmol_from_labels.py --mask stems.gpkg --lines lines.gpkg --out plot/trees

The mask is a GeoTIFF (anything above 0 is stem) or polygons. Polygons are rasterised on the grid
of --reference, usually the stem map being compared against, or at --pixel-size, which defaults
to the Analyzer's stem-map scale. Writes <out>.gpkg with the usual stems, vectors and nodes layers
and <out>_summary.json with plot totals. Lines sharing an ID stay one stem: `source_id` in the
`stems` layer holds that ID when lines are given.
"""
import argparse
import os
import sys

# Same pin as winmol_run.py, which the mask-only route imports: stem connection depends on
# set order, so an unpinned run would not be reproducible.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, "-u"] + sys.argv)

from shapely.ops import unary_union  # noqa: E402

from classes.Config import Config  # noqa: E402
from utils import ManualLabels as ML  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask", required=True, help="GeoTIFF or polygon file")
    ap.add_argument("--mask-layer")
    ap.add_argument("--lines", help="centre lines; without them the vectoriser finds the lines")
    ap.add_argument("--lines-layer")
    ap.add_argument("--id-field", default="ID", help="stem ID field of the lines (default: ID)")
    ap.add_argument("--reference", help="raster whose grid the polygons are rasterised on")
    ap.add_argument("--pixel-size", type=float, help="metres; default tile_size / img_width")
    ap.add_argument("--aoi", help="clip the lines to this polygon before measuring")
    ap.add_argument("--out", required=True, help="output prefix")
    return ap.parse_args(argv)


def run_lines(args, config, mask, grid):
    lines = ML.load_vector(args.lines, args.lines_layer, crs=grid["crs"])
    aoi = None
    if args.aoi:
        aoi = unary_union(ML.load_vector(args.aoi, crs=grid["crs"]).geometry)
    stems = ML.stems_from_lines(lines, args.id_field, config.measuring_point_spacing_m, aoi=aoi)
    if not stems:
        raise SystemExit("no centre lines left to measure")
    return ML.write_outputs(ML.measure_lines(stems, mask, grid, config), grid, args.out)


def run_mask_only(args, mask, grid):
    # Imported here because winmol_run pulls in the whole prediction stack.
    from winmol_run import ImageProcessing
    mask_path = ML.write_mask(mask, grid, f"{args.out}_mask.tif")
    processing = ImageProcessing(None, mask_path, mask_path, args.out, "Trees")
    plan = processing.build_plan(processing.detect_hardware())
    processing.run_vector_phase(plan, pred_path=mask_path)
    path = f"{args.out}.gpkg"
    if not os.path.exists(path):
        raise SystemExit("the vectoriser found no stems in this mask")
    return path


def main(argv=None):
    args = parse_args(argv)
    if args.aoi and not args.lines:
        raise SystemExit("--aoi only applies with --lines; clip the mask before passing it")
    config = Config()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pixel_size = args.pixel_size or config.tile_size / config.img_width
    mask, grid = ML.load_mask(args.mask, args.mask_layer, args.reference, pixel_size)
    path = run_lines(args, config, mask, grid) if args.lines else run_mask_only(args, mask, grid)
    summary = ML.summarise(path)
    summary["mode"] = "mask_and_lines" if args.lines else "mask_only"
    summary["pixel_size_m"] = abs(grid["transform"].a)
    ML.write_summary(summary, f"{args.out}_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
