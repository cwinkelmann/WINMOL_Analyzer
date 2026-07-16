#!/usr/bin/env python

################################################################################
"""Imports"""

import os
import sys

# Models are ONNX, loaded via IO.load_model_from_path (onnxruntime, no
# TensorFlow). A legacy Keras .hdf5 still loads if TensorFlow is installed
# (IO handles it and sets the legacy-Keras shim as needed).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classes.Config import Config
from classes.Timer import Timer
from utils import IO
from utils import Prediction as Pred
from utils import Quantification as Quant
from utils import Skeletonization as Skel
from utils import Vectorization as Vec


def run_pipeline(model_path, img_path, pred_dir, output_dir, config=None):
    """Run the full standalone WINMOL pipeline on a single orthomosaic.

    Loads the U-Net model, predicts the stem map, reconstructs and quantifies
    stems, and writes the stem-map raster plus a GeoPackage of the results.

    Returns a dict with the produced output paths and the stem count:
        {"stem_map_path": <.tiff>, "gpkg_path": <.gpkg>, "num_stems": int}
    """
    if config is None:
        config = Config()
    config.display()

    # Load the model (.onnx via onnxruntime, or .hdf5/.keras if TF present)
    model = IO.load_model_from_path(model_path)

    # Display a summary of the loaded model, if available
    if hasattr(model, "summary"):
        model.summary()

    # Extract the base name of the input image file
    file_name = os.path.splitext(os.path.basename(img_path))[0]

    # Load the input orthomosaic image and its profile using IO module
    img, profile = IO.load_orthomosaic(img_path, config)

    # Perform prediction on the input image with resampling
    pred, profile = Pred.predict_with_resampling_per_tile(
        img,
        profile,
        model,
        config
    )

    # Find stem segments in the predicted image using Skeletonization module
    segments = Skel.find_segments(pred, config, profile)

    # Restore geoinformation to the stem segments using Vectorization module
    segments = Vec.restore_geoinformation(segments, config, profile)

    # Build stem parts from the segments using Vectorization module
    stems = Vec.build_stem_parts(segments)

    # Connect individual stems to form complete structures
    stems = Vec.connect_stems(stems, config)

    # Rebuild endnodes from the connected stems
    Vec.rebuild_endnodes_from_stems(stems)

    # Quantify the properties of the identified stems
    stems = Quant.quantify_stems(stems, pred, profile)

    # Export the predicted stem map and stems information
    IO.export_stem_map(pred, profile, pred_dir, file_name)
    gpkg_path = IO.write_all_layers_to_gpkg(
        stems, profile, output_dir + file_name)

    stem_map_path = os.path.join(pred_dir, f'{file_name}.tiff')
    return {
        "stem_map_path": stem_map_path,
        "gpkg_path": gpkg_path,
        "num_stems": len(stems),
    }


if __name__ == '__main__':

    # Create a timer to measure the execution time of the script
    tt = Timer()
    tt.start()

    # Extract command-line arguments
    model_path = str(sys.argv[1])
    img_path = str(sys.argv[2])
    pred_dir = str(sys.argv[3])
    output_dir = str(sys.argv[4])

    result = run_pipeline(model_path, img_path, pred_dir, output_dir)
    print("Pipeline finished:", result)

    # Stop the timer and display the elapsed time
    tt.stop()
