#!/usr/bin/env python

################################################################################
"""Imports"""

import os
import sys

from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classes.Config import Config
from classes.Timer import Timer
from utils import IO
from utils import Prediction as Pred
from utils import Quantification as Quant
from utils import Skeletonization as Skel
from utils import Vectorization as Vec

if __name__ == '__main__':

    # Create a timer to measure the execution time of the script
    tt = Timer()
    tt.start()

    # Extract command-line arguments
    model_path = str(sys.argv[1])
    img_path = str(sys.argv[2])
    pred_dir = str(sys.argv[3])
    output_dir = str(sys.argv[4])

    # Create a Config instance and display its settings
    config = Config()
    config.display()

    # Load the model from the HDF5 file
    model = keras.models.load_model(model_path, compile=False)

    # Display a summary of the loaded model architecture
    model.summary()

    # Extract the base name of the input image file
    file_name = os.path.splitext(os.path.basename(img_path))[0]

    # Generate the name for the predicted image file
    pred_name = pred_dir + 'pred_' + file_name + '.tiff'

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
    end_nodes = Vec.rebuild_endnodes_from_stems(stems)

    # Quantify the properties of the identified stems
    stems = Quant.quantify_stems(stems, pred, profile)

    # Export the predicted stem map and stems information to GeoJSON
    IO.export_stem_map(pred, profile, pred_dir, file_name)
    IO.stems_to_geojson(stems, output_dir + file_name)

    # Stop the timer and display the elapsed time
    tt.stop()
