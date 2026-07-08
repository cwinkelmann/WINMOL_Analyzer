import os

# The pretrained WINMOL models are Keras 2 HDF5 artifacts and fail to load under
# the Keras 3 bundled with TensorFlow >= 2.16. Route tf.keras to the legacy
# Keras 2 shim (tf-keras). This runs at collection time, before any test module
# imports TensorFlow.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
