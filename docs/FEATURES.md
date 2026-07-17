### load models as keras format
https://www.tensorflow.org/tutorials/keras/save_and_load
hdf5 seems to deprecated, but should still work in 2.21

.keras model loading should work too.

### onnx loading
load models as onnx format, remove dependencies to tensorflow and keras to migrate to pytorch completely.


#### Speed up inference and the vectorisation later
use some JIT or other compilation tricks to speed it up.

### fix the QGIS plugin building
it should work on windows and linux with multiple QGIS versions


### Fix bugs found in the two code reviews.


### CI on GITHUB
the tests should run in github to ensure changes are not ruining something


### Fix QGIS plugin installation
currently the plugin fails to install on a mac but also on python 3.11 on linux with cuda. The venv installation process is quite flakey. In general multipe ways of installing exist. 
A classic windows install should work for QGIS 3.44 and 4, in linux one could install qgis into a conda environment or via apt. The plugin should be installable via the QGIS plugin manager and also via a zip file. The installation process should be documented in detail. On a mac options are slightly reduced. In all cases a CUDA GPU or Apple MPS should be usable. 

#### API connectivity
Instead of having a analyser locally, a worker on a beefy GPU machine could be used. The QGIS plugin would then send the orthomosaic to the worker and receive the results back. This would allow to use a GPU on a different machine than the one running QGIS. The API should be secure and support authentication and authorization. The API should be documented in detail.

#### QGIS features: manual canvas analysis
it Would be great if I can draw stem outlines and then look into what the segmentation model predicts, but also what the skeletonisation predicts. This would allow to compare the two methods and also to manually correct the predictions. The manual corrections should be saved and could be used to retrain the model.
For this the plugin would need to read either a 1-channel raster or shapefile which closed polygons.


### Local Environment Setup
currently the setup seems incomplete. There are a ton of scripts inthe root folder which are not documented and have broken imports, from qgis.PyQt.QtCore import QObject, pyqtSignal

### Multiple geotiff processing in QGIS
Assuming there are multiple geotiff raster layers in QGIS, it should be possible to run the winmol analyser on all of them in one go. The plugin should allow to select multiple layers and then run the analysis on all of them. The results should be saved in a separate folder/group with the same name as the input layer. The results should be added to the QGIS canvas automatically.


### Speedup the model inference
candidates would quantization, prunning, onnx speedup Especially on cpu it should run faster
#### Do a speed and feature benchmark
Are all the optimisations faster and by how much? How does accuracy change in the end.