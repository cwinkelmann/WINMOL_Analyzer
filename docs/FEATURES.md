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
currently the plugin fails to install on a mac but also on python 3.11 on linux with cuda. The venv installation process is quite flakey. 