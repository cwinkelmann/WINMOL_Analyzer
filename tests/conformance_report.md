# Model conformance report

Same crop (`tests/fixtures/crop_input.tif`), same config snapshot,
same pipeline for every model — only the model file differs.
Detections legitimately differ (different weights); every
structural check passed for the models listed as `ok`.

| model | loader | legacy keras | ok | load s | predict s | stem px % | parts | stems | length m | volume m³ |
|---|---|---|---|---|---|---|---|---|---|
| zenodo-unet-gends | Functional | 1 | ✅ | 0.73 | 1.82 | 12.896 | 258 | 62 | 597.2 | 75.629 |
| twostage-baseline-hdf5 | ? | 1 | ❌ | — | — | — | — | — | — | — |
| twostage-baseline-512transfer | Functional | 1 | ✅ | 0.65 | 1.72 | 6.901 | 542 | 97 | 687.9 | 19.465 |
| twostage-unet-onnx | OnnxSegmenter | 0 | ✅ | 1.15 | 1.54 | 8.271 | 202 | 55 | 525.5 | 42.807 |
| twostage-deeplab-onnx | OnnxSegmenter | 0 | ✅ | 3.33 | 0.26 | 7.065 | 152 | 52 | 448.6 | 33.855 |

## Pairwise stem-map agreement

| model A | model B | IoU (foreground) | pixel agreement |
|---|---|---|---|
| zenodo-unet-gends | twostage-baseline-512transfer | 0.265 | 0.8849 |
| zenodo-unet-gends | twostage-unet-onnx | 0.543 | 0.9374 |
| zenodo-unet-gends | twostage-deeplab-onnx | 0.480 | 0.9299 |
| twostage-baseline-512transfer | twostage-unet-onnx | 0.389 | 0.9332 |
| twostage-baseline-512transfer | twostage-deeplab-onnx | 0.347 | 0.9324 |
| twostage-unet-onnx | twostage-deeplab-onnx | 0.655 | 0.9680 |
