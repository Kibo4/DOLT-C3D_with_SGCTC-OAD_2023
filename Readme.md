# Online Action Detection : focus on earliness with OLT-C3D and CTC
## packages and roles

### Package Evaluation
This package contains three files:
* ExportBounds.py: this file exports, given a model, the bounds for sequences in the test set
* ExportBounds_MultiFold: this file exports, given a model,  the bounds for
sequences in the test set for the multi-fold cross-validation, it was used for MSRC12 and G3D
* ExportVisualResult.py: this file exports the visual results of the actions in the test set
### Package Model
Contains the model OLT-C3D. Two differents version : single and double-stream
### Package Tools
Contains multiples subpackages : 
* CallBacks: contain a callback used in training to visualize intermediate results
* DataSetFormaters: all the scripts to format datasets, exports/build splits
* Evaluation: contain a script to compute the metric used in the callback
* Gesture: contain a set of classes to represents joints, skeletons, sequences, labels...
* LossesAndMetrics: contain the losses and metrics used in the training
* RepresentationExtractor: contain the scripts to build the differents representations (heatmaps-based), used in preprocessing
* Strategy: contain the online strategy used in testing
## Requirements
Tested with python 2.10 and tensorflow 2.10.1. See the full list of requierements in the file requirements.txt

## Preprocessing

## Training

## Testing: export the bounds of the actions

## FAQ