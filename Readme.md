# Online Action Detection : focus on earliness with OLT-C3D and CTC
This code has been used to make the experiments of the following paper:

Early Action Detection at instance-level driven by a controlled CTC-Based Approach", William Mocaër; Eric Anquetil; Richard Kulpa. July 2023

```
@article{mocaer2023,
  title = {Early Action Detection at instance-level driven by a controlled CTC-Based Approach},
  author = {Mocaër, William and Anquetil, Eric and Kulpa, Richard},
  year = {2023},
  journal = {},
  volume = {},
  number = {},
  pages = {},
  doi = {},
}
```
## packages informations

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
Tested with python 3.10 and tensorflow 2.10.1. See the full list of requierements in the file environment.yml
Run the traning with libtcmalloc minimal ! (LD_PRELOAD=/usr/lib/libtcmalloc_minimal.so.4 python Training.py")
-> otherwise there will be memory problem during training due to data augmentation (known tf bug)
## Preprocessing
Contains the scripts to build the different representations (heatmaps-based), used in preprocessing
It takes as input (args) the db name, the "db.info" name and "hp.info" name
It consider that all database are the same root folder, and that the db.info and hp.info are in the folder root/dbname/
The absolute path root of the datasets is specified in the code (pathDB), you need to change it manually to make it work
the db.info and hp.info are in json format, 
db.info contains information about dataset which is necessary for the preprocessing and evaluation
hp.info contains information about the hyperparameters of the preprocessing (only, not model hp), which is necessary for the preprocessing
The dataset folder must contain the Split folder, which contains the splits of the dataset (train, [validation,] test). The split file name should be specified in the db.info file.
It export the preprocessed data into folders :
* pathDB + "PreprocessedData" for training data
* pathDB + "PreprocessedDataTest" for testing data
* pathDB + "PreprocessedDataValid" for validation data  (if no validation split is specified in the split file, it uses 10% of the training data, after shuffling)

The Folder "Tools/RepresentationExtractor" contains the scripts to build the differents representations (heatmaps-based), used in preprocessing
it contains the file "MapperIdVoxelizer" which map an id to a representation strategy
the id is specified in the hp.info file, in the "modeVoxelisation" field
Any "Voxelizer" designate a representation strategy


## Dataset format requirements
See https://www-shadoc.irisa.fr/oad-datasets/ for more information about the datasets and their format


## Training
Two files are available for training : Training.py and TraningNOSG.py. The content is the same, the difference is the hyperparameters specified. NOSG means No Segmentation-Guided CTC usage.
Hyperparameters of the model is specified directly in the training file in the dictionnary variabiable "config". It can be set automatically threw wandb process if needed.
The training file takes as required input (args) :
* db name (located in "pathDB" specified in code),
* db.info name (in pathDB)
* hp.info name (in pathDB) (needed to find the right proprocessed folder containing the preprocessed data)
* the Hyperparameter $\Psi$ (weight of the label prior) for training

## Testing: export the bounds of the actions
To evaluate with the evaluation framework (also available in another repository), you need to export the bounds of the actions of the test set.
WARNING: only the exported sequence will be evaluated, so be careful to export all the sequences of the test set which correspond to the official test set (for comparison).
Three files are available for exporting the bounds : ExportBounds.py, ExportBounds_MultiFold.py and ExportBounds_MultiFold_fromFile.py.
* ExportBounds.py contains the script to export the bounds of the actions for a single-fold model. Only one model is used.
* ExportBounds_MultiFold.py contains the script to export the bounds of the actions for a multi-fold case, it calls ExportBounds.py.
* ExportBounds_MultiFold_fromFile.py (preferred) is also for multi-fold case but it loads the model from a file, it calls ExportBounds.py
Note that multi-fold can be used to evaluate multiple models at the same time on the same

One should better used ExportBounds_MultiFold_fromFile.py.
This file contains a folder path which must contains all the files for evaluation. Each file contains a list of model. Each model is its name.

Configuration of the folders for the bound exportation : 
root/dbName:
* modelRunGrouped/ : source folder containing a list of file (name can be changed in the code)
  * Test1.txt : contains the list of models names (separated with ',') to evaluate for a given split. Each model should be in the expOut folder (otherwise it is downloaded threw wandb, but it should be configured to work).
  * Test2.txt
  * Test3.txt
  * ....
* expOut/ : target contains the exported bounds for each model (name can be changed in the code
  * Test1/
    * model1/ : contains the exported bounds for the model1
      * Weights/ : contains the weights of the model1 (should be here before execution, and wandb must be connected and the model available online)
        * config.txt : contains the concat dictionnary of hp.info and db.info, needed to know the split test folder
        * model/ : contains the necessary to reload the model with weights
        * Weights/ : optional, contains the output of summary of the model (weight numbers)
      * Bounds/ : generated by the script, contains the bounds and the class of the actions for each sequence
        * seq1
        * seq2
        * ....
      * Frames/ Generated by the script, contains the class prediction for each frame of each sequence 
        * seq1
        * seq2
        * ....
    * model2/ : contains the exported bounds for the model2
    * ....
    * ResultsMultiFold (After evaluation with eval framework)
  * Test2/
    * ...
  * Test3/
    * ...
  
All the results in testX independently will be aggregated and averaged in the evaluation framework.

_More details are given in the script files._
