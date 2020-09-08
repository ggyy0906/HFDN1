## Code for submission **Hierarchical Feature Disentangle Network for Universal Domain Adaptation** 

### Requirements
The code has been tested running under Python 3.7, with the following packages installed (along with their dependencies):

- pytorch == 1.2
- tensorboard == 2.0.0

### Files in the folder
- `mdata/`: tool functions of handling data input.
- `mground/`: utilities.
- `mtrain/`: training functions.
- `mmodel/`: impalements of models.
    - `mmodel/TFDN`: impalements of TFDN.

### Parameters
All parameters for training can be modified in `mmodel/basic_params.py` and `mmodel/TFDN/basic_params.py`.

### Datasets
Dataset should be placed in `DATASET/` folder with dataset name.
For example, put *OfficeHone* dataset at `DATASET/OFFICEHOME/`.