# SupervisedNMF (NESTEDCV PIPELINE)

## Description

This project performs Supervised NMF for feature engineering and evaluates the effect of the converted features.

# How to use

### Step 1: Clone the repository.

    git clone  https://gitlab.genesolutions.vn/ecd-data/supervisednmf.git

### Step 2:  Change to the project directory.

    cd supervisednmf

### Step 3: Run the `submit_run.ipynb` file with the following variables:

- `run_name`: Name of the run.

- `feature_name`: Name of the feature.

- `rank`: Number of components.

- `nmf_init_mode`: Initialization mode for SNMF. Options: `'random'`, `'nndsvd'`, `'nndsvda'`, `'nndsvdar'`.

- `loss_type`:  Supervised loss type. Options: `'LR'`, `'LDA'`, `'SVM'`.

- `iter`: Number of iteractions.

- `tolerance`: Tolerance level.

- `patience`: Number of epochs for early stopping.

- `alpha`: Alpha parameter for ADADELTA.

- `epsilon`: Epsilon parameter for ADADELTA.

### Step 4: Run the `find_best_SNMF.ipynb` file with the following variables:

- `feature_name`: Name of the feature.

    - Example: You see `Best SNMF =>  test_LR_random_EM_3_50000_1e-07_15_0.9_1e-06`, this is a best model with `case='test_LR_random_EM_3_50000_1e-07_15_0.9_1e-06'`

### Step 5: Run the `submit_run_final.ipynb` file with the following variables:

- `case`: Get from Step 4.

Go to `output_path`/feature.csv to get the transformed feature.


## Docker image

The Docker image used for this project is `gene110/samtools_python:v4`, which includes Python 3.8.10 and Samtools 1.16.
