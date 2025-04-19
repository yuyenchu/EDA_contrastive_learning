# EDA contrastive learning
Experiment replication of paper - [Contrastive Learning of Electrodermal Activity
Representations for Stress Detection](https://openreview.net/forum?id=bSC_xo8VQ1b)

## Installation
```
pip install -r requirements.txt
```
(Optional) Install clearml for CI/CD and monitoring
```
pip install clearml
```
**[NOTE]** If using tensorflow>=2.16.0 / keras>=3.0.0, uncomment following lines in `model.py`
```
# from keras import saving
...
# @saving.register_keras_serializable()
````

## Setup config
1. Edit `augment.json` for labeled and unlabled data augmentation, for functions and params, reference `augment_params.json`
2. Edit `USE_CLEARML` in `main.py` and `preprocessing.py` for using clearml or not
3. (Optional) If using clearml, run following command to config for your credentials
    ```
    cleaml-init
    ```


## Usage
- Preprocessing
    1. download [WESAD dataset](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
    2. unzip dataset to folder
    3. run script
        ```
        python preprocessing.py -p PATH_TO_DATASET_FOLDER
        ```
- Training
    1. run script
        ```
        python main.py
        ```