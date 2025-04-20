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
    1. Download [WESAD dataset](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
    2. Unzip dataset to folder
    3. Run script
        ```
        python preprocessing.py -p PATH_TO_DATASET_FOLDER
        ```
- Training
    1. Run script
        ```
        python main.py -d PATH_TO_DATASET_FOLDER -e EPOCHS -b BATCH_SIZE
        ```
- Hyper parameter search (with *clearml* and *optuna*):
    1. Install `clearml` and `optuna`
    2. Edit configuration of `optimizer` in `hp_search.py` as you need, e.g. `objective_metric_title`, `execution_queue`, `optimization_time_limit`.
    3. Run script
        ```
        python hp_search.py
        ```
- Add new augment class
    1. Go to `augmenter.py` and start editing
    2. Implement your new augment class, following these rules:
        - It needs to have `__init__` and `__call__` functions implemented with numpy operations only. 
        - The parameters for `__init__` should all have default values.
        - The function `__call__` should take in 3 parameters `x`, `left_buffer`, `right_buffer` each for eda data of shape (N,1) where N is the number of points per sample. 
        - Use the `@aug_export` dcorator to register your class. The name passed to the decorator will be used as key to access your class in `augment.json`.
        - To enable hyper parameter search for your class, make sure to add `hp` variable to the class.
    3. Your new class should be in following format
        ```python
        @aug_export('NewAugment_Key')
        class NewAugment:
            hp = {
                'param_1': (min_int_1, max_int_1),
                'param_2': (min_float_2, max_float_2),
            }
            ...
            def __init(self, param_1=default_int_1, param_2=default_float_2):
                ...
            
            def __call__(self, x, left_buffer, right_buffer):
                ...
                return np.array(out_arr).astype(np.float32)
        ```
    4. Run script to generate new `augment_params.json`
        ```
        python augmenter.py
        ```
