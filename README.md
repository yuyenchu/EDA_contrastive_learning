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
        python preprocessing.py -p PATH_TO_RAW_DATASET_FOLDER -d OUTPUT_DATASET_FOLDER 
        ```
- Training
    1. Run script
        ```
        python main.py -d PATH_TO_PROCESSED_DATASET_FOLDER -e EPOCHS -b BATCH_SIZE
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
## Experiment results
Values are selected from the experiment with highest accuracy and rounded to 4 decimals, with standard deviations of the top 3 experiments shown in parentheses. Augmentations labeled N/A do not have tunable parameters.
### Hyperparameter search
| **Augment Function** | **Parameter**       | **Best value** | **Accuracy**                | **F1 Score**               |
|-------------------------|--------------------------|---------------------|----------------------------------|---------------------------------|
| GaussianNoise_Det      | sigma_scale             | 1.0454 (0.0215)     | 0.8305 (0.0046)                  | 0.7685 (0.0126)                 |
| LowPassFilter_Det      | highcut_hz              | 0.093 (0.0087)      | 0.8049 (0.012)                   | 0.7175 (0.0177)                 |
| BandstopFilter_Det     | remove_freq             | 0.8421 (0.3448)     | 0.8191 (0.004)                   | 0.7605 (0.0106)                 |
| TimeShift_Det          | shift_len               | 11 (116.923)        | 0.8244 (0.0066)                  | 0.7387 (0.005)                  |
| HighFreqNoise_Det      | sigma_scale             | 0.4904 (0.2406)     | 0.8232 (0.0031)                  | 0.7529 (0.0047)                 |
| JumpArtifact_Det       | <p>max_n_jumps <p>shift_factor | <p>3 (0) <p>0.7229 (0.142) | 0.8233 (0.0227) | 0.738 (0.0382) |
| Permute_Det            | n_splits                | 4 (0.5773)          | 0.8057 (0.0026)                  | 0.7275 (0.0119)                 |
| ExtractPhasic           | N/A | N/A | 0.6678 (0.0079)     | 0.1511 (0.0605)                  |
| ExtractTonic            | N/A | N/A | 0.7678 (0.009)      | 0.6915 (0.0298)                  |
| Flip                    | N/A | N/A |0.7558 (0.0105)     | 0.6827 (0.0376)                  |

### Models
| **Model**              | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **ROC AUC** |
|---------------------------|-------------------|--------------------|-----------------|-------------------|------------------|
| Supervised(No aug)        | 0.8136 (0.0116)   | 0.7098 (0.0123)    | 0.7936 (0.0289) | 0.7493 (0.0185)   | 0.8390 (0.0029)  |
| Supervised(Top 1 aug)     | 0.8147 (0.0032)   | 0.7473 (0.0149)    | 0.7170 (0.0394) | 0.7802 (0.0113)   | 0.8692 (0.0018)  |
| Random Encoder(No aug)    | 0.6488 (0)        | 0 (0)              | 0 (0)           | 0 (0)             | 0.5 (0)          |
| Random Encoder(Top 1 aug) | 0.6488 (0)        | 0 (0)              | 0 (0)           | 0 (0)             | 0.5 (0)          |
| Contrastive(Top 1 aug)    | 0.8305 (0.0031)   | 0.7384 (0.0142)    | 0.8011 (0.0231) | 0.7685 (0.0051)   | 0.9104 (0.0075)  |
| Contrastive(Top 2 aug)    | 0.8369 (0.0019)   | 0.7734 (0.0248)    | 0.7549 (0.0455) | 0.7929 (0.0101)   | 0.915 (0.0046)   |
