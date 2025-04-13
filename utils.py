import h5py
import os
import numpy as np 
import time
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from glob import glob

def get_dataset(data_path='./', seg_shape=(240,1), batch_size=64, test_size=0.2, seed=42, unlabeled_aug=None, labeled_aug=None):
    unlabeled_eda = np.zeros((0,*seg_shape))
    labeled_eda = np.zeros((0,*seg_shape))
    unlabeled_lb = np.zeros((0,*seg_shape))
    unlabeled_rb = np.zeros((0,*seg_shape))
    labeled_lb = np.zeros((0,*seg_shape))
    labeled_rb = np.zeros((0,*seg_shape))
    nan = np.full((1,*seg_shape), np.nan)
    labels = []
    start = time.time()
    for fn in glob(os.path.join(data_path, 'S*.h5')):
        f = h5py.File(fn, 'r')
        unlabeled_eda = np.concat([unlabeled_eda, f['eda_unlabel'][:]], axis=0, dtype=np.float32)
        unlabeled_lb = np.concat([unlabeled_lb, nan, f['eda_unlabel'][:-1]], axis=0, dtype=np.float32)
        unlabeled_rb = np.concat([unlabeled_rb, f['eda_unlabel'][1:], nan], axis=0, dtype=np.float32)

        labeled_eda = np.concat([labeled_eda, f['eda'][:]], axis=0, dtype=np.float32)
        labeled_lb = np.concat([labeled_lb, nan, f['eda'][:-1]], axis=0, dtype=np.float32)
        labeled_rb = np.concat([labeled_rb, f['eda'][1:], nan], axis=0, dtype=np.float32)
        
        labels += f['label']
    print('gather time:', time.time()-start)

    start = time.time()
    labels = (np.array(labels)==2).astype(np.int32)
    X_train, X_test, Xl_train, _, Xr_train, _, y_train, y_test = train_test_split(labeled_eda, labeled_lb, labeled_rb, labels, test_size=test_size, random_state=seed)
    
    if (unlabeled_aug):
        unlabeled_train_ds = Dataset.from_tensor_slices((unlabeled_eda, unlabeled_lb, unlabeled_rb))
        unlabeled_train_ds = Dataset.zip(unlabeled_train_ds.map(unlabeled_aug), unlabeled_train_ds.map(unlabeled_aug))
    else:
        unlabeled_train_ds = Dataset.from_tensor_slices(unlabeled_eda)
        unlabeled_train_ds = Dataset.zip(unlabeled_train_ds, unlabeled_train_ds)
    unlabeled_train_ds = unlabeled_train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    if (labeled_aug):
        labeled_train_ds = Dataset.from_tensor_slices((X_train, Xl_train, Xr_train))
        labeled_train_ds = labeled_train_ds.map(labeled_aug)
        labeled_train_ds = Dataset.zip(labeled_train_ds, Dataset.from_tensor_slices(y_train))
    else:
        labeled_train_ds = Dataset.from_tensor_slices((X_train, y_train))
    labeled_train_ds = labeled_train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    labeled_test_ds = Dataset.from_tensor_slices((X_test, y_test)) \
        .batch(batch_size) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    print('ds build time:', time.time()-start)
    
    return unlabeled_train_ds, labeled_train_ds, labeled_test_ds

def build_encoder():
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
    return keras.Sequential(
        [
            layers.Input((240,1)),
            layers.Conv1D(4, kernel_size=7, strides=1, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool1D(),
            layers.Dropout(0.1),
            layers.Conv1D(16, kernel_size=7, strides=1, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool1D(),
            layers.Dropout(0.1),
            layers.Conv1D(32, kernel_size=7, strides=1, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool1D(),
            layers.Dropout(0.1),
            layers.Flatten(),
            layers.Dense(64, activation="relu", kernel_initializer=initializer),
        ],
        name="encoder",
    )

def build_projection_head():
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
    return keras.Sequential(
        [
            layers.Input((64,)),
            layers.Dense(32, activation="relu", kernel_initializer=initializer),
        ],
        name="projection_head",
    )

def build_classification_head():
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
    return keras.Sequential(
        [
            layers.Input((64,)),
            layers.Dense(1, activation="sigmoid", kernel_initializer=initializer),
        ],
        name="classification_head",
    )