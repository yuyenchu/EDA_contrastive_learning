import h5py
import os
import numpy as np 
import tensorflow as tf
from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split
from glob import glob

from augmenter import DataAugmenter

def get_dataset(data_path='./', batch_size=64, test_size=0.2, seed=42, unlabeled_aug=None, labeled_aug=None):
    unlabeled_eda = []
    labeled_eda = []
    labels = []
    for fn in glob(os.path.join(data_path, '*.h5')):
        f = h5py.File(fn, 'r')
        unlabeled_eda += f['eda_unlabel']
        labeled_eda += f['eda']
        labels += f['label']
    # labels = list(range(len(labeled_eda)))
    unlabeled_eda, labeled_eda = np.array(unlabeled_eda, np.float32), np.array(labeled_eda, np.float32)
    labels = (np.array(labels)==2).astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(labeled_eda, labels, test_size=test_size, random_state=seed)
    
    unlabeled_train_ds = Dataset.from_tensor_slices(unlabeled_eda)
    if (unlabeled_aug):
        unlabeled_train_ds = unlabeled_train_ds.map(unlabeled_aug)
    unlabeled_train_ds = unlabeled_train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    labeled_train_ds = Dataset.from_tensor_slices((X_train, y_train))
    if (labeled_aug):
        labeled_train_ds = labeled_train_ds.map(labeled_aug)
    labeled_train_ds = labeled_train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    labeled_test_ds = Dataset.from_tensor_slices((X_test, y_test)) \
        .batch(batch_size) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)

    return unlabeled_train_ds, labeled_train_ds, labeled_test_ds