import h5py
import os
import gc
import psutil
import numpy as np 
import time
import tensorflow as tf

from tensorflow.data import Dataset
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from glob import glob

class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    def __init__(self, collect_freq=200):
        self.collect_freq = collect_freq
        super().__init__()
    def on_batch_end(self, batch, logs=None):
        if batch % self.collect_freq == 0:
            gc.collect()
            #tf.keras.backend.clear_session()

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        #tf.keras.backend.clear_session()

class PrintMemoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_freq=200):
        self.last_usage = 0
        self.iteration = 0
        self.start_time=None
        self.delta_sum = 0 #starting to count at iteration==2
        if hasattr(self,"params"):
            self.steps_per_epoch=self.params.get('steps')
        else:
            self.steps_per_epoch=1000
        self.log_freq = log_freq
        super().__init__()

    def on_batch_end(self, batch, logs=None):
        if batch % self.log_freq == 0:
            if(self.start_time is None):
                self.start_time=time.time()
            usage = psutil.Process(os.getpid()).memory_info().rss
            print('**Batch {}**'.format(batch))
            print(f"Time since start: {(time.time()-self.start_time)/60:.2f} minutes")
            print('Memory usage: {}'.format(usage))
            delta = usage - self.last_usage
            print(f"Delta: {delta/(1024*1024):.2f} MiB")
            self.last_usage = usage
            if(self.iteration>2):
                self.delta_sum+=delta
                print(f"Average Delta since start: {(self.delta_sum/(self.iteration-2))/(1024*1024):.2f} MiB/iteration")

                print(f"Estimated growth per 1000 steps: {self.steps_per_epoch*(self.delta_sum/(self.iteration-2)/200)/(1024*1024):.2f} MiB")
            self.iteration+=1

    def on_epoch_begin(self, epoch, logs=None):
        print('**Begin Epoch {}**'.format(epoch))
        print('Memory usage: {}'.format(psutil.Process(os.getpid()).memory_info().rss))

    def on_epoch_end(self, epoch, logs=None):
        print('**End Epoch {}**'.format(epoch))
        print('Memory usage: {}'.format(psutil.Process(os.getpid()).memory_info().rss))

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
    print('unlabeled shape:', unlabeled_eda.shape, ', labeled shape:', labeled_eda.shape)
    print('gather time:', time.time()-start)
    
    start = time.time()
    labels = (np.array(labels)==2).astype(np.int32)
    unlabeled_eda, unlabeled_lb, unlabeled_rb = shuffle(unlabeled_eda, unlabeled_lb, unlabeled_rb, random_state=seed)
    X_train, X_test, Xl_train, _, Xr_train, _, y_train, y_test = train_test_split(labeled_eda, labeled_lb, labeled_rb, labels, test_size=test_size, random_state=seed)
    # # Stop magic stuff that eats up RAM:
    # options = tf.data.Options()
    # options.autotune.enabled = False
    # options.experimental_distribute.auto_shard_policy = (
    # tf.data.experimental.AutoShardPolicy.OFF)
    # options.experimental_optimization.inject_prefetch = False
    # # usage: ds = ds.with_options(options)

    if (unlabeled_aug):
        unlabeled_train_ds = Dataset.from_tensor_slices((unlabeled_eda, unlabeled_lb, unlabeled_rb))
        unlabeled_train_ds = Dataset.zip(unlabeled_train_ds.map(unlabeled_aug), unlabeled_train_ds.map(unlabeled_aug))
    else:
        unlabeled_train_ds = Dataset.from_tensor_slices(unlabeled_eda)
        unlabeled_train_ds = Dataset.zip(unlabeled_train_ds, unlabeled_train_ds)
    unlabeled_train_ds = unlabeled_train_ds.batch(batch_size)\
                                           .prefetch(tf.data.AUTOTUNE)
    
    if (labeled_aug):
        labeled_train_ds = Dataset.from_tensor_slices((X_train, Xl_train, Xr_train))
        labeled_train_ds = labeled_train_ds.map(labeled_aug)
        labeled_train_ds = Dataset.zip(labeled_train_ds, Dataset.from_tensor_slices(y_train))
    else:
        labeled_train_ds = Dataset.from_tensor_slices((X_train, y_train))
    labeled_train_ds = labeled_train_ds.batch(batch_size)\
                                       .prefetch(tf.data.AUTOTUNE)
    
    labeled_test_ds = Dataset.from_tensor_slices((X_test, y_test)) \
                             .batch(batch_size) \
                             .prefetch(tf.data.AUTOTUNE)
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

def check_memory_leak(ds):
    callbacks = [
        GarbageCollectionCallback(1000), 
        PrintMemoryCallback(1000)
    ]
    ds_iter=iter(ds.repeat(10))
    step=0
    while True:
        try:
            _=next(ds_iter)
            del _
            cb:tf.keras.callbacks.Callback
            for cb in callbacks:
                cb.on_batch_end(step,logs=None)
            step+=1
        except: 
            break

if __name__=='__main__':
    import argparse
    import json
    from augmenter import DataAugmenter

    parser = argparse.ArgumentParser(description='options for training')
    parser.add_argument('-d', '--dataset',  help='dataset path', type=str, default='./ds')
    args = parser.parse_args()  

    print('==> testing ds memory leak')
    with open('augment.json', 'r') as f:
        aug_cfg = json.load(f)
    unlabeled_aug = DataAugmenter(**aug_cfg.get('unlabeled_aug', {}))
    labeled_aug = DataAugmenter(**aug_cfg.get('labeled_aug', {}))
    ds1, ds2, ds3 = get_dataset(args.dataset)

    print('='*20, 'testing ds1', '='*20)
    check_memory_leak(ds1)
    print('='*20, 'testing ds2', '='*20)
    check_memory_leak(ds2)
    print('='*20, 'testing ds3', '='*20)
    check_memory_leak(ds3)