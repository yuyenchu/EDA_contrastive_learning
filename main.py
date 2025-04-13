import argparse
import json
import numpy as np

from glob import glob
from os import path
from tqdm import tqdm
from tensorflow import keras

from model import ContrastiveModel
from utils import get_dataset
from augmenter import DataAugmenter

USE_CLEARML = True

def get_parser():
    parser = argparse.ArgumentParser(description='options for training')
    parser.add_argument('-d', '--dataset',  help='dataset path or id',          type=str, default='./')
    parser.add_argument('-b', '--batch',    help='training batch size',         type=int, default=64)
    parser.add_argument('-e', '--epoch',    help='training epochs',             type=int, default=30)
    parser.add_argument('-l', '--lr',       help='training learning rate',      type=float, default=0.001)
    parser.add_argument('-t', '--temp',     help='contrastive loss temperature',type=float, default=1.0)
    return parser

if __name__=='__main__':
    with open('augment.json', 'r') as f:
        aug_cfg = json.load(f)
    if (USE_CLEARML):
        from clearml import Task, Dataset
        task = Task.init('EDA_contrastive', 'train')
        logger = task.get_logger()
        task.connect_configuration(aug_cfg, name='aug_cfg')
    args = get_parser().parse_args()  
    print('configs =',args)

    unlabeled_aug = DataAugmenter(**aug_cfg.get('unlabeled_aug', {}))
    labeled_aug = DataAugmenter(**aug_cfg.get('labeled_aug', {}))

    if (path.exists(args.dataset)):
        unlabeled_train_ds, labeled_train_ds, test_ds = get_dataset(args.dataset, batch_size=args.batch)
    elif (USE_CLEARML):
        ds = Dataset.get(dataset_id=args.dataset)
        unlabeled_train_ds, labeled_train_ds, test_ds = get_dataset(ds.get_local_copy(), batch_size=args.batch)
    else:
        raise ValueError('dataset not valid:', args.dataset)

    callbacks = []
    callbacks.append(keras.callbacks.TensorBoard(log_dir = './logs',
                                                    histogram_freq = 1,
                                                    profile_batch = '200,220'))
    callbacks.append(keras.callbacks.ModelCheckpoint(filepath='./model_ckpt.weights.h5',
                                                    monitor='val_p_acc',
                                                    mode='max',
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    initial_value_threshold=0.0,
                                                    verbose=1))

    # Contrastive pretraining
    pretraining_model = ContrastiveModel(args.temp)

    pretraining_model.compile(
        optimizer=keras.optimizers.AdamW(args.lr),
        train_mode='contrastive',
    )
    pretraining_history = pretraining_model.fit(
        unlabeled_train_ds, epochs=args.epoch, validation_data=test_ds, callbacks=callbacks
    )

    pretraining_model.compile(
        optimizer=keras.optimizers.AdamW(args.lr),
        train_mode='prediction',
    )
    predtraining_history = pretraining_model.fit(
        labeled_train_ds, epochs=args.epoch, validation_data=test_ds, callbacks=callbacks
    )
    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(predtraining_history.history["val_p_acc"]) * 100
        )
    )