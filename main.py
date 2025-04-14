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
print('tensorflow ==', tf.__version__)
print(tf.config.list_physical_devices())

def get_parser():
    parser = argparse.ArgumentParser(description='options for training')
    parser.add_argument('-d', '--dataset',  help='dataset path or id',          type=str, default='./')
    parser.add_argument('-m', '--model',    help='pretrained weight path or id',type=str, default=None)
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

    if (path.exists(args.dataset) and path.isdir(args.dataset)):
        unlabeled_train_ds, labeled_train_ds, test_ds = get_dataset(args.dataset, batch_size=args.batch, unlabeled_aug=unlabeled_aug, labeled_aug=labeled_aug)
    elif (USE_CLEARML):
        ds = Dataset.get(dataset_id=args.dataset)
        unlabeled_train_ds, labeled_train_ds, test_ds = get_dataset(ds.get_local_copy(), batch_size=args.batch, unlabeled_aug=unlabeled_aug, labeled_aug=labeled_aug)
    else:
        raise ValueError('dataset not valid:', args.dataset)

    if (path.exists(args.model) and path.isfile(args.model)):
        pretrain_path = args.model
    elif (USE_CLEARML):
        pretrain_path = Task.get_task(args.model).artifacts['checkpoint'].get_local_copy()
    else:
        pretrain_path = None
        print('[!Warning] model ckpt path not valid:', args.model)
    if (pretrain_path and pretrain_path.endswith('.weights.h5'))
        try:
            model.load_weights(pretrain_path)
            print('model weight load success')
        except:
            print('cannot load weight from:', pretrain_path)

    ckpt_path = './model_ckpt.weights.h5'
    callbacks = []
    callbacks.append(keras.callbacks.TensorBoard(log_dir = './logs',
                                                    histogram_freq = 1,
                                                    profile_batch = '200,220'))
    callbacks.append(keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                    monitor='val_p_acc',
                                                    mode='max',
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    initial_value_threshold=0.0,
                                                    verbose=1))

    # Contrastive pretraining
    model = ContrastiveModel(args.temp)

    model.compile(
        optimizer=keras.optimizers.AdamW(args.lr),
        train_mode='contrastive',
    )
    model.fit(
        unlabeled_train_ds, epochs=args.epoch, validation_data=test_ds, callbacks=callbacks
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(args.lr),
        train_mode='prediction',
    )
    history = model.fit(
        labeled_train_ds, epochs=args.epoch, validation_data=test_ds, callbacks=callbacks
    )

    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(history.history["val_p_acc"]) * 100
        )
    )
    model.load_weights(ckpt_path)
    model.save('model.keras')

    if (USE_CLEARML):
        task.upload_artifact('checkpoint', ckpt_path)