import argparse
import gc
import json
import numpy as np
import tensorflow as tf

from glob import glob
from os import path
from tqdm import tqdm
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from model import ContrastiveModel
from utils import get_dataset, GarbageCollectionCallback, PrintMemoryCallback
from augmenter import DataAugmenter

USE_CLEARML = False
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

    try:
        if (path.exists(args.model) and path.isfile(args.model)):
            pretrain_path = args.model
        elif (USE_CLEARML):
            pretrain_path = Task.get_task(args.model).artifacts['checkpoint'].get_local_copy()
    except:
        pretrain_path = None
        print('[!Warning] model ckpt path not valid:', args.model)
    if (pretrain_path and pretrain_path.endswith('.weights.h5')):
        try:
            model.load_weights(pretrain_path)
            print('model weight load success')
        except:
            print('cannot load weight from:', pretrain_path)

    ckpt_path = './model_ckpt.weights.h5'
    callbacks = []
    # callbacks.append(PrintMemoryCallback(672))
    callbacks.append(GarbageCollectionCallback())
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
    print('='*20, 'training start' , '='*20)
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
        'Best validation accuracy: {:.2f}%, loss {:.3f}'.format(
            max(history.history['val_p_acc']) * 100,
            max(history.history['val_p_loss'])
        )
    )
    model.load_weights(ckpt_path)
    model.save('model.keras')
    del unlabeled_train_ds
    del labeled_train_ds
    gc.collect()

    print('='*20, 'testing start' , '='*20)
    y_pred_score = model.predict(test_ds)
    y_pred = y_pred_score>0.5
    y_true = [l for d,l in test_ds.unbatch().as_numpy_iterator()]
    del test_ds
    gc.collect()

    cm = confusion_matrix(y_true, y_pred)
    print('confusion matrix:')
    print(cm)
    p, r, acc, f1, auc = (
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        roc_auc_score(y_true, y_pred_score)
    )
    print('precision:', p)
    print('recall:', r)
    print('accuracy:', acc)
    print('f1:', f1)
    print('roc_auc:', auc)

    if (USE_CLEARML):
        task.upload_artifact('checkpoint', ckpt_path)