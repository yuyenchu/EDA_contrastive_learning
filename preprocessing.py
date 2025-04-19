import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from glob import glob
from os import path
from tqdm import tqdm

USE_CLEARML = True
EDA_LABEL_CLS = 8 # count for eda label classes
EDA_SAMPLE_RATE = 4 # eda data sample rate per second
LABEL_SAMPLE_RATE = 700 # eda label sample rate per second

def get_parser():
    parser = argparse.ArgumentParser(description='options for training')
    parser.add_argument('-p', '--path',     help='path to WESAD root folder',   type=str, default='./WESAD')
    parser.add_argument('-d', '--dest',     help='path to output folder',       type=str, default='./')
    parser.add_argument('-s', '--size',     help='per sample size',             type=int, default='240')
    parser.add_argument('-o', '--overlap',  help='per sample overlapping size', type=int, default='1')
    return parser

def get_subject_name(subject_path):
    return path.splitext(path.basename(subject_path))[0]

def process_subject(subject_path, dest_path, size, overlap, logger):
    global EDA_LABEL_CLS, EDA_SAMPLE_RATE, LABEL_SAMPLE_RATE, USE_CLEARML

    with open(subject_path,'rb') as f:
        data = pickle.load(f, encoding='latin1')
    if (not data):
        return None, None, None
    eda = data['signal']['wrist']['EDA']
    label = data['label']
    assert label.shape[0] / LABEL_SAMPLE_RATE == eda.shape[0] / EDA_SAMPLE_RATE, "labels and EDA are not aligned"

    selections = np.round(np.arange(eda.shape[0], dtype=float)/EDA_SAMPLE_RATE*LABEL_SAMPLE_RATE).astype(np.int32)
    synced_label = label[selections]
    assert synced_label.shape[0] == eda.shape[0] , "synced labels and EDA are not aligned"

    l = eda.shape[0]/(size-overlap)+1
    m = np.sum(np.meshgrid(np.arange(size), np.arange(l)*size-np.arange(l)*overlap), axis=0)
    m = (m[m[:,-1]<eda.shape[0]]).astype(np.int32)
    eda_segs = eda[m]
    # print(eda_segs.shape, l, m)
    label_segs = synced_label[m]
    label_valid = np.apply_along_axis(lambda x: (x==x[0]).all(), 1, label_segs)

    edas, labels = eda_segs[label_valid], label_segs[label_valid][:,0]
    fn = path.join(dest_path, f'{get_subject_name(subject_path)}.h5')
    with h5py.File(fn, 'w') as f:
        eda_ds = f.create_dataset('eda', data=edas)
        label_ds = f.create_dataset('label', data=labels)
        eda_unlabel_ds = f.create_dataset('eda_unlabel', data=eda_segs)
        
    if (USE_CLEARML):
        f = plt.figure(figsize=(16,8))
        plt.plot(np.arange(size), eda_segs[0])
        logger.report_matplotlib_figure('EDA signal', path.split(path.dirname(subject_path))[-1], f, report_image=True)
    
    cls_len = [np.sum(labels==i) for i in range(EDA_LABEL_CLS)]
    return path.abspath(fn), [eda_segs.shape[0], edas.shape[0], *cls_len]

if __name__=='__main__':
    if (USE_CLEARML):
        from clearml import Task, TaskTypes, Dataset
        task = Task.init('EDA_contrastive', 'dataset_preprocess', task_type=TaskTypes.data_processing)
        task.set_user_properties(LABEL_SAMPLE_RATE=LABEL_SAMPLE_RATE, EDA_SAMPLE_RATE=EDA_SAMPLE_RATE)
        logger = task.get_logger()
    args = get_parser().parse_args()  
    print('configs =',args)

    subjects = glob(path.join(args.path, 'S*', 'S*.pkl'))
    subjects.sort(key=lambda x:int(get_subject_name(x).replace('S','')))
    print(subjects)
    if (len(subjects)>0):
        if (not path.exists(args.dest)):
            os.makedirs(args.dest, exist_ok=True)
        f = open(path.join(args.dest, 'count.csv'), 'w')
        f.write('subject,unlabeled,labeled,'+','.join([f'cls_{i}' for i in range(EDA_LABEL_CLS)])+'\n')
        ds = Dataset.create('WESAD_EDA', 'EDA_contrastive', ['WESAD', 'EDA']) if (USE_CLEARML) else None
        for i, s in enumerate(tqdm(subjects)):
            if (path.isfile(s)):
                saved_path, sample_counts = process_subject(s, args.dest, args.size, args.overlap, logger if (USE_CLEARML) else None)
                if (saved_path and path.isfile(saved_path)):
                    f.write(get_subject_name(s)+','+','.join([str(c) for c in sample_counts])+'\n')
                    if (ds):
                        ds.add_files(saved_path)
                        logger.report_scalar('data_length', 'labelled', sample_counts[1], i)
                        logger.report_scalar('data_length', 'unlabelled', sample_counts[0], i)
        f.close()
        if (ds):
            ds.add_files(path.join(args.dest, 'count.csv'))
            ds.upload()
            ds.finalize()
            print('==> Dataset id:', ds.id)
    else:
        print(f'no subject found in path: {args.path}')

    if (USE_CLEARML):
        task.close()