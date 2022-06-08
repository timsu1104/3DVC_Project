'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import os, glob, numpy as np, multiprocessing as mp, torch, argparse
import torch
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train/val/ test)', default='train')
parser.add_argument('--maxl', help='the length in one data file', type=int, default=2000)
opt = parser.parse_args()

split = opt.data_split
MAXLENGTH = opt.maxl
NUM_OBJECTS = 79
print('data split: {}'.format(split))

training_data_dir = "training_data/data"
testing_data_dir = "testing_data/data"
split_dir = "training_data/splits"

def get_split_files(split_name):
    if split_name == 'test':
        files = sorted(glob.glob(os.path.join(testing_data_dir, '*_datas.pth')))
        return files

    with open(os.path.join(split_dir, f"{split_name}.txt"), 'r') as f:
        prefix = [os.path.join(training_data_dir, line.strip()) for line in f if line.strip()]
        files = [p + "_datas.pth" for p in prefix]
    return files

files = get_split_files(split)
split_prefix = os.path.join(testing_data_dir, split) if split == 'test' else os.path.join(training_data_dir, split)

def f(fn):
    """
    fn: os.path.join(split_prefix, str(i) + '_data_aggregated.pth')
    """
    i = int(fn.split('/')[-1].split('_')[0])
    datas = files[i * MAXLENGTH : min((i + 1) * MAXLENGTH, len(files))]

    new_data = []
    for f in datas:
        data = torch.load(f)
        if split != 'test':
            rgb, depth, label, intrinsic = data
            box = [] # NumBoxes, 5
            sem = np.unique(label)
            sem = [i for i in sem if i < NUM_OBJECTS]
            for sem_label in sem:
                x, y = np.where(label == sem_label)
                box.append([x.min(), y.min(), x.max(), y.max(), sem_label])
            data = rgb, depth, label, intrinsic, box
        new_data.append(data)
        print('Processed ' + f)
                
    torch.save(new_data, fn)
    print('Saving to ' + fn)

splits = (len(files) + 1) // MAXLENGTH
target_files = [os.path.join(split_prefix, str(i) + '_data_aggregated.pth') for i in range(splits)]
if os.path.exists(split_prefix) == False:
    os.mkdir(split_prefix)

p = mp.Pool(processes=mp.cpu_count()//2) # Use all CPUs available
p.map(f, target_files)
p.close()
p.join()

"""
python prepare_data.py --data_split train
python prepare_data.py --data_split val
python prepare_data.py --data_split test
"""