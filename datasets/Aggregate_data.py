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

def f_test(fn):
    """
    fn: os.path.join(split_prefix, str(i) + '_data_aggregated.pth')
    """
    i = int(fn.split('/')[-1].split('_')[0])
    datas = files[i * MAXLENGTH : min((i + 1) * MAXLENGTH, len(files))]

    rgbs = []
    depths = []
    intrinsics = []
    for f in datas:
        rgb, depth, intrinsic = torch.load(f)
        rgbs.append(rgb)
        depths.append(depth)
        intrinsics.append(intrinsic)
        print('Processed ' + f)

    rgbs = torch.stack(rgbs, 0)
    depths = torch.stack(depths, 0)
    intrinsics = torch.stack(intrinsics, 0)
    
    torch.save((rgbs, depths, intrinsics), fn)
    print('Saving to ' + fn)

def f(fn):
    """
    fn: os.path.join(split_prefix, str(i) + '_data_aggregated.pth')
    """
    i = int(fn.split('/')[-1].split('_')[0])
    datas = files[i * MAXLENGTH : min((i + 1) * MAXLENGTH, len(files))]

    rgbs = []
    depths = []
    intrinsics = []
    labels = []
    boxs = []
    for f in datas:
        rgb, depth, label, intrinsic, box = torch.load(f)
        rgbs.append(rgb)
        depths.append(depth)
        labels.append(label)
        intrinsics.append(intrinsic)
        boxs.append(box)
        print('Processed ' + f)

    rgbs = torch.stack(rgbs, 0)
    depths = torch.stack(depths, 0)
    intrinsics = torch.stack(intrinsics, 0)
    labels = torch.stack(labels, 0)
    # Boxes are unaligned
    
    torch.save((rgbs, depths, labels, intrinsics, boxs), fn)
    print('Saving to ' + fn)

splits = (len(files) + 1) // MAXLENGTH
target_files = [os.path.join(split_prefix, str(i) + '_data_aggregated.pth') for i in range(splits + 1)]
if os.path.exists(split_prefix) == False:
    os.mkdir(split_prefix)

p = mp.Pool(processes=mp.cpu_count()//2) # Use all CPUs available
if opt.data_split == 'test':
    p.map(f_test, target_files)
else:
    p.map(f, target_files)
p.close()
p.join()

"""
python Aggregate_data.py --data_split train
python Aggregate_data.py --data_split val
python Aggregate_data.py --data_split test
"""