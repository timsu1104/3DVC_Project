'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import glob, numpy as np, multiprocessing as mp, torch, argparse
import torch
from PIL import Image
import pickle

NUM_OBJECTS = 79

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train/ test)', default='train')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))
files = sorted(glob.glob(split + 'ing_data/data/*_color_kinect.png'))
files2 = sorted(glob.glob(split + 'ing_data/data/*_depth_kinect.png'))
if split == 'train':
    files3 = sorted(glob.glob(split + 'ing_data/data/*_label_kinect.png'))
files4 = sorted(glob.glob(split + 'ing_data/data/*_meta.pkl'))
assert len(files) == len(files2)
if split == 'train':
    assert len(files) == len(files3)

def f_test(fn):
    fn2 = fn[:-16] + 'depth_kinect.png'
    fn4 = fn[:-16] + 'meta.pkl'
    rgb = np.array(Image.open(fn)) / 255   # convert 0-255 to 0-1
    depth = np.array(Image.open(fn2)) / 1000   # convert from mm to m
    meta = load_pickle(fn4)
    intrinsic = meta['intrinsic']

    rgb = torch.tensor(rgb, dtype=torch.float32)
    depth = torch.tensor(depth, dtype=torch.float32)
    intrinsic = torch.tensor(intrinsic, dtype=torch.float32)

    torch.save((rgb, depth, intrinsic), fn[:-16]+'datas.pth')
    print('Saving to ' + fn[:-16]+'datas.pth')


def f(fn):
    fn2 = fn[:-16] + 'depth_kinect.png'
    fn3 = fn[:-16] + 'label_kinect.png'
    fn4 = fn[:-16] + 'meta.pkl'
    print(fn)

    rgb = np.array(Image.open(fn)) / 255   # convert 0-255 to 0-1
    depth = np.array(Image.open(fn2)) / 1000   # convert from mm to m
    label = np.array(Image.open(fn3))
    meta = load_pickle(fn4)
    intrinsic = meta['intrinsic']

    box = [] # NumBoxes, 5
    sem = np.unique(label)
    sem = [i for i in sem if i < NUM_OBJECTS]
    for sem_label in sem:
        x, y = np.where(label == sem_label)
        box.append([x.min(), y.min(), x.max(), y.max(), sem_label])

    rgb = torch.tensor(rgb, dtype=torch.float32)
    depth = torch.tensor(depth, dtype=torch.float32)
    intrinsic = torch.tensor(intrinsic, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.int)
    box = torch.tensor(box, dtype=torch.float32)

    torch.save((rgb, depth, label, intrinsic, box), fn[:-16]+'datas.pth')
    print('Saving to ' + fn[:-16]+'datas.pth')

p = mp.Pool(processes=mp.cpu_count()//2) # Use all CPUs available
if opt.data_split == 'test':
    p.map(f_test, files)
else:
    p.map(f, files)
p.close()
p.join()

"""
python prepare_data.py --data_split train
python prepare_data.py --data_split test
"""