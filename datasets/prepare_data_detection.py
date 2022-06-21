'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import glob, numpy as np, multiprocessing as mp, argparse
import json
from PIL import Image
from detectron2.structures import BoxMode

NUM_OBJECTS = 79

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train/ test)', default='train')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))
files = sorted(glob.glob(split + 'ing_data/data/*_color_kinect.png'))
if split == 'train':
    files3 = sorted(glob.glob(split + 'ing_data/data/*_label_kinect.png'))

def f_test(fn_c):
    fn, id = fn_c
    output_f = fn[:-16] + 'datas.json'
    rgb = np.array(Image.open(fn)) / 255

    dict = {}
    dict['file_name'] = fn
    dict['image_id'] = id
    dict['height'] = rgb.shape[0]
    dict['width'] = rgb.shape[1]

    print('Saving to ' + fn[:-16]+'datas.json')
    with open(output_f, 'w') as f:
        json.dump(dict, f)

def f(fn_c):
    fn, id = fn_c
    fn3 = fn[:-16] + 'label_kinect.png'
    output_f = fn[:-16] + 'datas.json'

    rgb = np.array(Image.open(fn)) / 255
    label = np.array(Image.open(fn3))

    box = [] # NumBoxes, 5
    sem = np.unique(label)
    sem = [i for i in sem if i < NUM_OBJECTS]
    assert len(sem) > 0
    for sem_label in sem:
        x, y = np.nonzero(label == sem_label)
        box.append([int(x.min()), int(y.min()), int(x.max()), int(y.max()), sem_label])

    dict = {}
    dict['file_name'] = fn
    dict['image_id'] = id
    dict['height'] = rgb.shape[0]
    dict['width'] = rgb.shape[1]
    objs = []
    for bbox in box:
        obj = {}
        obj['bbox'] = bbox[:-1]
        obj['bbox_mode'] = BoxMode.XYXY_ABS
        obj['category_id'] = int(bbox[-1])
        obj['iscrowd'] = 0
        objs.append(obj)
    dict['annotations'] = objs

    print('Saving to ' + fn[:-16]+'datas.json')
    with open(output_f, 'w') as f:
        json.dump(dict, f)

target = [[f, i] for i, f in enumerate(files)]

p = mp.Pool(processes=mp.cpu_count()//2) # Use all CPUs available
if opt.data_split == 'test':
    p.map(f_test, target)
else:
    p.map(f, target)
p.close()
p.join()

"""
python prepare_data.py --data_split train
python prepare_data.py --data_split test
"""