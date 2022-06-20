import torch, sys, os
import numpy as np
from PIL import Image

sys.path.append(os.getcwd())
from utils.util import get_split_files

def dump_result(labels, output_path='test'):
    files = get_split_files('test', prefix='datasets')
    assert labels.size(0) == len(files)
    for label, file in zip(labels, files):
        p = file[:-10]
        label = np.array(label)
        im = Image.fromarray(label).convert('P')
        im.save(os.path.join(p + '_label_kinect.png'))



