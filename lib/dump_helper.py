import torch, sys, os
import numpy as np
from PIL import Image

sys.path.append(os.getcwd())
from utils.util import get_split_files

def dump_result(labels):
    files = get_split_files('test', prefix='datasets/')
    torch._assert(
        labels.size(0) == len(files),
        f"{labels.size()} {len(files)}"
    )
    for label, file in zip(labels, files):
        p = file[:-10]
        label = np.array(label.cpu())
        im = Image.fromarray(label).convert('P')
        im.save(os.path.join(p + '_label_kinect.png'))



