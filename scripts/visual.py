from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse, sys, os
from tqdm import tqdm

NUM_OBJECTS = 79
cmap = get_cmap('rainbow', NUM_OBJECTS)
COLOR_PALETTE = np.array([cmap(i)[:3] for i in range(NUM_OBJECTS + 3)])
COLOR_PALETTE = np.array(COLOR_PALETTE * 255, dtype=np.uint8)
COLOR_PALETTE[-3] = [119, 135, 150]
COLOR_PALETTE[-2] = [176, 194, 216]
COLOR_PALETTE[-1] = [255, 255, 225]

def vis(rgb_file, label_file):
    p = rgb_file.split('/')[-1].split('_')[0]
    rgb = np.array(Image.open(rgb_file)) / 255   # convert 0-255 to 0-1
    label = np.array(Image.open(label_file))
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.subplot(1, 3, 2)
    plt.imshow(COLOR_PALETTE[label])  # draw colorful segmentation
    plt.savefig('visualization/' + p + '_visualization.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', help='', type=int, default=-1)
    opt = parser.parse_args()
    id = opt.id
    sys.path.append(os.getcwd())
    from utils.util import get_split_files
    files = get_split_files('test', prefix='datasets/')
    if id == -1:
        for fn in tqdm(files):
            vis(fn[:-9] + 'color_kinect.png', fn[:-9] + 'label_kinect.png')
    else:
        fn = files[id]
        vis(fn[:-9] + 'color_kinect.png', fn[:-9] + 'label_kinect.png')