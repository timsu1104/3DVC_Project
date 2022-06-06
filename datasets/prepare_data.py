import numpy as np
import os, glob
from PIL import Image
import pickle
import open3d as o3d

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

training_data_dir = "training_data/data"
testing_data_dir = "testing_data/data"
split_dir = "training_data/splits"

def get_split_files(split_name):
    if split_name == 'test':
        rgb = sorted(glob.glob(os.path.join(testing_data_dir, '*_color_kinect.png')))
        depth = sorted(glob.glob(os.path.join(testing_data_dir, '*_depth_kinect.png')))
        meta = sorted(glob.glob(os.path.join(testing_data_dir, '*_meta.pkl')))
        return rgb, depth, meta

    with open(os.path.join(split_dir, f"{split_name}.txt"), 'r') as f:
        prefix = [os.path.join(training_data_dir, line.strip()) for line in f if line.strip()]
        rgb = [p + "_color_kinect.png" for p in prefix]
        depth = [p + "_depth_kinect.png" for p in prefix]
        label = [p + "_label_kinect.png" for p in prefix]
        meta = [p + "_meta.pkl" for p in prefix]
    return rgb, depth, label, meta