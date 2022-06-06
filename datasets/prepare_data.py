import numpy as np, torch
import os, glob
from PIL import Image
import pickle
from tqdm.contrib import tzip

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

def read_file(rgb_file, depth_file, label_file, meta_file):
    rgb = np.array(Image.open(rgb_file)) / 255   # convert 0-255 to 0-1
    depth = np.array(Image.open(depth_file)) / 1000   # convert from mm to m
    label = np.array(Image.open(label_file))
    meta = load_pickle(meta_file)
    return rgb, depth, label, meta

def read_test_file(rgb_file, depth_file, meta_file):
    rgb = np.array(Image.open(rgb_file)) / 255   # convert 0-255 to 0-1
    depth = np.array(Image.open(depth_file)) / 1000   # convert from mm to m
    meta = load_pickle(meta_file)
    return rgb, depth, meta

def Aggregate_data(split_name: str) -> None:
    print("Split: {}".format(split_name))
    if split_name == 'test':
        rgb_files, depth_files, meta_files = get_split_files(split_name)
        rgbs = []
        depths = []
        intrinsics = []
        for rgb_file, depth_file, meta_file in tzip(rgb_files, depth_files, meta_files):
            rgb, depth, meta = read_test_file(rgb_file, depth_file, meta_file)
            rgbs.append(rgb)
            depths.append(depth)
            intrinsics.append(meta['intrinsic'])
        rgbs = np.stack(rgbs, axis=0)
        depths = np.stack(depths, axis=0)
        intrinsics = np.stack(intrinsics, axis=0)
        rgbs = torch.tensor(rgbs)
        depths = torch.tensor(depths)
        intrinsics = torch.tensor(intrinsics)
        torch.save((rgbs, depths, intrinsics), split_name + '.pth')
        
    else:
        rgb_files, depth_files, label_files, meta_files = get_split_files(split_name)
        rgbs = []
        depths = []
        labels = []
        intrinsics = []
        for rgb_file, depth_file, label_file, meta_file in tzip(rgb_files, depth_files, label_files, meta_files):
            rgb, depth, label, meta = read_file(rgb_file, depth_file, label_file, meta_file)
            rgbs.append(rgb)
            depths.append(depth)
            labels.append(label)
            intrinsics.append(meta['intrinsic'])
        rgbs = np.stack(rgbs, axis=0)
        depths = np.stack(depths, axis=0)
        labels = np.stack(labels, axis=0)
        intrinsics = np.stack(intrinsics, axis=0)
        rgbs = torch.tensor(rgbs)
        depths = torch.tensor(depths)
        labels = torch.tensor(labels)
        intrinsics = torch.tensor(intrinsics)
        torch.save((rgbs, depths, labels, intrinsics), split_name + '.pth')

if __name__ == '__main__':
      Aggregate_data('train')
      Aggregate_data('val')
      Aggregate_data('test')