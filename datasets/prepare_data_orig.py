from genericpath import exists
import numpy as np, torch
import os, glob, argparse
from tqdm import tqdm

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

def Aggregate_data(split_name: str) -> None:
    print("Split: {}".format(split_name))
    if split_name == 'test':
        split_prefix = os.path.join(testing_data_dir, split_name)
        if os.path.exists(split_prefix) == False:
            os.mkdir(split_prefix)
        files = get_split_files(split_name)
        splits = (len(files) + 1) // MAXLENGTH
        for i in range(splits):
            fn = os.path.join(split_prefix, str(i) + '_data_aggregated.pth')
            rgbs = []
            depths = []
            intrinsics = []
            for file in tqdm(files[i * MAXLENGTH : min((i + 1) * MAXLENGTH, len(files))]):
                rgb, depth, intrinsic = torch.load(file)
                rgbs.append(rgb)
                depths.append(depth)
                intrinsics.append(intrinsic)
            rgbs = np.stack(rgbs, axis=0)
            depths = np.stack(depths, axis=0)
            intrinsics = np.stack(intrinsics, axis=0)
            rgbs = torch.tensor(rgbs)
            depths = torch.tensor(depths)
            intrinsics = torch.tensor(intrinsics)
            torch.save((rgbs, depths, intrinsics), fn)
        
    else:
        split_prefix = os.path.join(training_data_dir, split_name)
        if os.path.exists(split_prefix) == False:
            os.mkdir(split_prefix)
        files = get_split_files(split_name)
        splits = (len(files) + 1) // MAXLENGTH
        print("files {}\nsplits {}".format(len(files), splits))
        for i in range(splits):
            fn = os.path.join(split_prefix, str(i) + '_data_aggregated.pth')
            rgbs = []
            depths = []
            labels = []
            intrinsics = []
            for file in tqdm(files[i * MAXLENGTH : min((i + 1) * MAXLENGTH, len(files))]):
                rgb, depth, label, intrinsic = torch.load(file)
                rgbs.append(rgb)
                depths.append(depth)
                labels.append(label)
                intrinsics.append(intrinsic)
            rgbs = np.stack(rgbs, axis=0)
            depths = np.stack(depths, axis=0)
            labels = np.stack(labels, axis=0)
            intrinsics = np.stack(intrinsics, axis=0)
            rgbs = torch.tensor(rgbs)
            depths = torch.tensor(depths)
            labels = torch.tensor(labels)
            intrinsics = torch.tensor(intrinsics)
            torch.save((rgbs, depths, labels, intrinsics), fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxl', help='the length in one data file', type=int, default=10000)
    opt = parser.parse_args()

    MAXLENGTH = opt.maxl
    Aggregate_data('train')
    Aggregate_data('val')
    Aggregate_data('test')