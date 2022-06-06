import sys, os, glob, numpy as np
import pandas as pd
from tqdm.contrib import tzip
import torch
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.getcwd())
from utils.data_processing import *

class Dataset:
    def __init__(self): 
        self.batch_size = 4
        self.train_workers = 4
        self.val_workers = 4
        with open('datasets/models_pointnet.json', 'r') as f:
            self.MODELS = json.load(f)
    
    def trainLoader(self, logger):
        datas, meta_files = get_split_files('train')
        datas, meta_files = datas[:16], meta_files[:16]
        # for rgb_file, depth_file, label_file, meta_file in tzip(rgb_files, depth_files, label_files, meta_files):
        #     self.train_files.append(read_file(rgb_file, depth_file, label_file, meta_file))
        self.train_files = [[torch.load(data), load_pickle(meta_file)] for data, meta_file in tzip(datas, meta_files)]

        self.NumTrainSamples = len(self.train_files)
        logger.info('Training samples: {}'.format(self.NumTrainSamples))
        assert self.NumTrainSamples > 0

        train_set = list(range(self.NumTrainSamples))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    
    
    def valLoader(self, logger):
        # rgb_files, depth_files, label_files, meta_files = get_split_files('val')
        # self.val_files = []
        # for rgb_file, depth_file, label_file, meta_file in tzip(rgb_files, depth_files, label_files, meta_files):
        #     self.val_files.append(read_file(rgb_file, depth_file, label_file, meta_file))
            
        datas, meta_files = get_split_files('val')
        datas, meta_files = datas[:4], meta_files[:4]
        # for rgb_file, depth_file, label_file, meta_file in tzip(rgb_files, depth_files, label_files, meta_files):
        #     self.train_files.append(read_file(rgb_file, depth_file, label_file, meta_file))
        self.val_files = [[torch.load(data), load_pickle(meta_file)] for data, meta_file in tzip(datas, meta_files)]
        self.val_df = pd.read_csv('datasets/training_data/objects_v1.csv')

        self.NumValSamples = len(self.val_files)
        logger.info('Validation samples: {}'.format(self.NumValSamples))
        assert self.NumValSamples > 0

        val_set = list(range(self.NumValSamples))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge, num_workers=self.val_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    
    
    def testLoader(self, logger):
        rgb_files, depth_files, label_files, meta_files = get_split_files('test')
        self.test_files = []
        for rgb_file, depth_file, label_file, meta_file in tzip(rgb_files, depth_files, label_files, meta_files):
            self.test_files.append([*read_file(rgb_file, depth_file, label_file, meta_file), rgb_file])
        self.test_df = pd.read_csv('datasets/testing_data/objects_v1.csv')

        self.NumTestSamples = len(rgb_files)
        logger.info('Testing samples: {}'.format(self.NumTestSamples))
        assert self.NumTestSamples > 0

        test_set = list(range(self.NumTestSamples))
        self.test_data_loader = DataLoader(test_set, batch_size=self.batch_size, collate_fn=self.testMerge, num_workers=self.test_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    

    def trainMerge(self, id):
        geo_syms = []
        model_pc = []
        image_pc = []
        gt_poses = []
        for idx in id:
            data, meta = self.train_files[idx]
            rgb, _, label, coords, gt_pose = data
            selected = self.train_df.loc[self.train_df['object'].isin(meta['object_names'])]
            symmetry = selected['geometric_symmetry'].to_list()
            loc = selected['location'].to_list()
            ind = selected.index.to_list()
            assert len(loc) == len(ind)

            cnt = -1
            sel_ind = []
            for directory, index in zip(loc, ind):
                cnt += 1
                scale = np.array(meta['scales'][index])
                
                source = np.array(self.MODELS[directory]['coords']) * scale
                source -= source.mean(axis=0) # zero centering

                # Read image target pose
                image_points = coords[label == index].reshape((-1,3))
                if image_points.shape[0] == 0:
                    continue
                target = image_points @ np.linalg.inv(meta['extrinsic'])[:3, :3].T + np.linalg.inv(meta['extrinsic'])[:3, 3]
                target -= target.mean(axis=0) # zero centering

                model_pc.append(source)
                image_pc.append(torch.from_numpy(target).float())
                geo_syms.append(symmetry[cnt])
                sel_ind.append(cnt)
            gt_poses.append(torch.tensor(gt_pose[sel_ind], dtype=torch.float32))

        model_pc = torch.tensor(model_pc, dtype=torch.float32)
        image_pc = pad_sequence(image_pc, batch_first=True)# torch.tensor(image_pc, dtype=torch.float32)
        gt_poses = torch.cat(gt_poses, 0)

        return {
            "model_pc": model_pc, 
            "image_pc": image_pc, 
            # "image_mask": image_mask, 
            'symmetry': geo_syms, 
            "poses_world": gt_poses
            }
     
    def valMerge(self, id):
        geo_syms = []
        model_pc = []
        image_pc = []
        gt_poses = []
        image_mask = []
        MaxLenImage = 0
        for idx in id:
            data, meta = self.train_files[idx]
            # rgb, depth, label, meta = read_file(rgb_file, depth_file, label_file, meta_file)
            # coords, gt_pose, _ = export_one_scan(depth, meta)
            rgb, _, label, coords, gt_pose = data
            selected = self.val_df.loc[self.val_df['object'].isin(meta['object_names'])]
            symmetry = selected['geometric_symmetry'].to_list()
            loc = selected['location'].to_list()
            ind = selected.index.to_list()
            assert len(loc) == len(ind)

            cnt = -1
            for directory, index in zip(loc, ind):
                cnt += 1
                scale = meta['scales'][index]
                
                # Read Model
                source = np.concatenate(
                    [
                        np.array(self.MODELS[directory]['coords']) * scale, 
                        np.array(self.MODELS[directory]['normals'])
                    ], axis=1)

                # Read image target pose
                image_points = coords[label == index].reshape((-1,3))
                select_colors = rgb[label == index].reshape((-1,3))
                if image_points.shape[0] == 0:
                    continue
                target = image_points @ np.linalg.inv(meta['extrinsic'])[:3, :3].T + np.linalg.inv(meta['extrinsic'])[:3, 3]
                target = np.concatenate([target, select_colors], axis=1)

                model_pc.append(source)
                image_pc.append(target)
                if MaxLenImage < target.shape[0]: MaxLenImage = len(target)
                geo_syms.append(symmetry[cnt])
                gt_poses.append(gt_pose[cnt])
        
        for i, pc in enumerate(image_pc):
            if pc.shape[0] < MaxLenImage:
                pad = np.zeros((MaxLenImage -  pc.shape[0], 6))
                image_pc[i] = np.concatenate([pc, pad], axis=0)
            image_mask.append(np.concatenate([np.ones(pc.shape[0]), np.zeros(MaxLenImage -  pc.shape[0])], axis=0))

        model_pc = torch.tensor(model_pc, dtype=torch.float32)
        image_pc = torch.tensor(image_pc, dtype=torch.float32)
        image_mask = torch.tensor(image_mask, dtype=torch.float32)
        gt_poses = torch.tensor(gt_poses, dtype=torch.float32)

        return {
            "model_pc": model_pc, 
            "image_pc": image_pc, 
            "image_mask": image_mask, 
            'symmetry': geo_syms, 
            "poses_world": gt_poses
            }

    def testMerge(self, id):
        geo_syms = []
        model_pc = []
        image_pc = []
        image_mask = []
        scene_name = []
        p2s_map = []
        labels = []
        MaxLenImage = 0
        for bid, idx in enumerate(id):
            rgb, depth, label, meta, rgb_file = self.train_files[idx]
            # rgb, depth, label, meta = read_file(rgb_file, depth_file, label_file, meta_file)
            scene_name.append(rgb_file.split('/')[-1].split('_')[0])
            coords, _ = export_one_scan(depth, meta, task='test')
            selected = self.train_df.loc[self.train_df['object'].isin(meta['object_names'])]
            symmetry = selected['geometric_symmetry'].to_list()
            loc = selected['location'].to_list()
            ind = selected.index.to_list()
            assert len(loc) == len(ind)

            cnt = -1
            for directory, index in zip(loc, ind):
                cnt += 1
                scale = meta['scales'][index]
                
                # Read Model
                source = np.concatenate(
                    [
                        np.array(self.MODELS[directory]['coords']) * scale, 
                        np.array(self.MODELS[directory]['normals'])
                    ], axis=1)

                # Read image target pose
                image_points = coords[label == index].reshape((-1,3))
                select_colors = rgb[label == index].reshape((-1,3))
                if image_points.shape[0] == 0:
                    continue
                target = image_points @ np.linalg.inv(meta['extrinsic'])[:3, :3].T + np.linalg.inv(meta['extrinsic'])[:3, 3]
                target = np.concatenate([target, select_colors], axis=1)

                model_pc.append(source)
                image_pc.append(target)
                if MaxLenImage < target.shape[0]: MaxLenImage = len(target)
                geo_syms.append(symmetry[cnt])
                p2s_map.append(bid)
                labels.append(index)
        
        for i, pc in enumerate(image_pc):
            if pc.shape[0] < MaxLenImage:
                pad = np.zeros((MaxLenImage -  pc.shape[0], 6))
                image_pc[i] = np.concatenate([pc, pad], axis=0)
            image_mask.append(np.concatenate([np.ones(pc.shape[0]), np.zeros(MaxLenImage -  pc.shape[0])], axis=0))

        model_pc = torch.tensor(model_pc, dtype=torch.float32)
        image_pc = torch.tensor(image_pc, dtype=torch.float32)
        image_mask = torch.tensor(image_mask, dtype=torch.float32)

        return {
            "model_pc": model_pc, 
            "image_pc": image_pc, 
            "image_mask": image_mask, 
            'symmetry': geo_syms, 
            "label": labels,
            "p2s_map": p2s_map,
            "scene_name": scene_name
            }