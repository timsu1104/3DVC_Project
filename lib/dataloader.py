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
    
    def trainLoader(self, logger):
        rgbs, depths, labels, intrinsics = torch.load('datasets/train.pth')
        self.train_files = [[rgb, depth, label, intrinsic] for rgb, depth, label, intrinsic in tzip(rgbs, depths, labels, intrinsics)]

        logger.info('Training samples: {}'.format(len(self.train_files)))
        assert len(self.train_files) > 0

        train_set = list(range(len(self.train_files)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    
    
    def valLoader(self, logger):
        rgbs, depths, labels, intrinsics = torch.load('datasets/val.pth')
        self.val_files = [[rgb, depth, label, intrinsic] for rgb, depth, label, intrinsic in tzip(rgbs, depths, labels, intrinsics)]

        logger.info('Validation samples: {}'.format(len(self.val_files)))
        assert len(self.val_files) > 0

        val_set = list(range(len(self.val_files)))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge, num_workers=self.val_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    
    
    def testLoader(self, logger):
        rgbs, depths, labels, intrinsics = torch.load('datasets/test.pth')
        self.test_files = [[rgb, depth, intrinsic] for rgb, depth, intrinsic in tzip(rgbs, depths, intrinsics)]

        logger.info('Testing samples: {}'.format(len(self.test_files)))
        assert len(self.test_files) > 0

        test_set = list(range(len(self.test_files)))
        self.test_data_loader = DataLoader(test_set, batch_size=self.batch_size, collate_fn=self.testMerge, num_workers=self.test_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    

    def trainMerge(self, id):
        rgbs = []
        depths = []
        labels = []
        metas = []
        for idx in id:
            rgb, depth, label, intrinsic = self.train_files[idx]
            rgbs.append(rgb)
            depths.append(depth)
            labels.append(label)
            metas.append(intrinsic)
        
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        labels = torch.stack(labels, 0)
        metas = torch.stack(metas, 0)

        return {
            "rgb": rgbs, 
            "depth": depths, 
            "meta": metas, 
            "gt": labels
            }
     
    def valMerge(self, id):
        rgbs = []
        depths = []
        labels = []
        metas = []
        for idx in id:
            rgb, depth, label, intrinsic = self.val_files[idx]
            rgbs.append(rgb)
            depths.append(depth)
            labels.append(label)
            metas.append(intrinsic)
        
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        labels = torch.stack(labels, 0)
        metas = torch.stack(metas, 0)

        return {
            "rgb": rgbs, 
            "depth": depths, 
            "meta": metas, 
            "gt": labels
            }

    def testMerge(self, id):
        rgbs = []
        depths = []
        metas = []
        for idx in id:
            rgb, depth, intrinsic = self.test_files[idx]
            rgbs.append(rgb)
            depths.append(depth)
            metas.append(intrinsic)
        
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        metas = torch.stack(metas, 0)

        return {
            "rgb": rgbs, 
            "depth": depths, 
            "meta": metas
            }