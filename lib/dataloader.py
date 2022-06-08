import sys, os, glob, numpy as np
import pandas as pd
from tqdm.contrib import tzip
from tqdm import tqdm
import torch
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.getcwd())

NUM_OBJECTS = 79

class Dataset:
    def __init__(self): 
        self.batch_size = 4
        self.train_workers = 4
        self.val_workers = 4
    
    def trainLoader(self, logger):
        f_train = sorted(glob.glob(os.path.join('datasets/training_data/data/train', "*_data_aggregated.pth")))
        self.train_files = []
        for f in f_train:
            data = torch.load(f)
            self.train_files += [[rgb, depth, label, intrinsic, box] for rgb, depth, label, intrinsic, box in tqdm(data)]

        logger.info('Training samples: {}'.format(len(self.train_files)))
        assert len(self.train_files) > 0

        train_set = list(range(len(self.train_files)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    
    
    def valLoader(self, logger):
        f_val = sorted(glob.glob(os.path.join('datasets/training_data/data/val', "*_data_aggregated.pth")))
        self.val_files = []
        for f in f_val:
            data = torch.load(f)
            self.val_files += [[rgb, depth, label, intrinsic, box] for rgb, depth, label, intrinsic, box in tqdm(data)]

        logger.info('Validation samples: {}'.format(len(self.val_files)))
        assert len(self.val_files) > 0

        val_set = list(range(len(self.val_files)))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge, num_workers=self.val_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    
    
    def fpn_testLoader(self, logger):
        f_test = sorted(glob.glob(os.path.join('datasets/testing_data/data/test', "*_data_aggregated.pth")))
        self.test_files = []
        for f in f_test:
            rgbs, depths, labels, intrinsics = torch.load(f)
            self.test_files += [[rgb, depth, label, intrinsic] for rgb, depth, label, intrinsic in tzip(rgbs, depths, labels, intrinsics)]

        logger.info('Testing samples: {}'.format(len(self.test_files)))
        assert len(self.test_files) > 0

        test_set = list(range(len(self.test_files)))
        self.test_data_loader = DataLoader(test_set, batch_size=self.batch_size, collate_fn=self.fpn_testMerge, num_workers=self.test_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)

    def testLoader(self, logger):
        f_test = sorted(glob.glob(os.path.join('datasets/testing_data/data/test', "*_data_aggregated.pth")))
        self.test_files = []
        for f in f_test:
            rgbs, depths, labels, intrinsics = torch.load(f)
            with open(os.path.join('datasets/2Dproposal.json'), "r") as f:
                preds = json.load(f)
            self.test_files += [[rgb, depth, label, intrinsic, pred] for rgb, depth, label, intrinsic, pred in tzip(rgbs, depths, labels, intrinsics, preds)]

        logger.info('Testing samples: {}'.format(len(self.test_files)))
        assert len(self.test_files) > 0

        test_set = list(range(len(self.test_files)))
        self.test_data_loader = DataLoader(test_set, batch_size=self.batch_size, collate_fn=self.testMerge, num_workers=self.test_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    

    def trainMerge(self, id):
        rgbs = []
        depths = []
        labels = []
        metas = []
        boxes = []
        for idx in id:
            rgb, depth, label, intrinsic, box = self.train_files[idx]
            rgbs.append(torch.tensor(rgb))
            depths.append(torch.tensor(depth))
            labels.append(torch.tensor(label))
            metas.append(torch.tensor(intrinsic))
            boxes.append(torch.tensor(box))
        
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        labels = torch.stack(labels, 0)
        metas = torch.stack(metas, 0)
        boxes = torch.stack(boxes, 0)

        return {
            "rgb": rgbs, 
            "depth": depths, 
            "meta": metas, 
            "box": boxes,
            "gt": labels
            }
     
    def valMerge(self, id):
        rgbs = []
        depths = []
        labels = []
        metas = []
        boxes = []
        for idx in id:
            rgb, depth, label, intrinsic, box = self.val_files[idx]
            rgbs.append(torch.tensor(rgb))
            depths.append(torch.tensor(depth))
            labels.append(torch.tensor(label))
            metas.append(torch.tensor(intrinsic))
            boxes.append(torch.tensor(box))
        
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        labels = torch.stack(labels, 0)
        metas = torch.stack(metas, 0)
        boxes = torch.stack(boxes, 0)

        return {
            "rgb": rgbs, 
            "depth": depths, 
            "meta": metas, 
            "box": boxes,
            "gt": labels
            }

    def fpn_testMerge(self, id):
        rgbs = []
        depths = []
        metas = []
        for idx in id:
            rgb, depth, intrinsic = self.test_files[idx]
            rgbs.append(torch.tensor(rgb))
            depths.append(torch.tensor(depth))
            metas.append(torch.tensor(intrinsic))
        
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        metas = torch.stack(metas, 0)

        return {
            "rgb": rgbs, 
            "depth": depths, 
            "meta": metas
            }
    
    def testMerge(self, id):
        rgbs = []
        depths = []
        metas = []
        boxes = []
        labels = []
        scores = []
        for idx in id:
            rgb, depth, intrinsic, pred = self.test_files[idx]
            box, label, score = pred
            rgbs.append(torch.tensor(rgb))
            depths.append(torch.tensor(depth))
            metas.append(torch.tensor(intrinsic))
            boxes.append(box)
            labels.append(label)
            scores.append(score)
        
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        metas = torch.stack(metas, 0)
        boxes = torch.stack(boxes, 0)
        labels = torch.stack(labels, 0)
        scores = torch.stack(scores, 0)

        return {
            "rgb": rgbs, 
            "depth": depths, 
            "meta": metas,
            "box": boxes,
            "label": labels,
            "score": scores
            }