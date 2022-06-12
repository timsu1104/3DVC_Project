import sys, os, glob, numpy as np
from tqdm.contrib import tzip
from tqdm import tqdm
import torch
import json
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from utils.util import get_split_files

NUM_OBJECTS = 79

class Dataset:
    def __init__(self): 
        self.batch_size = 8
        self.train_workers = 4
        self.val_workers = 4
    
    def trainLoader(self, logger, TOY):
        if TOY:
            f_train = get_split_files('train', prefix="datasets/")
            self.train_files = [torch.load(data) for data in tqdm(f_train[:100])]
        else:
            f_train = sorted(glob.glob(os.path.join('datasets/training_data/data/train', "1*_data_aggregated.pth")))
            self.train_files = []
            for f in f_train:
                print("Loading", f)
                rgbs, depths, labels, intrinsics, boxs = torch.load(f)
                print("Loaded", f)
                for rgb, depth, label, intrinsic, box in zip(rgbs, depths, labels, intrinsics, boxs):
                    self.train_files.append([rgb, depth, label, intrinsic, box])

        logger.info('Training samples: {}'.format(len(self.train_files)))
        assert len(self.train_files) > 0

        train_set = list(range(len(self.train_files)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    
    
    def valLoader(self, logger, TOY):
        if TOY:
            f_val = get_split_files('val', prefix="datasets/")
            self.val_files = [torch.load(data) for data in tqdm(f_val[:20])]
        else:
            f_val = sorted(glob.glob(os.path.join('datasets/training_data/data/val', "*_data_aggregated.pth")))
            self.val_files = []
            for f in f_val:
                rgbs, depths, labels, intrinsics, boxs = torch.load(f)
                for rgb, depth, label, intrinsic, box in zip(rgbs, depths, labels, intrinsics, boxs):
                    self.val_files.append([rgb, depth, label, intrinsic, box])

        logger.info('Validation samples: {}'.format(len(self.val_files)))
        assert len(self.val_files) > 0

        val_set = list(range(len(self.val_files)))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge, num_workers=self.val_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    
    
    def fpn_testLoader(self, logger):
        f_test = sorted(glob.glob(os.path.join('datasets/testing_data/data/test', "*_data_aggregated.pth")))
        self.test_files = []
        for f in f_test:
            rgbs, depths, labels, intrinsics = torch.load(f)
            for rgb, depth, label, intrinsic in zip(rgbs, depths, labels, intrinsics):
                self.test_files.append([rgb, depth, label, intrinsic])

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
            for rgb, depth, label, intrinsic, pred in zip(rgbs, depths, labels, intrinsics, preds):
                self.test_files.append([rgb, depth, label, intrinsic, pred])

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
            if rgb.size(0) == 0 or rgb.size(1) == 0:
                continue
            rgbs.append(rgb)
            depths.append(depth)
            labels.append(label)
            metas.append(intrinsic)
            assert len(box.shape) == 2
            # new_box = []
            # for single_box in box:
            #     for x1, y1, x2, y2, lbl in single_box:
            #         if x2 > x1 and y2 > y1:
            #             new_box.append([x1, y1, x2, y2, lbl])
            # new_box = torch.tensor(new_box)
            x_diff = box[:, 2] - box[:, 0]
            y_diff = box[:, 3] - box[:, 1]
            box = box[(x_diff > 0) * (y_diff > 0)]
            boxes.append(box)
        
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        labels = torch.stack(labels, 0)
        metas = torch.stack(metas, 0)

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
            rgbs.append(rgb)
            depths.append(depth)
            labels.append(label)
            metas.append(intrinsic)
            new_box = []
            # for single_box in box:
            #     for x1, y1, x2, y2, lbl in single_box:
            #         if x2 > x1 and y2 > y1:
            #             new_box.append([x1, y1, x2, y2, lbl])
            # new_box = torch.tensor(new_box)
            x_diff = box[:, 2] - box[:, 0]
            y_diff = box[:, 3] - box[:, 1]
            box = box[(x_diff > 0) * (y_diff > 0)]
            boxes.append(new_box)
        
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        labels = torch.stack(labels, 0)
        metas = torch.stack(metas, 0)

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
    
    def testMerge(self, id):
        rgbs = []
        depths = []
        metas = []
        boxes = []
        scores = []
        for idx in id:
            rgb, depth, intrinsic, pred = self.test_files[idx]
            box, label, score = pred
            box = torch.cat([box, label.unsqueeze(1)], dim=1)
            rgbs.append(rgb)
            depths.append(depth)
            metas.append(intrinsic)
            boxes.append(box)
            scores.append(score)
        
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        metas = torch.stack(metas, 0)
        # boxes = torch.stack(boxes, 0)
        # scores = torch.stack(scores, 0)

        return {
            "rgb": rgbs, 
            "depth": depths, 
            "meta": metas,
            "box": boxes,
            "score": scores
            }