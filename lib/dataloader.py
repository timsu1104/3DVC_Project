import sys, os, glob
from tqdm import tqdm
import torch
import json
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from utils.util import get_split_files

NUM_OBJECTS = 79

class Dataset:
    def __init__(self): 
        self.batch_size = 4
        self.train_workers = 4
        self.val_workers = 4
        self.test_workers = 4
    
    def trainLoader(self, logger, TOY=False):
        if TOY:
            f_train = get_split_files('train', prefix="datasets/")
            self.train_files = [torch.load(data) for data in tqdm(f_train[:100])]
        else:
            f_train = get_split_files('train', prefix="datasets/")
            self.train_files = [torch.load(data) for data in tqdm(f_train)]

        logger.info('Training samples: {}'.format(len(self.train_files)))
        assert len(self.train_files) > 0

        train_set = list(range(len(self.train_files)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    
    
    def valLoader(self, logger, TOY=False):
        if TOY:
            f_val = get_split_files('val', prefix="datasets/")
            self.val_files = [torch.load(data) for data in tqdm(f_val[:20])]
        else:
            f_val = get_split_files('val', prefix="datasets/")
            self.val_files = [torch.load(data) for data in tqdm(f_val)]

        logger.info('Validation samples: {}'.format(len(self.val_files)))
        assert len(self.val_files) > 0

        val_set = list(range(len(self.val_files)))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge, num_workers=self.val_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    

    def testLoader(self, logger):
        f_test = sorted(glob.glob(os.path.join('datasets/testing_data/data/test', "*_data_aggregated.pth")))
        self.test_files = []
        for f in f_test:
            rgbs, depths, intrinsics = torch.load(f)
            with open(os.path.join('datasets/2Dproposal.json'), "r") as f:
                preds = json.load(f)
            for rgb, depth, intrinsic, pred in zip(rgbs, depths,intrinsics, preds):
                self.test_files.append([rgb, depth, intrinsic, pred])

        logger.info('Testing samples: {}'.format(len(self.test_files)))
        assert len(self.test_files) > 0

        test_set = list(range(len(self.test_files)))
        self.test_data_loader = DataLoader(test_set, batch_size=self.batch_size, collate_fn=self.testMerge, num_workers=self.test_workers, shuffle=False, sampler=None, drop_last=True, pin_memory=True)    

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
            assert len(box.shape) == 2
            x_diff = box[:, 2] - box[:, 0]
            y_diff = box[:, 3] - box[:, 1]
            lbl = box[:, 4]
            mask = (x_diff > 0) * (y_diff > 0) * (lbl < 79)
            if torch.sum(mask) == 0:
                continue
            box = box[mask]
            torch._assert(
                len(box) != 0,
                f"x_diff {x_diff}, y_diff {y_diff}, lbl {lbl}, box {box}"
            )

            label = torch.clamp(label, min=0, max=79)

            rgbs.append(rgb)
            depths.append(depth)
            labels.append(label)
            metas.append(intrinsic)
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
            if rgb.size(0) == 0 or rgb.size(1) == 0:
                continue
            assert len(box.shape) == 2
            x_diff = box[:, 2] - box[:, 0]
            y_diff = box[:, 3] - box[:, 1]
            lbl = box[:, 4]
            mask = (x_diff > 0) * (y_diff > 0) * (lbl < 79)
            if torch.sum(mask) == 0:
                continue
            box = box[mask]
            torch._assert(
                len(box) != 0,
                f"x_diff {x_diff}, y_diff {y_diff}, lbl {lbl}, box {box}"
            )
            
            label = torch.clamp(label, min=0, max=79)

            rgbs.append(rgb)
            depths.append(depth)
            labels.append(label)
            metas.append(intrinsic)
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

    def testMerge(self, id):
        rgbs = []
        depths = []
        metas = []
        boxes = []
        scores = []
        for idx in id:
            rgb, depth, intrinsic, pred = self.test_files[idx]
            box, label, score = pred
            box = torch.tensor(box)
            label = torch.tensor(label).long()
            score = torch.tensor(score)
            box = torch.cat([box[:, torch.tensor([1, 0, 3, 2])], label.unsqueeze(1)], dim=1)
            rgbs.append(rgb)
            depths.append(depth)
            metas.append(intrinsic)
            box = box[score > 0.3]
            if box.size(0) > 10:
                box = box[:10]
            boxes.append(box)
            scores.append(score)
        
        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        metas = torch.stack(metas, 0)

        return {
            "rgb": rgbs, 
            "depth": depths, 
            "meta": metas,
            "box": boxes,
            "score": scores
            }