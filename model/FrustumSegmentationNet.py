import torch, torch.nn as nn
import numpy as np
from torchsparse import PointTensor
from torchsparse.utils import sparse_collate
import sys, os

sys.path.append(os.getcwd())
from utils.spvcnn_utils import *

class FrustumSegmentationNet(nn.Module):
    def __init__(self) -> None:
        super(FrustumSegmentationNet, self).__init__()
        # self.segment = SegmentModule(in_channels=3, voxel_size=0.2, num_classes=2)
        self.f = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU()
        )
        self.h = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def image2pc(self, depth, intrinsic):
        """
        Takes in the cropped depth and intrinsic data, return the pointcloud. 
        """
        z = depth
        # print(z.shape)
        v, u = np.indices(z.shape)
        v = torch.from_numpy(v).cuda()
        u = torch.from_numpy(u).cuda()
        # v, u = torch.from_numpy(np.indices(z.shape)).cuda()
        uv1 = torch.stack([u + 0.5, v + 0.5, torch.ones_like(z)], axis=-1)
        coords = uv1 @ torch.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
        return coords

    def forward(self, rgb, depth, intrinsic, box):
        """
        Parameters
        ------------
        rgb: torch.Tensor, (BatchSize, H, W, 3)
        depth: torch.Tensor, (BatchSize, H, W)
        intrinsic: torch.Tensor, (BatchSize, 3, 3)
        box: List[torch.Tensor], (BatchSize, M, 5)
        
        Return
        ---------
        label: torch.Tensor, (BatchSize, H, W)
        """
        labels = []
        ### TODO: Crop depth and segment
        for bind, single_box in enumerate(box):
            # print(single_box)
            label = torch.zeros((79, *rgb.shape[1:-1])).cuda()
            for x1, y1, x2, y2, lbl in single_box.long():
                if x2 <= x1 or y2 <= y1:
                    continue
                cropped_pc = self.image2pc(depth[bind, x1 : x2, y1 : y2], intrinsic[bind])
                x = torch.cat([cropped_pc, rgb[bind, x1 : x2, y1 : y2]], dim=-1)
                orig_shape = x.shape[:-1]
                x = self.f(x).view(-1, 1024)
                global_feats = torch.max(x, dim=0)[0]
                x = torch.cat([x, torch.repeat_interleave(global_feats.unsqueeze(0), x.size(0), dim=0)], dim=1)
                x = self.h(x).view((*orig_shape, -1))
                seg = torch.nonzero(torch.argmax(x, dim=-1) == 1)
                if seg.size(0) == 0:
                    continue
                xind, yind = seg[:, 0], seg[:, 1]
                xind += x1
                yind += y1
                label[lbl, xind, yind] = 1.
            labels.append(label)
        labels = torch.stack(labels, 0)

        return labels
        

def model_fn_decorator(test=False):
    def model_fn(batch, model):
        rgb = batch['rgb'].cuda()
        depth = batch['depth'].cuda()
        intrinsic = batch['meta'].cuda()
        box = batch['box']
        gt = batch['gt'].cuda().long()

        Criterion = nn.CrossEntropyLoss(reduction='none')

        pred = model(rgb, depth, intrinsic, box)
        loss = Criterion(pred, gt)

        return loss, pred
        
    def test_model_fn(batch, model):
        rgb = batch['rgb'].cuda()
        depth = batch['depth'].cuda()
        intrinsic = batch['meta'].cuda()
        box = batch['box']

        pred = model(rgb, depth, intrinsic, box)

        return pred

    if test:
        return test_model_fn
    return model_fn