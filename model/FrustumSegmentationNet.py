import torch, torch.nn as nn
import numpy as np
from torchsparse import PointTensor
from torchsparse.utils import sparse_collate
from pytorch3d.ops import ball_query
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
        self.getlabel = nn.Linear(1024, 82)
        self.h = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def sample(self, pc: torch.tensor, num: int = 4000):
        """
        Iteratively sample points out of a point cloud by Farthest Point Sampling (FPS). 

        Parameter
        -----------
        Pointcloud: torch.tensor, (N, 3)
            The input pointcloud. 
        num: int, 
            Number of points to be selected. 

        Return
        -----------
        index: torch.tensor,  (M, 3)
            Sampled points' index. 
        """
        select = []
        seed = np.random.randint(0, num)
        select.append(seed)
        dist = torch.sum((pc - pc[seed]) ** 2, dim=1)
        for _ in range(num-1):
            seed = torch.argmin(dist)
            select.append(seed)
            new_dist = torch.sum((pc - pc[seed]) ** 2, dim=1)
            dist = torch.minimum(dist, new_dist)
        return torch.tensor(select).cuda().long()

    def image2pc(self, depth, intrinsic):
        """
        Takes in the cropped depth and intrinsic data, return the pointcloud. 
        """
        z = depth
        v, u = np.indices(z.shape)
        v = torch.from_numpy(v).cuda()
        u = torch.from_numpy(u).cuda()
        uv1 = torch.stack([u + 0.5, v + 0.5, torch.ones_like(z)], axis=-1)
        coords = uv1 @ torch.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
        return coords
    
    def set_abstraction(self, pc, num=4000):
        """
        Iteratively sample points out of a point cloud by Farthest Point Sampling (FPS). 

        Parameter
        -----------
        Pointcloud: torch.tensor, (M, K, 3 + C)
            The input pointcloud. 
        num: int, 
            Number of points to be selected. 

        Return
        -----------
        index: torch.tensor,  (M, 3)
            Sampled points' index. 
        """
        index = self.sample(pc, num=num)
        selected = pc[index]
        grouped_feats = ball_query(selected, ) # (M, K, 3 + C)
        output_feats = torch.max(grouped_feats, dim=1)
        return output_feats

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
        for bind, single_box in enumerate(box):
            label = torch.zeros((82, *rgb.shape[1:-1])).cuda()
            single_box = single_box.cuda().long()
            for x1, y1, x2, y2, _ in single_box:
                # Cropping and lifting
                cropped_pc = self.image2pc(depth[bind, x1 : x2, y1 : y2], intrinsic[bind])

                # 3D PointCloud Segmentation
                x = torch.cat([cropped_pc, rgb[bind, x1 : x2, y1 : y2]], dim=-1)
                orig_shape = x.shape[:-1]
                x = self.f(x).view(-1, 1024)
                global_feats = torch.max(x, dim=0)[0]
                lbl = self.getlabel(global_feats)
                x = torch.cat([x, torch.repeat_interleave(global_feats.unsqueeze(0), x.size(0), dim=0)], dim=1)
                x = self.h(x).view((*orig_shape, -1))

                # Extract Segmentation
                seg = torch.nonzero(torch.argmax(x, dim=-1) == 1)
                torch._assert(
                    x.size(-1) == 2,
                    "x.size is {}, x is ".format(x.shape, x)
                )
                if seg.size(0) == 0:
                    print("2 continue")
                    continue
                xind, yind = seg[:, 0], seg[:, 1]
                xind += x1
                yind += y1

                # Scaling by Kernel
                center = torch.stack([x1 + x2, y1 + y2]).float() / 2
                kernel = 1 / torch.norm(seg - center, p=2, dim=1)

                label[:, xind, yind] = label[:, xind, yind] + lbl.unsqueeze(1) * kernel
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

        Criterion = nn.CrossEntropyLoss()

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