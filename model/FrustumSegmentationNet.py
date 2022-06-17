import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchsparse import PointTensor
from torchsparse.utils import sparse_collate
# from pytorch3d.ops import ball_query
import sys, os

sys.path.append(os.getcwd())
from utils.spvcnn_utils import *

class PointNet(nn.Module):
    def __init__(self, input_channel=3, output_channel=6) -> None:
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(3 + input_channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.dense1 = torch.nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dense2 = torch.nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dense3 = torch.nn.Linear(256, output_channel)

        for net in [
            self.conv1,
            self.conv2,
            self.conv3,
            self.dense1,
            self.dense2,
            self.dense3,
        ]:
            torch.nn.init.xavier_uniform_(net.weight)

    def forward(self, target):
        """
        Parameters
        ------------
        target: torch.tensor, (BatchSize, NumPoints, 6)
        """
        points = target[..., :3]
        colors = target[..., 3:]
        length = (points.max(dim=1, keepdim=True)[0] - points.min(dim=1, keepdim=True)[0]) / 2
        points = points * 10 / length
        x = torch.cat([points, colors], dim=-1)

        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.dense1(x)))
        x = F.relu(self.bn5(self.dense2(x)))
        Raw_result = self.dense3(x) # 6 + 3
        
        return Raw_result

class FrustumSegmentationNet(nn.Module):
    def __init__(self) -> None:
        super(FrustumSegmentationNet, self).__init__()
        self.pointnet = PointNet(output_channel=128)
        self.get_score = nn.Linear(3 + 3 + 128, 80)

    def sample(self, pc: torch.tensor, num: int = 500):
        """
        Iteratively sample points out of a point cloud by Farthest Point Sampling (FPS). 

        Parameter
        -----------
        Pointcloud: torch.tensor, (N, 3 + C)
            The input pointcloud. 
        num: int, 
            Number of points to be selected. 

        Return
        -----------
        index: torch.tensor,  (M, )
            Sampled points' index. 
        """
        xyz = pc[:, :3]
        select = []
        iternum = min(pc.shape[0], num)
        seed = np.random.randint(iternum)
        select.append(seed)
        dist = torch.sum((xyz - xyz[seed].unsqueeze(0)) ** 2, dim=-1)
        for _ in range(iternum-1):
            seed = torch.argmin(dist)
            select.append(seed)
            new_dist = torch.sum((xyz - xyz[seed].unsqueeze(0)) ** 2, dim=-1)
            dist = torch.minimum(dist, new_dist)
        return torch.tensor(select).cuda().long()
    
    def ball_query(self, pc, sel_pts, K=50, radius=1e-3):
        """
        Grouping Layer. 

        Parameter
        -----------
        Pointcloud: torch.tensor, (N, 3 + C)
            The input pointcloud. 
        sel_pts: torch.tensor, (N', 3 + C)
        K: int, 
            Max points number.
        radius: double, 
            Max ball radius 

        Return
        -----------
        grouped_pts: torch.tensor,  (N', K, 3 + C)
            Sampled points. 
        """
        xyz = pc[:, :3]
        sel_xyz = sel_pts[:, :3]
        dist = torch.norm(torch.repeat_interleave(sel_xyz.unsqueeze(1), xyz.shape[0], dim=1) - xyz.unsqueeze(0), p=2, dim=2) # (N', N)
        value, index = torch.sort(dist, dim=-1)
        output = pc[index] # (N', N, 3)
        mask = torch.sum(value < radius, dim=-1).clamp(min=0, max=K)
        output = output * torch.broadcast_to(torch.unsqueeze(value < radius, -1), output.shape)
        output = output[:, :K]
        return output, index, mask
    
    def center(self, pc):
        xyz = pc[..., :3]
        feats = pc[..., 3:]
        xyz = xyz - torch.mean(xyz, dim=-2, keepdim=True)
        length = xyz.max(dim=-2, keepdim=True)[0] - xyz.min(dim=-2, keepdim=True)[0]
        xyz = xyz / length # scale
        centered_pc = torch.cat([xyz, feats], dim=-1)
        return centered_pc

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
    
    def set_abstraction(self, pc, num=500):
        """
        Set abstraction layer. 

        Parameter
        -----------
        Pointcloud: torch.tensor, (N, 3 + C)
            The input pointcloud. 
        num: int, 
            Number of points to be selected. 

        Return
        -----------
        output_feats: torch.tensor,  (M, 3 + C')
            Sampled points' index. 
        allocate_index: torch.tensor,  (M, K)
        """
        index = self.sample(pc, num=num)
        selected = pc[index]
        grouped_feats, allocate_index, mask = self.ball_query(pc, selected) # (M, K, 3 + C)
        grouped_feats = self.center(grouped_feats)
        output_feats = self.pointnet(grouped_feats) # (M, 3 + C')
        return output_feats, allocate_index, mask
    
    def segmentation(self, pc, abs_feats, index, masks):
        """
        Segmentation layer. 

        Parameter
        -----------
        Pointcloud: torch.tensor, (N, C)
            The input pointcloud. 
        abs_feats: torch.tensor,  (M, C')
            Abstract features for each cluster.
        index: torch.tensor,  (M, K) 
            The points in each cluster.

        Return
        -----------
        seg: 
        """
        N, _ = pc.shape
        f_ind = torch.zeros(N).long()
        for clus_id, (ind, mask) in enumerate(zip(index, masks)):
            f_ind[ind[:mask]] = clus_id
        seg = torch.cat([pc, abs_feats[f_ind]], dim=-1)
        
        seg = self.get_score(seg)
        return seg

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
        label: torch.Tensor, (BatchSize, H, W, 80)
        """
        labels = []
        for bind, single_box in enumerate(box):
            label = Variable(torch.zeros((80, *rgb.shape[1:-1]))).cuda()
            if len(single_box) == 0: continue
            single_box = single_box.cuda().long()
            for x1, y1, x2, y2, _ in single_box:
                # Cropping and lifting
                cropped_pc = self.image2pc(depth[bind, x1 : x2, y1 : y2], intrinsic[bind])

                # 3D PointCloud Segmentation
                x = torch.cat([cropped_pc, rgb[bind, x1 : x2, y1 : y2]], dim=-1)
                orig_shape = x.shape
                x = x.view((-1, orig_shape[-1]))
                mid_params = self.set_abstraction(x)

                # Extract Segmentation
                seg = self.segmentation(x, *mid_params)
                assert x.size(0) == seg.size(0)
                seg = seg.view((*orig_shape[:-1], -1)) # H, W, 80

                if seg.size(0) == 0:
                    continue
                # xind, yind = seg[:, 0], seg[:, 1]
                # xind += x1
                # yind += y1

                # Scaling by Kernel
                # center = torch.stack([x1 + x2, y1 + y2]).float() / 2
                # kernel = 1 / torch.norm(cropped_pc - center, p=2, dim=1)

                label[:, x1 : x2, y1 : y2] = seg.permute(2, 0, 1) # * kernel
            
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
        gt.clamp_(max=79)

        Criterion = nn.CrossEntropyLoss()

        pred = model(rgb, depth, intrinsic, box)
        torch._assert(
                pred.grad_fn != None,
                "Gradient disappeared!"
            )
        torch._assert(
                pred.size(1) == 80,
                f"Size Error: pred {pred.shape}, gt {gt.shape}"
            )
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