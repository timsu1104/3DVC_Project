import torch, torch.nn as nn
from minkunet import MinkUNet as SegmentModule
import torchsparse
import torchsparse.nn as spnn
from torchsparse import PointTensor
from torchsparse.utils import sparse_collate

sys.path.append(os.getcwd())
from utils.spvcnn_utils import *

class FrustumSegmentationNet(nn.Module):
    def __init__(self) -> None:
        super(FrustumSegmentationNet, self).__init__()
        self.segment = SegmentModule(in_channels=3, voxel_size=0.2, num_classes=2)

    def image2pc(self, depth, intrinsic):
        """
        Takes in the cropped depth and intrinsic data, return the pointcloud. 
        """
        z = depth
        v, u = torch.indices(z.shape)
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
        box: boxes, (BatchSize, M, 5)
        
        Return
        ---------
        label: torch.Tensor, (BatchSize, H, W)
        """
        label = torch.ones_like(rgb)
        ### TODO: Crop depth and segment
        for x1, y1, x2, y2, lbl in box:
            cropped_pc = self.image2pc(depth[x1 : x1 + 1, y1 : y2 + 1], intrinsic)
            coords, feats = sparse_collate([cropped_pc], [rgb[x1 : y1 + 1, x2 : y2 + 1]], coord_float=True)
            x = PointTensor(feats, coords).cuda()
            output = self.segment(x)
            xind, yind = (output == 1)
            xind += x1
            yind += y1
            label[xind, yind] = lbl

        ### TODO: Combine
        

def model_fn_decorator(test=False):
    def model_fn(batch, model):
        rgb = batch['rgb'].cuda()
        depth = batch['depth'].cuda()
        intrinsic = batch['meta'].cuda()
        box = batch['box'].cuda()
        gt = batch['gt'].cuda()

        Criterion = nn.CrossEntropyLoss(reduction='none')

        pred = model(rgb, depth, intrinsic, box)
        loss = Criterion(pred, gt)

        return loss, pred
        
    def test_model_fn(batch, model):
        rgb = batch['rgb'].cuda()
        depth = batch['depth'].cuda()
        intrinsic = batch['meta'].cuda()

        pred = model(rgb, depth, intrinsic, test=True)

        return pred

    if test:
        return test_model_fn
    return model_fn