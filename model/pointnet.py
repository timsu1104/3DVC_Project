import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, time

sys.path.append(os.getcwd())
from lib.loss_helper import get_loss

NUMOBJECTS = 79
K = 64

class PointNet(nn.Module):
    def __init__(self) -> None:
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.dense1 = torch.nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dense2 = torch.nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dense3 = torch.nn.Linear(256, 3 + 6)

        for net in [
            self.conv1,
            self.conv2,
            self.conv3,
            self.dense1,
            self.dense2,
            self.dense3,
        ]:
            torch.nn.init.xavier_uniform_(net.weight)

    def rot_6d(self, x, y):
        x = F.normalize(x, dim=-1)
        y = y - x * (x * y).sum(-1, keepdims=True)
        y = F.normalize(y, dim=-1)
        z = torch.cross(x, y, -1)
        return torch.stack([x, y, z], dim=-1)

    def forward(self, target, center_trans, scales):
        """
        Parameters
        ------------
        target: torch.Tensor, (BatchSize, NumPoints, 6)
           Pointcloud of the appearance. Centered and scaled already.
        center_trans: torch.Tensor, (BatchSize, 3)
            centering bias, to be added to the translation predicted. 
        scales: torch.Tensor, (BatchSize, 3)
            scales, to be added to the translation predicted. 
        
        Return
        ---------
        RotMat: torch.Tensor, (BatchSize, 3, 3)
            Predicted rotation matrix. 
        Trans: torch.Tensor, (BatchSize, 3)
            Predicted translation. 
        """
        points = target[..., :3]
        colors = target[..., 3:]
        length = (points.max(dim=1, keepdim=True)[0] - points.min(dim=1, keepdim=True)[0]) / 2
        points = points * 10 / length
        x = torch.cat([points, colors], dim=2)

        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.dense1(x)))
        x = F.relu(self.bn5(self.dense2(x)))
        Raw_result = self.dense3(x) # 6 + 3

        # generate prediction
        Trans = Raw_result[..., :3] * length.squeeze() * scales / 10 + center_trans #(B, 3)
        a_1 = Raw_result[..., 3:6]
        a_2 = Raw_result[..., 6:9]
        RotMat = self.rot_6d(a_1, a_2)
        return RotMat, Trans # pred

def model_fn_decorator(test=False):
    def model_fn(batch, model):
        target = batch['image_pc'].cuda()
        gt = batch['poses_world'].cuda()
        sym = batch['symmetry']
        center_trans = batch['center_trans'].cuda()
        scales = batch['scales'].cuda()

        pred = model(target, center_trans, scales)
        loss_pack = get_loss(pred, gt, sym)

        return loss_pack
        
    def test_model_fn(batch, model):
        target = batch['image_pc'].cuda()
        center_trans = batch['center_trans'].cuda()
        scales = batch['scales'].cuda()

        RotMat, Trans = model(target, center_trans, scales)

        pred = torch.cat([RotMat, Trans.unsqueeze(2)], dim=2)
        foot = torch.tensor([[[0., 0., 0., 1.]]], dtype=torch.float32).cuda()
        foot = torch.repeat_interleave(foot, pred.shape[0], dim=0)
        pred = torch.cat([pred, foot], dim=1) # (B, 4, 4)

        return pred

    if test:
        return test_model_fn
    return model_fn