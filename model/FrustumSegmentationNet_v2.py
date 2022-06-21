import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class FrustumSegmentationNet(nn.Module):
    def __init__(self, inp_dim=3, hid_dim=128, oup_dim=512) -> None:
        super(FrustumSegmentationNet, self).__init__()
        self.f1 = nn.Linear(3 + inp_dim, 3 + hid_dim)
        self.f2 = nn.Linear(3 + hid_dim, 3 + hid_dim)
        self.f3 = nn.Linear(3 + hid_dim, 3 + hid_dim)
        self.h1 = nn.Linear(3 + hid_dim, 3 + oup_dim)
        self.h2 = nn.Linear(3 + oup_dim, 3 + oup_dim)
        self.h3 = nn.Linear(3 + oup_dim, 3 + oup_dim)
        
        self.get_score = nn.Linear(9 + inp_dim + hid_dim + oup_dim, 1)
    
    def pointnet2(self, inp):
        hidden = F.relu(self.f1(inp))
        hidden = F.relu(self.f2(hidden))
        hidden = F.relu(self.f3(hidden))
        x = torch.max(hidden, dim=0)[0]
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        return hidden, x
    
    def center(self, pc, type='xyz'):
        if type == 'xyz':
            xyz = pc[..., :3]
            feats = pc[..., 3:]
            xyz = xyz - torch.mean(xyz, dim=-2, keepdim=True)
            length = (xyz.max(dim=-2, keepdim=True)[0] - xyz.min(dim=-2, keepdim=True)[0]).clamp(min=1e-5)
            
            xyz = xyz / length # scale
            centered_pc = torch.cat([xyz, feats], dim=-1)
            return centered_pc
        else:
            pc = pc - torch.mean(pc, dim=-2, keepdim=True)
            length = (pc.max(dim=-2, keepdim=True)[0] - pc.min(dim=-2, keepdim=True)[0]).clamp(min=1e-5)
            pc = pc / length
            return pc

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
        label: torch.Tensor, (BatchSize, H, W, 2)
        """
        segments = []
        for bind, single_box in enumerate(box):
            single_box = single_box.cuda().long()
            segs = []
            for x1, y1, x2, y2, _ in single_box:
                # if x1 >= x2-1 or y1 >= y2-1: continue
                # Cropping and lifting
                cropped_pc = self.image2pc(depth[bind, x1 : x2, y1 : y2], intrinsic[bind])

                # 3D PointCloud Segmentation
                x = torch.cat([cropped_pc, rgb[bind, x1 : x2, y1 : y2]], dim=-1)
                x = self.center(x)
                orig_shape = x.shape
                x = x.view((-1, orig_shape[-1]))

                h_feats, abs_feats = self.pointnet2(x)
                seg = torch.cat([x, h_feats, torch.repeat_interleave(abs_feats.unsqueeze(0), x.size(0), dim=0)], dim=-1)

                assert x.size(0) == seg.size(0) > 0
                seg = seg.view((*orig_shape[:-1], -1)) # H, W, 9 + i + h + o

                seg = self.get_score(seg).squeeze(-1) # > 0 yes; < 0 no
                A, B = seg.max(), seg.min()
                seg = seg - (A + B) / 2
                if A > B:
                    seg = 2 * seg / (A - B)
                segs.append(seg)
            segments.append(segs)
        return segments
        

def model_fn_decorator(val=False, test=False):
    def model_fn(batch, model):
        rgb = batch['rgb'].cuda()
        depth = batch['depth'].cuda()
        intrinsic = batch['meta'].cuda()
        box = batch['box']
        gt = batch['gt'].cuda().long()
        gt.clamp_(max=79)

        preds = model(rgb, depth, intrinsic, box)

        loss = 0
        for label, single_box, pred in zip(gt, box, preds):
            single_box = single_box.cuda().long()
            for (x1, y1, x2, y2, lbl), pred_seg in zip(single_box, pred):
                # if x1 >= x2-1 or y1 >= y2-1: continue
                crop = label[x1 : x2, y1 : y2]
                try:
                    loss = loss + (torch.sum(1 - pred_seg[crop == lbl]) + torch.sum(1 + pred_seg[crop != lbl])) / ((x2 - x1) * (y2 - y1))
                except:
                    print(crop.size(), lbl, pred_seg.size(), x1, x2, y1, y2)
                    assert False

        return loss, pred
    
    def val_model_fn(batch, model):
        rgb = batch['rgb'].cuda()
        depth = batch['depth'].cuda()
        intrinsic = batch['meta'].cuda()
        box = batch['box']

        preds = model(rgb, depth, intrinsic, box)

        labels = []
        for single_box, pred in zip(box, preds):
            single_box = single_box.cuda().long()
            label = torch.zeros(rgb.shape[1:-1]).cuda()
            conf = torch.zeros(rgb.shape[1:-1]).cuda() # confidence
            mask = torch.ones(rgb.shape[1:-1]).cuda().bool()
            for (x1, y1, x2, y2, lbl), pred_seg in zip(single_box, pred):
                # if x1 >= x2-1 or y1 >= y2-1: continue
                label[x1 : x2, y1 : y2] = lbl * (pred_seg > conf[x1 : x2, y1 : y2])
                conf[x1 : x2, y1 : y2] = torch.maximum(pred_seg, conf[x1 : x2, y1 : y2])
                mask[x1 : x2, y1 : y2] *= (pred_seg <= 0)
            label[mask] = 79
            labels.append(label)
        labels = torch.stack(labels, 0)
                
        return labels


    def test_model_fn(batch, model):
        rgb = batch['rgb'].cuda()
        depth = batch['depth'].cuda()
        intrinsic = batch['meta'].cuda()
        box = batch['box']
        score = batch['score']

        preds = model(rgb, depth, intrinsic, box)

        labels = []
        for single_box, single_score, pred in zip(box, score, preds):
            single_box = single_box.cuda().long()
            single_score = single_score.cuda()
            label = torch.zeros(rgb.shape[1:-1]).cuda()
            conf = torch.zeros(rgb.shape[1:-1]).cuda() # confidence
            mask = torch.ones(rgb.shape[1:-1]).cuda().bool()
            for (x1, y1, x2, y2, lbl), scr, pred_seg in zip(single_box, single_score, pred):
                if x1 >= x2-1 or y1 >= y2-1: continue
                label[x1 : x2, y1 : y2] = lbl * (scr * pred_seg > conf[x1 : x2, y1 : y2])
                conf[x1 : x2, y1 : y2] = torch.maximum(scr * pred_seg, conf[x1 : x2, y1 : y2])
                mask[x1 : x2, y1 : y2] *= (pred_seg <= 0)
            label[mask] = 79
            labels.append(label)
        labels = torch.stack(labels, 0)
                
        return labels

    if test:
        return test_model_fn
    if val:
        return val_model_fn
    return model_fn