import torch
import numpy as np
from transforms3d.euler import euler2mat

def get_shape_agnostic_loss(pred, gt, type='Geodesic'):
    if type == 'Geodesic':
        rot_loss = torch.arccos(0.5 * (torch.trace(gt @ pred.T) - 1))
    elif type == 'MAE':
        # rot_loss = torch.norm(pred - gt, p='fro')
        rot_loss = torch.sum(torch.abs(pred - gt))
    else:
        assert type == 'Euclidean'
        rot_loss = torch.norm(pred - gt, p=2)
    return rot_loss

def eval_rdiff_with_sym_axis(pred_rotation, gt_rotation, sym_axis, type):
    """
    Compute rotation error (unit: degree) based on symmetry axis.
    """
    if sym_axis == "x":
        x1, x2 = pred_rotation[:, 0], gt_rotation[:, 0]
        diff = torch.sum(x1 * x2)
        loss = get_shape_agnostic_loss(x1, x2, type=type)
    elif sym_axis == "y":
        y1, y2 = pred_rotation[:, 1], gt_rotation[:, 1]
        diff = torch.sum(y1 * y2)
        loss = get_shape_agnostic_loss(y1, y2, type=type)
    elif sym_axis == "z":
        z1, z2 = pred_rotation[:, 2], gt_rotation[:, 2]
        diff = torch.sum(z1 * z2)
        loss = get_shape_agnostic_loss(z1, z2, type=type)
    else:  # sym_axis == "", i.e. no symmetry axis
        mat_diff = torch.matmul(pred_rotation, gt_rotation.T)
        diff_tr = mat_diff.trace()
        diff = (diff_tr - 1) / 2.0
        loss = get_shape_agnostic_loss(pred_rotation, gt_rotation, type=type)
    diff = torch.clip(diff, min=-1.0, max=1.0)
    return loss, torch.arccos(diff).detach() / np.pi * 180  # degree

def eval_rdiff(pred_rotation, gt_rotation, geometric_symmetry, type):
    """
    Compute rotation error (unit: degree) based on geometric symmetry.
    """
    syms = geometric_symmetry.split("|")
    sym_axis = ""
    sym_N = np.array([1, 1, 1])  # x, y, z
    for sym in syms:
        if sym.find("inf") != -1:
            sym_axis += sym[0]
        elif sym != "no":
            idx = ord(sym[0]) - ord('x')
            value = int(sym[1:])
            sym_N[idx] = value
    if len(sym_axis) >= 2:
        return 0.0
    
    assert sym_N.min() >= 1

    gt_rotations = []
    for xi in range(sym_N[0]):
        for yi in range(sym_N[1]):
            for zi in range(sym_N[2]):
                R = euler2mat(
                    2 * np.pi / sym_N[0] * xi,
                    2 * np.pi / sym_N[1] * yi,
                    2 * np.pi / sym_N[2] * zi,
                )
                gt_rotations.append(gt_rotation @ torch.tensor(R, dtype=torch.float32).cuda())

    r_diffs = []
    r_losses = []
    for gt_rotation in gt_rotations:
        r_loss, r_diff = eval_rdiff_with_sym_axis(pred_rotation, gt_rotation, sym_axis, type=type)
        r_losses.append(r_loss)
        r_diffs.append(r_diff)
    
    r_losses = torch.stack(r_losses, 0)
    r_diffs = torch.stack(r_diffs, 0)
    return r_losses.min(), r_diffs.min()

def get_rotation_loss(preds, gts, symmetry, type='Geodesic'):
    rot_loss = []
    rot_diff = []
    for pred, gt, sym in zip(preds, gts, symmetry):
        r_loss, r_diff = eval_rdiff(pred, gt, sym, type)
        rot_loss.append(r_loss)
        rot_diff.append(r_diff)
    
    rot_loss = torch.stack(rot_loss, 0)
    rot_loss = rot_loss / 10
    rot_diff = torch.stack(rot_diff, 0)

    return rot_loss, rot_diff

def get_translation_loss(pred, gt):
    return torch.sum(torch.abs(pred - gt), dim=1).squeeze() * 2, torch.norm(pred - gt, p=2, dim=1).squeeze() * 100

def get_loss(pred, gt, sym):
    rotation_pred, translation_pred = pred
    # rotation_pred = pred[:, :3, :3]
    # translation_pred = pred[:, :3, 3]
    rotation_gt = gt[:, :3, :3]
    translation_gt = gt[:, :3, 3]

    rotation_loss, rotation_diff = get_rotation_loss(rotation_pred, rotation_gt, sym, type='MAE')
    translation_loss, translation_diff = get_translation_loss(translation_pred, translation_gt)
    loss = rotation_loss + translation_loss
    # print(rotation_diff,'\n', translation_diff, '\n')
    Trans_shots = torch.sum(translation_diff < 1)
    Relation_shots = torch.sum(rotation_diff < 5)
    Shots = torch.sum(rotation_diff[translation_diff < 1] < 5)
    Total = loss.shape[0]
    loss = loss.sum()
    diff = rotation_diff.sum()

    shots = Shots, Total, Trans_shots, Relation_shots

    return loss, shots, diff, rotation_loss.sum(), translation_loss.sum()