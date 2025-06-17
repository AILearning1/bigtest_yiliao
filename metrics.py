import torch

def calculate_dice_score(pred, target, smooth=1e-5):
    """
    计算Dice系数
    pred: 预测掩码 (B, 1, H, W)
    target: 目标掩码 (B, 1, H, W)
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()

def calculate_iou(pred, target, smooth=1e-5):
    """
    计算IoU (Intersection over Union)
    pred: 预测掩码 (B, 1, H, W)
    target: 目标掩码 (B, 1, H, W)
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item() 