import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from config.config import Config
from data.dataset import MoNuSegDataset, get_transforms
from models.transunet import TransUNet
from utils.metrics import calculate_dice_score, calculate_iou

def visualize_results(image, pred_mask, true_mask, save_path):
    """
    可视化分割结果
    :param image: 原始图像 (C, H, W)
    :param pred_mask: 预测掩码 (H, W)
    :param true_mask: 真实掩码 (H, W)
    :param save_path: 保存路径
    """
    # 转换为numpy数组
    image = image.cpu().numpy().transpose(1, 2, 0)
    pred_mask = pred_mask.cpu().numpy()
    true_mask = true_mask.cpu().numpy()
    
    # 归一化图像到0-1范围
    image = (image - image.min()) / (image.max() - image.min())
    
    # 创建图像
    plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(131)
    plt.imshow(image)
    plt.title('原始图像')
    plt.axis('off')
    
    # 显示预测掩码
    plt.subplot(132)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('预测分割')
    plt.axis('off')
    
    # 显示真实掩码
    plt.subplot(133)
    plt.imshow(true_mask, cmap='gray')
    plt.title('真实分割')
    plt.axis('off')
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()

def evaluate():
    # 设置设备
    device = torch.device(Config.DEVICE)
    
    # 创建保存可视化结果的目录
    vis_dir = os.path.join(Config.SAVE_DIR, 'visualization_results')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 创建测试数据集和数据加载器
    test_dataset = MoNuSegDataset(
        root_dir=Config.TEST_DATASET_PATH,
        transform=get_transforms(mode='test', image_size=Config.IMAGE_SIZE),
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    # 创建模型
    model = TransUNet(
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        hidden_size=Config.HIDDEN_SIZE,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS
    ).to(device)
    
    # 加载最佳模型
    model_path = os.path.join(Config.SAVE_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 评估
    test_dice = 0
    test_iou = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            
            dice_score = calculate_dice_score(pred_masks, masks)
            iou_score = calculate_iou(pred_masks, masks)
            
            test_dice += dice_score
            test_iou += iou_score
            
            # 保存每个样本的可视化结果
            for i in range(images.size(0)):
                save_path = os.path.join(vis_dir, f'sample_{batch_idx}_{i}.png')
                visualize_results(
                    images[i],
                    pred_masks[i, 0],  # 取第一个通道
                    masks[i, 0],       # 取第一个通道
                    save_path
                )
            
            pbar.set_postfix({
                'dice': dice_score,
                'iou': iou_score
            })
    
    # 计算平均指标
    avg_test_dice = test_dice / len(test_loader)
    avg_test_iou = test_iou / len(test_loader)
    
    print("\n最终评估结果:")
    print(f"测试集 Dice 分数: {avg_test_dice:.4f}")
    print(f"测试集 IoU 分数: {avg_test_iou:.4f}")
    print(f"\n可视化结果已保存到: {vis_dir}")

if __name__ == '__main__':
    evaluate() 