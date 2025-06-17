import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from config.config import Config
from data.dataset import MoNuSegDataset, get_transforms
from models.transunet import TransUNet
from utils.metrics import calculate_dice_score, calculate_iou

def plot_loss_curves(train_losses, val_losses, save_path):
    """绘制损失函数曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def train():
    # 创建保存目录
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # 设置设备
    device = torch.device(Config.DEVICE)
    
    # 创建数据集和数据加载器
    train_dataset = MoNuSegDataset(
        root_dir=Config.TRAIN_DATASET_PATH,
        transform=get_transforms(mode='train', image_size=Config.IMAGE_SIZE),
        mode='train'
    )
    
    test_dataset = MoNuSegDataset(
        root_dir=Config.TEST_DATASET_PATH,
        transform=get_transforms(mode='test', image_size=Config.IMAGE_SIZE),
        mode='test'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
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
    
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(Config.LOG_DIR)
    
    # 用于记录损失值
    train_losses = []
    val_losses = []
    
    # 训练循环
    best_dice = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算指标
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            dice_score = calculate_dice_score(pred_masks, masks)
            iou_score = calculate_iou(pred_masks, masks)
            
            # 更新进度条
            train_loss += loss.item()
            train_dice += dice_score
            train_iou += iou_score
            
            pbar.set_postfix({
                'loss': loss.item(),
                'dice': dice_score,
                'iou': iou_score
            })
        
        # 计算训练集平均指标
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]')
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                dice_score = calculate_dice_score(pred_masks, masks)
                iou_score = calculate_iou(pred_masks, masks)
                
                val_loss += loss.item()
                val_dice += dice_score
                val_iou += iou_score
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'dice': dice_score,
                    'iou': iou_score
                })
        
        # 计算验证集平均指标
        avg_val_loss = val_loss / len(test_loader)
        avg_val_dice = val_dice / len(test_loader)
        avg_val_iou = val_iou / len(test_loader)
        val_losses.append(avg_val_loss)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Dice/train', avg_train_dice, epoch)
        writer.add_scalar('IoU/train', avg_train_iou, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Dice/val', avg_val_dice, epoch)
        writer.add_scalar('IoU/val', avg_val_iou, epoch)
        
        # 保存最佳模型
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, 'best_model.pth'))
        
        print(f'Epoch {epoch+1}/{Config.NUM_EPOCHS}:')
        print(f'Train - Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}, IoU: {avg_train_iou:.4f}')
        print(f'Val - Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}, IoU: {avg_val_iou:.4f}')
    
    # 绘制损失曲线
    plot_loss_curves(train_losses, val_losses, os.path.join(Config.LOG_DIR, 'loss_curves.png'))
    writer.close()

if __name__ == '__main__':
    train() 