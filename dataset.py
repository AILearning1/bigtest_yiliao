import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET

class MoNuSegDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.mask_dir = os.path.join(os.path.dirname(root_dir), 'Annotations')
        all_images = [f for f in os.listdir(root_dir) if f.endswith('.tif')]
        if mode == 'train':
            self.image_files = []
            for img in all_images:
                xml_name = img.replace('.tif', '.xml')
                xml_path = os.path.join(self.mask_dir, xml_name)
                if os.path.exists(xml_path):
                    self.image_files.append(img)
        else:
            self.image_files = all_images
    
    def __len__(self):
        return len(self.image_files)
    
    def _create_mask_from_xml(self, xml_path, image_shape):
        """从XML文件创建掩码"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 创建空白掩码
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        
        # 解析XML中的多边形标注
        for region in root.findall('.//Region'):
            points = []
            for vertex in region.findall('.//Vertex'):
                x = float(vertex.get('X'))
                y = float(vertex.get('Y'))
                points.append([x, y])
            
            if points:
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)
        
        return mask
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mode == 'train':
            xml_name = img_name.replace('.tif', '.xml')
            xml_path = os.path.join(self.mask_dir, xml_name)
            if not os.path.exists(xml_path):
                raise ValueError(f"无法找到标注文件: {xml_path}")
            mask = self._create_mask_from_xml(xml_path, image.shape)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # 添加通道维度
            mask = mask.unsqueeze(0)  # 将 [H, W] 转换为 [1, H, W]
        
        return image, mask

def get_transforms(mode='train', image_size=128):
    if mode == 'train':
        return A.Compose([
            A.RandomCrop(height=image_size, width=image_size, p=0.5),  # 随机裁剪
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # 添加高斯噪声
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]) 