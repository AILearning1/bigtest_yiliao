class Config:
    # 数据集配置
    TRAIN_DATASET_PATH = r"D:\shiyan\Azhihui\bigtest\MoNuSeg 2018 Training Data\MoNuSeg 2018 Training Data\Tissue Images"
    TEST_DATASET_PATH = r"D:\shiyan\Azhihui\bigtest\MoNuSegTestData\MoNuSegTestData"
    IMAGE_SIZE = 128  # 减小图像尺寸
    BATCH_SIZE = 2  # 减小批量大小
    NUM_WORKERS = 2
    
    # 模型配置
    IN_CHANNELS = 3
    NUM_CLASSES = 1
    HIDDEN_SIZE = 384  # 减小隐藏层大小
    NUM_HEADS = 6  # 减少注意力头数
    NUM_LAYERS = 6  # 减少Transformer层数
    DROPOUT = 0.1
    
    # 训练配置
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    DEVICE = "cpu"
    
    # 数据增强配置
    AUGMENTATION = True
    
    # 保存配置
    SAVE_DIR = "checkpoints"
    LOG_DIR = "logs" 