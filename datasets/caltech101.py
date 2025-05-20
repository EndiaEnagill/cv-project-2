import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

def get_caltech101_loaders(data_dir, batch_size=32):
    """
    获取标准划分的Caltech-101数据加载器
    :param data_dir: 数据集根目录
    :param batch_size: 批量大小
    :return: (train_loader, test_loader, class_names)
    """
    # 数据增强和归一化
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # 加载完整数据集
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # 标准划分：每类前30张训练
    train_indices = []
    test_indices = []
    class_counts = {}
    
    for idx, (img_path, _) in enumerate(full_dataset.samples):
        class_name = os.path.basename(os.path.dirname(img_path))
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        if class_counts[class_name] <= 30:
            train_indices.append(idx)
        else:
            test_indices.append(idx)
    
    # 创建子集
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, full_dataset.classes