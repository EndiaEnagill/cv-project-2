import torch.nn as nn
from torchvision import models

def get_alexnet(num_classes=101, pretrained=True, fine_tune=True):
    """
    获取AlexNet模型
    :param num_classes: 输出类别数
    :param pretrained: 是否使用预训练权重
    :param fine_tune: 是否微调卷积层
    :return: 配置好的AlexNet模型
    """
    model = models.alexnet(pretrained=pretrained)
    
    # 冻结卷积层参数（如果不微调）
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    
    # 修改分类器最后一层
    model.classifier[6] = nn.Linear(4096, num_classes)
    
    # 初始化新层权重
    nn.init.normal_(model.classifier[6].weight, 0, 0.01)
    nn.init.constant_(model.classifier[6].bias, 0)
    
    return model