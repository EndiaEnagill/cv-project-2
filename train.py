import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.alexnet import get_alexnet
from datasets.caltech101 import get_caltech101_loaders
from utils.logger import TrainingLogger
import yaml
import os
from tqdm import tqdm
from datetime import datetime

def evaluate(model, dataloader, criterion, device):
    """评估模型在验证集/测试集上的表现"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    
    return loss, accuracy

def train(config_path):
    """主训练函数"""
    # 初始化设备
    with open(os.path.join(config_path, 'alexnet.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化日志记录器
    logger = TrainingLogger(os.path.join(config_path, 'logs'))
    
    try:
        # 加载数据
        train_loader, val_loader, class_names = get_caltech101_loaders(
            config['data_dir'], 
            batch_size=config['batch_size']
        )
        print(f"Dataset loaded with {len(class_names)} classes")
        print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        
        # 初始化模型
        model = get_alexnet(
            num_classes=len(class_names),
            pretrained=config['pretrained'],
            fine_tune=config['fine_tune']
        ).to(device)
        print("Model initialized:")
        print(f"Using pretrained weights: {config['pretrained']}")
        print(f"Fine-tuning enabled: {config['fine_tune']}")
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        
        # 设置不同层的学习率
        params_to_optimize = []
        if config['fine_tune']:
            # 卷积层使用较小学习率
            params_to_optimize.append({
                'params': model.features.parameters(),
                'lr': config['lr'] * 0.1
            })
        else:
            # 冻结卷积层参数
            for param in model.features.parameters():
                param.requires_grad = False
        
        # 分类层使用正常学习率
        params_to_optimize.append({
            'params': model.classifier.parameters(),
            'lr': config['lr']
        })
        
        optimizer = optim.Adam(params_to_optimize, weight_decay=config['weight_decay'])
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.1, 
            patience=5, 
            verbose=True
        )
        
        print("\nStarting training...")
        best_acc = 0.0
        best_epoch = 0
        
        for epoch in range(config['epochs']):
            model.train()
            running_loss = 0.0
            
            # 训练阶段
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}') as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    pbar.set_postfix({'loss': loss.item()})
            
            # 验证阶段
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            # 计算epoch训练损失
            epoch_loss = running_loss / len(train_loader.dataset)
            
            # 记录所有指标
            logger.log_metrics(epoch+1, epoch_loss, val_loss, val_acc)
            
            # 更新学习率
            scheduler.step(val_acc)
            
            # 打印信息
            print(f'Epoch {epoch+1}/{config["epochs"]} | '
                  f'Train Loss: {epoch_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {val_acc*100:.2f}% | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), f'{config_path}/best_alexnet.pth')
                print(f"New best model saved with accuracy {val_acc*100:.2f}%")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
    finally:
        # 确保日志被正确关闭
        logger.close()
        print(f"Training logs saved to {logger.log_dir}")
        
        if 'best_epoch' in locals():
            print(f"\nBest model at epoch {best_epoch} with accuracy {best_acc*100:.2f}%")
    
    return model

if __name__ == '__main__':
    # 加载配置文件
    for i in range(30, 33):
        config_path = f'model_data/{i+1}'
        with open(os.path.join(config_path, 'alexnet.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    
        print("Configuration:")
        for key, value in config.items():
            print(f"{key:15}: {value}")
        print()
    
        # 开始训练
        trained_model = train(config_path)