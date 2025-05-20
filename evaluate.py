import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.alexnet import get_alexnet
from datasets.caltech101 import get_caltech101_loaders
import argparse
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def evaluate_model(model, dataloader, criterion, device, class_names):
    """评估模型并返回详细结果"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    loss = running_loss / len(dataloader.dataset)
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    # 生成分类报告和混淆矩阵
    clf_report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'classification_report': clf_report,
        'confusion_matrix': cm
    }

def save_evaluation_results(results, class_names, save_dir):
    """保存评估结果和可视化（不使用seaborn）"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存文本结果
    with open(os.path.join(save_dir, f'evaluation_results_{timestamp}.txt'), 'w') as f:
        f.write(f"Test Loss: {results['loss']:.4f}\n")
        f.write(f"Test Accuracy: {results['accuracy']*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(
            results['labels'], 
            results['predictions'], 
            target_names=class_names
        ))
    
    # 2. 保存分类报告为CSV
    report_df = pd.DataFrame(results['classification_report']).transpose()
    report_df.to_csv(os.path.join(save_dir, f'classification_report_{timestamp}.csv'))
    
    # 3. 使用matplotlib绘制混淆矩阵（替代seaborn）
    plt.figure(figsize=(15, 12))
    cm = results['confusion_matrix']
    
    # 绘制矩阵
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # 添加标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{timestamp}.png'), dpi=300)
    plt.close()
    
    # 4. 保存预测错误样本
    error_indices = np.where(np.array(results['predictions']) != np.array(results['labels']))[0]
    if len(error_indices) > 0:
        error_samples = pd.DataFrame({
            'True Label': [class_names[results['labels'][i]] for i in error_indices],
            'Predicted Label': [class_names[results['predictions'][i]] for i in error_indices]
        })
        error_samples.to_csv(os.path.join(save_dir, f'misclassified_samples_{timestamp}.csv'), index=False)

def test(model_path, data_dir, batch_size):
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    _, test_loader, class_names = get_caltech101_loaders(data_dir, batch_size=batch_size)
    print(f"Loaded test set with {len(test_loader.dataset)} samples")
    
    # 加载模型
    model = get_alexnet(num_classes=len(class_names))
    model.load_state_dict(torch.load(os.path.join(model_path, 'best_alexnet.pth')))
    model = model.to(device)
    print(f"Model loaded from {model_path}")
    
    # 评估
    criterion = nn.CrossEntropyLoss()
    results = evaluate_model(model, test_loader, criterion, device, class_names)
    
    # 保存结果
    save_dir = os.path.join(model_path, "logs/evaluate")
    save_evaluation_results(results, class_names, save_dir)
    
    # 打印摘要
    print("\nEvaluation Summary:")
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"\nResults saved to {save_dir}")
    
    # 打印每个类别的准确率
    print("\nClass-wise Accuracy:")
    class_acc = results['classification_report']
    for cls in class_names:
        print(f"{cls:20}: {class_acc[cls]['precision']:.2f}")

# model_path 需要是模型保存的专用文件夹
if __name__ == '__main__':
    config_path = 'model_data/33'
    data_dir = './data/caltech101'
    batch_size = 32
    
    # 开始测试
    test(config_path, data_dir, batch_size)