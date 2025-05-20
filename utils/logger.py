import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        # 创建日志目录
        self.log_dir = log_dir
        self.tb_dir = os.path.join(log_dir, "tensorboard")
        self.csv_dir = os.path.join(log_dir, "training_logs")
        self.plot_dir = os.path.join(log_dir, "plots")
        
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # 初始化TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tb_writer = SummaryWriter(os.path.join(self.tb_dir, timestamp))
        
        # 初始化CSV日志
        self.csv_path = os.path.join(self.csv_dir, f"{timestamp}.csv")
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc'])
        
        # 存储数据用于绘图
        self.epoch_data = []
    
    def log_metrics(self, epoch, train_loss, val_loss, val_acc):
        # 记录到TensorBoard
        self.tb_writer.add_scalar('Loss/train', train_loss, epoch)
        self.tb_writer.add_scalar('Loss/val', val_loss, epoch)
        self.tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # 记录到CSV
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_acc])
        
        # 存储数据
        self.epoch_data.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
    
    def save_plots(self):
        # 准备数据
        epochs = [d['epoch'] for d in self.epoch_data]
        train_losses = [d['train_loss'] for d in self.epoch_data]
        val_losses = [d['val_loss'] for d in self.epoch_data]
        val_accs = [d['val_acc'] for d in self.epoch_data]
        
        # 保存loss曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        loss_plot_path = os.path.join(self.plot_dir, 'loss_curve.png')
        plt.savefig(loss_plot_path)
        plt.close()
        
        # 保存accuracy曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, val_accs, label='Validation Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        acc_plot_path = os.path.join(self.plot_dir, 'accuracy_curve.png')
        plt.savefig(acc_plot_path)
        plt.close()
    
    def close(self):
        self.tb_writer.close()
        self.save_plots()