# -*- coding: utf-8 -*-
"""
@Author: ShouqinGuan
工具函数模块
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import json
import gc
from torch.utils.tensorboard import SummaryWriter

class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    def __init__(self, name, fmt=':f', window_size=100):
        self.name = name
        self.fmt = fmt
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []
        self.window = deque(maxlen=self.window_size)
        self.window_avg = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(val)
        
        # 更新滑动窗口平均值
        self.window.append(val)
        self.window_avg = sum(self.window) / len(self.window)
        
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ExperimentLogger:
    """
    实验日志记录器，支持TensorBoard和本地文件记录
    """
    def __init__(self, log_dir, exp_name=None):
        if exp_name:
            self.log_dir = os.path.join(log_dir, exp_name)
        else:
            self.log_dir = log_dir
            
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.metrics = {}
        self.log_file = os.path.join(self.log_dir, "training_log.txt")
        
        # 初始化日志文件
        with open(self.log_file, "w") as f:
            f.write(f"# 训练日志 - {exp_name}\n")
            f.write("# 格式: step,loss,lr,samples_per_second\n")
    
    def add_scalar(self, tag, value, step):
        """记录标量值到TensorBoard"""
        self.writer.add_scalar(tag, value, step)
        
        # 同时更新内部指标记录
        if tag not in self.metrics:
            self.metrics[tag] = AverageMeter(tag)
        self.metrics[tag].update(value)
        
    def log_metrics(self, step, **kwargs):
        """记录多个指标"""
        log_str = f"Step: {step}"
        for key, value in kwargs.items():
            self.add_scalar(key, value, step)
            log_str += f", {key}: {value:.6f}"
            
        # 写入日志文件
        with open(self.log_file, "a") as f:
            metrics_str = ",".join([f"{v:.6f}" for k, v in kwargs.items()])
            f.write(f"{step},{metrics_str}\n")
            
        return log_str
    
    def plot_loss_curve(self, save_path=None):
        """Plot the loss curve with improved styling"""
        if 'loss' not in self.metrics:
            print("No loss data available to plot")
            return
            
        # Set plot style
        plt.style.use('ggplot')
        plt.figure(figsize=(12, 7))
        
        # Plot original loss with better styling
        steps = list(range(1, len(self.metrics['loss'].history) + 1))
        plt.plot(steps, self.metrics['loss'].history, 'b-', alpha=0.3, linewidth=1, label='Raw Loss')
        
        # Calculate and plot moving average with improved styling
        window_size = min(100, len(steps))
        if window_size > 0:
            smoothed_losses = []
            for i in range(len(steps)):
                if i < window_size - 1:
                    smoothed_losses.append(np.mean(self.metrics['loss'].history[:i+1]))
                else:
                    smoothed_losses.append(np.mean(self.metrics['loss'].history[i-window_size+1:i+1]))
            plt.plot(steps, smoothed_losses, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        
        # Add min loss marker
        if smoothed_losses:
            min_loss_idx = np.argmin(smoothed_losses)
            min_loss = smoothed_losses[min_loss_idx]
            plt.scatter(steps[min_loss_idx], min_loss, color='green', s=100, zorder=5)
            plt.annotate(f'Min: {min_loss:.4f}', 
                        (steps[min_loss_idx], min_loss),
                        xytext=(10, -30), 
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='green'))
        
        # Improve labels and styling
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', frameon=True, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add experiment info if available
        if hasattr(self, 'log_dir') and os.path.basename(self.log_dir):
            plt.figtext(0.02, 0.02, f"Experiment: {os.path.basename(self.log_dir)}", 
                       fontsize=8, style='italic')
        
        # Improve layout
        plt.tight_layout()
        
        # Save with higher DPI for better quality
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'loss_curve.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        return save_path
    
    def save_metrics(self):
        """保存所有指标到JSON文件"""
        metrics_data = {}
        for name, meter in self.metrics.items():
            metrics_data[name] = {
                'history': meter.history,
                'average': meter.avg,
                'window_average': meter.window_avg
            }
            
        with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=4)
    
    def close(self):
        """关闭日志记录器"""
        self.save_metrics()
        self.plot_loss_curve()
        self.writer.close()

def free_memory():
    """释放内存"""
    gc.collect()
    torch.cuda.empty_cache()

# 在utils.py文件末尾添加以下函数

def print_model_parameters(model, min_param_count=10000, max_depth=2):
    """
    美观精简地打印模型参数信息
    
    参数:
        model: 要分析的模型
        min_param_count: 只显示参数数量大于此值的模块 (默认10,000)
        max_depth: 最大显示深度 (默认2)
    """
    print("\n📊 Model Parameters Summary")
    print("=" * 60)
    
    total_params = 0
    module_info = {}
    layer_params = 0
    layer_submodules = set()
    layer_count = 0  # 新增：记录layer层数
    
    # 收集模块信息
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # 检查是否是layer模块
        parts = name.split('.')
        if len(parts) > 1 and parts[0] == 'layers':
            # 合并所有layers的参数
            layer_params += param_count
            if len(parts) > max_depth:
                layer_submodules.add('.'.join(parts[1:2] + parts[max_depth:max_depth+1]))  # 包含layer编号
            layer_count = max(layer_count, int(parts[1])) + 1  # 统计layer数量
            continue
        
        # 提取模块路径
        module_path = '.'.join(parts[:max_depth])
        
        # 更新模块信息
        if module_path not in module_info:
            module_info[module_path] = {
                'count': 0,
                'submodules': set(),
                'params': []
            }
        
        module_info[module_path]['count'] += param_count
        if len(parts) > max_depth:
            module_info[module_path]['submodules'].add('.'.join(parts[max_depth:max_depth+1]))
        module_info[module_path]['params'].append((name, param.shape, param_count))
    
    # 如果有layer参数，添加到模块信息
    if layer_params > 0:
        # 计算实际子模块总数（层数×每层子模块数）
        unique_submodules_per_layer = len(set(s.split('.')[1] for s in layer_submodules)) if layer_submodules else 0
        total_submodules = layer_count * unique_submodules_per_layer if unique_submodules_per_layer > 0 else len(layer_submodules)
        
        # 确保tok_embeddings.weight排在前面
        if 'tok_embeddings.weight' in module_info:
            module_info = {'tok_embeddings.weight': module_info['tok_embeddings.weight'], 
                          'layers': {
                              'count': layer_params,
                              'submodules': layer_submodules,
                              'params': [],
                              'total_submodules': total_submodules
                          }}
            # 添加其他模块
            for k, v in module_info.items():
                if k not in ['tok_embeddings.weight', 'layers']:
                    module_info[k] = v
        else:
            module_info['layers'] = {
                'count': layer_params,
                'submodules': layer_submodules,
                'params': [],
                'total_submodules': total_submodules
            }
    
    # 打印主要模块
    print(f"{'Module':<30} | {'Params':>15} | {'%':>6}")
    print("-" * 60)
    
    # 按特定顺序排序（确保tok_embeddings.weight第一）
    custom_order = ['tok_embeddings.weight', 'layers']
    remaining_modules = [m for m in module_info.keys() if m not in custom_order]
    sorted_modules = [(k, module_info[k]) for k in custom_order if k in module_info] + \
                    sorted([(m, module_info[m]) for m in remaining_modules], 
                          key=lambda x: -x[1]['count'])
    
    for module, info in sorted_modules:
        if info['count'] < min_param_count:
            continue
            
        param_percent = info['count'] / total_params * 100
        submodule_count = info.get('total_submodules', len(info['submodules']))
        
        # 主模块行
        print(f"{module:<30} | {info['count']:>15,} | {param_percent:>5.1f}%")
        
        # 如果有子模块且参数较多，显示子模块总数
        if submodule_count > 0 and info['count'] > min_param_count * 10:
            print(f"  └ [merged {submodule_count} submodules]")
    
    # 打印汇总信息
    print("=" * 60)
    print(f"🔹 Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 计算并打印未显示的小参数
    shown_params = sum(info['count'] for _, info in sorted_modules if info['count'] >= min_param_count)
    if shown_params < total_params:
        small_params = total_params - shown_params
        print(f"🔸 Small params (<{min_param_count:,}): {small_params:,} ({small_params/total_params:.1%})")
    
    print("=" * 60)
    
    return total_params / 1e6  # 返回模型大小（百万参数）

def save_checkpoint(epoch, step, model, optimizer, scheduler, total_loss, processed_samples, path, rank=0, logger=None, is_best=False):
    """保存训练状态到检查点"""
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'total_loss': total_loss,
        'processed_samples': processed_samples
    }
    
    torch.save(checkpoint, path)
    
    # 如果是最佳模型，保存一个副本
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "checkpoint-best.pt")
        torch.save(checkpoint, best_path)
    
    if rank == 0:
        print(f"Checkpoint saved at epoch {epoch}, step {step} to {path}")
        if logger:
            logger.add_scalar('checkpoint_saved', 1.0, step)