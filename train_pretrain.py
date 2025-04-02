# -*- coding: utf-8 -*-
"""
author: shouqinguan
优化的多卡训练脚本
torchrun --nproc_per_node=num_gpus train_ddp.py
"""
import os
import gc
import math
import time
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
from transformers import get_linear_schedule_with_warmup
from model import SLM, LMConfig
from utils import ExperimentLogger, free_memory, print_model_parameters, save_checkpoint

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免tokenizer并行警告

def get_lr(current_step, total_steps, lr):
    """余弦学习率调度"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def setup_ddp():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
        # 初始化进程组
        init_process_group(backend='nccl')
        
        # 获取分布式训练参数
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # 设置当前设备
        torch.cuda.set_device(local_rank)
        
        # 设置随机种子以确保各进程一致性
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        return True, local_rank, rank, world_size
    return False, 0, 0, 1

def is_master_process(rank):
    """检查是否为主进程"""
    return rank == 0

def create_optimizer(model, lr=1e-4, weight_decay=0.01, beta1=0.9, beta2=0.95):
    """创建优化器，使用参数分组以应用不同的权重衰减"""
    # 将参数分为需要和不需要权重衰减的两组
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(nd in name.lower() for nd in ['bias', 'norm', 'layernorm']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer_grouped_params = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    return optim.AdamW(optimizer_grouped_params, lr=lr, betas=(beta1, beta2))

def load_checkpoint(path, model, optimizer, scheduler, device, rank):
    """从检查点恢复训练状态"""
    if not os.path.exists(path):
        if rank == 0:
            print(f"Checkpoint {path} not found. Starting from scratch.")
        return 0, 0, 0.0, 0
    
    # 确保所有进程等待主进程加载检查点
    if dist.is_initialized():
        dist.barrier()
    
    checkpoint = torch.load(path, map_location=device)
    
    # 加载模型状态
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器和调度器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 恢复其他训练状态
    start_epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('step', 0)
    total_loss = checkpoint.get('total_loss', 0.0)
    processed_samples = checkpoint.get('processed_samples', 0)
    
    if rank == 0:
        print(f"Resuming training from checkpoint {path}")
        print(f"Resuming from epoch {start_epoch}, step {global_step}")
    
    return start_epoch, global_step, total_loss, processed_samples

class PretrainDataset(Dataset):
    """预训练数据集"""
    def __init__(self, dataset, max_length=2048):
        self.dataset = dataset
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 处理数据
        input_ids = np.array(item['input_ids'], dtype=np.int64)
        attention_mask = np.array(item['attention_mask'], dtype=np.int64)
        
        # 截断到最大长度
        if len(input_ids) > self.max_length:
            start_idx = np.random.randint(0, len(input_ids) - self.max_length)
            input_ids = input_ids[start_idx:start_idx + self.max_length]
            attention_mask = attention_mask[start_idx:start_idx + self.max_length]
        elif len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = np.pad(input_ids, (0, pad_len), 'constant')
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant')
        
        # 准备输入和目标
        return {
            'input_ids': torch.from_numpy(input_ids[:-1]),
            'labels': torch.from_numpy(input_ids[1:]),
            'attention_mask': torch.from_numpy(attention_mask[1:])
        }

def train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    device,
    epochs,
    start_epoch=0,
    global_step=0,
    total_loss=0.0,
    processed_samples=0,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    checkpoint_dir="checkpoints",
    checkpoint_interval=1000,
    log_interval=100,
    rank=0,
    world_size=1,
    logger=None,
    best_loss=float('inf')
):
    """训练循环"""
    model.train()
    is_master = rank == 0
    
    # 创建检查点目录
    if is_master:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练循环
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_samples = 0
        
        # 设置数据加载器的epoch，确保分布式采样器正确洗牌
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            step_start_time = time.time()
            
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 计算损失
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # 更新参数
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 更新统计信息
                global_step += 1
                
                # 在多卡训练中收集所有进程的损失
                if dist.is_initialized():
                    # 将损失收集到所有进程
                    loss_tensor = torch.tensor([loss.item() * gradient_accumulation_steps], device=device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    loss_value = loss_tensor.item() / world_size
                else:
                    loss_value = loss.item() * gradient_accumulation_steps
                
                total_loss += loss_value
                epoch_loss += loss_value
                
                batch_samples = input_ids.size(0)
                processed_samples += batch_samples * world_size  # 考虑所有GPU的样本
                epoch_samples += batch_samples * world_size
                
                # 计算学习率和每秒处理的样本数
                lr = scheduler.get_last_lr()[0]
                step_time = time.time() - step_start_time
                samples_per_second = (batch_samples * world_size) / step_time
                
                # 记录日志
                if is_master:
                    if logger:
                        logger.log_metrics(
                            global_step,
                            loss=loss_value,
                            lr=lr,
                            samples_per_second=samples_per_second
                        )
                    
                    if global_step % log_interval == 0:
                        # 计算剩余时间和步数
                        steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
                        total_steps_all_epochs = steps_per_epoch * epochs
                        remaining_steps = total_steps_all_epochs - global_step
                        
                        # 计算平均每步时间（使用最近的步骤时间）
                        avg_step_time = step_time * gradient_accumulation_steps  # 考虑梯度累积
                        
                        # 估计剩余时间
                        remaining_time = remaining_steps * avg_step_time
                        remaining_hours = int(remaining_time // 3600)
                        remaining_minutes = int((remaining_time % 3600) // 60)
                        
                        # 格式化输出
                        print(f"Epoch [{epoch}/{epochs-1}], Step {global_step}, Loss: {loss_value:.4f}, "
                              f"LR: {lr:.8f}, Samples/sec: {samples_per_second:.2f}, "
                              f"Processed: {processed_samples}, "
                              f"Remaining: {remaining_steps}/{total_steps_all_epochs} steps "
                              f"({remaining_hours}h {remaining_minutes}m)")
                
                # 保存检查点
                if is_master and global_step % checkpoint_interval == 0:
                    # 保存最新检查点
                    latest_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-latest.pt")
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}.pt")
                    
                    # 检查是否是最佳模型
                    is_best = epoch_loss / (step + 1) < best_loss
                    if is_best:
                        best_loss = epoch_loss / (step + 1)
                    
                    save_checkpoint(
                        epoch, global_step, model, optimizer, scheduler,
                        total_loss, processed_samples, checkpoint_path, rank,
                        logger, is_best
                    )
                    
                    # 同时保存为latest
                    torch.save(torch.load(checkpoint_path), latest_checkpoint_path)
                    
                    # 绘制损失曲线
                    if logger:
                        logger.plot_loss_curve()
                    
                    # 释放内存
                    free_memory()
        
        # 计算epoch统计信息
        epoch_time = time.time() - epoch_start_time
        epoch_avg_loss = epoch_loss / (step + 1)
        
        if is_master:
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s, Avg Loss: {epoch_avg_loss:.4f}")
            
            # 每个epoch结束保存一次检查点
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch}.pt")
            save_checkpoint(
                epoch, global_step, model, optimizer, scheduler,
                total_loss, processed_samples, checkpoint_path, rank,
                logger
            )
            
            # 更新最佳损失
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                best_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint-best.pt")
                torch.save(torch.load(checkpoint_path), best_checkpoint_path)
                print(f"新的最佳模型已保存! 平均损失: {best_loss:.4f}")
    
    # 训练结束，保存最终模型
    if is_master:
        final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint-final.pt")
        save_checkpoint(
            epochs, global_step, model, optimizer, scheduler,
            total_loss, processed_samples, final_checkpoint_path, rank,
            logger
        )
        
        # 保存模型权重（不包含优化器状态）
        if isinstance(model, DDP):
            model.module.save_pretrained(os.path.join(checkpoint_dir, "final_model"))
        else:
            model.save_pretrained(os.path.join(checkpoint_dir, "final_model"))
        
        # 绘制最终损失曲线
        if logger:
            logger.plot_loss_curve()
    
    return total_loss, processed_samples

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="分布式训练")
    parser.add_argument("--num_gpus", type=int, default=None, help="要使用的GPU数量，默认使用所有可用GPU")
    parser.add_argument("--master_port", type=str, default="29500", help="主进程端口")
    parser.add_argument("--config", type=str, default=None, help="训练配置文件路径")
    parser.add_argument("--resume", action="store_true", help="是否从检查点恢复训练")
    parser.add_argument("--exp_name", type=str, default=None, help="实验名称")
    return parser.parse_args()

def validate_config(config):
    """验证配置文件的有效性"""
    required_fields = ['dim', 'n_layers', 'n_heads', 'vocab_size']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"配置文件缺少必要字段: {field}")
    
    if config['dim'] % config['n_heads'] != 0:
        raise ValueError("dim 必须能被 n_heads 整除")

def setup_training_env(num_gpus=None, master_port="29500"):
    """设置训练环境"""
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 支持才能进行训练")
    
    # 复制当前环境变量
    current_env = os.environ.copy()
    
    # 设置PYTHONPATH
    current_env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    
    # 设置NCCL环境变量以提高稳定性
    current_env["NCCL_DEBUG"] = "INFO"
    current_env["NCCL_SOCKET_IFNAME"] = "eth0"
    current_env["NCCL_IB_DISABLE"] = "1"
    current_env["NCCL_P2P_DISABLE"] = "1"
    
    # 设置CUDA相关环境变量
    current_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # 确定GPU数量
    available_gpus = torch.cuda.device_count()
    if num_gpus is None:
        num_gpus = available_gpus
    num_gpus = min(num_gpus, available_gpus)
    
    if num_gpus > 0:
        current_env["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(num_gpus)])
    
    # 设置分布式训练环境变量
    current_env["WORLD_SIZE"] = str(num_gpus)
    current_env["MASTER_PORT"] = master_port
    
    return current_env, num_gpus

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置训练环境
    env, num_gpus = setup_training_env(args.num_gpus, args.master_port)
    
    # 创建实验目录
    if args.exp_name is None:
        args.exp_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 初始化默认配置
    config = {
        'dim': 1024,
        'n_layers': 24,
        'n_heads': 32,
        'n_kv_heads': 4,
        'vocab_size': 151664,
        'norm_eps': 1e-6,
        'max_seq_len': 2048,
        'rope_theta': 1e6
    }
    
    # 如果提供了配置文件，则加载并覆盖默认配置
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)  # 使用文件配置更新默认配置
        
        # 保存配置副本到实验目录
        with open(os.path.join(exp_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=4)
    else:
        # 即使没有配置文件，也保存默认配置
        with open(os.path.join(exp_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=4)
        print("使用默认配置进行训练")
    
    # 验证配置有效性
    validate_config(config)
    
    # 更新环境变量
    os.environ.update(env)
    
    # 设置分布式训练
    is_ddp, local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 训练配置
    data_dir = r"/root/.jupyter/3SAILab/pretrain_dataset"  # 数据目录
    batch_size = 1  # 批量大小
    max_seq_length = config.get('max_seq_len', 2048)  # 序列长度
    lr = 1e-4  # 学习率
    epochs = 1  # 训练轮数
    gradient_accumulation_steps = 8  # 梯度累积步数
    max_grad_norm = 1.0  # 梯度裁剪阈值
    save_interval = 5000  # 保存检查点间隔
    log_interval = 10  # 日志记录间隔
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = ExperimentLogger(exp_dir) if rank == 0 else None
    
    if rank == 0:
        print(f"找到 {num_gpus} 个GPU")
        print(f"实验目录: {exp_dir}")
        print(f"数据目录: {data_dir}")
        print(f"批量大小: {batch_size}")
        print(f"序列长度: {max_seq_length}")
        print(f"学习率: {lr}")
        print(f"训练轮数: {epochs}")
        print(f"梯度累积步数: {gradient_accumulation_steps}")
    
    # 创建模型
    model = SLM(LMConfig(**config))
    model.to(device)
    
    if rank == 0:
        # 打印模型参数信息
        model_size = print_model_parameters(model)
        print(f"模型大小: {model_size:.2f}M 参数")
        print(f"GPU内存使用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, lr=lr)
    
    # 加载数据集
    try:
        from datasets import load_dataset
        if rank == 0:
            print("正在加载数据集...")
        
        dataset = load_dataset('parquet', data_files={'train': f'{data_dir}/*.parquet'}, split='train')
        
        if rank == 0:
            print(f"数据集加载成功，共 {len(dataset)} 个样本")
    except Exception as e:
        if rank == 0:
            print(f"加载数据集时出错: {e}")
            import traceback
            traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    
    # 确保所有进程都正确加载了数据集
    if dist.is_initialized():
        dist.barrier()
    
    # 计算总步数
    total_rows = len(dataset)
    total_steps = (total_rows // (batch_size * world_size * gradient_accumulation_steps)) * epochs
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.01 * total_steps),
        num_training_steps=total_steps
    )
    
    # 将模型包装在DDP中
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # 创建数据集和数据加载器
    train_dataset = PretrainDataset(dataset, max_length=max_seq_length)
    train_sampler = DistributedSampler(train_dataset) if is_ddp else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    if rank == 0:
        print(f"数据加载器创建完成，总步数: {total_steps}")
        print(f"GPU内存使用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    
    # 初始化训练状态
    start_epoch = 0
    global_step = 0
    total_loss = 0.0
    processed_samples = 0
    best_loss = float('inf')
    
    # 恢复训练检查点
    if args.resume:
        latest_checkpoint = os.path.join(checkpoint_dir, "checkpoint-latest.pt")
        if os.path.exists(latest_checkpoint):
            start_epoch, global_step, total_loss, processed_samples = load_checkpoint(
                latest_checkpoint, model, optimizer, scheduler, device, rank
            )
            if rank == 0:
                print(f"从检查点恢复训练，从步骤 {global_step} 开始")
        else:
            if rank == 0:
                print("未找到检查点，从头开始训练")
    
    # 开始训练
    try:
        total_loss, processed_samples = train(
            model=model,
            train_dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
            start_epoch=start_epoch,
            global_step=global_step,
            total_loss=total_loss,
            processed_samples=processed_samples,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=save_interval,
            log_interval=log_interval,
            rank=rank,
            world_size=world_size,
            logger=logger,
            best_loss=best_loss
        )
        
        if rank == 0:
            print("训练完成!")
            print(f"总损失: {total_loss}")
            print(f"处理样本数: {processed_samples}")
    
    except Exception as e:
        if rank == 0:
            print(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 保存错误检查点
            error_path = os.path.join(checkpoint_dir, f"error_step{global_step}.pt")
            save_checkpoint(
                epoch=start_epoch, 
                step=global_step, 
                model=model, 
                optimizer=optimizer, 
                scheduler=scheduler,
                total_loss=total_loss, 
                processed_samples=processed_samples, 
                path=error_path, 
                rank=rank,
                logger=logger
            )
            print(f"错误检查点已保存到 {error_path}")
    
    finally:
        # 等待所有进程完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 清理分布式环境
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()