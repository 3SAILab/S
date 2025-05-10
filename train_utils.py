import random
import os
import torch
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if using multi-GPU
        # Ensure deterministic behavior (may impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def plot_loss(losses, steps, output_file="loss_curve.png", title="Training Loss Curve", beta=0.1):
    if not steps:
        print("No loss data recorded.")
        return
    # Calculate exponential moving average
    ema_losses = []
    if losses:
        ema_losses = [losses[0]]  # Initial value
        for i in range(1, len(losses)):
            ema = beta * losses[i] + (1 - beta) * ema_losses[-1]
            ema_losses.append(ema)
    
    plt.figure(figsize=(10, 6), dpi=150)
    plt.title(title)
    plt.fill_between(steps, losses, ema_losses, alpha=0.2, color="#1f77b4")
    plt.plot(steps, ema_losses, label='Smoothed Loss (EMA)', linewidth=1, color="#1f77b4")
    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Loss curve saved to {output_file}")

def plot_lr(learning_rates, steps, output_file="lr_curve.png", title="Learning Rate Schedule"):
    """
    绘制学习率变化曲线
    
    参数:
        learning_rates: 学习率列表
        steps: 训练步骤列表
        output_file: 输出图像文件路径
        title: 图表标题
    """
    if not steps or not learning_rates:
        print("没有学习率数据可绘制。")
        return
        
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(steps, learning_rates, linewidth=2, color="#2ca02c")
    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.7)
    # plt.yscale('log')  # 使用对数刻度更好地显示学习率变化
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Learning Rate curve saved to {output_file}")


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
