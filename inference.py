# -*- coding: utf-8 -*-
"""
@Author: ShouqinGuan
模型推理脚本 - 用于加载训练好的模型并生成文本
"""
import os
import time
import json
import argparse
import torch
import numpy as np
import warnings
from tqdm import tqdm
from model import SLM, LMConfig
from transformers import AutoTokenizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径，可以是检查点文件或目录")
    parser.add_argument("--config_path", type=str, default=None, help="模型配置文件路径")
    parser.add_argument("--tokenizer_path", type=str, default="/model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="分词器路径")
    parser.add_argument("--prompt", type=str, default="我是", help="提示文本")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="生成的最大新token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p采样参数")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k采样参数")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚参数")
    parser.add_argument("--device", type=str, default=None, help="设备，例如'cuda:0'或'cpu'")
    parser.add_argument("--seed", type=int, default=int(time.time()), help="随机种子")
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径")
    # 添加模型配置参数
    parser.add_argument("--dim", type=int, default=1024, help="模型维度")
    parser.add_argument("--n_layers", type=int, default=24, help="层数")
    parser.add_argument("--n_heads", type=int, default=32, help="注意力头数")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--vocab_size", type=int, default=151664, help="词汇表大小")
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_model(args):
    """加载模型和配置"""
    # 如果提供了配置文件路径，从文件加载配置
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        model_config = LMConfig(**config_dict)
    else:
        # 否则使用命令行参数创建配置
        model_config = LMConfig(
            dim=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seq_len=args.max_seq_len,
            vocab_size=args.vocab_size,
            norm_eps=1e-5,
            rope_theta=1000000.0
        )
    
    # 创建模型
    model = SLM(model_config)
    
    # 加载模型权重
    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location="cpu", weight_onlys=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"从检查点加载模型: {args.model_path}")
    else:
        raise ValueError(f"模型路径无效: {args.model_path}")
    
    # 将模型移动到设备上并设置为评估模式
    model = model.to(args.device)
    model.eval()
    
    return model

def load_tokenizer(tokenizer_path):
    """加载分词器"""    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        # 确保有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"加载分词器时出错: {e}")

def generate_text(model, tokenizer, prompt, args):
    """生成文本"""
    # 编码输入文本
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)
    
    # 生成文本
    generated = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    print("\n=== 生成的ids ===")
    print(generated[0])
    # 解码生成的文本
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    return generated_text

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {args.device}")
    
    # 加载分词器
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    # 加载模型
    model = load_model(args)
    
    # 生成文本
    if args.prompt:
        print(f"提示: {args.prompt}")
        generated_text = generate_text(model, tokenizer, args.prompt, args)
        
        # 打印生成的文本
        print("\n=== 生成的文本 ===")
        print(generated_text)
        
        # 保存到文件
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            print(f"生成的文本已保存到: {args.output_file}")
    else:
        # 交互式模式
        print("进入交互式模式，输入'exit'退出")
        while True:
            prompt = input("\n请输入提示 >>> ")
            if prompt.lower() == 'exit':
                break
            
            generated_text = generate_text(model, tokenizer, prompt, args)
            
            print("\n=== 生成的文本 ===")
            print(generated_text)

if __name__ == "__main__":
    main()