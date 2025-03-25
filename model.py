# -*- coding: utf-8 -*-
"""
author: shouqinguan
date: 2025-3-21
description: S architecture of a language model
"""
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class LMConfig(PretrainedConfig):
    model_type = "S"

    def __init__(
        self,
        dim: int = 1536,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: int = 4,
        vocab_size: int = 151664,
        hidden_dim: int = None,
        multiple_of: int = 64,
        norm_eps: float = 1e6,
        max_seq_len: int = 2048,
        rope_theta: int = 1e6,
        **kwargs
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        super().__init__(**kwargs)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        
def percompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    # TODO why freqs
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    # TODO why outer
    freqs = torch.outer(t, freqs).float()
    # TODO why polar
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)
    return pos_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # freqs_cis (b, d)  x (a, b, c, d) ->  freqs_cis (1, b, 1, d)
    ndim = x.ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, pos_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(pos_cis, xq_)
    xq_out = torch.view_as_real(freqs_cis * xq_).flatten(3)
    xk_out = torch.view_as_real(freqs_cis * xk_).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_req: int) -> torch.Tensor:
    bs, seq_len, n_kv_heads, head_dim = x.shape
    if n_req == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, seq_len, n_kv_heads, n_req, head_dim)
        .reshape(bs, seq_len, n_kv_heads * n_req, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        assert config.n_heads % self.n_kv_heads == 0
        self.n_local_heads = config.n_heads
        self.n_local_kv_heads = config.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
    
    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache=False):
        # TODO kv_cache why
        bs, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xv = apply_rotary_emb(xq, xk, pos_cis)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            is_causal=True
        )
        output = output.transpose(1, 2).reshape(bs, seq_len, -1)
        return self.wo(output), past_kv

class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of) 
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x) * self.w3(x)))
        

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attn = Attention(config)

        self.layer_id = layer_id
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, past_key_value=None, use_cache: bool = False):      
        h_attn, past_kv = self.attn(
            self.attn_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + h_attn
        out = h + self.ffn(self.ffn_norm(h))
        return out, past_kv
    
class SLM(PreTrainedModel):
    config_class = LMConfig
    def __init__(self, config: LMConfig = None):
        self.config = config
        super().__init__(self.config)
        self.vocab_size, self.n_layers = config.vocab_size, config.n_layers
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(l, config) for l in range(self.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        self.register_buffer(
            "pos_cis",
            percompute_pos_cis(dim=config.dim // config.n_heads, end=config.max_seq_len * 2, theta=config.rope_theta),
            persistent=False
        )
        self.out = CausalLMOutputWithPast()

    def forward(self, input_ids: Optional[torch.Tensor] = None, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, use_cache: bool = False, **args):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get('start_pos', 0)
        h = self.tok_embeddings(input_ids)
        pos_cis = self.pos_cis[start_pos: start_pos + input_ids.size(1)]
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis, 
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )
            past_kvs.append(past_kv)
        logits = self.output(self.norm(h))
        self.out.__setitem__("logits", logits)
        self.out.__setitem__('past_key_value', past_kvs)
        return self.out
    
    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=512, temperature=0.75, top_p=0.9, stream=False, rp=1, use_cache=True, pad_token_id=0, **args):
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat([seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)], dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_token, temperature, top_p, rp, use_cache, **args):
        start = input_ids.shape[1]  # 记录初始长度
        past_kvs = None
        new_token_idx = 0
        while new_token_idx < max_new_token:  # 基于新令牌数循环
            if past_kvs is None:  # 首轮使用完整输入
                out = self(input_ids, past_key_values=past_kvs, use_cache=use_cache)
            else:  # 后续仅用最后一个令牌
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            
            # 应用重复惩罚和温度调整
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            
            # Top-p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            
            # 采样并更新输入
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, input_ids_next], dim=-1)
            new_token_idx += 1
            
            # 遇到EOS则提前终止
            if input_ids_next.item() == eos_token_id:
                print(input_ids_next)
                break
            yield input_ids[:, start:]  # 仅返回新生成的令牌

def print_model_parameters(model):
    print("Layer Name & Parameters")
    print("-----------------------------")
    total_params = 0
    for name, parameter in model.named_parameters():
        param_size = parameter.size()
        param_count = torch.prod(torch.tensor(param_size)).item()
        total_params += param_count
        print(f"{name:50} | Size: {str(param_size):30} | Count: {str(param_count):20}")
    print("-----------------------------")
    print(f"Total Parameters: {total_params} ({total_params / 1000000:.1f} M)")

if __name__ == "__main__":
    LMConfig_Dense = LMConfig()
    S_Dense = SLM(LMConfig_Dense)

    # 生成测试输入 - 形状为 (4, 2047) 的整数张量，值在 0-999 之间
    test_input = torch.randint(low=0, high=151643, size=(4, 2047))
    
    # 测试模型
    with torch.no_grad():
        # 前向传播
        output = S_Dense(test_input)
        logits = output.logits
        # 检查输出形状
        print(f"Output shape: {logits.shape}")  # 应该是 (4, 2047, vocab_size)
        
        # 检查输出值
        print(f"Sample output values (first 5 of first sequence):")
        # 检查概率分布是否合理
        probs = torch.softmax(logits, dim=-1)
        print("\nProbability sums (should be ~1.0):")
        print(probs[0, :5, :].sum(dim=-1))  # 检查概率是否归一化
