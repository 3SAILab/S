import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

# --- Configuration ---
@dataclass
class CustomLlamaConfig:
    vocab_size: int = 32000  # Size of the vocabulary
    hidden_size: int = 768    # Dimension of the hidden states (d_model)
    intermediate_size: int = 2048 # Dimension of the MLP intermediate layer
    num_hidden_layers: int = 6     # Number of transformer blocks
    num_attention_heads: int = 8   # Number of query heads (n_heads)
    num_key_value_heads: int = 4   # Number of key/value heads (n_kv_heads) - for GQA
    head_dim: int = hidden_size // num_attention_heads # Dimension of each attention head
    max_seq_len: int = 1024 # Maximum sequence length
    rms_norm_eps: float = 1e-5     # Epsilon for RMSNorm
    rope_theta: float = 10000.0    # RoPE base frequency

# Instantiate a default config for demonstration
config = CustomLlamaConfig()

# --- 1. RMSNorm Implementation ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter (learnable)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate Root Mean Square along the last dimension (features)
        # (batch_size, seq_len, hidden_size) -> computes RMS over hidden_size
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms

    def forward(self, x):
        # Normalize the input and scale by the learnable gamma parameter
        output = self._norm(x.float()).type_as(x) # Calculate in float32 for stability
        return output * self.weight

# --- 2. Rotary Positional Embedding (RoPE) ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the rotational embeddings (complex numbers cis(theta*m))
    needed for RoPE. Calculated in float32 for precision.
    """
    # Generates frequencies based on the dimension and theta parameter
    # Formula: freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # Generates position indices (m) from 0 to end-1
    t = torch.arange(end, dtype=torch.float32) # Positions m = 0, 1, ..., seq_len-1

    # Calculates outer product of positions and frequencies (theta * m)
    freqs = torch.outer(t, inv_freq) # Shape: (seq_len, dim / 2)

    # Converts freqs to complex numbers in polar form: R * exp(i * theta) = R * (cos(theta) + i*sin(theta))
    # Here R=1, so it's just cis(theta*m) = cos(theta*m) + i*sin(theta*m)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # Shape: (seq_len, dim / 2)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor, # Query tensor (batch, seq_len, num_heads, head_dim)
    xk: torch.Tensor, # Key tensor   (batch, seq_len, num_kv_heads, head_dim)
    freqs_cis: torch.Tensor, # Precomputed complex frequencies (seq_len, head_dim / 2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Positional Embedding to query and key tensors.
    """
    # Reshape xq and xk to view the last dimension as complex numbers
    # (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, num_heads, head_dim/2, 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis to match the sequence length dimension of xq_ and xk_
    # freqs_cis shape is (seq_len, head_dim/2) -> needs broadcasting for batch and head dims
    # Add dimensions for batch and heads: (1, seq_len, 1, head_dim/2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # Add batch and head dims for broadcasting
    seq_len = xq_.shape[1]
    freqs_cis = freqs_cis[:, :seq_len, :, :] # Ensure freqs match sequence length

    # Apply rotation by complex multiplication:
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    # xq_out = xq_ * freqs_cis
    # xk_out = xk_ * freqs_cis
    # view_as_real converts complex back to (..., 2) shape
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # Flatten the last two dims
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk) # Cast back to original dtype

# --- 3. Grouped Query Attention (GQA) ---
class Attention(nn.Module):
    def __init__(self, config: CustomLlamaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads # Number of Q heads sharing one K/V head
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        # Linear layers for Q, K, V projections
        self.wq = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)

        # Output projection layer
        self.wo = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat K/V heads n_rep times to match the number of query heads.
        Input shape: (batch, seq_len, num_kv_heads, head_dim)
        Output shape: (batch, seq_len, num_heads, head_dim)
        """
        if n_rep == 1:
            return x
        batch, seq_len, num_kv_heads, head_dim = x.shape
        x = x.unsqueeze(3).expand(batch, seq_len, num_kv_heads, n_rep, head_dim)
        return x.reshape(batch, seq_len, num_kv_heads * n_rep, head_dim)


    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None # Causal mask
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. Project to Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. Reshape for multi-head attention
        # (batch, seq_len, num_heads * head_dim) -> (batch, seq_len, num_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 3. Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 4. Repeat K and V heads for GQA
        xk = self._repeat_kv(xk, self.num_queries_per_kv) # (batch, seq_len, num_heads, head_dim)
        xv = self._repeat_kv(xv, self.num_queries_per_kv) # (batch, seq_len, num_heads, head_dim)

        # 5. Transpose for attention calculation: (batch, num_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 6. Scaled Dot-Product Attention
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if mask is not None:
            # mask shape should be (1, 1, seq_len, seq_len) or broadcastable
            scores = scores + mask # Masked positions get large negative values

        # Softmax converts scores to probabilities/weights
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(xq)

        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, xv)

        # 7. Concatenate heads and project output
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        output = output.transpose(1, 2).contiguous()
        # (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, hidden_size)
        output = output.view(batch_size, seq_len, self.hidden_size)

        # Final linear projection
        output = self.wo(output)
        return output

# --- 4. SwiGLU Feed-Forward Network ---
class FeedForward(nn.Module):
    def __init__(self, config: CustomLlamaConfig):
        super().__init__()
        hidden_dim = config.hidden_size
        intermediate_dim = config.intermediate_size

        # Llama uses SwiGLU FFN: w2(swish(w1(x)) * w3(x))
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False) # Gate projection
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False) # Up projection
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False) # Down projection

    def forward(self, x):
        # Apply Swish (SiLU) to the gate projection w1(x)
        swish_gate = F.silu(self.w1(x))
        # Project x up with w3(x)
        x_up = self.w3(x)
        # Element-wise multiply the Swish gate and the up-projected tensor
        gated = swish_gate * x_up
        # Project down to the hidden dimension
        output = self.w2(gated)
        return output

# --- 5. Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, config: CustomLlamaConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Pre-normalization and Attention + Residual connection
        # h = x + Attention(RMSNorm(x))
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)

        # Pre-normalization and FeedForward + Residual connection
        # out = h + FeedForward(RMSNorm(h))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# --- 6. Overall Llama Model ---
class CustomLlamaModel(nn.Module):
    def __init__(self, config: CustomLlamaConfig):
        super().__init__()
        self.config = config

        # Token Embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer Blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # Final normalization layer
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language Model Head (Output layer)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            # RoPE is typically applied to a fraction of the head dimension (e.g., 128)
            # Or sometimes the full head dimension. Let's use head_dim for simplicity here.
            # Need to confirm exact Llama3 RoPE dimension application if critical.
            self.config.head_dim,
            self.config.max_seq_len * 2, # Multiply by 2 for flexibility, might need adjustment
            theta=self.config.rope_theta
        )

        # Cache for the causal mask (optional, can be built on the fly)
        self._mask_cache: Optional[torch.Tensor] = None

    def _build_causal_mask(self, seq_len: int) -> torch.Tensor:
        # Builds an upper-triangular mask for causal attention.
        # If mask exists and is large enough, reuse it.
        if self._mask_cache is not None and self._mask_cache.shape[-1] >= seq_len:
            return self._mask_cache[:, :, :seq_len, :seq_len]

        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1) # Creates upper triangle matrix of -inf
        self._mask_cache = mask.type(torch.get_default_dtype()) # Use model's default dtype
        return self._mask_cache


    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        assert seq_len <= self.config.max_seq_len, \
            f"Sequence length {seq_len} exceeds model max length {self.config.max_seq_len}"

        # 1. Get Token Embeddings
        h = self.tok_embeddings(tokens) # (batch, seq_len, hidden_size)

        # 2. Retrieve precomputed RoPE frequencies for the current sequence length
        # Needs to be on the same device as the input tensor `h`
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seq_len]

        # 3. Build or retrieve the causal attention mask
        mask = self._build_causal_mask(seq_len).to(h.device)

        # 4. Pass through Transformer Blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        # 5. Final Normalization
        h = self.norm(h)

        # 6. Language Model Head (Output logits)
        output = self.output(h) # (batch, seq_len, vocab_size)

        # Return logits (usually in float32 for stability in loss calculation)
        return output.float()

# --- Example Usage ---
if __name__ == "__main__":
    # Use the default config defined earlier
    model_config = CustomLlamaConfig()
    print("Model Configuration:")
    print(model_config)

    # Instantiate the custom model
    model = CustomLlamaModel(model_config)
    print(f"\nModel Instantiated. Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create a dummy input tensor (batch_size=2, seq_len=10)
    # Token IDs should be within the vocab_size range
    dummy_input = torch.randint(0, model_config.vocab_size, (2, 10))
    print(f"\nDummy Input Shape: {dummy_input.shape}")

    # Set model to evaluation mode (if not training)
    model.eval()

    # Perform a forward pass
    with torch.no_grad(): # Disable gradient calculation for inference
        logits = model(dummy_input)

    print(f"Output Logits Shape: {logits.shape}") # Should be (batch_size, seq_len, vocab_size)

    # You can now use this 'model' object in your training loop.
    # Calculate loss using logits and target token IDs (e.g., with nn.CrossEntropyLoss)
    # Example loss calculation:
    # criterion = nn.CrossEntropyLoss()
    # dummy_targets = torch.randint(0, model_config.vocab_size, (2, 10))
    # # Reshape logits and targets for CrossEntropyLoss: (Batch * SeqLen, VocabSize) and (Batch * SeqLen)
    # loss = criterion(logits.view(-1, model_config.vocab_size), dummy_targets.view(-1))
    # print(f"Example Loss: {loss.item()}") # Requires dummy_targets
