import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import re
from collections import Counter
import math



class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)  #  ** weight qkv's **  
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # causal mask: (1,1,T,T) buffer
        mask = torch.tril(torch.ones(block_size, block_size))  ## lower triangular of ones, [T,T]
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))  ## attaches [1,1,T,T] mask to model
        
    def forward(self, x, return_attn=False, ablated_heads=None):
        B, T, C = x.shape                     # Batch, Token, Channels  ... note T can be < block_size
        qkv = self.qkv(x)                     # (B,T,3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, nh, T, hd)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B,nh,T,T)

        # apply causal mask (only allow <= current position)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))  ## Replaces top right block with -inf's 
        att_probs = F.softmax(att, dim=-1)

        out = att_probs @ v                                                   # (B,nh,T,hd)
        if ablated_heads is not None: out[:, ablated_heads, :, :] = 0         # kill ablated heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)                  # (B,T,C)
        out = self.proj(out)

        if return_attn:
            return out, att_probs
        return out
    
    
    
    
class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, return_attn=False, ablated_heads=None):
        if return_attn:
            att_out, att_probs = self.attn(self.ln1(x), return_attn=True, ablated_heads=ablated_heads)
            x = x + att_out
            x = x + self.mlp(self.ln2(x))
            return x, att_probs
        x = x + self.attn(self.ln1(x), ablated_heads=ablated_heads)
        x = x + self.mlp(self.ln2(x))
        return x
    
    
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, d_model=64, n_blocks=4, n_heads=4, d_ff=256):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, block_size) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  ## Final layer 

        # weight tying (saves params ... )
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, return_attn=False, ablate=None):
        B, T = idx.shape
        assert T <= self.block_size

        pos = torch.arange(T, device=idx.device).unsqueeze(0)         # (1,T)
        x = self.tok_emb(idx) + self.pos_emb(pos)                     # (B,T,d)

        attn_maps = []
        for i, blk in enumerate(self.blocks):
            ablated_heads = None
            if ablate is not None:
                ablated_heads = [h for (b, h) in ablate if b==i]      # heads to kill in this block
            if return_attn:
                x, att = blk(x, return_attn=True, ablated_heads=ablated_heads)
                attn_maps.append(att)                                 # each: (B,nh,T,T)
            else:
                x = blk(x, ablated_heads=ablated_heads)

        x = self.ln_f(x)
        logits = self.lm_head(x)                                      # (B,T,vocab)

        if return_attn:
            return logits, attn_maps
        return logits
    
    
    
    
    
    
    
    
    
    