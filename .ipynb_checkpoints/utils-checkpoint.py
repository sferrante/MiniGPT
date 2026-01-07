import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import time
import re
import math


def diag_means(M, max_off=5):
    out = {}
    for off in range(-max_off, max_off+1):
        d = np.diagonal(M, offset=off)
        if len(d): out[off] = d.mean()
    return out

def plot_delta_heads(
    idx_rep, idx_ctl, attn_rep, attn_ctl, itos, zoom, # zoom=True plots the k->k-1 block; zoom=False plots full A[h]
    L=16, k=1,   
    cmap="viridis",
    print_block=1,
    max_off=3,
    figsize_scale=2.0,
):
    """
    idx_rep/idx_ctl: (B,T) token ids
    attn_rep/attn_ctl: list length n_blocks; each is (B, n_heads, T, T)
    Plots Δ = mean(attn_rep) - mean(attn_ctl) per (block, head).
    """
    n_blocks = len(attn_rep)
    n_heads  = attn_rep[0].shape[1]
    T = idx_rep.shape[1]

    fig, axes = plt.subplots(n_blocks, n_heads, figsize=(figsize_scale*n_heads, figsize_scale*n_blocks),
                             squeeze=False)

    for b in range(n_blocks):
        A_rep = attn_rep[b].mean(dim=0).detach().cpu().numpy()  # (H,T,T)
        A_ctl = attn_ctl[b].mean(dim=0).detach().cpu().numpy()
        A = A_rep - A_ctl                                       # (H,T,T)

        for h in range(n_heads):
            ax = axes[b, h]

            if zoom == False:
                M = A[h][L:4*L, L:4*L]
            if zoom == True:
                # k->k-1 block (queries in chunk k, keys in chunk k-1)
                q0, q1 = k*L, (k+1)*L
                p0, p1 = (k-1)*L, k*L
                M = A[h, q0:q1, p0:p1]

            v = np.max(np.abs(M)) + 1e-12
            ax.imshow(M, cmap=cmap, vmin=-v, vmax=v, aspect="auto", origin="upper")
            ax.set_title(f"Block {b}, Head {h}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

            # if b == print_block:
            #     # raw (non-delta) diagonals on the same block (useful sanity check)
            #     q0, q1 = k*L, (k+1)*L
            #     p0, p1 = (k-1)*L, k*L
            #     Mraw = attn_rep[b].mean(0)[h, q0:q1, p0:p1].detach().cpu().numpy()
            #     print(f"[b={b}, h={h}] diag_means(raw):", diag_means(Mraw, max_off=max_off))
            #     # full-matrix offsets for copy vs induction in your chunked setting
            #     print(f"[b={b}, h={h}] Δ copy diag (offset=-L):     {np.diagonal(A[h], offset=-L).mean():.6g}")
            #     print(f"[b={b}, h={h}] Δ induction diag (offset=-(L-1)): {np.diagonal(A[h], offset=-(L-1)).mean():.6g}")

    plt.tight_layout()
    plt.show()










## ------------------------------------ Ablations ------------------------------------ ##



@torch.no_grad()
def infer_n_blocks_heads(model, idx_example):
    device = next(model.parameters()).device
    _, attn = model(idx_example[:1].to(device), return_attn=True)
    n_blocks = len(attn)
    n_heads  = attn[0].shape[1]
    return n_blocks, n_heads

@torch.no_grad()
def rep_ctl_chunk_delta(
    model,  idx_rep,  idx_ctl,
    L=16,  chunk_ks=None,  micro_bs=200,
    ablate=None,          # expects: list of (block, head) tuples, e.g. [(1,2),(3,0)]
):
    """
    Scores next-token NLL *inside* chunk k (excluding first token in chunk),
    returns mean losses (avg CE) and delta = ctl_loss - rep_loss (bigger => repeat helped).
    """
    device = next(model.parameters()).device
    idx_rep = idx_rep.to(device)
    idx_ctl = idx_ctl.to(device)

    B, T = idx_rep.shape    # B = batch size , T = tokens (16*4)
    assert idx_ctl.shape == (B, T)
    assert T % L == 0, f"T={T} must be multiple of L={L}"
    n_chunks = T // L
    if chunk_ks is None: chunk_ks = list(range(1, n_chunks))  # typically ignore k=0 (no prior chunk to induce from)

    # positions t where we score CE(logits[t], target[t+1]) and t stays inside chunk
    # exclude first token in chunk: start = kL+1
    # exclude last token of chunk as a predictor (since it predicts next chunk): end = (k+1)L-1 (exclusive)
    pos_slices = {}
    for k in chunk_ks:
        start = k * L + 1
        end   = (k + 1) * L - 1   # positions of start, end T values of the chunks  (17 - 31 for first chunk)
        if end <= start:
            raise ValueError(f"Chunk k={k} too small for scoring with L={L}")
        pos_slices[k] = slice(start, end)

    rep_sum = {k: 0.0 for k in chunk_ks}
    ctl_sum = {k: 0.0 for k in chunk_ks}
    rep_cnt = {k: 0 for k in chunk_ks}
    ctl_cnt = {k: 0 for k in chunk_ks}

    # print(f'rep_sum is {rep_sum}')

    for s in range(0, B, micro_bs):
        # print(f'{s} / {B}')
        rep_mb = idx_rep[s:s+micro_bs]
        ctl_mb = idx_ctl[s:s+micro_bs]
        b = rep_mb.shape[0]  # could be less than micro_bs if overflows > B

        both = torch.cat([rep_mb, ctl_mb], dim=0)     # (2b, T)
        logits = model(both, ablate=ablate)           # (2b, T, V)
        V = logits.shape[-1]

        # per-position NLL (negative log-likelihood) over t=0..T-2 predicting token t+1
        nll = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, V),  ## :-1 drops the last token, since there is no "next" one to predict
            both[:, 1:].reshape(-1),          
            reduction="none",
        ).view(2*b, T-1)

        nll_rep = nll[:b]
        nll_ctl = nll[b:]

        for k in chunk_ks:
            sl = pos_slices[k]
            rep_sum[k] += float(nll_rep[:, sl].sum().item())
            # print(f'rep_sum is {rep_sum}')
            ctl_sum[k] += float(nll_ctl[:, sl].sum().item())
            rep_cnt[k] += int(nll_rep[:, sl].numel())
            ctl_cnt[k] += int(nll_ctl[:, sl].numel())

    rep_loss = {k: rep_sum[k] / max(rep_cnt[k], 1) for k in chunk_ks}
    ctl_loss = {k: ctl_sum[k] / max(ctl_cnt[k], 1) for k in chunk_ks}
    delta    = {k: ctl_loss[k] - rep_loss[k] for k in chunk_ks}

    rep_mean = float(np.mean([rep_loss[k] for k in chunk_ks]))
    ctl_mean = float(np.mean([ctl_loss[k] for k in chunk_ks]))
    del_mean = float(np.mean([delta[k] for k in chunk_ks]))

    return {
        "chunk_ks": list(chunk_ks),
        "rep_loss": rep_loss,
        "ctl_loss": ctl_loss,
        "delta": delta,
        "rep_loss_mean": rep_mean,
        "ctl_loss_mean": ctl_mean,
        "delta_mean": del_mean,
    }


def heatmap_df(df, n_blocks, n_heads, value_col="delta_drop"):
    A = np.full((n_blocks, n_heads), np.nan, dtype=float)
    for _, r in df.iterrows():
        A[int(r["block"]), int(r["head"])] = float(r[value_col])
    return A


    