 # MiniGPT 

A small (1-10M parameters) GPT-style language model in **PyTorch** from scratch.
The model is trained on **Project Gutenberg** text (Jane Austen novels), then analyzed with **mechanistic interpretability** tools—attention heatmaps and targeted head ablations—to identify and validate **induction heads**.

---

## Overview

- **Model:** decoder-only GPT-style Transformer (token+pos embeddings, causal multi-head self-attention, MLP blocks, residuals/LayerNorm) implemented from scratch in PyTorch (no `nn.Transformer`).
- **Training:** trained on public-domain Project Gutenberg text (Jane Austen novels).
- **Analysis:** mechanistic interpretability study of **induction heads** using attention heatmaps and targeted head ablations (measuring loss impact).

--- 

## Contents

- `TransformerModel.py` — Causal self-attention (masking), Transformer blocks, and the GPT model definition.
- `MiniGPT.ipynb` —  Data loading + tokenization, training loop, evaluation, and checkpoint saving.
- `InductionHeads.ipynb` — Induction-head investigation: attention heatmaps, head ablations, and quantitative loss/behavior checks.
- `utils.py` — Shared helpers (plotting, ablation utilities, cross-entropy evaluation, etc.).
- `Data/` — Training text (`.txt`) from Project Gutenberg (three Jane Austen novels).
- `Plots/` — Generated attention heatmaps + ablation visualizations for different model sizes.
- `miniGPT_2Mparams.pt` — Example saved checkpoint (model weights + config) for a 2M-parameter run.

---

## Results

### Induction-head probe: repeated-prefix vs randomized-prefix prompts

To test for induction behavior, we compare two types of prompts:

- **Repeated-prefix (control):**  
  `[base, base, base, base]`  
  The same token sequence `base` is repeated multiple times. If the model has an induction head, it can learn the pattern “when I see a repeated context, copy the token that followed it last time.”

- **Randomized-prefix (baseline):**  
  `[base, rand1, rand2, rand3]`  
  Only the first chunk is `base`; the rest are unrelated random chunks. This removes the repeated-context signal while keeping prompt length and token statistics similar.

For each position, we measure an induction-style attention score \(A(\cdot)\): the amount of attention mass assigned to the **“copy-from-previous-occurrence”** location.  
In practice, an induction head shows up as a bright stripe on the **+1 off-diagonal** of the attention matrix: token \(t\) attends strongly to \(t'\) where \(t'\) is the previous occurrence of the matching context, shifted by one (so it can copy the *next* token).

The plots below visualize this in two ways:
1) **Attention heatmaps:** qualitative evidence of the +1 off-diagonal stripe (stronger = more induction-like copying).
2) **Head ablations:** quantitative evidence that specific heads causally support the behavior. We compute a metric (e.g., \(\Delta\)) before/after ablating a head; a large drop indicates that head contributes materially to induction.

---

### Attention patterns (2M vs 5.6M)

<!-- Full attention heatmaps -->
**2M Parameters**: 

![2M attention heatmap](Plots/Attention_HeatMap_full_MiniGPT-2M.png)

**5.6M Parameters**

![5.6M attention heatmap](Plots/Attention_HeatMap_full_MiniGPT-5p6M.png)

<!-- Zoomed attention heatmaps (look for +1 off-diagonal stripe) -->
**2M Parameters**:

![2M zoomed attention heatmap](Plots/Attention_HeatMap_zoom_MiniGPT-2M.png)

**5.6M Parameters**:

![5.6M zoomed attention heatmap](Plots/Attention_HeatMap_zoom_MiniGPT-5p6M.png)

---

### Causal tests via head ablations (2M vs 5.6M)

For each model, we ablate individual attention heads at inference time and recompute the induction metric (reported as \(\Delta\)).  
The ablation plots show how \(\Delta\) changes when a head is removed: heads that strongly reduce \(\Delta\) are candidates for induction heads.

![2M head ablation impact](Plots/HeadAblation_HeatMap_MiniGPT-2M.png)
![5.6M head ablation impact](Plots/HeadAblation_HeatMap_MiniGPT-5p6M.png)


## Installation

```bash
git clone https://github.com/sferrante/Neuro-Symbolic-Reasoning.git
cd Neuro-Symbolic-Reasoning

pip install numpy
pip install matplotlib
pip install sklearn
pip install z3-solver
pip install torch --index-url https://download.pytorch.org/whl/cpu
