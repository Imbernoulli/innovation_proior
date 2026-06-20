Measured result — alternating (derandomized) flip; completes the airbench94 training,
arXiv:2404.00498 §3.6 (Table 1, Table 2). Metric: A100-seconds to reach 94% mean accuracy, **lower is
better**; accuracy higher is better.

**Final airbench94 training:** 94.01% mean accuracy in **9.9 epochs / 3.83 A100-seconds** (`airbench94.py`,
n=1000 runs). A non-algorithmic `torch.compile` reduces this 14% to **3.29 A100-seconds**
(`airbench94_compiled.py`, §3.7) while remaining mathematically equivalent.

**Table 1 — training-distribution options** (mean accuracy, fixed budget):

| random reshuffling | alternating flip | mean accuracy |
|---|---|---|
| No | No | 93.40% |
| No | Yes | 93.48% |
| Yes | No | 93.92% |
| Yes | Yes | **94.01%** |

Both derandomizing techniques improve accuracy independently. **Table 2 — effective speedup from switching
random→alternating flip** (selected; the airbench94 no-Cutout no-TTA rows): 27.1% at 20 epochs, 38.3% at
40 epochs. The paper estimates alternating flip contributes the final ~10% of the speedup over prior work,
and shows (Table 3, ImageNet ResNet-18) that it improves over random flip in every setting where flipping
helps at all.
