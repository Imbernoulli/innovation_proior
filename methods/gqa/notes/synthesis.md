# GQA synthesis notes

## Pain point
Autoregressive decoder inference is memory-bandwidth bound. Each decode step loads (a) the
decoder weights and (b) the entire KV cache from HBM, but does only O(d) arithmetic per token.
The accelerator's compute units starve waiting on memory. Roofline: a kernel is memory-bound
when arithmetic intensity (FLOPs / byte) < machine balance.

## KV-cache memory:compute derivation (self-checked, matches Shazeer 2019 §2.4.1, §3.1)
Notation: b batch, n seq len, d model dim, h heads, k=d/h head dim.
- Compute over n decode steps: Θ(b n d²) (QKVO projections dominate).
- MHA memory access: KV cache touched Σ_i b·h·i·k = Θ(b n² d); weights n·d². Total Θ(bn²d + nd²).
  Ratio mem/compute = (bn²d + nd²)/(bnd²) = **n/d + 1/b**. → bottleneck when n≈d or b small.
- MQA: K,V single head, cache Θ(b n² k)=Θ(bn²d/h). Total Θ(bnd + bn²k + nd²).
  Ratio = **1/d + n/(dh) + 1/b**. Offending n/d term cut by factor h.
- GQA-G: G KV heads, cache ∝ G. Ratio term = n·G/(dh). G=1→MQA, G=h→MHA. KV cache ∝ G.

## MQA downsides → motivates GQA + uptraining
- Quality degradation (single KV head = aggressive capacity cut, worsens as models scale h).
- Training instability (loss spikes, divergence on long-input finetuning — appendix A).
- Don't want to train a separate inference-only model from scratch.

## GQA
Partition H query heads into G groups, each group shares one K head and one V head.
G=1 is MQA, G=H is MHA. KV cache ∝ G. Keep proportional bandwidth/capacity cut as h scales.
Larger models suffer less from bandwidth (KV ∝ d, FLOPs ∝ d²) so can afford intermediate G.
Sharding replicates the single MQA KV head across model partitions (waste); GQA with G≈#partitions
removes that waste. Chose G=8 as middle ground. Not applied to encoder self-attn (parallel, not
bandwidth-bound) — applied to decoder self-attn and cross-attn.

## Uptraining (recipe inspired by sparse upcycling, Komatsuzaki 2022)
Two steps: (1) convert checkpoint — mean-pool the K,V projection matrices of the h/G heads in
each group into one matrix; (2) continue pretraining for α≈5% of original steps on same recipe.
Mean-pool beats select-one-head beats random-init (ordered by info preserved). GQA already decent
right after conversion; MQA needs the uptraining. 5% gets most gains, diminishing by 10%.

## Canonical implementation (HF LLaMA modeling_llama.py)
- k_proj/v_proj output num_key_value_heads*head_dim (= G*k), q_proj outputs num_heads*head_dim.
- num_key_value_groups = num_heads // num_key_value_heads = H/G = group size.
- repeat_kv(x, n_rep): expand KV heads [b,G,s,k] → [b,H,s,k] by repeat_interleave so each KV head
  serves its n_rep=H/G query heads. Then standard scaled-dot-product attention.
- Conversion: W_k^g = mean over h/G member heads of W_k^(j) (einops reduce 'g r ... -> g ...', mean).

## Design-decision → why
- Share K,V not Q: bandwidth bottleneck is the cached K,V (Q for current token is tiny); reducing
  query heads would cut representational mixing without helping the cache.
- Mean-pool not select/random: preserves the averaged learned representation → least adaptation.
- α=5%: enough to adapt to new structure, cheap; 10% diminishing.
- G=8: speed cost rises slowly from MQA then steeper near MHA; 8 near-MQA speed, near-MHA quality.
