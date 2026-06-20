# Context: ShinkaEvolve's discovered entropy-weighted under-utilization hinge loss

## Research question

Global-batch load-balancing loss (Qiu et al. 2025) balances MoE expert usage while preserving
specialization, by measuring the token frequency `f_i` over the whole corpus rather than each
micro-batch. But it equalizes only the *average* usage: an expert that has fallen well below its
fair share — a nearly-dead expert, the residue of collapse — is penalized no more sharply than one
that is merely a little cold, because the smooth `f·P` term has a weak gradient in the
under-utilized tail. The question: *is there a load-balancing loss, discovered rather than
hand-designed, that specifically rescues the under-used experts without over-suppressing the
specialization the global-batch term protects?*

## The discovered loss

ShinkaEvolve (Lange et al. 2025, arXiv:2509.19349, Sec. 4.4, Eq. 1) evolved the Python of the
balancing loss with an LLM-driven evolutionary search, scored by fitness `r = −(L_CE + L_imb)` on
real MoE pretraining. The discovered loss keeps the global-batch term and **adds** an
entropy-weighted under-utilization hinge:

```
L = N_E · (1/L) Σ_ℓ Σ_i f_{ℓ,i} P_{ℓ,i}                      [global-batch LBL]
    + (0.1/L) Σ_ℓ s(P_ℓ) · Σ_i max(0, τ − f_{ℓ,i})           [discovered hinge]

  s(P_ℓ) = 0.5 + (1 − H(P_ℓ) / log N_E)        entropy-complement weight
  τ       = 0.064 / N_E                          per-expert usage floor
```

The hinge fires **only** for experts below the usage floor `τ` (a per-expert minimum), and its
strength is scaled by `s(P_ℓ)`, the normalized complement of the router-distribution entropy in
that layer: when the router is peaked (low entropy `H`, collapse-prone), `s` is large and the
rescue is strong; when the router is already near-uniform, `s` is small and the hinge barely fires.
This is the targeted dead-expert rescue the global-batch term lacks, gated to act exactly when and
where collapse is happening, without flattening a healthy router.

## What is measured

`L_CE` (perplexity), `L_imb = ½ Σ_i |f_i − 1/N|`, fitness `r = −(L_CE + L_imb)`.

## The substrate and scale

ShinkaEvolve's evaluation used a 556M-param MoE, `N_E=64` experts, `K=8` active (82M active), ~2.10B
FineWeb tokens, `λ=0.01`, with a 2.7B/~30B-token scaling run, beating DeepSeek/Qwen global-batch
LBL on seven downstream benchmarks. **This is a small reproduction** — a tiny MoE (`N=8`, top-`K=2`,
two MoE layers, latent-topic next-token task) — that reproduces the mechanism and ordering, not the
scale. The discovered formula was confirmed verbatim against the paper (the released repo at
github.com/SakanaAI/ShinkaEvolve does not ship the MoE example code; the authoritative artifact is
Eq. 1 of arXiv:2509.19349, which matches the form above exactly: first-term coefficient `N_E/L`,
hinge coefficient `0.1/L`, weight `s(P_ℓ)=0.5+(1−H(P_ℓ)/log N_E)`, floor `τ=0.064/N_E`).
