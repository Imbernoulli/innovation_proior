GeGLU gave the cleanest possible isolation of one variable, and its row tells me the *structure* was right but the gate is unsettled. On seed 42 it lands $val\_loss = 2.2952$, with WikiText-2 ppl 44.13, LAMBADA 68.73, and downstream {arc_easy 54.88, hellaswag 32.90, piqa 64.15, winogrande 50.36}. The gating structure carried its weight — it didn't blow up, didn't need a schedule change. But two numbers in that row sag exactly where I should worry: LAMBADA at 68.73 and hellaswag at 32.90 are the weakest of the GLU fills I'm comparing, and those are precisely the metrics — long-range completion and commonsense continuation — that depend on the per-position transformation passing a useful, well-scaled signal forward. That points straight at the part of GeGLU I chose most casually: I put GELU on the gate because it was the activation the default MLP already ran, not because I derived it as the right gate. The structure is fixed and good; the gate is an open knob, and this row is the reason to turn it.

I propose **SwiGLU** — keep the GLU structure and the matched 8/3 budget exactly, and replace the gate's activation, $\text{GELU}\to\text{SiLU}$ (Swish at $\beta=1$):
$$\text{MLP}(x) = \big(\,\text{SiLU}(xW) \otimes (xV)\,\big)\,W_2, \qquad \text{SiLU}(z) = z\,\sigma(z).$$
The change is literally one function on the gate path. This is deliberately the family parameterized by exactly that choice of $f$ — sigmoid gives GLU, ReLU gives ReGLU, GELU gives GeGLU, identity gives Bilinear, and Swish/SiLU gives SwiGLU — so swapping $f$ is free and fully controlled: any $val\_loss$ delta from here is attributable to the gate's activation alone, the same clean isolation that made GeGLU's number interpretable.

Why SiLU should beat GELU rather than merely differ, I have to argue honestly, because the two are close. Both are "value $\times$ smooth gate of the value": $\text{GELU}(z)=z\,\Phi(z)$ weights $z$ by the standard-normal CDF, $\text{SiLU}(z)=z\,\sigma(z)$ weights it by the logistic. At $\beta=1$ the two curves are nearly indistinguishable — GELU's own cheap approximation is $z\,\sigma(1.702\,z)$, a Swish with $\beta\approx 1.702$ — so on the curve alone I do not expect a large gap, and I won't pretend I'm reaching for a dramatically different function. What I *am* reaching for is the gate that the modern at-scale FFNs converged on for this precise slot: SwiGLU is the form PaLM, LLaMA, DeepSeek, and Qwen all settled on for a gated, $\sim$8/3-width, bias-free feed-forward sublayer, which is why it earns the *next* rung rather than sitting as a sibling of GeGLU — it is the empirically-selected gate for exactly this construction, where GeGLU was the conservative carry-over.

The mechanistic case for *why this could lift the metrics that sagged* turns on the gate's behavior in two regimes. For large positive preactivations the two gates agree closely (both approach identity-like passthrough), so the difference is concentrated near and below zero. There $\sigma(z)$ is the smoother, slightly *softer* gate around the origin and has a marginally more pronounced non-monotonic dip for small negative $z$. Two consequences matter. First, the value path is linear in both, so the gradient-highway argument GeGLU was built around is untouched: $\nabla[X\otimes f(X)]$ still has the leading term $\nabla X \otimes f(X)$ that scales the upstream gradient by the gate *value*, not by an activation derivative — switching from a $\Phi$-shaped to a $\sigma$-shaped gate only reshapes *which* units the highway is open on, it does not reintroduce a derivative factor. Second, the gate value is what multiplies the carried content, and SiLU's gently non-monotonic, smooth profile lets a slightly larger band of moderate-magnitude preactivations pass content at near-or-above unit gain before saturating, with a softer roll-off into the negative regime. For tasks that depend on accumulating a precise, well-scaled signal across many tokens — LAMBADA's long-range completion, hellaswag's commonsense continuation — a gate that is a touch more generous and smoother in the moderate regime, and slightly less prone to hard suppression, is the kind of change that nudges those metrics. I assert the *direction*, not the magnitude: the curves are close, so I expect a small effect.

The budget bookkeeping is identical to GeGLU and worth confirming, because the matched-budget premise is the only reason these numbers compare. SwiGLU has the identical three-matrix layout — gate $W$, value $V$, down $W_2$, the same shapes — so parameters are again $3\,d\,d_{ff}'$, FLOPs again three $d\times d_{ff}'$ matmuls, and the matched condition $3\,d\,d_{ff}' = 2\,d\,(4d)$ gives the same $d_{ff}' = \tfrac{8}{3}d$. At $n\_embd = 1024$ that is $\lfloor\tfrac83\cdot1024\rfloor = 2730$ rounded up to a multiple of 64 (2752) — the same width GeGLU used. So SwiGLU and GeGLU differ in the *single* function on the gate path and in nothing else: not width, not matrix count, not biases (`config.bias=False` throughout), not the schedule. In the task's edit surface the forward is $x\to w_1(x)$ (gate) and $x\to w_3(x)$ (value), both to $(B,T,\text{hidden})$; apply SiLU to the gate, multiply elementwise, project down with $c_{proj}$, dropout — one line, `F.silu(self.w1(x)) * self.w3(x)`, where GeGLU had `F.gelu`. `CONFIG_OVERRIDES` stays empty: I am changing the gate's activation, not the learning rate or weight decay, and the entire value of the step is that it isolates that one function.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — step 2: SwiGLU MLP (8/3 width)
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 / 3 * config.n_embd)
        # Round to nearest multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: SiLU(xW1) * (xW3) then project back
        return self.dropout(self.c_proj(F.silu(self.w1(x)) * self.w3(x)))


# training-setup hook left at the default — only the gate activation changed:
CONFIG_OVERRIDES = {}
```
