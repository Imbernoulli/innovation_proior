When I push a decoder-only Transformer deeper, I pay for the depth twice. Each block writes the output of an attention or feed-forward sublayer back onto a shared residual stream, $x_{l+1} = x_l + F_l(x_l)$, and the identity skip is the highway that makes deep nets trainable — it gives gradients a clean path backward and lets each block begin as a small nudge to the identity. Yet as the stack grows the run becomes jittery, with loss and gradient spikes that worsen with depth and force smaller learning rates or a warmup just to avoid divergence; and when a deep model finally trains, probing shows the deepest layers do almost nothing — I can prune the top third with negligible change in loss. So depth destabilizes the run and the depth I bought is half-dead, and I want both fixed at once, cheaply: no meaningful new parameters, no surgery on the sublayers, and ideally working across pre-norm, post-norm, and the hybrids.

The reason the deep layers die becomes precise once I trace the pre-norm recursion $x_{l+1} = x_l + F(\mathrm{Norm}(x_l))$. The normalization fixes the scale of the branch *input* but does nothing to the stream $x_l$ I keep adding onto, so the stream variance accumulates with depth: under zero-mean normal weights $\sigma^2_{x_l} = \sigma^2_{x_1}\cdot\Theta\!\left(\prod_{k=1}^{l-1}(1 + 1/\sigma_{x_k})\right)$, bounded at depth $L$ between linear and exponential, $\Theta(L) \le \sigma^2_{x_L} \le \Theta(\exp L)$. The damage is in the block Jacobian: $\partial\,\mathrm{Pre\text{-}LN}(x)/\partial x = I + (\partial f/\partial\mathrm{Norm})(\partial\mathrm{Norm}/\partial x)$, and because $\mathrm{Norm}$ divides by the stream's own scale, $\partial\mathrm{Norm}/\partial x$ carries a $1/\sigma_{x_l}$ that shrinks as the variance grows. The second term collapses, the block Jacobian tends to $I$, and a block whose Jacobian is $I$ is becoming locally identity-like — changing its input barely changes its output. Chained across the stack, $\|\partial y_L/\partial x_1\|_2 \le \prod_l (1 + A/\sigma_{x_l} + B/\sigma^2_{x_l})$ converges to a finite constant on the exponential branch. The variance explosion is not a numerical annoyance; it is the mechanism that turns deep layers into dead weight, exactly the redundancy the pruning probes keep finding.

Knowing this, the existing fixes fall into families that all share one shape. LayerNorm Scaling cures the variance directly by dividing the branch input by $\sqrt{l}$, $x_{l+1} = x_l + F(\mathrm{Norm}(x_l)/\sqrt{l})$, flattening the curve toward linear and keeping deep Jacobians off $I$ — but the $1/\sqrt{l}$ factor is purely depth-dependent and static, baked in for the whole run, so it keeps strangling the deepest branches late in training when they should be learning at full strength. The bounded-update line — Fixup, T-Fixup, DeepNorm — instead pins down that the exploding *model update* is the disease and bounds it by a constant; DeepNorm uses $x_{l+1} = \mathrm{Norm}(\alpha\,x_l + F_\beta(x_l))$ with a constant skip scale $\alpha=(2L)^{1/4}$ and branch scale $\beta=(8L)^{-1/4}$, provably bounding the update at init and training past a thousand layers — but the bound is derived from, and held at, initialization, applied uniformly even through the long stable phase where a constraint sized for the worst instant of training is needlessly conservative. And the scalar-on-the-branch family, ReZero and SkipInit, gets the most important piece right: put one scalar on each branch, $x_{i+1} = x_i + \alpha_i\,F[W_i](x_i)$, and initialize every $\alpha_i$ to $0$, so the net is *exactly* the identity at init (the toy stack $x_L = (1+\alpha w)^L x_0$ has Jacobian $(1+\alpha w)^L$, and $\alpha=0$ preserves the signal where $\alpha=1$ would explode it). Being near identity at init is precisely what De and Smith showed is healthy — with $\mathrm{Var}(x_\ell)\approx\ell$ the new branch accounts for only a $1/(\ell+1)$ slice of the output variance, so a good deep net starts dominated by its skips. But ReZero and SkipInit *learn* $\alpha_i$, independently per layer, and that is the crux of where they leave me hanging: nothing forces the shallow branches to turn on before the deep ones. Given how entangled the stack is — a deep layer's input is the running output of every shallow layer, and a shallow layer's gradient is back-propagated through every deep layer above it — a deep branch firing early with random weights does two bad things at once, injecting noise into the representations the upper layers consume and corrupting the gradients the lower layers learn from. A learned scalar gives no control over this; its only guarantee is "start at zero," and after step one it is the optimizer's call. Every lever — depth-aware init, the constant DeepNorm $\alpha$, the static $1/\sqrt{l}$, the zero-init learnable scalar — sets up the first step and then freezes or hands the wheel to the optimizer. None is training-phase-aware, even though two facts about the trajectory stare back at me: training is staged (chaotic warmup, then a long stable phase, then decay), and the layers converge unevenly — shallow layers settle into their final representation earlier than deep ones. The missing ingredient is *when*, and in what depth order, each branch is allowed to start contributing.

I propose ProRes (Progressive Residual Warmup). The branch scalar should not be a parameter at all; it should be a predefined function $\alpha(l,t)$ of layer index $l$ and global step $t$ that I impose, that ramps from $0$ up to $1$ over training, and whose ramp is slower for deeper layers. The residual write becomes

$$x_{l+1} = x_l + \alpha(l,t)\, F(\mathrm{Norm}(x_l)),$$

and the defining linear schedule is

$$\alpha(l,t) = \min\!\left(\frac{t}{T\,l},\,1\right),\qquad l = 1,\dots,L,$$

where $T$ is the warmup length of the first layer, so layer $l$ reaches $\alpha=1$ at step $T\,l$ and the whole model finishes warming up at $T\,L$, after which every $\alpha=1$ and the model runs exactly as vanilla. Each of the three properties I wanted falls out rather than being asserted. Identity at init: $\alpha(l,0)=\min(0,1)=0$, so $x_{l+1}=x_l$ exactly — ReZero's clean start for free, made exact and parameter-free, and the variance cannot explode early because nothing is being added to the stream yet. Bounded update across both depth and time: early on only the shallow layers have nonzero $\alpha$, and they contribute only a fraction $\alpha<1$, so the model update is throttled; but the constraint relaxes itself layer by layer as training proceeds, tight when updates are chaotic and fully released once they are stable — the temporal awareness DeepNorm's constant $\alpha$ lacks. And the shallow-before-deep ordering: a deep layer's $\alpha$ stays near $0$ while the shallow layers below it do their large early updates, so by the time it climbs enough to matter those shallow layers have largely settled, and the deep layer builds on a stabilized input instead of amplifying random init noise or poisoning the gradient path. The activation explosion is cured the same way LayerNorm Scaling cures it but in time rather than statically — early deep $\alpha$ near zero adds almost nothing, the increments come on gradually so the variance climbs gently, and because $\alpha\to1$ there is no permanent late-phase clamp.

Several design choices are load-bearing. The scalar must be *predefined, not learned*: learning it is exactly ReZero, and there is no term in the loss that rewards "let the shallow layers settle first" — the shallow-before-deep order is a prior I hold about how training should proceed, drawn from the convergence-order and dependency facts, and a prior is something I impose, not something I ask the data to rediscover. The specific *shape* matters too, which I can see by ruling out alternatives. An equal schedule $\alpha(l,t)=\min(t/T,1)$ with no $l$-dependence still delays the chaotic init but ignores ordering — all deep branches switch on simultaneously with the shallow ones, and a long shared warmup can dump a coordinated burst of representation and gradient noise into the stack. A reverse schedule $\tau_l = T(L-l+1)$, deep-first, actively does the harmful thing, letting randomly-initialized deep branches dominate optimization early while the shallow layers starve — the very Post-LN divergence pattern warmup exists to avoid. Freezing the constraint instead of relaxing it — holding $\alpha(l)=1/\sqrt{l}$ forever, which is just LayerNorm Scaling — strangles the deep layers through the entire stable phase, whereas the relaxing ramp hands back their full capacity once they are past the dangerous early regime; that the relaxing version should win is the clean, falsifiable consequence of the whole training-phase-aware thesis. The linear-in-$l$ stagger $\tau_l = T\,l$ is simply the simplest schedule that encodes "deeper takes longer" with a single knob. I multiply only the *branch* $F(\mathrm{Norm}(x_l))$ by $\alpha$ and leave the skip at weight $1$: if I scaled the skip I would attenuate the gradient highway, and at $\alpha=0$ the network would collapse to zero output instead of to the identity — keeping the skip at $1$ makes $\alpha=0$ exactly identity and $\alpha=1$ exactly vanilla, a clean interpolation between the two endpoints I want. The principle is architecture-agnostic: wherever a block has a residual branch I multiply that branch by $\alpha(l,t)$ — Pre-LN $x_l + \alpha\,F(\mathrm{Norm}(x_l))$, Post-LN $\mathrm{Norm}(x_l + \alpha\,F(x_l))$, Sandwich-LN $x_l + \alpha\,\mathrm{Norm}(F(\mathrm{Norm}(x_l)))$, DeepNorm $\mathrm{Norm}(\alpha_{\text{const}}x_l + \alpha(l,t)\,F_\beta(x_l))$, LayerNorm Scaling $x_l + \alpha(l,t)\,F(\mathrm{Norm}(x_l)/\sqrt{l})$ — with the post-norm-flavored variants benefiting most because they are the ones whose deep layers normally dominate and destabilize. There is one free number, $T$. Too small and I barely delay anything past the chaotic init; too large and $T\,L$ eats the training budget so the deep layers never get enough full-strength steps. For the sizes I care about $T=1000$ lands well — at $32$ layers that is a $32{,}000$-step total warmup, comfortably inside a $100$k-step run — and I do not even tune it, except at extreme depth where I drop it to $T=500$ so $T\,L$ stays $\le$ the total steps and every layer actually reaches full strength. A handful of curvature variants share the same total warmup $T\,L$ and only change the early bend: $\left(\min(t/(T l),1)\right)^2$ eases in more gently for touchier Post-LN, $\left(\min(t/(T l),1)\right)^{1/2}$ comes on faster; linear is the natural default.

The implementation has one wrinkle: $\alpha(l,t)$ depends on the global step, which the block does not see, so the schedule lives in the model's forward loop. I carry the step as a non-trainable buffer; in the nanoGPT-style harness the block already returns $x + \delta$, so I recover the contribution as $\delta = \texttt{block\_out} - x$ and write $x \leftarrow x + \alpha\,(\texttt{block\_out} - x)$, scaling exactly the branch and leaving the skip at weight $1$. I read the step before the loop so the first training forward is truly $t=0$ with every branch off, then advance the buffer after the forward. Once a layer's $\alpha$ reaches $1$ I take $\texttt{block\_out}$ directly, identical to vanilla. Two extra lines of arithmetic, one integer buffer, zero learnable parameters, and the optimizer untouched.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# existing primitives: CausalSelfAttention, MLP, LayerNorm, GPTConfig


class Block(nn.Module):
    """Vanilla Pre-LN block — unchanged. The residual schedule is applied in GPT.forward."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # ProRes: T = first-layer warmup length; layer l warms up over T*l steps,
        # the whole model over T*L. The step counter is a buffer, not a parameter.
        self.prores_T = 1000
        self.register_buffer('_prores_step', torch.zeros(1, dtype=torch.long))

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))

        step = self._prores_step.item()
        T = self.prores_T

        for i, block in enumerate(self.transformer.h):
            block_out = block(x)                  # block_out = x + delta (Pre-LN residual)
            if self.training and step < T * (i + 1):
                layer_idx = i + 1                 # 1-indexed layer l
                alpha = min(step / (T * layer_idx), 1.0)   # alpha(l,t) = min(t/(T*l), 1)
                x = x + alpha * (block_out - x)   # scale only the branch: x <- x + alpha*delta
            else:
                x = block_out                     # alpha == 1: exact vanilla Pre-LN

        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        if self.training:
            self._prores_step += 1               # first training forward uses t=0
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # unchanged: ProRes adds no learnable parameters
        decay = [p for p in self.parameters() if p.dim() >= 2 and p.requires_grad]
        nodecay = [p for p in self.parameters() if p.dim() < 2 and p.requires_grad]
        optim_groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
```
