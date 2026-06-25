The default run told me exactly what is wrong, and in the per-seed numbers. On `p=59` the three seeds grokked at 10000, 5500, and *never*: the 456 seed sat at `test_accuracy 0.982` with `grok_rate 0` because it ran out of the 30000-step budget before the late jump arrived, dragging the prime's mean steps-to-grok to 15167 and its grok rate to 0.667. On `p=97` and `p=113` every seed reached a clean `1.0`, but even there the mean steps-to-grok were 2833 and 2333 — thousands of full-batch steps spent memorizing before the held-out curve moved. The recipe is not broken; weight decay is plainly doing its job. What is broken is *timing*: the grok is slow, and on the smallest table the late jump sits so close to the ceiling that one unlucky split costs an entire grok. I have to shrink the delay without touching the architecture, the loss, or the optimizer — all of which are doing what they should — which leaves exactly one place: the gradient hook the scaffold exposes between `loss.backward()` and `optimizer.step()`.

The training curve that produced those numbers has two things happening on wildly different timescales in the *same* run: the training loss falls off a cliff in well under a thousand steps, while the held-out loss barely moves for thousands of steps and then drops. Both curves are downstream of one process — the parameters moving — so the parameter motion must contain a fast-changing part and a slow-changing part mixed together. The fast part memorizes the table; the slow part eventually lands the Fourier circuit and generalizes. That is the working hypothesis, and it is sharp and actionable: if I can separate the slow part of the parameter motion from the fast part, the slow part is the friend I want to encourage, and encouraging it is exactly how I would make the 456 seed grok before step 30000 and pull the easy primes in from ~2800.

The method I propose is the **Grokfast-EMA gradient hook**: amplify the slow-varying component of each parameter's gradient before AdamW reads it. To make "slow" and "fast" precise, read the per-step update of a single scalar parameter, $u(t) = \theta(t{+}1) - \theta(t)$, as a *signal* indexed by training step. The natural language for splitting a signal into slow- and fast-varying parts is frequency: take its discrete-time Fourier transform $U(\omega) = \sum_t u(t)\,e^{-i\omega t}$, and "slow" is the low-frequency content, "fast" the high. The hypothesis restated is that the low-frequency content drives generalization and the high-frequency content drives memorization, so accelerating generalization means *boosting the low-frequency content of the parameter motion*.

I do not act on $u(t)$ directly — the optimizer produces it from the gradient stream $g(t)$ — but for a first-order optimizer the update is built linearly out of the gradients (SGD is $u=-\eta g$, momentum a running average of $g$, AdamW an EMA of $g$ rescaled and shifted by decoupled decay). So I reshape $g$ and let it propagate into $u$: boost the low frequencies of the gradient signal in the hook, before AdamW reads `p.grad`. The cleanest way to get an *amplifier* rather than an information-destroying filter is to take a low-pass-filtered copy of the signal and add it back: with $h(t)$ a low-pass filter and $*$ convolution,

$$\hat g(t) = g(t) + h(t) * g(t), \qquad \hat G(\omega) = \big(1 + H(\omega)\big)\,G(\omega).$$

The effective gain at frequency $\omega$ is $1 + H(\omega)$. If $H$ is low-pass — big at low $\omega$, small at high $\omega$ — then the slow component is multiplied up while the fast component passes through with gain $\approx 1$. Keeping the high frequencies is essential: the rapid early descent that fits the training table is the fast part doing useful work, and I still need the model to fit the table. I am not denoising the gradient; I am *emphasizing* its slow part relative to its fast part. This is why *adding* the low-pass copy is right and the obvious simpler move — *replacing* $g$ with $h*g$ — is wrong: training purely on a moving average of recent gradients is training with a giant smoothed gradient and no fast content, which would make training slow and shaky, exactly the wrong thing when the problem is slowness.

For $h$ I want a low-pass filter with $O(1)$ state, not a windowed moving average (a finite-impulse-response filter literally stores $w$ gradient copies per parameter — a non-starter as a general tool and a per-step copy cost besides). The way to get the same behavior with constant memory is a one-pole infinite-impulse-response filter, i.e. an exponential moving average, the same recursion that lives inside momentum and Adam: keep one buffer $\mu$ per parameter and update

$$\mu(t) = \alpha\,\mu(t{-}1) + (1-\alpha)\,g(t), \qquad \hat g(t) = g(t) + \lambda\,\mu(t).$$

Its impulse response is geometric, $h(t) = \lambda(1-\alpha)\alpha^t$ for $t\ge 0$, so the transfer function is $H(\omega) = \lambda(1-\alpha)\big/\big(1-\alpha e^{-i\omega}\big)$, convergent since $\alpha<1$. At DC, $H(0)=\lambda$, so the slowest component is multiplied by $1+\lambda$ — directly what $\lambda$ buys. At Nyquist, $H(\pi)=\lambda(1-\alpha)/(1+\alpha)$, which for $\alpha$ near 1 is tiny (at $\alpha=0.98$, about $0.01\lambda$), so the fastest component passes with gain $\approx 1$. Low frequencies up by $1+\lambda$, high frequencies barely touched, with $\alpha$ setting where the rolloff sits and $\lambda$ the gain — exactly the asymmetric high-boost I wanted, with constant memory.

There is a debt to pay before I trust acting on $g$ instead of $u$, because the optimizer in the loop is AdamW, whose second-moment rescaling and decoupled decay look frequency-dependent. Write a generic first-order optimizer as a linear time-invariant system with scalar state $x$: $x(t)=Ax(t{-}1)+Bg(t)$, $u(t)=Cx(t)+Dg(t)$ (momentum is $A=\mu,B=1-\tau,C=-\eta,D=0$; Nesterov adds $D=-\eta$). In the frequency domain its input-to-output transfer function is $H_{io}(\omega)=BC/(1-Ae^{-i\omega})+D$. Run the filtered gradient through the *same* optimizer — same $A,B,C,D$, only the input changed — and the update ratio is $\hat U(\omega)/U(\omega) = [H_{io}\hat G]/[H_{io}G] = 1+H(\omega)$: the $H_{io}$ term cancels because it is linear and unchanged. So for any linear first-order optimizer, filtering the gradient by $h$ is identical to filtering the *update* by the same $h$ — my hypothesis about the slow component of the motion is faithfully served by acting on the gradient. The honest caveat is that AdamW's second-moment rescaling is not strictly linear-time-invariant, so the equivalence is exact only in the linearized reading; but the buffer EMA varies slowly relative to the per-step gradient, so the boost still lands on the slow part of the motion in practice, and this is the same AdamW the default rung already runs successfully — I am not changing it, only feeding it a low-frequency-boosted gradient.

That equivalence also tells me *where* the code goes: I can act on the gradients sitting in `p.grad` after `backward()` and before `step()` and get the same effect regardless of which optimizer consumes them — no custom optimizer object. That is exactly the contract `TrainHook.post_grad(step)` exposes. So the fill seeds a per-parameter EMA buffer from the first gradient it sees, then on every call decays each buffer by $\alpha$, mixes in $(1-\alpha)$ of the current `p.grad`, and adds $\lambda$ times the buffer back into `p.grad` in place. The default architecture and `WarmupAdamW(wd=1.0)` are left untouched, so the rung isolates the EMA hook. I set the gain modest and the decay long: $\lambda=2.0$ (low-frequency gain $1+\lambda=3$) and $\alpha=0.98$ (effective memory $\approx 1/(1-\alpha)\approx 50$ steps), which reaches down to genuinely slow structure without being so inert it stops tracking. One distinction from simply raising AdamW's $\beta_1$ is worth being precise about: momentum *consumes* the smoothed gradient as the update, whereas here I add the smoothed gradient as a *residual* on top of the raw gradient, $\hat g = g + \lambda\mu$, and only then hand it to AdamW, which still does its own EMA and rescaling. The mixing happens before the optimizer and is independent of it — that independence is what lets me leave the proven optimizer exactly as it is.

The falsifiable expectations against the default numbers are concrete. Weight decay's slow grind toward the flat solution is still present; I am now amplifying the very component of the trajectory that grind was producing, so every grok should arrive earlier. On `p=59`, the 456 seed that ran out of budget should now grok inside it, lifting `grok_rate` from 0.667 toward 1.0 and `test_accuracy` from 0.992 toward 1.0, with mean steps-to-grok falling well below 15167. On `p=97` and `p=113`, accuracies should stay at 1.0 while steps-to-grok drop below 2833 and 2333. If steps-to-grok do not move, or move up, the slow-gradient hypothesis is wrong on this task; if accuracies fall, $\lambda$ is drowning the fast component the model needs to fit the table.

```python
# EDITABLE region of custom_strategy.py — step 2: default model/optimizer + Grokfast-EMA hook
class GrokTransformer(nn.Module):
    """One-layer decoder-only transformer (same as Nanda 2023)."""

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512):
        super().__init__()
        self.p = p
        self.vocab_size = p + 1
        self.eq_token = p
        self.tok_embed = nn.Embedding(self.vocab_size, d_model)
        self.pos_embed = nn.Embedding(3, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp, bias=False),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model, bias=False),
        )
        self.unembed = nn.Linear(d_model, p, bias=False)
        self._init_baseline_weights(d_model)

    def _init_baseline_weights(self, d_model: int) -> None:
        hidden_std = 1.0 / math.sqrt(d_model)
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=hidden_std)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=hidden_std)
        nn.init.normal_(self.attn.in_proj_weight, mean=0.0, std=hidden_std)
        nn.init.normal_(self.attn.out_proj.weight, mean=0.0, std=hidden_std)
        nn.init.normal_(self.mlp[0].weight, mean=0.0, std=hidden_std)
        nn.init.normal_(self.mlp[2].weight, mean=0.0, std=hidden_std)
        nn.init.normal_(self.unembed.weight, mean=0.0, std=1.0 / math.sqrt(self.p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        eq = torch.full((B, 1), self.eq_token, dtype=torch.long, device=x.device)
        seq = torch.cat([x, eq], dim=1)
        pos = torch.arange(3, device=x.device).unsqueeze(0).expand(B, 3)
        h = self.tok_embed(seq) + self.pos_embed(pos)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = h + attn_out
        h = h + self.mlp(h)
        return self.unembed(h[:, -1, :])


def build_model(p: int, config: TaskConfig) -> nn.Module:
    return GrokTransformer(p=p, d_model=128, n_heads=4, d_mlp=512)


class WarmupAdamW(torch.optim.AdamW):
    """AdamW with Nanda's 10-step linear warmup to lr=1e-3."""

    def __init__(self, params, *, warmup_steps: int = 10, **kwargs):
        super().__init__(params, **kwargs)
        self.warmup_steps = warmup_steps
        self._step_count_for_warmup = 0
        self._base_lrs = [group["lr"] for group in self.param_groups]

    def step(self, closure: Callable | None = None):
        self._step_count_for_warmup += 1
        scale = min(self._step_count_for_warmup / self.warmup_steps, 1.0)
        for group, base_lr in zip(self.param_groups, self._base_lrs):
            group["lr"] = base_lr * scale
        return super().step(closure)


def make_optimizer(model: nn.Module, config: TaskConfig) -> torch.optim.Optimizer:
    """AdamW lr=1e-3 wd=1.0 (Grokfast keeps the baseline optimizer untouched)."""
    return WarmupAdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        weight_decay=1.0,
    )


class TrainHook:
    """Grokfast-EMA gradient filter.

    Maintains an EMA of each parameter's gradient and adds `lamb * EMA` to
    the raw gradient before opt.step().
    """

    def __init__(self, model: nn.Module, config: TaskConfig):
        self.model = model
        self.config = config
        self.alpha = 0.98
        self.lamb = 2.0
        self.grads: dict[str, torch.Tensor] = {}

    def post_grad(self, step: int) -> None:
        for name, param in self.model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            if name not in self.grads:
                self.grads[name] = param.grad.detach().clone()
            else:
                self.grads[name].mul_(self.alpha).add_(param.grad.detach(), alpha=1.0 - self.alpha)
            param.grad.add_(self.grads[name], alpha=self.lamb)
```
