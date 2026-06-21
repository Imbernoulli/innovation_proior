NAdam came back where I predicted: val_loss 2.3231, essentially a vanilla-AdamW result, with the downstream accuracies at the low end of the field (arc_easy 55.18, hellaswag 32.75). Refining the momentum of a *diagonal* step leaves the step diagonal — and the cosine schedule anneals the learning rate late in training, exactly where the look-ahead's advantage is thinnest, so there was nothing structural to exploit. The diagnosis is now confirmed by the number: to move validation loss I have to change the *shape* of the update, not its momentum. The cheapest place to start is to ask what the diagonal step actually *is*.

Strip the moving averages away — set $\beta_1=\beta_2=0$, drop $\epsilon$ — and AdamW's update becomes $g/\sqrt{g^2} = \text{sign}(g)$. So AdamW, and the NAdam I just ran, is sign descent with smoothing: it normalizes each coordinate of the gradient *in isolation* to roughly $\pm1$, and the second moment merely softens that normalization. The family already throws away gradient magnitude in the limit. The question, then, is whether to throw it away *deliberately and uniformly* rather than through a noisy per-coordinate $\sqrt{v}$ estimate that costs a whole second-moment buffer. That reframing is the rung.

I propose Lion: step every coordinate by the same magnitude — the *sign* of a momentum-blended gradient — and drop Adam's second moment entirely, one buffer instead of two. Two arguments justify giving up adaptivity, and both matter here. First, the sign discards the gradient's magnitude, which injects a controlled, uniform noise into each update; injected update noise is a known regularizer, the same family as added gradient noise or sharpness-aware methods, nudging training toward flatter, more robust minima. On a 7.1B-token run at fixed budget, where NAdam's representation metrics came in depressed, a regularizer that pushes toward flat minima is exactly the lever I want. Second, one momentum buffer instead of two is genuinely cheaper at 355M parameters. The flip side I have to respect: signing a *noisy* gradient gives a bad direction, so this should like large batches — and the substrate's effective batch (micro-batch 96 × grad-accum 6 × 2 GPUs) is large, so the precondition is met.

The piece I would never have hand-written is *how* the momentum enters. A single EMA signed — signSGD-with-momentum — ties what the optimizer remembers to how it steps, because one constant governs both. Lion separates them. The step is the sign of a blend that puts real weight on the *current* gradient,

$$\theta_t = \theta_{t-1} - \eta\,\text{sign}\big(\beta_1 m_{t-1} + (1-\beta_1) g_t\big),$$

so the move reacts to fresh information; but the buffer itself updates on its own, slower constant,

$$m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t,$$

a longer memory of gradient history. Two constants doing two jobs: a long stable history ($\beta_2$) and a recency-weighted step ($\beta_1$). This decoupling is exactly what NAdam lacked — NAdam mixes the fresh moment with the gradient for the step but never separates the *tracking* rate from the *application* rate — and it is load-bearing, not cosmetic: the degenerate single-EMA version underperforms at both $\beta=0.9$ and $\beta=0.99$, so both the long memory and the recency step are needed.

I should be honest about the rule's provenance, because it is not a hand-derivation like Nesterov-into-Adam. It is the surviving point of an evolutionary program search over update rules, seeded with AdamW itself, mutated by inserting/deleting/editing statements over arrays with AdamW's two-buffer memory signature, and selected by a proxy task with meta-validation to survive the proxy-to-scale gap. The raw winner was a mess — a `clip`, an `arcsin`, two odd-constant interpolations, a chain computing $m \cdot m \to \sqrt{\cdot} \to m/\sqrt{m\cdot m}$, and a stray `cosh` — but it simplifies. The trailing `cosh` is dead (overwritten next iteration); the `clip`/`arcsin` ablate away with no quality loss; and $\sqrt{m^2}$ is just $|m|$, so $m/|m| = \text{sign}(m)$ collapses three statements to one. The two interpolations compose into a single $\approx0.99$ EMA feeding the signed step, dropping below Adam's two tracked buffers. What is left is the single-buffer, sign-of-blended-momentum rule above — reassuring, because it means an outward search from AdamW actually converges to this structure rather than my imposing it.

I have to be careful that the baseline I am editing is *this task's* Lion, not the canonical recipe, because the harness fixes choices the standalone method would set differently. Standalone Lion uses betas $(0.9, 0.99)$ — a $\sim$10× longer buffer memory — and a learning rate $\sim$0.1× AdamW's with weight decay scaled up $\sim$10× to hold the effective decoupled decay $\eta\lambda$ constant against the larger sign-update norm. This task does none of that. It passes the harness's own betas $(0.9, 0.95)$ straight through, so $\beta_2 = 0.95$ here — a much *shorter* buffer memory than the method's 0.99, the long-horizon stability arm turned down to whatever AdamW was using. It scales the learning rate by **0.3**, not 0.1 — a more aggressive step, absorbing the sign update's large norm while the rest of the schedule rides the same cosine shape. And it does *not* raise the weight decay: it reuses the substrate's $\lambda=0.1$ on the 2D params only (1D at 0.0), applied as a decoupled shrink *before* the sign step ($p \leftarrow p(1-\eta\lambda)$, then $p \leftarrow p - \eta\,\text{sign}(c_t)$). The 1D-vs-2D grouping is the substrate's default, unchanged. The consequence I carry forward: with $\beta_2$ only 0.95 the buffer's memory is short — about a ten-step horizon rather than the $\sim$hundred-step horizon 0.99 intends — so the "remember"/"step" decoupling exists (this is genuinely more than signSGD-with-momentum) but its long-horizon arm is weaker than the method wants. The 0.3 factor is baked into the constructor; since the cosine schedule and grad-clip are left as the substrate sets them, I set no `CONFIG_OVERRIDES`.

So the delta from NAdam is the first real change of geometry on the ladder: stop scaling each coordinate by its own $\sqrt{v}$ and step every coordinate by the same magnitude, with momentum split across two constants and a single buffer. The regularizing noise of the sign, plus the large reliable batch that makes the sign trustworthy, should pull val_loss *meaningfully* below NAdam's 2.3231, with arc_easy and hellaswag — depressed at 55.18 and 32.75 and tracking representation quality — moving most. What I am *not* expecting is the ceiling: the sign step still treats a weight matrix as a bag of independent scalars, equalizing entries rather than directions, blind to the matrix's operator structure. That residual blindness is the next rung's target.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — step 2: Lion (sign-momentum), task config
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        class Lion(torch.optim.Optimizer):
            """Lion optimizer — sign-based updates with EMA momentum."""
            def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
                defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
                super().__init__(params, defaults)
            @torch.no_grad()
            def step(self):
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        grad = p.grad
                        state = self.state[p]
                        if len(state) == 0:
                            state['exp_avg'] = torch.zeros_like(p)
                        exp_avg = state['exp_avg']
                        beta1, beta2 = group['betas']
                        # Weight decay first (decoupled, before update)
                        if group['weight_decay'] != 0:
                            p.mul_(1 - group['lr'] * group['weight_decay'])
                        update = exp_avg * beta1 + grad * (1 - beta1)
                        p.add_(torch.sign(update), alpha=-group['lr'])
                        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        optimizer = Lion(optim_groups, lr=learning_rate * 0.3, betas=betas)
        print("using Lion optimizer")
        return optimizer
```
