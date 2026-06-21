Stochastic depth shaved another $\sim 5\%$ of wall-clock off the depth axis at accuracy held near baseline, and the whole stack is now in place: a corrected optimizer, anti-aliased downsamples, softened targets, cheaper early steps, channel attention, averaged weights, and rows, columns, and whole blocks dropped for speed. The throughput levers have done their job — each step is dramatically cheaper than the floor recipe's. But a tension keeps recurring: every time I cash in accuracy headroom for speed (progressive resizing, ColOut, stochastic depth), I must refill it with a quality lever (BlurPool, label smoothing, SE, EMA) just to stay above 76.6%. The binding constraint now is *quality margin* — if I had more accuracy in hand, I could shorten the schedule even more aggressively and still clear the bar. So the last and strongest move should buy the most generalization I can get, even at a real throughput cost, because that margin is what lets the rest of the stack run leaner. And I want a sharper question than "how do I regularize more": I already have a pile of penalty-style regularizers hitting diminishing returns, so stacking another dropout-flavored term will not move much. I want to change *what kind of minimum* SGD finds. Two models can have identical training loss but very different test loss, and the distinction that keeps coming up is *sharpness*: a minimum at the bottom of a narrow, steep ravine versus one in a wide, flat basin. Flatness matters because the test loss surface is a slightly shifted version of the train surface (finite data, train→test distribution shift). At the bottom of a sharp ravine, a small shift moves the bottom out from under me and the loss shoots up — the sharp minimum is fragile to exactly the perturbation that separates train from test. A wide flat minimum has low loss over a whole neighborhood, so even after the surface shifts the loss stays low. Plain SGD and all my current methods minimize the *point-wise* training loss $L(\theta)$ — nothing in that objective cares whether the neighborhood around $\theta$ is flat or sharp, so SGD will happily descend into a sharp ravine. The information about sharpness lives in the *neighborhood* of $\theta$, and the point-wise loss throws it away.

The method I propose is **Sharpness-Aware Minimization (SAM)**. The reframing is to put the neighborhood into the objective: instead of minimizing $L(\theta)$, minimize the worst-case loss in a small ball around $\theta$,

$$\min_\theta\; \max_{\|\varepsilon\| \le \rho}\; L(\theta + \varepsilon).$$

Read carefully, this asks for a $\theta$ such that even the worst nearby point within radius $\rho$ has low loss. A sharp minimum fails — there is a nearby point just up the ravine wall with much higher loss, so the inner max is large — while a flat minimum passes, because every nearby point is also low. Minimizing this worst-case-over-a-neighborhood objective *is* seeking flatness, with $\rho$ the knob for how big a ball I demand low loss over. The inner max has to be made tractable, since I cannot search the whole ball each step. For small $\rho$, take a first-order Taylor expansion $L(\theta + \varepsilon) \approx L(\theta) + \varepsilon^\top \nabla L(\theta)$; maximizing the linear term over $\|\varepsilon\| \le \rho$ is solved by pointing $\varepsilon$ straight along the gradient and pushing to the ball's edge:

$$\varepsilon^* \approx \rho\,\frac{\nabla L(\theta)}{\|\nabla L(\theta)\|}.$$

So the worst nearby point is approximately $\theta + \varepsilon^*$ — climb the local hill a distance $\rho$ in the *normalized* gradient direction. The SAM objective's gradient is then, to this order, just $\nabla L$ evaluated *at that perturbed point* $\theta + \varepsilon^*$, not at $\theta$. That is the whole trick: compute where the loss is locally worst nearby and take your descent step using the gradient *there*, which lowers the loss at the worst nearby point and drags the whole neighborhood down — flattening.

Concretely each update is a two-gradient procedure. First, compute the gradient at $\theta$ (an ordinary forward-backward), use it to form $\varepsilon^* = \rho\,g/\|g\|$, and *temporarily* move the weights to $\theta + \varepsilon^*$ — `first_step`, climbing to the local maximum. Second, compute the gradient *again* at $\theta + \varepsilon^*$ (a second forward-backward), then *undo* the perturbation back to $\theta$ and let the base optimizer — my `DecoupledSGDW` — step with that second gradient — `second_step`. So each SAM step is: ascend to the nearby worst point, read the gradient there, return, and descend with it. The cost is unavoidable: *two* forward-backward passes per step instead of one, which roughly halves training throughput. That is the price of flatness, and it is steep enough that I have to manage it or it wipes out everything the throughput levers bought. The knob I add is an `interval`: do not run the full two-gradient step every iteration — run it once every $k$ steps and take ordinary single-gradient base-optimizer steps in between, amortizing the doubled cost. Lower interval means more SAM, more flatness, slower; higher interval is cheaper but weaker — it is the throughput-vs-quality dial. Running SAM every $10$th step is the practical sweet spot, most of the generalization benefit at a fraction of the cost; when I raise the interval I scale $\rho$ up proportionally, since the periodic ascent steps need to be larger to have comparable effect, so `interval=10` pairs with a larger $\rho$ than the `interval=1` setting of $\rho=0.05$.

A few realities make it work in practice. The two-gradient structure needs the training step expressed as a *closure* the optimizer can call twice on demand, because the standard `optimizer.step()` is only invoked once. There is a distributed nicety too: the *first* of the two gradients does not need to be all-reduced across GPUs — it only locates $\varepsilon^*$ locally — so skipping that sync recovers some lost throughput and, by letting each GPU climb to a slightly different nearby point, adds a little helpful stochasticity. (SAM also has to coexist with mixed precision, which normally fights closures via the gradient scaler; that is handled in the trainer.) One scope check on when SAM helps: its whole mechanism is robustness to the train→test perturbation, which presumes the model sees the training data enough times to overfit — flatness matters when fitting a finite set repeatedly. On a task that sees each example once (effectively infinite data, like large-scale LM pretraining) there is no overfitting for flatness to protect against and SAM gives nothing, but multi-epoch ImageNet classification is exactly the overfitting regime it is built for, so this is the right task. SAM is the finale because it is the strongest generalization lever in the recipe — it costs throughput, but it buys the largest quality margin of anything in the stack, and that margin is precisely what lets every speed lever below it run at its aggressive setting and still clear the standard-schedule **76.6% top-1 on ImageNet**, reached not in the baseline's $\sim 3.5$ hours on 8×A100 but in a small fraction of that wall-clock and cost. The core is the SAM optimizer's two-step, interval-gated and driven by a forward-backward closure.

```python
@torch.no_grad()
def first_step(self):
    grad_norm = self._grad_norm()
    for group in self.param_groups:
        scale = group['rho'] / (grad_norm + group['epsilon'])
        for p in group['params']:
            if p.grad is None:
                continue
            e_w = p.grad * scale.to(p)
            p.add_(e_w)                       # climb to the local maximum "w + e(w)"
            self.state[p]['e_w'] = e_w

@torch.no_grad()
def second_step(self):
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None or 'e_w' not in self.state[p]:
                continue
            p.sub_(self.state[p]['e_w'])      # back to "w" from "w + e(w)"
    self.base_optimizer.step()                # the sharpness-aware update

@torch.no_grad()
def step(self, closure=None):
    assert closure is not None, 'SAM requires a closure (full forward-backward pass)'
    closure = torch.enable_grad()(closure)
    loss = None
    if (self.global_step + 1) % self.interval == 0:
        loss = closure(ddp_sync=False)        # gradient at (w), per-GPU, no sync
        if loss:
            self.first_step()                 # set weights to (w + e(w))
            loss_dict = {}
            if closure(loss_dict=loss_dict):  # gradient at (w + e(w))
                self.second_step()            # reset to (w) and step base optimizer
            else:
                self.sub_e_w()                # second pass failed: restore (w)
    else:
        loss = closure()                      # ordinary single-gradient step
        if loss:
            self.base_optimizer.step()
    self.global_step += 1
    return loss
```
