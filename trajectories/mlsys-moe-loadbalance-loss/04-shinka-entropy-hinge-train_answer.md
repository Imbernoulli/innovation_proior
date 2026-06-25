The global-batch loss landed where I expected — imbalance $0.0561$, in the same good band as the Switch loss, with cross-entropy no worse and slightly better — and it ended by naming exactly what it leaves undone. The smooth $f \cdot P$ penalty equalizes the *average* usage but treats an expert at a tenth of its fair share the same as one at nine-tenths: both are just small terms in the sum, so the gradient that would resurrect a nearly-dead expert is weakest precisely in the tail where I need it most. Balancing the mean is not resurrecting the dying. I want a term that singles out the under-utilized experts and pushes specifically on them — and I have to be careful, because the obvious ways to do that are exactly the ways to wreck specialization.

A targeted under-use penalty should fire on an expert that has fallen below some floor of usage and leave alone every expert at or above its fair share. That shape is a one-sided **hinge**: penalize $\tau - f_i$ only when it is positive, zero otherwise. An expert comfortably above the floor contributes nothing; an expert below it contributes in proportion to how far below — the opposite of the smooth $f \cdot P$ term, which spreads its attention evenly across all experts. The floor $\tau$ should be a small fraction of the uniform share, because I do not want to demand that every expert hit $1/N$ exactly, only that none wither below a minimum. But a bare hinge is dangerous, and I have to see the danger before trusting it: if it fires hard whenever an expert dips below the floor, it will fire even when the router is healthy and the dip is benign finite-batch roughness, and pushing hard there flattens legitimate structure and raises the cross-entropy — exactly the micro-batch failure again. So the hinge needs a gate that asks *is this real collapse or just normal variation?* The router already carries a clean signal for that: the entropy $H$ of its probability distribution over experts. When the router is peaked — low entropy, mass on a few experts — that is the collapse regime, a cold expert really is being starved, and the rescue should be strong; when the router is near-uniform — high entropy — the system is healthy, momentary under-use is noise, and the hinge should barely fire.

I propose ShinkaEvolve's discovered loss: keep the global-batch term and add an entropy-weighted under-utilization hinge on top, both averaged over the $L$ layers,
$$L = N_E \cdot \tfrac{1}{L} \sum_\ell \sum_i f_{\ell,i}\, P_{\ell,i} \;+\; \tfrac{0.1}{L} \sum_\ell s(P_\ell) \sum_i \max\!\big(0,\ \tau - f_{\ell,i}\big),$$
with the peakedness weight and floor
$$s(P_\ell) = 0.5 + \Big(1 - \tfrac{H(P_\ell)}{\log N_E}\Big), \qquad \tau = \frac{0.064}{N_E}.$$
Each piece earns its place. The hinge $\max(0, \tau - f_{\ell,i})$ is nonzero *only* for experts below the floor $\tau$, so healthy experts are untouched and all the force concentrates on the cold tail. The weight $s(P_\ell)$ reads peakedness by normalizing the router entropy by its maximum $\log N_E$ (the entropy of the uniform distribution) so it runs in $[0,1]$, taking one minus that so peaked routers score near one and uniform routers near zero, then offsetting by a half so the weight is never quite zero and the floor is always at least gently enforced. That gives $s \approx 1.5$ when the router has collapsed and $s \approx 0.5$ when it is uniform — the modulation that turns the hinge into a collapse-triggered rescue that waits, mostly idle, and surges only when the router peaks and experts start dying. The global term keeps the $N_E$ scale that makes its balanced optimum scale-free; the hinge carries a small coefficient of one-tenth so it sharpens the cold tail without overwhelming the global balancing it sits on.

I should be candid about the provenance of the exact constants — the half-offset, the floor at $0.064/N_E$ (about six percent of $1/N_E$), the one-tenth hinge weight. I did not derive them from first principles; they are the artifacts of an *evolutionary search over the loss function itself*. ShinkaEvolve, the program-evolution search, evolved the Python of the balancing loss, scored by the very fitness used throughout this ladder, $r = -(L_{CE} + L_{imb})$, on real MoE pretraining, and this is the form it converged to. My reasoning is the reconstruction of *why* the discovered form makes sense, not the path that found it; what I can verify is that each piece plays the role the mechanism needs — the hinge for targeting, the floor for the threshold, the entropy complement for the collapse gate, the small coefficient for not overwhelming the global term.

One implementation point decides whether the hinge does anything at all. As written on the count $f_i$, the hinge is non-differentiable, so its gradient would be zero just like the bare count penalty of the first balancing rung. The count can only *select* which experts are under the floor; the gradient that actually raises a cold expert's usage must flow through the differentiable probability $P_i$ of those selected experts. So I let $f$ decide membership in the under-used set — `under = (tau - f.detach() > 0)` — and apply the differentiable pressure to the $P$ of that set, pushing the router's probability mass up on exactly the experts the floor test flagged as dying via $\mathrm{clamp}(\tau - P,\, 0)$. That is what makes the hinge a real training signal rather than a decorative zero, and the entropy weight $s$ is itself detached so it modulates without contributing its own gradient.

I expect this to hold the cross-entropy where the global-batch loss had it — the hinge is idle when the router is healthy, so it should not cost specialization — while pushing the imbalance below what the global term alone managed, because now the cold tail is actively rescued rather than merely averaged over. The best joint point of cross-entropy and imbalance should be here. And this is where the ladder stops among hand-reachable losses: this is the form a dedicated program-evolution search discovered against this exact fitness, on a $556\text{M}$-parameter, $82\text{M}$-active MoE over two-billion-plus FineWeb tokens; my run is a small reproduction of its mechanism, not its scale, and there is no further hand-designed rung above it that I have reason to believe does better on the joint objective. The one variant in the ladder with still lower imbalance, the global loss paired with DeepSeek's loss-free bias, wins through a non-gradient hard-count selection controller — a complementary mechanism that stacks with this loss rather than competing with it. Among load-balancing *losses* scored on the ShinkaEvolve fitness, the entropy-hinge is the endpoint.

```python
import math
import torch

def layer_f_P(probs, topi, N):
    counts = torch.bincount(topi.reshape(-1), minlength=N).float()
    f = counts / counts.sum()
    P = probs.mean(0)
    return f, P

def balance_loss_shinka(probs_list, topi_list, N):
    """ShinkaEvolve discovered loss:
       global-batch LBL + entropy-weighted under-utilization hinge."""
    L = len(probs_list)
    tau = 0.064 / N
    term1 = 0.0
    term2 = 0.0
    for probs, topi in zip(probs_list, topi_list):
        f, P = layer_f_P(probs, topi, N)
        term1 = term1 + N * (f.detach() * P).sum()                 # global-batch LBL
        Pn = P / (P.sum() + 1e-9)
        H = -(Pn * (Pn + 1e-9).log()).sum()                        # router entropy
        s = 0.5 + (1.0 - H / math.log(N))                          # entropy-complement weight
        under = (tau - f.detach() > 0).float()                     # experts below the floor
        # hinge gradient flows through P of the under-used experts (raise their mass)
        term2 = term2 + s.detach() * (under * torch.clamp(tau - P, min=0.0)).sum()
    return (term1 / L) + (0.1 / L) * term2
```
