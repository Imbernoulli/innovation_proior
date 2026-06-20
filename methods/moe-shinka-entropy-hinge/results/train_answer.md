The global-batch load-balancing loss balanced the MoE's average expert usage while leaving the
cross-entropy intact, but it had a blind spot I could name precisely: the smooth product of token
fraction and router probability treats an expert at a tenth of its fair share the same as one at
nine-tenths, so the gradient that would resurrect a nearly-dead expert is weakest exactly in the
tail where I need it. Balancing the mean is not resurrecting the dying. I wanted a term that singles
out the under-utilized experts and pushes specifically on them, while being careful not to flatten
the specialization the global term protects.

The method is the load-balancing loss that ShinkaEvolve discovered by evolutionary search over the
loss function itself, scored on the very fitness I use — the negative of cross-entropy plus imbalance
— during real MoE pretraining. It keeps the global-batch term and adds an entropy-weighted
under-utilization hinge. The hinge is one-sided: it penalizes how far an expert sits below a small
per-expert usage floor, and contributes nothing for any expert at or above that floor, so it ignores
the healthy experts entirely and concentrates all its force on the cold tail. A bare hinge would be
dangerous, though, because it would fire on benign momentary under-use and flatten structure just as
the micro-batch loss did. So it is gated by the router's own peakedness: I take the entropy of the
router's expert distribution, normalize it by its maximum, take one minus that and offset it by a
half, giving a weight that is large when the router is peaked — the collapse regime, where a cold
expert really is starving — and small when the router is near-uniform and any under-use is just
noise. The hinge becomes a collapse-triggered rescue that waits mostly idle and surges only when
experts start dying.

I am candid that I did not derive the exact constants — the half-offset, the floor at roughly six
percent of the uniform share, the tenth weighting of the hinge relative to the global term — from
first principles; they are what the search converged to, and my account is the reconstruction of why
the discovered form makes sense, with each piece playing the role the mechanism needs. One
implementation point is essential: the hinge is written on the count, which is non-differentiable, so
as written its gradient is zero. The count can only select which experts are under the floor; the
gradient that actually raises a cold expert's usage must flow through the differentiable probability
of those selected experts, so I let the count decide membership in the under-used set and apply the
pressure to the probability of that set. Measured against the global-batch loss, this is the better
balance-versus-cross-entropy point — it cuts the imbalance markedly while holding the cross-entropy,
because the hinge is idle when the router is healthy and only acts in the dying tail. This is the
endpoint of the load-balancing-loss ladder; my run reproduces its mechanism at small scale, not the
paper's 64-expert, billions-of-tokens scale.

```python
import math
import torch

def layer_f_P(probs, topi, N):
    counts = torch.bincount(topi.reshape(-1), minlength=N).float()
    f = counts / counts.sum()
    P = probs.mean(0)
    return f, P


def balance_loss_shinka(probs_list, topi_list, N):
    """ShinkaEvolve's discovered load-balancing loss (arXiv:2509.19349, Eq. 1):

        L = N * (1/L) sum_l sum_i f_{l,i} P_{l,i}                  [global-batch LBL]
          + (0.1/L) sum_l s(P_l) * sum_i max(0, tau - f_{l,i})     [entropy-hinge]

        s(P_l) = 0.5 + (1 - H(P_l) / log N)   entropy-complement (peakedness) weight
        tau    = 0.064 / N                     per-expert usage floor

    The hinge fires only for under-floor experts, scaled by router peakedness; its
    gradient flows through the differentiable P of those experts."""
    L = len(probs_list)
    tau = 0.064 / N
    term1 = 0.0
    term2 = 0.0
    for probs, topi in zip(probs_list, topi_list):
        f, P = layer_f_P(probs, topi, N)
        term1 = term1 + N * (f.detach() * P).sum()              # global-batch LBL
        Pn = P / (P.sum() + 1e-9)
        H = -(Pn * (Pn + 1e-9).log()).sum()                    # router entropy
        s = 0.5 + (1.0 - H / math.log(N))                      # peakedness weight
        under = (tau - f.detach() > 0).float()                 # experts below the floor
        term2 = term2 + s.detach() * (under * torch.clamp(tau - P, min=0.0)).sum()
    return (term1 / L) + (0.1 / L) * term2
```
