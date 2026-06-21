The Switch aux loss did what it was meant to: it cut imbalance $2.2\times$ against the control, from $0.1286$ to $0.0587$, at essentially flat cross-entropy — proof that the $f \cdot P$ surrogate really does steer the router toward uniform. But the previous rung named its own suspicion precisely, and it stands: the counts $f_i$ were measured on the micro-batch, the small noisy slice one forward pass sees, and demanding uniformity of every slice punishes legitimate per-slice specialization. Some of the balancing pressure was therefore spent flattening structure I want to keep, which is why the cross-entropy could not come down as far as it might. The diagnosis points at exactly one knob, and I want to turn only that knob: the set of tokens over which the frequency $f_i$ is computed.

The granularity is the whole story. The property I actually care about is corpus-level — across all the data, every expert should pull roughly its fair share so that no expert dies and the capacity is used — and I emphatically do *not* care that one micro-batch of two dozen sequences is internally uniform. If a micro-batch happens to be all code, the right behavior is for the code experts to light up and the rest to stay quiet; specialization *is* per-slice skew that averages out to balance over the corpus. The micro-batch penalty conflates the code slice's benign skew with the pathological skew of collapse and penalizes both, enforcing the right constraint at the wrong scope. So I propose the **global-batch load-balancing loss**: keep the penalty form exactly as it was,
$$L_{LBL} = \alpha \cdot N \cdot \sum_i f_i \, P_i, \qquad \alpha = 10^{-2},$$
same coefficient, same multiply-by-$N$ that keeps the balanced optimum scale-free, same detached counts and differentiable probabilities — and change only that $f_i$ is now the fraction of tokens routed to expert $i$ across the **whole batch** rather than each micro-batch alone. The penalty now asks: across all this data, is the usage uniform? — and it is silent about whether any individual slice is uniform. A micro-batch is free to be as specialized as its content demands, as long as the specializations of different slices cover all the experts when summed. The router keeps its learned structure and still satisfies the balance constraint, which is exactly the freedom the micro-batch version denied it (Qiu et al. 2025). In this single-process reproduction there is one device and the training batch already *is* the global batch, so the contrast I make real is between the previous rung's penalty on four micro-splits and this rung's penalty on the full batch — a faithful stand-in for the per-device-versus-global distinction, even if the absolute gap is muted at a scale where the slices are already fairly representative of each other.

Alongside this rung sits a complementary lever worth running, even though it is not a loss at all: DeepSeek's auxiliary-loss-free scheme. It keeps a per-expert bias $b_i$ used *only* to break ties in the top-K selection — added to the routing scores for ranking, but excluded from the gate weights that actually combine the experts — and nudges it once per step,
$$b_i \leftarrow b_i + u \cdot \mathrm{sign}(\bar c - c_i), \qquad u = 10^{-3},$$
where $c_i$ is expert $i$'s recent load and $\bar c$ the mean, so overloaded experts get cooled and underloaded ones warmed. It carries no gradient by design; it is a slow control loop on the counts running beside the differentiable penalty. Pairing it with the global-batch loss is attractive because the two act on different surfaces: the loss gives the router a gradient about balance through $P$, while the bias adjusts the hard selection directly, catching imbalance the smooth gradient is slow to fix. I run the global-batch loss both alone and with this bias to see whether the count-level controller buys anything on top.

I expect imbalance in the same good band as the Switch loss — both are the same penalty form and both break collapse — with cross-entropy at least as low, because the global scope stops over-constraining individual slices. The honest caveat I carry into the next rung is this: the global-batch loss equalizes the *average* usage but has nothing special to say about the experts that have fallen *well below* their fair share. To the smooth $f \cdot P$ term, an expert at a tenth of its fair share and one at nine-tenths of it are both just small contributions to the sum, so the gradient that would resurrect a nearly-dead expert is weak exactly in the tail where rescue matters most. Balancing the mean is not the same as resurrecting the dying, and that under-utilized tail — left on the table by the global term, and not addressed by the count-only bias either — is the opening the next rung has to attack.

```python
import torch

def layer_f_P(probs, topi, N):
    counts = torch.bincount(topi.reshape(-1), minlength=N).float()
    f = counts / counts.sum()
    P = probs.mean(0)
    return f, P

def balance_loss_global(probs_list, topi_list, N, alpha=1e-2):
    """Global-batch LBL: alpha * N * sum_i f_i P_i, f computed over the GLOBAL batch."""
    total = 0.0
    for probs, topi in zip(probs_list, topi_list):
        f, P = layer_f_P(probs, topi, N)          # f over the full (global) batch
        total = total + N * (f.detach() * P).sum()
    return alpha * total / len(probs_list)

# Optional DeepSeek auxiliary-loss-free bias (selection-only, no aux gradient):
#   biased top-k uses (router_logits + b);  after each step, per layer:
#   f, _ = layer_f_P(probs, topi, N);  c = f * N;  cbar = c.mean()
#   b += u * torch.sign(cbar - c)         # u = 1e-3
```
