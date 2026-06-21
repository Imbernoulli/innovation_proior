We want to distill a large, well-trained teacher LLM into a smaller student in the cheapest possible way: run the teacher once over the entire pre-training corpus, write its output to disk, and then train as many students as we like against that cache without ever touching the teacher again. The teacher runs on whatever cheap compute is available, the students train on small clusters with no teacher in memory, and dozens of design ablations stop paying for the teacher's forward pass each time. The only thing standing in the way is the size of what must be cached. Distribution-matching distillation wants the teacher's full probability vector over the vocabulary at every training token, and modern vocabularies run to roughly $100\mathrm{k}$ entries; at one byte per probability, a trillion tokens of soft targets is on the order of $100$ PB. That is not expensive, it is impossible. So the full distribution cannot be stored; each token's teacher distribution has to be summarized *sparsely* — a handful of numbers, ideally about ten — and a student trained on that summary must end up where it would have ended up trained on the full teacher: the same loss, the same calibration, the same downstream scores. The whole problem reduces to choosing that sparse per-token summary well.

To know what "well" means I have to look at the object the summary feeds. The teacher gives a target distribution $t$ over the vocab, the student produces $p = \mathrm{softmax}(x)$ from its logits $x$, and I fit $p$ to $t$ with the forward KL loss $L = \sum_i t_i \log(t_i / p_i)$. What actually trains the model is the gradient this puts on the logits, so I compute it once and keep it un-simplified. With the softmax Jacobian $\partial p_i / \partial x_j = p_i(\mathbf{1}\{i=j\} - p_j)$,
$$\frac{\partial L}{\partial x_j} = -\sum_i t_i \frac{1}{p_i}\frac{\partial p_i}{\partial x_j} = \Big(\sum_i t_i\Big) p_j - t_j.$$
I deliberately do *not* set $\sum_i t_i = 1$ yet, because the moment the target is truncated that sum stops being one and the leading factor comes back to bite me. When the target is a genuine distribution — the full teacher, or a one-hot label under cross-entropy — $\sum_i t_i = 1$ and the gradient collapses to the clean $p_j - t_j$, whose only zero is $p = t$. So full distillation drives the student to exactly the teacher distribution and produces beautifully calibrated students; that is the behavior a sparse summary has to reproduce. Full distillation is the ceiling, plain cross-entropy on the one-hot label (same gradient with $t = e_{\mathrm{label}}$, calibrated but discarding all the teacher's dark knowledge) is the floor, and I need a sparse thing that lands near the ceiling.

The summary everybody reaches for is top-$K$: keep the $K$ highest-probability teacher tokens, zero the rest. It even has a local justification. If I keep $K$ entries and renormalize them by their kept mass $a = \sum_{i\in K} t_i$, the $L_1$ error to the true $t$ works out to $2(1-a)$, minimized by taking the largest probabilities — top-$K$ is the $L_1$-optimal $K$-subset. Yet the known failure mode is exactly what $L_1$ does not predict: as $K$ shrinks, students become over-confident and badly miscalibrated, and the cache needs far more kept tokens than the budget allows. The poison shows up in the un-simplified gradient. With a top-$K$ target the sum $\sum_i t_i$ is no longer one but $a < 1$, so the gradient is $a\,p_j - t_j$. For a token outside the kept set, $t_j = 0$ and the gradient $a\,p_j$ only vanishes at $p_j = 0$ — the loss actively drives every tail token, including rare ground-truth tokens, to zero. For a kept token it vanishes at $p_j = t_j / a$, the teacher probability inflated by $1/a > 1$. So at the optimum the student is over-confident on the head and under-confident to the point of zero on everything else, and the smaller $a$ is, the worse the inflation. The $L_1$ story was about reconstructing one token's distribution; it said nothing about what the KL gradient does to a head-only target.

Curing the up-scaling directly is possible but not enough. The ghost-token trick adds one bucket carrying the residual mass on both sides — teacher $1 - \sum_K t$ and student $1 - \sum_K p$ — and takes KL over the $K$ kept tokens plus that bucket. Working through its gradient, on the kept tokens it adds $(1 - \sum_K t)\,p_j$ to $a\,p_j - t_j$ and the two recombine to exactly $p_j - t_j$, restoring full distillation on the head; off the head it leaves a residual correction $\big(\sum_K(t_i - p_i)/(1 - \sum_K p_i)\big)p_j$ proportional to the student's own tail mass. So it fixes the in-$K$ inflation and gets the total tail mass right — but it collapses the entire tail into one undifferentiated bucket, with no per-token tail signal at all, and the tail is exactly where a Zipfian teacher carries real information about rare tokens. Label smoothing (spread the residual uniformly) and the naive fix (dump it on the ground-truth token) share the same original sin: they all begin by *deleting* the tail and then try to compensate. Truncation-plus-repair is a dead end.

So stop truncating, and name the defect precisely. Truncation reports a tail token's probability as $t^s_i = 0$ when its true value is $t_i > 0$; the summary is *biased*, and that bias is what the gradient propagated into the scaling and the miscalibration. What I want is a sparse summary whose *expectation*, over whatever randomness builds it, is the true teacher $t$, tail included. "Expectation" is the tell — this is an estimation problem, and the standard machine for it is estimating $\mathbb{E}_{x\sim t}[f(x)] = \sum_x f(x)\,t(x)$ by drawing from a proposal $q$ and reweighting: $s_q = \frac{1}{n}\sum_i f(x_i)\,t(x_i)/q(x_i)$ with $x_i \sim q$ is unbiased, $\mathbb{E}_q[s_q] = \mathbb{E}_t[f]$, for *any* proposal $q$ that is non-zero wherever $t f$ is. The line that names my disease is the condition: the estimate is unbiased only if $q > 0$ everywhere $t f \neq 0$, and top-$K$ is precisely a proposal that is zero on the entire tail. That violated support condition is why it is biased.

I propose Random Sampling Knowledge Distillation (RS-KD): build the sparse summary by sampling token ids from the teacher and caching the empirical counts. Concretely, draw $N$ token ids with replacement, each with probability $t$, let $c_i$ be the number of times token $i$ appears (so $\sum_i c_i = N$), and define the sparse target as the empirical distribution $t^s_i = c_i / N$. This is sparse by construction — at most $N$ non-zero entries, and far fewer distinct tokens in practice because the head repeats — and its support is whatever was *sampled*, so the tail is reached by chance rather than structurally deleted. Because the counts are multinomial with $N$ trials and per-trial probability $t_i$, $\mathbb{E}[c_i] = N t_i$, and therefore
$$\mathbb{E}[t^s_i] = \frac{\mathbb{E}[c_i]}{N} = t_i \quad \text{for every } i, \text{ head and tail alike}.$$
That is the unbiasedness top-$K$ could never have. What I actually care about is the gradient, so I check it: with a sparse target the logit gradient is $p_j - t^s_j$ *provided the target sums to one*, which $t^s = c/N$ does automatically since $\sum_i c_i/N = 1$. This is load-bearing — because $t^s$ is a proper distribution the $\sum_i t_i$ factor in $(\sum_i t_i)p_j - t_j$ is exactly one, so the $1/a$ inflation that broke top-$K$ simply cannot occur; sampling produces a stochastic target where truncation produced a sub-stochastic one. Taking expectations,
$$\mathbb{E}[\,p_j - t^s_j\,] = p_j - t_j,$$
the full-distillation gradient, token by token, tail included. A pre-training batch is enormous, so the noisy per-token targets average out and the student is driven by essentially the full-KD signal while I store a dozen numbers per token.

The loss is cheaper than it looks. With $t^s_i = c_i/N$, the forward KL is $\sum_i (c_i/N)\log(c_i/N) - \sum_i (c_i/N)\log p_i$; the first term is fixed by the cached target and drops out of the student's gradient, leaving $-\frac{1}{N}\sum_i c_i \log p_i$, which is exactly the average cross-entropy of the student against the $N$ sampled ids. Distillation here is just cross-entropy against tokens drawn from the teacher instead of against the one ground-truth token, so no probability vector or explicit KL is even needed.

Sampling from $t$ is one choice of proposal; importance sampling says any supported $q$ is unbiased after reweighting, and the reason to consider others is variance, since a noisy target gives a noisy gradient. The variance-minimizing proposal is the classic $q^* \propto t\,|f|$, which reaches zero variance with one sample for a non-negative integrand — but it needs the very normalizer I am trying to compute, so it is a guidepost, not a recipe. The usable knob is a tempered teacher $q \propto t^\tau$. At $\tau < 1$ the proposal flattens toward uniform, spreading draws over more tail tokens with much noisier weights; at $\tau > 1$ it sharpens toward the head and under-samples the tail, drifting back toward the truncation pathology; at $\tau = 0$ it is uniform over the whole vocabulary, still unbiased but wildly noisy against a peaked teacher. For general $\tau \neq 1$ each draw of token $i$ contributes $(t_i/q_i)/N$, carrying the importance weight. But at $\tau = 1$ the proposal *is* $t$, the likelihood ratio $t_i/q_i = 1$ for every sample, the weight collapses to a bare $1/N$, and $t^s_i$ reduces to the plain count fraction $c_i/N$ — no ratios to track, no heavy-tailed weights to wreck the variance. Numerically the estimator variance is flat across roughly $\tau \in [0.8, 1.2]$, so there is no strong reason to leave $\tau = 1$, which I ship for its simplicity.

The remaining choices follow from the same accounting. The number of rounds $N$ sets both fidelity and storage, since I cache one entry per *unique* sampled token; more draws lower the variance but accumulate more unique tokens, and that accumulation is sublinear (coupon-collector / power-law) because head tokens get redrawn while new tail tokens appear only occasionally. About $N = 50$ rounds yields on the order of $12$ distinct tokens — the storage scale I wanted, with many averaging rounds. There is a free quantization bonus: at $N = 50$ every probability is $x/50$ for an integer $x \le 50 < 2^7$, so it stores *exactly* in $7$ bits by keeping the numerator, with $17$ bits for a $V \approx 100\mathrm{k}$ vocab id, totaling $24$ bits = $3$ bytes per unique token and no quantization error below $N = 128$. Forward KL is the loss the whole argument hinges on — its gradient $(\sum t)p_j - t_j$ is what makes an unbiased target give an unbiased gradient, and it is mean-seeking, spreading the student to cover the teacher's support, unlike mode-seeking reverse KL which would abandon the tail. One offline correctness trap: the teacher's distribution at a position depends on its prefix context, so the teacher (caching) and student (training) must see identical packing and shuffling, or the cached targets correspond to the wrong context and quietly become garbage. The whole method, then, is: sample the teacher, train on the counts.

```python
import torch
import torch.nn.functional as F


def downsample_random_sampling(teacher_probs, N=50):
    """RS-KD downsampler (tau = 1): sample N token ids ~ teacher, accumulate counts/N.
    Unbiased estimate of the teacher: E[c_i / N] = t_i for every token i."""
    sampled_idx = torch.multinomial(teacher_probs, N, replacement=True)      # [B, N]
    values = torch.full((teacher_probs.size(0), N), 1.0 / N,                 # each draw -> 1/N
                        device=teacher_probs.device, dtype=teacher_probs.dtype)
    sparse_target = torch.zeros_like(teacher_probs)                          # [B, V]
    sparse_target.scatter_add_(1, sampled_idx, values)                       # accumulate counts/N
    return sparse_target                                                     # sums to 1, <= N nonzeros


def distillation_loss(student_logits, sparse_target):
    """Forward KL( sparse_target || student )."""
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(student_log_probs, sparse_target, reduction="batchmean")
```

As a single distillation-loss body that takes student/teacher logits over the same tokens, with `labels` marking completion positions (`-100` on padding/prompt):

```python
import torch
import torch.nn.functional as F


def compute_distill_loss(student_logits, teacher_logits, labels=None,
                         N=50, reduction="batchmean"):
    # RS-KD: sample N ids ~ teacher per position, train on the empirical counts/N target.
    # tau = 1 -> proposal is the teacher, importance ratio t_i/q_i == 1, weight 1/N per draw.
    B, T, V = student_logits.shape
    teacher_probs = F.softmax(teacher_logits, dim=-1).reshape(B * T, V)

    sampled_idx = torch.multinomial(teacher_probs, N, replacement=True)      # [B*T, N]
    sparse_target = torch.zeros_like(teacher_probs)                          # [B*T, V]
    sparse_target.scatter_add_(
        1, sampled_idx,
        torch.full_like(sampled_idx, 1.0 / N, dtype=teacher_probs.dtype),    # each draw -> 1/N
    )                                                                        # t^s, sums to 1

    student_log_probs = F.log_softmax(student_logits, dim=-1).reshape(B * T, V)
    per_token = F.kl_div(student_log_probs, sparse_target,
                         reduction="none").sum(dim=-1)                       # [B*T]

    if labels is not None:                                                   # mask pad/prompt
        mask = labels.reshape(-1) != -100
        per_token = per_token[mask]
        denom = mask.sum().clamp_min(1)
    else:
        denom = torch.tensor(max(per_token.numel(), 1),
                             device=per_token.device, dtype=per_token.dtype)

    if reduction == "batchmean":
        return per_token.sum() / denom
    elif reduction == "sum":
        return per_token.sum()
    elif reduction == "mean":
        return per_token.mean()
    return per_token
```
