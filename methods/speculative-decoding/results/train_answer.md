Generating $K$ tokens from a large autoregressive Transformer $M_p$ costs $K$ strictly serial forward passes: to sample $x_t$ the model must already know $x_{<t}$, so pass $t+1$ cannot begin until pass $t$ has emitted its token. As models grew to billions of parameters this serial latency became the dominant cost of deployment, and we want to cut the *number of serial passes* through the large model — but under three constraints that rule out almost everything. First, the output distribution must stay *exactly* intact: a production system validated on a specific checkpoint cannot silently start emitting different tokens. Second, no retraining and no architecture surgery, so the method must run on off-the-shelf models. Third, the gain must show up as real wall-clock latency on real accelerators, not a tidier asymptotic count. The obvious families all fail one of these. Uniform-cost reduction — distilling, quantizing, sparsifying, swapping in cheaper attention — makes every pass cheaper but produces a *different* model with a different output distribution. Adaptive-computation and early-exit methods spend compute proportional to instance difficulty by taking a heuristic shortcut, which means a changed architecture or training procedure and, fatally, a token the large model might not have emitted — the distribution drifts. Blockwise parallel decoding and shallow aggressive decoding guess several tokens at once and verify, which is the right spirit, but they are greedy-only (no stochastic sampling), need custom trained heads or copy-style tasks, and target downstream quality rather than the exact distribution. Nothing simultaneously keeps the exact distribution, supports general stochastic sampling, and works on unmodified checkpoints.

To find the waste I look at where the time goes inside one decode pass rather than counting passes. At batch size 1 a forward pass reads every weight and the whole KV cache out of high-bandwidth memory, multiplies, and produces one token's logits — an arithmetic intensity far below what the accelerator can do, so the wall-time is set by *bytes moved*, not FLOPs, and the compute units sit idle. That is the opening, because the model reads its entire weight set whether it scores one token or fifty: given a block $x_1,\dots,x_K$ already in hand, a single causally-masked pass returns $p(\cdot\mid x_{<1}),\dots,p(\cdot\mid x_{<K})$ for all positions at once, and being memory-bound, that block pass costs almost the same wall-time as a single-token pass. So *verifying* a block of guessed tokens is cheap (one pass) while *generating* it is expensive ($K$ passes) only because generation discovers tokens one at a time. The leverage is that a small, fast model gets most of the easy positions right, so I let it run ahead.

I propose speculative decoding. A cheap approximation model $M_q$ drafts the next $\gamma$ tokens autoregressively — $\gamma$ cheap passes — and then the large target model $M_p$ runs *once* over the whole drafted block to score all $\gamma+1$ positions in parallel; an accept/reject rule called speculative sampling decides how many guesses to keep and provably emits tokens distributed *exactly* as a draw from $M_p$ alone. The technical heart is generalizing speculative execution to the stochastic setting: "what $M_p$ would produce" is not one right token but a distribution $p(\cdot)$, so I need a rule that accepts $M_q$'s guess often, costs nothing extra to repair on rejection, and leaves a sample that is exactly $p$-distributed. Strip it to one position: I hold the small model's $q(x)$ and the big model's $p(x)$ over the same vocabulary, I have drawn $x\sim q$, and I want a sample from $p$. Classical rejection sampling accepts $x\sim q$ with probability $p(x)/(M\,q(x))$ for $M=\max_x p(x)/q(x)$, but that global maximum crushes the accept rate toward $1/M$ — tiny whenever any token has $p\gg q$, which with a small $q$ and a large vocabulary is always — and every rejection retries from scratch, throwing away the very parallel pass I wanted to reuse. Importance sampling produces *weighted* samples, not unweighted draws, so it cannot emit a token at all. Both are dead ends.

The rule that works looks only at each token's own $p$ and $q$, with no global constant. Where $q(x)\le p(x)$ the small model is cautious — it never over-proposes — so I accept those draws outright. Where $q(x)>p(x)$ it over-proposes, so I keep only a $p(x)/q(x)$ fraction. Combining the cases, I accept the guess $x$ with probability
$$\min\!\Big(1,\ \frac{p(x)}{q(x)}\Big).$$
On rejection I must repair the shortfall exactly. The probability that I drew $x$ from $q$ and accepted it is $q(x)\min(1,p(x)/q(x))=\min(p(x),q(x))$, so after the accept step token $x$ has received $\min(p(x),q(x))$ but is owed $p(x)$; the deficit is $\max(0,p(x)-q(x))$. So when I reject I resample from the residual distribution
$$p'(x)=\mathrm{norm}\big(\max(0,\,p(x)-q(x))\big),$$
the part of $p$ that $q$ failed to cover — a quick fix-up needing no model re-run. The exactness follows by accounting for both branches. The accept branch contributes $\min(q(x'),p(x'))$. For the reject branch, using $\max(0,p-q)=p-\min(p,q)$, the residual's normalizer is $Z=\sum_x\big(p(x)-\min(p(x),q(x))\big)=1-\sum_x\min(p(x),q(x))$, while the overall accept probability is $\beta=\sum_x q(x)\min(1,p(x)/q(x))=\sum_x\min(p(x),q(x))$, so $Z=1-\beta$. The reject probability is exactly $1-\beta$ too, so it cancels the normalizer: $P(\text{reject},x')=(1-\beta)\,p'(x')=p(x')-\min(q(x'),p(x'))$. Adding the branches,
$$P(x=x')=\min(p(x'),q(x'))+\big(p(x')-\min(p(x'),q(x'))\big)=p(x').$$
Nothing was assumed about $q$ — no support coverage, no ratio bound — so *any* $q$ yields the exact target distribution; a poor $q$ merely rejects more often. That is the non-negotiable guarantee.

Lifting this to the block of $\gamma$ guesses, position $i$ was drafted from $q_i=q(\cdot\mid\text{prefix},x_1,\dots,x_{i-1})$ and the one parallel target pass gives $p_1,\dots,p_{\gamma+1}$ — one distribution per position plus one extra past the last guess. I apply the single-position test in order, accepting $x_i$ iff $r_i\le p_i(x_i)/q_i(x_i)$, and I *must stop at the first rejection*: the downstream $p_{n+2},\dots$ were conditioned on the rejected token $x_{n+1}$, which I am about to replace, so they are invalid. At the first rejected position I emit the residual token $p'=\mathrm{norm}(\max(0,p_{n+1}-q_{n+1}))$; if every guess is accepted I take a free bonus token straight from the spare $p_{\gamma+1}$ already computed in the same pass. So each target pass emits between $1$ and $\gamma+1$ tokens, always at least one (the fix-up even when $x_1$ itself is rejected). Every sampling scheme — argmax, top-$k$, nucleus, temperature — is first folded into "sample from an adjusted categorical," so a single rule covers all of them.

The payoff is quantifiable. Writing $\beta=\sum_x\min(p,q)=1-D(p,q)$ where $D$ is total-variation distance, and $\alpha:=E(\beta)$ the expected acceptance rate, the i.i.d. assumption makes the number of accepted guesses geometric capped at $\gamma$, so
$$E[\#\text{tokens}]=\sum_{i=0}^{\gamma}\alpha^i=\frac{1-\alpha^{\gamma+1}}{1-\alpha},$$
which is $1$ at $\alpha\to0$ and $\gamma+1$ at $\alpha\to1$, as it must be. With $c$ the ratio of one $M_q$ run to one $M_p$ run, a step costs $T(c\gamma+1)$ and the wall-time speedup over standard decoding is
$$\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(c\gamma+1)}.$$
At $\gamma=1$ this is $(1+\alpha)/(1+c)$, so any $\alpha>c$ already wins, and $\alpha>c$ is necessary for larger $\gamma$ too; the integer $\gamma$ is swept numerically. This is a latency-for-compute trade — total arithmetic can rise by $(1-\alpha)(\gamma\hat c+\gamma+1)/(1-\alpha^{\gamma+1})$ when guesses are rejected — but it is exactly worth it in the memory-bound regime, where the compute was idle and the genuine win is that target weights and KV are read once per step instead of once per token, shrinking memory traffic by the same $(1-\alpha^{\gamma+1})/(1-\alpha)$ factor. And $M_q$ need not be a neural net: a bigram table or a copy-from-context heuristic is a valid guesser with $c\approx0$, since the proof never cared what $q$ was. Two extensions follow the same logic: a lenience knob compares $l\,q(x)$ to $p(x)$ to raise acceptance at the cost of exactness, and beam search uses a wider helper beam $u\ge w$, accepting while $\mathrm{top}_w(M_p)\subseteq\mathrm{top}_u(M_q)$.

```python
import torch
from torch.nn import functional as F


def top_k_top_p_filter(logits, top_k=0, top_p=0.0):
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < kth[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        logits[remove.scatter(1, sorted_idx, remove)] = float("-inf")
    return logits


def norm_logits(logits, temperature, top_k, top_p):
    # Standardize any sampling scheme into a single categorical distribution.
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        probs.scatter_(1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
        return probs
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    return F.softmax(logits, dim=-1)


def sample(probs, num_samples=1):
    return torch.multinomial(probs, num_samples=num_samples)


@torch.no_grad()
def autoregressive_decoding(prefix, model, max_len, temperature=1.0, top_k=0, top_p=0.0):
    T = prefix.shape[1] + max_len
    while prefix.shape[1] < T:
        logits = model(prefix).logits
        probs = norm_logits(logits[:, -1, :], temperature, top_k, top_p)
        prefix = torch.cat((prefix, sample(probs)), dim=1)
    return prefix


@torch.no_grad()
def fast_decoding(prefix, helper_model, target_model, max_len, lookahead=4,
                  temperature=1.0, top_k=0, top_p=0.0):
    """Sample from target_model with the output distribution unchanged, using
    helper_model to propose lookahead tokens per target pass. Batch size 1."""
    assert prefix.shape[0] == 1
    seq_len = prefix.shape[1]
    T = seq_len + max_len

    while prefix.shape[1] < T:
        prefix_len = prefix.shape[1]

        # 1. Draft: the helper proposes lookahead tokens autoregressively.
        x = prefix
        for _ in range(lookahead):
            q_logits = helper_model(x).logits
            q_next = norm_logits(q_logits[:, -1, :], temperature, top_k, top_p)
            x = torch.cat((x, sample(q_next)), dim=1)

        # Recompute q_i for the drafted positions.
        q = helper_model(x).logits
        for i in range(q.shape[1]):
            q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p)

        # 2. Verify: one parallel target pass scores all lookahead+1 positions.
        p = target_model(x).logits
        for i in range(p.shape[1]):
            p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p)

        # 3. Accept the longest matching prefix; n = end index of valid prefix.
        is_all_accept = True
        n = prefix_len - 1
        for i in range(lookahead):
            drafted = x[:, prefix_len + i].unsqueeze(-1)
            p_i = p[:, prefix_len + i - 1, :].gather(1, drafted).squeeze(1)
            q_i = q[:, prefix_len + i - 1, :].gather(1, drafted).squeeze(1)
            accept_prob = torch.minimum(torch.ones_like(p_i), p_i / q_i.clamp_min(1e-12))
            if (torch.rand_like(accept_prob) <= accept_prob).item():
                n += 1
            else:
                # 4a. Rejection: resample from the residual at this position.
                residual = torch.clamp(p[:, n, :] - q[:, n, :], min=0.0)
                residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                t = sample(residual)
                is_all_accept = False
                break

        prefix = x[:, :n + 1]
        if is_all_accept:
            # 4b. All accepted: free bonus token from the extra target distribution.
            t = sample(p[:, -1, :])
        prefix = torch.cat((prefix, t), dim=1)

    return prefix[:, :T]
```
