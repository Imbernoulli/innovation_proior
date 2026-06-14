Let me start from what actually blocks me, which is storage, not modeling. I want to distill a big teacher LLM into a small student in the cheapest possible way: run the teacher once over the whole pre-training corpus, write its output to disk, and then train as many students as I like against that cache without ever touching the teacher again. The teacher runs on whatever cheap compute I can scrounge, the students train on small clusters, and I can do dozens of design ablations without paying the teacher's forward pass each time. That is the dream, and the only thing standing in the way is the size of what I have to cache. Distribution-matching distillation wants the teacher's full probability vector over the vocabulary at every single training token. For a modern model that vocabulary is on the order of a hundred thousand entries, and at, say, one byte per probability, a trillion tokens of soft targets is something like a hundred petabytes. That is not "expensive," that is "impossible." So I cannot store the full distribution. I have to store a *sparse* summary of each token's teacher distribution — a handful of numbers per token, ideally around ten — and I need a student trained on that summary to end up where it would have ended up trained on the full teacher: same loss, same calibration, same downstream scores. The whole problem reduces to choosing that sparse per-token summary well.

What does "well" even mean here? Let me be precise about the object I'm distilling into, because that tells me what a good summary has to preserve. The teacher gives me a target distribution `t` over the vocab, the student produces `p = softmax(x)` from its logits `x`, and I fit `p` to `t` with the forward KL loss `L = sum_i t_i log(t_i / p_i)`. The thing I actually care about is the gradient this puts on the student's logits, because that is the entire training signal. Let me just compute it, because I'll be staring at this quantity for the rest of the derivation. The softmax Jacobian is `dp_i/dx_j = p_i (1{i=j} - p_j)`. So
```
dL/dx_j = - sum_i t_i (1/p_i) dp_i/dx_j
        = - sum_i t_i (1/p_i) p_i (1{i=j} - p_j)
        = - sum_i t_i (1{i=j} - p_j)
        = sum_i t_i p_j - sum_i t_i 1{i=j}
        = (sum_i t_i) p_j - t_j.
```
There it is, and I want to hold onto the *un-simplified* form `dL/dx_j = (sum_i t_i) p_j - t_j` rather than immediately setting `sum_i t_i = 1`, because the moment I start truncating the target, that sum stops being one and the leading factor will come back to bite me. When the target is a genuine distribution — the full teacher, or a one-hot label under cross-entropy — `sum_i t_i = 1` and the gradient is the clean `dL/dx_j = p_j - t_j`, whose only zero is `p = t`. So full distillation drives the student to *exactly* the teacher distribution, and students trained this way come out beautifully calibrated, matching the teacher's confidence. That's the behavior I want a sparse summary to reproduce. Cross-entropy on the one-hot label is the same gradient with `t = e_{label}`, and it's also calibrated, but it throws away all of the teacher's soft structure — the relative mass on the wrong tokens, the "dark knowledge" that made distillation worth doing in the first place. So full distillation is my ceiling, CE is my floor, and I need a sparse thing that lands near the ceiling.

The obvious sparse summary is the one everybody reaches for: keep the top-`K` highest-probability tokens of the teacher and throw the rest away. Set `t^s_i = t_i` for `i` in the top-`K` set, zero otherwise. It feels right, and I can even justify it locally. Suppose I'm forced to keep only `K` entries and renormalize them to sum to one; which `K` minimize the error to the true `t`? Let me measure error in `L1`. Write the kept-and-renormalized distribution `v_i = t_i / a` for `i` in the kept set `K`, zero otherwise, with `a = sum_{i in K} t_i`. Then
```
L1 = sum_i |t_i - v_i|
   = sum_{i in K} |t_i - t_i/a| + sum_{i not in K} |t_i - 0|
   = (1/a - 1) sum_{i in K} t_i + (1 - sum_{i in K} t_i)
   = (1/a - 1) a + (1 - a)
   = (1 - a) + (1 - a)
   = 2(1 - a).
```
So the `L1` error is `2(1 - a)`, and to minimize it I should make `a`, the kept mass, as large as possible — which means keeping the `K` *largest* probabilities. Top-`K` is the `L1`-optimal `K`-subset. Great, the intuition checks out... and yet the known failure mode of top-`K` caches is exactly the thing the `L1` story does not predict: as `K` gets small, the student becomes over-confident and badly miscalibrated, and a sparse cache needs far more kept tokens than the storage budget wants. The teacher is calibrated, CE students are calibrated, full-distillation students are calibrated — but top-`K` students are not. Something about the truncation is poisoning the target in a way that "least `L1` error per token" completely fails to capture. Wall.

Let me go back to the gradient I was careful to keep un-simplified, because that's where the poison must be. With a top-`K` target, the sum `sum_i t_i` in the gradient is no longer one — it's `sum_{i in K} t_i = a < 1`. So the logit gradient under top-`K` KL is
```
dL/dx_j = (sum_{i in K} t_i) p_j - t_j = a * p_j - t_j.
```
Now ask where this vanishes. For a token `j` *outside* the kept set, `t_j` was set to zero, so the gradient is `a * p_j`, which is positive and only vanishes when `p_j = 0` — the loss is actively pushing every non-kept token's probability *to zero*, including rare ground-truth tokens that happened to fall outside the top-`K`. For a token `j` *inside* the kept set, the gradient `a p_j - t_j` vanishes at `p_j = t_j / a`, which is the teacher probability *scaled up* by `1/a > 1`. So at the optimum the student is over-confident on the kept tokens — each kept probability inflated by `1/a` — and under-confident, indeed driven to zero, on everything else. That's the miscalibration, derived. And it's not a small effect: if I keep little mass (`a` small), `1/a` is a big inflation factor, which is exactly why calibration gets worse as `K` shrinks. The `L1`-optimality argument was about reconstructing a single token's distribution; it said nothing about what the *KL gradient* does when fed a head-only target, and the gradient is what trains the model.

So the up-scaling is one disease. Can I cure it directly? The clean idea is to stop pretending the kept mass is the whole story and explicitly carry the leftover. Add one "ghost" bucket that holds the residual mass for *both* sides — teacher residual `t^s_ghost = 1 - sum_{i in K} t_i` and student residual `p_ghost = 1 - sum_{i in K} p_i` — and take the KL over the `K` kept tokens plus this one extra bucket. Let me check that this actually fixes the gradient. The ghost term is `L_ghost = (1 - sum_K t) log[(1 - sum_K t)/(1 - sum_K p)]`. Its derivative with respect to a logit `x_j` only enters through the student residual `1 - sum_K p`, and `d/dx_j (1 - sum_{i in K} p_i) = - sum_{i in K} dp_i/dx_j = - sum_{i in K} p_i(1{i=j} - p_j)`. So
```
dL_ghost/dx_j = - (1 - sum_K t)/(1 - sum_K p) * d/dx_j (1 - sum_K p)
             =   (1 - sum_K t)/(1 - sum_K p) * sum_{i in K} p_i(1{i=j} - p_j).
```
For `j` in `K`, `sum_{i in K} p_i 1{i=j} = p_j`, and `sum_{i in K} p_i p_j = p_j sum_K p`, so this is `(1 - sum_K t)/(1 - sum_K p) * (p_j - p_j sum_K p) = (1 - sum_K t)/(1 - sum_K p) * p_j (1 - sum_K p) = (1 - sum_K t) p_j`. Add that to the top-`K` KL gradient `a p_j - t_j = (sum_K t) p_j - t_j`, and the two `p_j` terms combine: `(sum_K t) p_j + (1 - sum_K t) p_j - t_j = p_j - t_j`. So on the kept tokens the ghost-token loss restores *exactly* the full-distillation gradient `p_j - t_j`. For `j` outside `K`, write `b = sum_K p`. The ghost gradient is `-((1 - a)/(1 - b)) b p_j`, while the top-`K` part is `a p_j`, so the combined gradient is
```
a p_j - ((1 - a)/(1 - b)) b p_j
= ((a(1 - b) - (1 - a)b)/(1 - b)) p_j
= ((a - b)/(1 - b)) p_j
= (sum_K (t_i - p_i) / (1 - sum_K p_i)) p_j.
```
That is not a per-token tail target from the teacher; it is a residual correction proportional to the student's own tail probabilities. So the ghost token genuinely cures the kept-token up-scaling, and it stops treating the whole tail as zero, but it still folds the entire tail into one undifferentiated bucket. The kept tokens get faithful supervision, the *total* tail mass is right, but within the tail there's no per-token signal at all — the student learns "the leftover mass is `1 - a`" and nothing about how that mass is distributed among the millions of tail tokens. And the tail is where rare ground-truth tokens live; a Zipf distribution has a long, heavy tail that carries real information. Patching the in-`K` mass isn't enough; I need actual supervision *in* the tail, per token, not a lumped bucket. Second wall, and it tells me truncation-plus-repair is a dead end: any scheme that starts by deleting the tail and then tries to compensate is fighting the structure of the problem.

Let me also rule out the lazy patches so I don't circle back to them. Spreading the residual mass uniformly over all tokens (label smoothing) does restore calibration, but a uniform tail is a terrible model of a Zipf tail — real token mass falls off like `1/i`, not flat — so it distorts the teacher. Dumping the residual onto the ground-truth token (a "naive fix") gives one tail token signal but still does not reconstruct the teacher's tail distribution. Every one of these is downstream of the same original sin: *truncation*. I keep the head and lose the tail, and then no amount of bucketing the lost mass recovers the per-token tail signal.

So stop truncating. What is the truncation, really? Truncating to top-`K` is choosing a sparse summary by *deterministically picking the largest entries*. The defining feature is that it assigns zero to everything in the tail. The defining symptom is that the summary is a *biased* picture of the teacher: a tail token `i` has true probability `t_i > 0`, but my summary always reports `t^s_i = 0`, so the expected value of my summary at `i` is `0`, not `t_i`. Bias. The summary systematically misrepresents the teacher, and that bias is exactly what the gradient `(sum_K t) p_j - t_j` propagated into the scaling and the calibration. What I want instead is a sparse summary `t^s` that is *unbiased* — one whose expectation, over whatever randomness I use to build it, is the true teacher distribution `t`, including on the tail.

The word "expectation" is the tell. I'm trying to summarize `t` by a sparse object whose expectation recovers `t`. That's an estimation problem, and there's a standard machine for it: estimating an expectation under one distribution while drawing from another. If I want `E_{x~t}[f(x)] = sum_x f(x) t(x)` and direct draws from `t` are awkward, I can draw from a proposal `q` and reweight: `s_q = (1/n) sum_i f(x_i) t(x_i)/q(x_i)` with `x_i ~ q`. The reweighting makes this unbiased for *any* proposal `q`, as long as `q(x) > 0` wherever `t(x) f(x) != 0`:
```
E_q[s_q] = E_q[ f(x) t(x)/q(x) ] = sum_x q(x) f(x) t(x)/q(x) = sum_x f(x) t(x) = E_t[f].
```
And here's the line that names my disease precisely: the estimate is unbiased *only* if `q` is non-zero everywhere `t f` is non-zero. The instant `q = 0` somewhere `t` isn't, the estimator is biased — and top-`K` is exactly a proposal that is *zero on the entire tail*. That's why top-`K` is biased; it violates the one condition that buys unbiasedness. The fix is to use a proposal that has support everywhere `t` does, so the tail is never structurally zeroed. The most natural such proposal, and the one that keeps everything trivially simple, is to sample from `t` itself.

Let me make that concrete. Forget the reweighting machinery for a second and just *sample tokens from the teacher distribution*. Draw `N` token ids, with replacement, each with probability `t`. Count how often each token came up: let `c_i` be the number of times token `i` appeared in the `N` draws, so `sum_i c_i = N`. Define my sparse summary as the empirical distribution `t^s_i = c_i / N`. This is sparse by construction — at most `N` of the `c_i` are non-zero, and in practice far fewer than `N` distinct tokens appear because the head repeats — and it has support determined by what was *sampled*, which can include tail tokens, never structurally excluded the way truncation excludes them. Is it unbiased? The counts `c_i` are multinomial with `N` trials and per-trial probability `t_i`, so `E[c_i] = N t_i`, and therefore
```
E[t^s_i] = E[c_i] / N = N t_i / N = t_i,   for every i, head and tail alike.
```
Unbiased on the nose, for all `i`. That's the property top-`K` could never have. The tail isn't deleted; it's sampled — a token with tiny `t_i` rarely shows up, but when it does it carries the right expected mass, and on average over the draws the summary reconstructs `t` exactly.

Now I should check that unbiasedness of the *target* actually buys me what I care about, which is the *gradient*. Recall the full-distillation gradient is `g_j = p_j - t_j` (here `sum t = 1`). With a sparse target `t^s` instead, the gradient at the logit is `g^s_j = p_j - t^s_j` — *provided* I feed the loss a target that sums to one, which `t^s = c/N` does, since `sum_i c_i/N = N/N = 1`. That last point matters: because the empirical distribution `t^s` is automatically normalized, the `sum t` factor in the general gradient `(sum t) p_j - t_j` is exactly one, so there is no up-scaling, no `1/a` inflation — the very pathology that broke top-`K` simply cannot occur, because sampling produces a proper distribution while truncation produced a sub-stochastic one. Take expectations of the sampled gradient:
```
E[g^s_j] = p_j - E[t^s_j] = p_j - t_j = g_j.
```
The sampled-target gradient equals the full-distillation gradient *in expectation*, token by token, tail included. So over a batch — and a pre-training batch is enormous — the noisy per-token sparse targets average out, and the student is driven by essentially the full-distillation gradient, while I only ever stored a dozen numbers per token. That is the whole method in one identity: an unbiased sparse target gives an unbiased gradient, so sampling from the teacher recovers full distillation in expectation at a tiny fraction of the storage.

Let me nail the loss form, because I want it cheap. With `t^s_i = c_i/N`, the forward KL between the sparse target and the student is `sum_i t^s_i log(t^s_i / p_i) = sum_i (c_i/N) log(c_i/N) - sum_i (c_i/N) log p_i`. The first term depends only on the (fixed, cached) target, not on the student, so it drops out of the gradient — and what's left, `- sum_i (c_i/N) log p_i`, is just `(1/N) sum_i c_i (- log p_i)`, i.e. the *average cross-entropy of the student against each of the `N` sampled token ids*. So I don't even need to materialize a probability vector and call a KL: training on the sampled target is identical to taking the `N` sampled ids and summing the student's cross-entropy over them. Cheap, and it makes the connection to ordinary cross-entropy training obvious — distillation here is "cross-entropy, but against tokens drawn from the teacher instead of against the one ground-truth token."

I've been sampling from `t` itself, which is one choice of proposal. Importance sampling tells me *any* supported proposal `q` is unbiased after reweighting, so should I pick a smarter `q`? The reason to bother is variance: unbiasedness is necessary but I also want the sparse estimate to be *low-variance*, because a noisy target gives a noisy gradient. The variance of the importance-sampling estimator is `Var[s_q] = Var[t f / q]/n`, and the variance-minimizing proposal is the classic `q*(x) ∝ t(x) |f(x)|` — for a non-negative integrand it even drives the variance to zero with a single sample (the catch being that `q*` needs the normalizer I'm trying to compute, so it's a guidepost, not a usable recipe). The lesson I take from it isn't "use `q*`," which I can't, but "the shape of `q` controls variance, and `q = t` is not necessarily the sweet spot." So let me consider a tunable family that brackets `t`: a tempered teacher, `q(x) ∝ t(x)^tau`, where `tau` is a sampling temperature. `tau = 1` is sampling from `t` directly. `tau < 1` flattens the proposal toward uniform, spreading draws over more tail tokens — broader coverage, but much noisier weights for a peaked teacher distribution. `tau > 1` sharpens toward the head — accurate on the head, but it under-samples the tail, drifting back toward the truncation pathology. And `tau = 0`? That's uniform sampling over the whole vocabulary, every token equally likely; the reweighting still makes it unbiased, but the variance is enormous because a hundred thousand tokens sampled uniformly is a wildly noisy estimate of a peaked teacher distribution. So there's a genuine bias-free, variance-controlled knob in `tau`, with a sweet spot somewhere around `tau = 1` where I'm sampling roughly proportionally to the teacher — accurate on the head, still covering the tail.

For general `tau != 1` I do have to carry the importance weight. Each occurrence of token `i` drawn from `q` contributes `(t_i / q_i) / N` to that coordinate. I do not self-normalize those weighted counts if the goal is the clean unbiased estimator; the expectation is what gives me back `t_i`. But at `tau = 1` the proposal *is* `t`, so the likelihood ratio `t_i / q_i = 1` for every sample — no reweighting at all, the weight of each draw is exactly `1/N`, and `t^s_i` collapses back to the bare count fraction `c_i / N`. That's a real simplification: `tau = 1` means "just sample token ids from the teacher and count," no ratios to track, no risk of a heavy-tailed weight distribution wrecking the variance. So unless a different `tau` buys a lot, `tau = 1` is the one to ship for its sheer simplicity. Numerically simulating the estimator variance across `tau`, the minimum sits in a band roughly `0.8` to `1.2`, and across that band the estimate quality is flat enough that there is no strong reason to leave `tau = 1`, where the weights vanish. A genuinely optimal proposal (via optimal experimental design, say) might shave variance further, but `tau = 1` gives the gradient identity I need with the simplest cache format. `tau = 1` it is.

Now, how many sampling rounds `N`? This sets both the fidelity and the storage, since I cache one entry per *unique* sampled token, and the storage is what I'm fighting. More draws mean lower variance (the empirical `c_i/N` concentrates on `t_i` as `N` grows) but more unique tokens to store. The relationship between `N` and the number of distinct tokens that show up isn't linear — it's a coupon-collector-flavored saturation, because the head tokens get redrawn over and over and only occasionally does a new tail token appear, so unique-token count grows much slower than `N` (empirically close to a power law in `N`). So I get to amortize: I can take many rounds to pin down the head accurately while only slowly accumulating tail entries. Around `N = 50` draws gives me on the order of `12` distinct tokens, which is the storage scale I was aiming for while still reducing sampling variance by averaging many draws. So `N ≈ 50`, about `12` unique tokens, is the operating point I'd target — a dozen entries per token rather than a large top-`K` cache, and astronomically below the full distribution.

There's a quiet bonus in the `c_i/N` representation for storage that I should not miss. If `N = 50`, then every sparse probability is of the form `x/50` for an integer `x` between `1` and `50`. There are at most `50` distinct values, which is fewer than `2^7 = 128`, so I can store each probability *exactly* in `7` bits by just storing the integer numerator `x` — no quantization error at all. Pair that with `17` bits for a vocab id (`log2` of a hundred-thousand vocab), and a unique token costs `24` bits, exactly `3` bytes, byte-aligned. With a dozen unique tokens per training token, that's about `36` bits... wait, `12 * 3` bytes `= 36` bytes per training token, which for `100`B tokens is on the order of a few terabytes — far below large top-`K` caches and petabytes for the full distribution. The sampling representation isn't just statistically clean; it quantizes for free. (Only if I push `N` past `128` would the numerator stop fitting in `7` bits and I'd switch to a ratio encoding of sorted probabilities.)

One thing I have to be careful about in the offline setting, because it's a correctness trap rather than a modeling choice. The teacher's distribution at a position depends on the *prefix context* — the tokens before it. When I cache offline, the teacher saw some particular packing and shuffling of documents; when I later train the student, it must see the *same* prefix context at each position, or the cached target corresponds to a different context than the one the student is conditioning on. If I pack documents to a fixed sequence length and don't mask across document boundaries (for efficiency), then a different shuffle seed between teacher inference and student training desynchronizes the contexts after the first document boundary, and the cached targets quietly become wrong, especially at short sequence lengths where neighboring-document tokens leak more into the distribution. The fix is just to align teacher and student sequences exactly (same packing, same order). Not glamorous, but it's the difference between the cache being valid and being garbage.

Let me also sanity-check the loss/divergence choice while I'm here, since I committed to forward KL. The reason isn't aesthetic — it's that the whole unbiasedness-of-the-gradient argument is a *forward-KL* fact. Forward KL's logit gradient is `(sum t) p_j - t_j`, which under an unbiased, normalized sampled target becomes `p_j - t^s_j` with expectation `p_j - t_j`; that exact-in-expectation property is what makes sampling equivalent to full distillation, and it's special to this loss. Reverse KL is mode-seeking — it would have the student chase the teacher's dominant mode and ignore the tail — which is the opposite of what I want when the point is to faithfully reproduce the *whole* teacher distribution, tail included. Forward KL is mean-seeking, it spreads the student to cover the teacher's support, which is precisely the behavior the tail problem demands. So forward KL is the loss whose gradient the sampling argument is built around.

So let me assemble the actual code, filling the two empty slots — the downsampling that builds the cache-able sparse target, and the loss that consumes it. The downsampler at `tau = 1` is just: draw `N` token ids from the teacher distribution with replacement, give each a weight `1/N`, and scatter those into a sparse vector (so a token drawn `c_i` times accumulates `c_i/N`). The loss is the forward KL of the student log-probabilities against that sparse target.

```python
import torch
import torch.nn.functional as F


def downsample(teacher_probs, N=50):
    """Random Sampling KD downsampler (tau = 1): sample N token ids ~ teacher,
    accumulate counts/N into a sparse target. Unbiased: E[c_i/N] = t_i for all i."""
    # sample N token ids per position, with replacement, with probability teacher_probs
    sampled_idx = torch.multinomial(teacher_probs, N, replacement=True)   # [B, N]
    # each draw contributes 1/N; a token drawn c_i times accumulates c_i/N
    values = torch.full((teacher_probs.size(0), N), 1.0 / N,
                        device=teacher_probs.device,
                        dtype=teacher_probs.dtype)                         # [B, N]
    sparse_target = torch.zeros_like(teacher_probs)                       # [B, V]
    sparse_target.scatter_add_(1, sampled_idx, values)                    # accumulate counts/N
    return sparse_target                                                  # sparse: <= N nonzeros


def distillation_loss(student_logits, sparse_target):
    """Forward KL( sparse_target || student ). For tau=1 this equals the average
    cross-entropy of the student against the N sampled token ids."""
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(student_log_probs, sparse_target, reduction="batchmean")
```

And to be faithful to the single-loss surface I'd actually plug this into — where both teacher and student logits over the same tokens are handed to me, with `labels` marking the completion positions and `-100` on padding/prompt — the same idea, written as one loss body that samples the teacher, builds the counts/N target, takes forward KL, and masks the non-completion positions before reducing:

```python
import torch
import torch.nn.functional as F


def compute_distill_loss(student_logits, teacher_logits, labels=None,
                         N=50, reduction="batchmean"):
    # Random Sampling KD: sample from the teacher, train on the empirical counts/N target.
    # tau = 1 (the proposal is the teacher itself), so each sampled draw has weight 1/N
    # and the importance ratio t_i/q_i is identically 1 -- no reweighting needed.
    B, T, V = student_logits.shape
    teacher_probs = F.softmax(teacher_logits, dim=-1).reshape(B * T, V)   # full teacher dist

    # Sample N token ids ~ teacher per position; accumulate counts/N into a sparse target.
    # Unbiased estimator of the teacher: E[c_i / N] = t_i for every token i (head AND tail),
    # so the logit gradient (p_j - t^s_j) equals the full-distillation gradient in expectation.
    sampled_idx = torch.multinomial(teacher_probs, N, replacement=True)   # [B*T, N]
    sparse_target = torch.zeros_like(teacher_probs)                       # [B*T, V]
    sparse_target.scatter_add_(
        1, sampled_idx,
        torch.full_like(sampled_idx, 1.0 / N, dtype=teacher_probs.dtype)  # each draw -> 1/N
    )                                                                     # t^s sums to 1

    # Forward KL( t^s || p ); for tau=1 this is the avg cross-entropy over the sampled ids.
    student_log_probs = F.log_softmax(student_logits, dim=-1).reshape(B * T, V)
    per_token = F.kl_div(student_log_probs, sparse_target,
                         reduction="none").sum(dim=-1)                    # [B*T]

    if labels is not None:                                               # exclude pad/prompt
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

Let me trace the causal chain once, to make sure the pieces are load-bearing and not decorative. I started unable to afford caching the full teacher distribution, so I had to summarize it sparsely. The intuitive summary, top-`K`, is `L1`-optimal per token but it creates miscalibrated students; computing the KL logit gradient showed why — truncation makes the target sub-stochastic, the `(sum t) p_j - t_j` gradient then scales the kept probabilities up by `1/a` (over-confidence) and drives the tail to zero (no rare-token supervision). The ghost token repaired the in-`K` gradient back to `p_j - t_j` but collapsed the tail into one bucket, so it still lacked per-token tail signal; label smoothing and naive fixes shared the same root flaw, because all of them begin by *deleting* the tail. Reframing top-`K` as a proposal with zero probability on the tail revealed the precise defect: it violates the support condition that makes an importance-sampling estimate unbiased, which is exactly why the summary is biased. Sampling token ids from the teacher and caching the empirical distribution `c_i/N` fixes this — multinomial counts give `E[c_i/N] = t_i` for every token including the tail, the target is automatically normalized so there's no `1/a` up-scaling, and the resulting logit gradient `p_j - t^s_j` equals the full-distillation gradient `p_j - t_j` in expectation, so large batches average toward the full-distillation signal while storing a dozen numbers. The tempered proposal `q ∝ t^tau` is a bias-free variance knob; `tau = 1` makes the importance weights exactly one and the variance is near its minimum, so I sample straight from the teacher and count. Forward KL is the loss whose gradient the whole argument hinges on, and at `tau = 1` the objective is just the average cross-entropy against the sampled ids. About `N = 50` rounds yield about `12` unique tokens, and the `c_i/N` form quantizes exactly into `7` bits, giving roughly `3` bytes per unique token. Align the teacher and student sequences so the cached contexts match. That's the method: sample the teacher, train on the counts.
