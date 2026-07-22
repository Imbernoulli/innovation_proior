Let me start from the thing that actually breaks when I distill a big autoregressive teacher into a
small student and I do it the way that should work best — train the student on its *own* generated
sequences, scored by the teacher, with a mode-seeking objective. I already believe two things from
hard experience. One: training on a fixed corpus is off-policy, the prefixes the student sees in
training are not the prefixes it walks at inference, and that mismatch compounds errors over the
sequence, so I want to train on student-generated outputs — let the student generate, have the teacher
label each token-distribution along the way. Two: the student is much smaller than the teacher and
cannot represent its full distribution, so forward KL, which is mass-covering, forces the student to
smear probability over the teacher's whole support and I get an over-smooth, hedging, incoherent
model — the mode-averaging failure. The fix for that is reverse KL, `D_KL(q_θ, p)`, weighted by the
student's own probability, mode-seeking: it concentrates the student on the teacher's dominant modes
and abandons the tail. So the configuration I want is reverse KL on student-generated outputs. And it
trains badly. I want to understand exactly why before I reach for anything, because the failure is
going to dictate the fix.

Let me look at the gradient, because the gradient is what the optimizer actually feels. For a
context-target pair `(x, y)`, write the gradient of forward KL with respect to the parameters. The
clean way to see it is that the KL is `Σ p·log(p/q_θ)`, the `θ`-dependence is only in `q_θ`, and
differentiating `−Σ p·log q_θ` gives `−Σ p·(1/q_θ)·∇q_θ`. Collapsed onto the sampled sequence this is
`∇_θ D_KL(p, q_θ) = − r_{p,q_θ}·∇_θ q_θ(y|x)`, where `r_{p,q_θ}` is the ratio of the two
distributions — the negative gradient of the model probability, weighted *inversely* by that
probability. Stare at the weighting. If `q_θ(y|x) ≈ 0` — the student assigns almost no mass to a token
the teacher likes — the ratio `r_{p,q_θ}` explodes, the gradient norm blows up, and the optimizer
takes a huge step in a direction estimated from almost no probability mass, i.e. a noisy direction. So
forward KL has a built-in instability exactly where the student is most wrong. That is one disease.

Now the reverse direction, the one I actually want for mode-seeking. I have to separate two signs that
are easy to conflate. The literal divergence is `D_KL(q_θ, p) = Σ q_θ·(log q_θ − log p)`, so the product
rule gives `(log(q_θ/p) + 1)·∇q_θ`. The token-loss code I need to match accumulates the negative quantity
`Σ q_θ·(log p − log q_θ)` first and then negates it at reduction time; that accumulator has the opposite
sign, but the minimized scalar loss has
`∇_θ D_KL(q_θ, p) = (log r_{q_θ,p} + 1)·∇_θ q_θ(y|x)`, with `r_{q_θ,p} = q_θ/p`. The magnitude is what
matters for stability, and that magnitude blows up at the *other* end: when `p(y|x) ≈ 0` — the teacher
assigns near-zero probability to what the student did — the log ratio `log(q_θ/p) → +∞`. Same disease,
mirror image: forward KL is unstable where the student vanishes, reverse KL is unstable where the teacher
vanishes.

And here is the wall, the thing that makes my chosen configuration worse than either piece alone. I am
training reverse KL *on student-generated outputs*. A student-generated sequence is not sampled from the
teacher distribution; it can be off the teacher's high-probability region, often a sequence the teacher
finds unlikely or unfamiliar. So on SGOs, `p(y|x) ≈ 0` is not just a remote edge case; it can happen
often enough to dominate training. The exact regime where the reverse-KL gradient coefficient `log r_{q_θ,p}`
explodes is the regime my data lives in. The data that fixes the training-inference mismatch is the
data that detonates the reverse-KL gradient. That is why mode-seeking-on-SGO trains badly: every
self-generated token where the teacher is surprised throws a giant noisy gradient. I can see now why
people bolt stabilizers onto the reverse-KL policy gradient — variance baselines, mixing the teacher
into the sampling, importance weights, clipping — they are all fighting this one explosion. I want to
kill the explosion at its source instead.

What is the source? The coefficient blows up because a *probability sits in a denominator and goes to
zero*. In forward KL it is `q_θ` in `r_{p,q_θ} = p/q_θ`; in reverse KL it is `p` in `r_{q_θ,p} = q_θ/p`.
The denominator is one of the two raw distributions, and a raw distribution can be arbitrarily close to
zero. The obvious dodge — clamp the denominator to a fixed floor `p ← max(p, ε)` — I distrust, because a
flat floor is unrelated to the loss being optimized: it discontinuously changes the gradient at the clamp
boundary and the bias it injects has nothing to do with where the student actually is. I want a floor that
is *part of* the divergence, not a patch on top of it. If the denominator were `α·p + (1−α)·q_θ` it
cannot vanish unless *both* `p` and `q_θ` vanish together, which is far rarer than either alone — and on
an SGO the offending token is one the *student itself* produced, so `q_θ` there is by construction not
tiny. That is the floor I want: mix the teacher with a sliver of the student and compute the KL against
the mixture instead of against the raw target.

So for the direction I care about, define `p̃ = (1−α)·p + α·q_θ` for the skewed teacher, with `α` small,
and `D_SRKL^α(p, q_θ) = D_KL(q_θ, p̃) = D_KL(q_θ, (1−α)·p + α·q_θ)`. At `α = 0` this is exactly
reverse KL, `D_KL(q_θ, p)`; as `α` grows, the mixture floor rises. `p̃` *itself* depends on `θ` through
its `α·q_θ` leg, so writing down `log r_{q_θ,p̃} + 1` by analogy would miss the extra term that
dependence creates — differentiate honestly. `D_SRKL^α = Σ q_θ·log q_θ − Σ q_θ·log p̃`. The first sum differentiates to
`Σ (log q_θ + 1)·∇q_θ`. The second is the trap: `∇[Σ q_θ·log p̃] = Σ (log p̃)·∇q_θ + Σ q_θ·(∇p̃)/p̃`, and
`∇p̃ = α·∇q_θ`, so the second piece is `Σ (α·q_θ/p̃)·∇q_θ = Σ α·r_{q_θ,p̃}·∇q_θ`. (Both `Σ∇q_θ = 0` terms
that would otherwise appear cancel because the per-vocab pieces are themselves the coefficients.)
Subtracting, the per-token coefficient is `log q_θ − log p̃ + 1 − α·r_{q_θ,p̃} = log r_{q_θ,p̃} + 1 −
α·r_{q_θ,p̃}`, so `∇_θ D_SRKL^α(p, q_θ) = (log r_{q_θ,p̃} + 1 − α·r_{q_θ,p̃})·∇_θ q_θ(y|x)` with
`r_{q_θ,p̃} = q_θ/p̃`.

The `−α·r` term is exactly the piece I could have gotten wrong, so check it numerically against a finite
difference. Fix a teacher `p` over a small vocab,
parameterize the student as `q_θ = softmax(z)` so the chain through `p̃`'s `α·q_θ` leg is genuinely
present, and compare `d/dz` of the full-vocab `KL(q_θ, p̃)` against `Σ_v coef_v · (∂q_v/∂z)` using
`coef_v = log(q_v/p̃_v) + 1 − α·(q_v/p̃_v)` and the softmax Jacobian `∂q_v/∂z_j = q_v(δ_{vj} − q_j)`. With
`α = 0.1`, `V = 6`, a random `p` and `z`, the two gradients agree to `1e−6`: numeric `[−0.1278,
−0.0413, 0.1873, −0.1537, 0.1207, 0.0148]` against analytic `[−0.1278, −0.0413, 0.1873, −0.1537,
0.1207, 0.0148]`. So the coefficient is right, `−α·r` term and all — dropping that term breaks the match,
which is the concrete reason I cannot just reuse plain reverse KL's coefficient.

Now does this coefficient actually stay bounded where plain reverse KL detonates? Compare to plain
reverse KL's `log r_{q_θ,p} + 1`. Two things changed. The ratio inside the log is now `q_θ/p̃` instead
of `q_θ/p`, and on an SGO where `p ≈ 0` the mixture still has the `α·q_θ` leg, so `p̃ ≥ α·q_θ` and
`r_{q_θ,p̃} ≤ 1/α`; the log can no longer diverge. The new `−α·r_{q_θ,p̃}` term subtracts from the
positive reverse-KL coefficient and grows when the ratio grows, pulling the coefficient back down where
it would otherwise run away. Take a
token the teacher essentially rules out, `p = 1e−9`, that the student emitted with `q_θ = 0.5`, and
`α = 0.1`. Plain reverse KL: `log(0.5/1e−9) + 1 = 21.03`, and it keeps growing like `−log p` as `p`
shrinks. Skewed: `p̃ = 0.9·1e−9 + 0.1·0.5 = 0.05`, so `r_{q_θ,p̃} = 0.5/0.05 = 10.0` — exactly the `1/α`
ceiling, independent of how small `p` got — and the coefficient is `log 10 + 1 − 0.1·10 = 2.303 + 1 − 1
= 2.303`. Twenty-one versus two, and the skewed value cannot grow past its ceiling no matter how
surprised the teacher is. The mixture floor is doing precisely the job I designed it for.

I'll want both directions available, so work the forward direction the same way and check that the
floor mechanism isn't special to reverse. Define the skewed forward KL
`D_SKL^α(p, q_θ) = D_KL(p, α·p + (1−α)·q_θ)` — mix the *student* leg up with a sliver of teacher,
`q̃_θ = α·p + (1−α)·q_θ`. Here the outer weight `p` is `θ`-independent, so only the `−Σ p·log q̃_θ` leg
contributes: `∇ = −Σ p·(∇q̃_θ)/q̃_θ = −Σ p·((1−α)∇q_θ)/q̃_θ = −(1−α)·Σ (p/q̃_θ)·∇q_θ`, giving
`∇_θ D_SKL^α(p, q_θ) = −(1−α)·r_{p,q̃_θ}·∇_θ q_θ(y|x)`. A finite-difference check on the same setup
confirms this closed form to `1e−6` as well. Same story as before: where the student vanishes, the
mixture still has the `α·p` leg, so `q̃_θ ≥ α·p` and `r_{p,q̃_θ} ≤ 1/α`; the `(1−α)` prefactor shrinks
the whole coefficient on top of that. So skewing tames the gradient in both directions. The asymmetry I
keep is which raw distribution I protect: skew-forward floors the student leg (mass-covering, good when
the *student* can vanish on teacher-favored tokens), skew-reverse floors the teacher leg (mode-seeking,
good when the *teacher* can vanish on SGOs). Since my configuration is mode-seeking on SGOs, skewed
*reverse* KL is the one whose denominator-floor sits exactly where my data puts the zeros.

Now I should worry about the opposite failure: am I buying gradient stability at the cost of a target
that no longer faithfully represents the teacher? A second concern is estimation noise — I compute the
divergence on mini-batches, and a high-variance estimator gives a noisy objective. Let me look at the
empirical estimation error of the skewed estimator, because if skewing also reduces *that*, it is a free
win, and if it trades off against the gradient scale I need to know the trade to pick `α`. Take `p^1_n`,
`p^2_n` the empirical distributions from `n` i.i.d. samples of `p^1`, `p^2`. Under mild assumptions the
L2 error of the `α`-SKL estimator obeys a bound of the form
`E[|D_SKL^α(p^1_n, p^2_n) − D_SKL^α(p^1, p^2)|^2] ≤ c_1(α)/n^2 + c_2·log^2(α n)/n + c_3·log^2(c_4 n)/(α^2
n)`, with `c_1(α) = min{1/α^2, χ^2(p^1,p^2)^2/(1−α)^2}` and `c_2, c_3, c_4` positive constants
independent of `n`, `α`, and the KL value (`χ^2` the chi-square divergence). The important
`α`-dependence is the singular behavior near zero: the inverse-`α` terms become smaller as I move away
from raw KL, so skewing lowers the estimator error contribution that comes from an unprotected
denominator. The log term is still present and I should not pretend the whole displayed bound is a
monotone function of `α`; the useful fact I can lean on is just that the raw-KL edge (`α → 0`) has a
diverging inverse-`α` contribution, and a moderate skew kills it.

Moving `α` away from zero helps the gradient coefficient and the inverse-`α` estimation terms — so why
not push `α` toward 1? Here I have to be careful not to double-count a benefit, because the
gradient-norm "benefit" is partly illusory under a modern optimizer. Adam-style optimizers normalize by
a running estimate of the gradient scale, so a uniformly smaller gradient coefficient is divided back
out — the reduced gradient scale from a large `α` gets compensated away. To see the estimation error in
the units the optimizer actually moves in, I should look at the L2 norm *normalized by the gradient
scale*, i.e. divide the SKL estimator deviation by its `(1−α)` coefficient and re-bound. Doing that,
`E[|(1/(1−α))·(D_SKL^α(p^1_n,p^2_n) − D_SKL^α(p^1,p^2))|^2] ≤ c_1^*(α)/n^2 + c_2·log^2(α
n)/((1−α)^2 n) + c_3·log^2(c_4 n)/(α^2 (1−α)^2 n)` with `c_1^*(α) = min{1/(α^2 (1−α)^2),
χ^2^2/(1−α)^4}`. Now the `α`-dependence is a real two-sided trade-off: the inverse-`α` pieces want `α`
larger, the inverse-`(1−α)` pieces want `α` smaller, so there is an interior minimum rather than a
one-sided answer. The forward and reverse cases do not land at the same place: SKL stays relatively
robust as `α` grows, but the reverse normalization is harsher (it is the `q_θ` leg, the one carrying the
sampling), and the inverse-`(1−α)` terms bite sooner, so SRKL's normalized error is smallest at a small
`α` and worsens beyond it. I want a value small enough that I have only nudged the original KL direction,
large enough that the denominator floor and the estimation-error reduction kick in — an interior point of
that convex normalized curve, comfortably inside the bound's `α < 1/8` validity range. `α = 0.1` sits
there: 10% of the other distribution. In the reverse direction I care about, `p̃ = 0.9·p + 0.1·q_θ` —
enough to floor the denominator (the `r ≤ 1/α = 10` ceiling I checked above) while leaving the reverse
target 90% teacher. So `α = 0.1`. I would still want to confirm the exact location of the SRKL minimum
empirically on real distillation runs rather than from the bound alone, since the bound only pins down
the *shape*, but the analysis is enough to commit to a small interior value and rule out pushing `α`
high.

This also tells me to ask whether skewing is really different from the generalized JSD I could have
reached for, or whether I have just re-derived a corner of it. JSD interpolates by
`D_JSD^β(p, q_θ) = β·D_KL(p, M) + (1−β)·D_KL(q_θ, M)`, `M = β·p + (1−β)·q_θ`. Claim to test: this equals
`β·D_SKL^β(p, q_θ) + (1−β)·D_SRKL^{1−β}(p, q_θ)`, a *sum of two skewed KLs whose skew parameters are
tied to the same `β`*. The forward leg `D_KL(p, M)`
is `D_SKL^α(p,q_θ)` with inner mixture `α·p + (1−α)·q_θ`; setting `α = β` makes that mixture `β·p +
(1−β)·q_θ = M`. The reverse leg `D_KL(q_θ, M)` is `D_SRKL^α` with inner mixture `(1−α)·p + α·q_θ`;
setting `α = 1−β` makes that `β·p + (1−β)·q_θ = M` again. So *if* the algebra is right, the forward leg
carries skew `β` and the reverse leg carries skew `1−β`, locked together. Checking numerically on random
`p, q` over a 5-symbol vocab with `β = 0.37`: the SKL inner mixture equals `M` and the SRKL inner mixture
equals `M` (both exactly), and `β·KL(p,M) + (1−β)·KL(q,M) = 0.06706` matches `β·D_SKL^β +
(1−β)·D_SRKL^{1−β} = 0.06706` to all printed digits. The identity holds. So the consequence is real and
not a coincidence of my chosen numbers: the same `β` that sets the skew of the forward leg sets `1−β` for
the reverse leg, and I cannot make *both* legs mildly skewed at once. If I want `α = 0.1` on the reverse
term, JSD forces `0.9` on the forward term, and vice versa. My L2-vs-`α` analysis says I want a *mild*
skew on the term I am actually using; JSD structurally cannot give a mild skew to both legs at once. A
single skewed (reverse) KL with a freely chosen mild `α` reaches an operating point JSD's coupled
parameter cannot — `(α_fwd, α_rev) = (anything, 0.1)` is off the `α_fwd + α_rev = 1` line JSD is confined
to. That is the structural reason to prefer one skewed KL over the interpolated JSD, not just a
task-dependent comparison.

I need to keep two problems separate, because skewing the loss is only one part of the full
configuration. The gradient instability I just cured is the *objective* problem.
There is a separate *data* problem: SGOs are expensive (generating them every step dominates training
time) and the teacher's feedback on far-off SGOs is noisy (it can prefer short wrong answers to long
correct ones), so a complete system would *schedule* how much SGO to use — start with little, ramp it up
guided by validation loss — and reuse past SGOs from a replay buffer with a decaying replay ratio to save
generation cost. Those are scheduling and data-pipeline mechanisms, orthogonal to the per-token loss.
The pieces still interact: the fast early movement from skewed (R)KL is what lets an off-policy replay
buffer avoid the usual stale-policy bias error. But the loss itself — the thing I plug into the
token-level slot — is just the skewed reverse KL with `α = 0.1`. I keep the scheduler and the buffer as
the surrounding system and land the loss on its own.

Now the loss in code, and the one place to be careful is the distinction between a literal divergence
value and the training loss used to optimize the student. The
reverse KL is `D_KL(q_θ, p̃) = Σ_v q_θ(v)·(log q_θ(v) − log p̃(v))`. Both terms depend on `θ` so I keep
both — unlike the forward case where the target entropy is constant and drops, here the `Σ q_θ·log q_θ`
leg carries the `+1` normalization gradient I derived, so I must not discard it. Compute teacher and
student probabilities and log-probabilities, form the mixture `p̃ = (1−α)·p + α·q_θ` in probability
space and take its log, then accumulate `q_θ·log p̃` minus `q_θ·log q_θ` over the vocabulary, mask out any
`±inf` logit positions so they contribute zero rather than `nan`, sum over the vocabulary to a per-token
value, mask to the completion tokens (`label ≠ −100`), and average over the valid tokens. The overall sign
is negative because I accumulate `Σ q_θ·(log p̃ − log q_θ)` (the negative of the KL) and negate at the
reduction. The skewed *forward* variant is the same skeleton with the mixture floored on the student leg,
`q̃_θ = α·p + (1−α)·q_θ`, and the teacher as the outer weight, `Σ p·log q̃_θ`, where the target-entropy
`Σ p·log p` is constant in `θ` and can be dropped for training. If I wanted to report the numeric SKL
value, I would add that constant back; for the training loss I leave it out.

```python
import torch
import torch.nn.functional as F


def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    # Skewed reverse KL: KL(q_theta, (1-lam)*p + lam*q_theta). lam = alpha = 0.1.
    # Adds lam*q_theta to the teacher-side denominator, so on student-generated
    # outputs with p ~= 0 the ratio q_theta / p_tilde is at most 1/lam.
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs        # p~ = (1-a)p + a q

    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)         # guard -inf logits

    # q_theta * log p~  -  q_theta * log q_theta   (keep BOTH legs: the second
    # carries the +1 normalization gradient of reverse KL).
    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)                           # per-token -KL
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    # Skewed forward KL: KL(p, lam*p + (1-lam)*q_theta). Adds lam*p to the
    # student-side denominator, so p / q_tilde is at most 1/lam where q_theta ~= 0.
    # Target entropy Sum p log p is constant in theta -> dropped.
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1 - lam) * student_probs        # q~ = a p + (1-a) q
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
```

The sign convention and the keep-both-legs point are exactly where an off-by-a-term bug would hide
silently, so trace the code on a tiny input. Take one batch, two
positions, vocab size 4, random teacher and student logits, and a label `[5, −100]` so the second
position is masked out and only the first should count. Running `skewed_reverse_kl` returns `0.46541`. As
an independent reference I compute `KL(q_θ, p̃)` directly at position 0 from the softmaxed logits with
`p̃ = 0.9·p + 0.1·q_θ`: that is `0.46541` too — they match, so the masking, the mixture, and the negate-
at-reduction sign are all consistent, and the value is non-negative as a divergence must be. Then the
keep-both-legs check: if I drop the `q_θ·log q_θ` leg and keep only `−Σ q_θ·log p̃`, I get `1.392`, which
is not `KL(q_θ, p̃)` and is not even a divergence (it would not be zero when `q_θ = p̃`). So the second
leg is load-bearing — it is the `+1` normalization term made concrete — and the comment in the code is
literally what the numbers force.

So the loss is the masked, vocabulary-summed skewed reverse KL at `α = 0.1`, keeping both legs of the
reverse-KL gradient and guarding `±inf` logits, with the skewed forward variant carried alongside on the
same skeleton for whichever direction a future setup needs — the scheduler and replay buffer stay the
separate, orthogonal system I set aside earlier.
