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

Now the reverse direction, the one I actually want for mode-seeking. I follow the same sample-wise sign
convention as the token implementation: inspect the accumulated term `Σ q_θ·(log p − log q_θ)`, then the
scalar loss negates that accumulated value to minimize the KL. The `θ`-dependence is now in *both* the
outer weighting `q_θ` and the `log q_θ` inside. The product rule gives `Σ (∇q_θ)·(log p − log q_θ)` from
the weighting and `−Σ q_θ·∇log q_θ = −Σ ∇q_θ` from the inner log, so the sample-wise coefficient is
`−(log(q_θ/p) + 1)`. Collapsed onto the sampled sequence, the coefficient written for reverse KL is
`∇_θ D_KL(q_θ, p) = −(log r_{q_θ,p} + 1)·∇_θ q_θ(y|x)`, with `r_{q_θ,p} = q_θ/p`. The magnitude is what
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
zero. I need the denominator to contain a floor from the other distribution. If the denominator is
`α·p + (1−α)·q_θ` it cannot vanish unless *both* `p` and `q_θ` vanish together, which is far rarer than
either alone. Skew the KL by computing it against a mixture instead of against the raw target.

For the direction I care about, mix the *teacher* with a sliver of the student and take the reverse KL
of the student against that mixture. Write `p̃ = (1−α)·p + α·q_θ` for the skewed teacher, with `α` small,
and define `D_SRKL^α(p, q_θ) = D_KL(q_θ, p̃) = D_KL(q_θ, (1−α)·p + α·q_θ)`. At `α = 0` this is exactly
reverse KL, `D_KL(q_θ, p)`; as `α` grows, the mixture floor rises. In the same sample-wise convention,
the inner ratio is now `q_θ/p̃`, and `p̃` itself depends on `θ` through its `α·q_θ` leg. Carrying that
dependence through, the coefficient becomes `log r_{q_θ,p̃} + 1 − α·r_{q_θ,p̃}`, i.e.
`∇_θ D_SRKL^α(p, q_θ) = −(log r_{q_θ,p̃} + 1 − α·r_{q_θ,p̃})·∇_θ q_θ(y|x)`, with `r_{q_θ,p̃} = q_θ/p̃`.
Compare to plain reverse KL's `log r_{q_θ,p} + 1`. Two things changed and both help. The ratio inside
the log is now `q_θ/p̃` instead of `q_θ/p`, and on an SGO where `p ≈ 0` the mixture still has the
`α·q_θ` leg, so `p̃ ≥ α·q_θ` and `r_{q_θ,p̃} ≤ 1/α`; the log no longer diverges. The new
`−α·r_{q_θ,p̃}` term subtracts from the coefficient and grows when the ratio grows, pulling the
coefficient back down where it would otherwise run away. So the skewed reverse KL has a bounded,
well-behaved gradient on precisely the teacher-unfamiliar SGOs that wrecked plain reverse KL. The
interpolation prevents the denominator of the ratio from reaching zero, which is what makes the gradient
stable.

For completeness let me confirm the forward direction behaves the same way, because I will want both as
options. Define the skewed forward KL `D_SKL^α(p, q_θ) = D_KL(p, α·p + (1−α)·q_θ)` — mix the *student*
leg up with a sliver of teacher. Its gradient comes out `∇_θ D_SKL^α(p, q_θ) = −(1−α)·r_{p,q̃_θ}·∇_θ
q_θ(y|x)` with `q̃_θ = α·p + (1−α)·q_θ`. Same story: where the student vanishes, the mixture still has
the `α·p` leg, so `q̃_θ ≥ α·p` and `r_{p,q̃_θ} ≤ 1/α`; the `(1−α)` prefactor shrinks the whole
coefficient as well. So skewing tames the gradient in both directions. The
asymmetry I keep is which raw distribution I protect: skew-forward floors the student leg (mass-covering,
good when the *student* can vanish on teacher-favored tokens), skew-reverse floors the teacher leg
(mode-seeking, good when the *teacher* can vanish on SGOs). Since my configuration is mode-seeking on
SGOs, skewed *reverse* KL is the one whose denominator-floor sits exactly where my data puts the zeros.

Now I should worry about the opposite failure: am I buying gradient stability at the cost of a target
that no longer faithfully represents the teacher? A second concern is estimation noise — I compute the
divergence on mini-batches, and a high-variance estimator gives a noisy objective. Let me check the
empirical estimation error of the skewed estimator, because if skewing also reduces *that*, it is a free
win, and if it trades off against the gradient scale I need to know the trade to pick `α`. Take `p^1_n`,
`p^2_n` the empirical distributions from `n` i.i.d. samples of `p^1`, `p^2`. Under mild assumptions the
L2 error of the `α`-SKL estimator obeys a bound of the form
`E[|D_SKL^α(p^1_n, p^2_n) − D_SKL^α(p^1, p^2)|^2] ≤ c_1(α)/n^2 + c_2·log^2(α n)/n + c_3·log^2(c_4 n)/(α^2
n)`, with `c_1(α) = min{1/α^2, χ^2(p^1,p^2)^2/(1−α)^2}` and `c_2, c_3, c_4` positive constants
independent of `n`, `α`, and the KL value (`χ^2` the chi-square divergence). The important
`α`-dependence is the singular behavior near zero: the inverse-`α` terms become smaller as I move away
from raw KL, so skewing lowers the estimator error contribution that comes from an unprotected
denominator. The log term is still present, so I should not pretend the whole displayed bound is a
monotone function of `α`; the useful fact is that the raw-KL edge has high estimation error and a
moderate skew makes the mini-batch objective track its true value more tightly.

Moving `α` away from zero helps the gradient coefficient and the inverse-`α` estimation terms — does
that mean push `α` toward 1? No, and the reason is that the gradient-norm "benefit" is partly illusory
under a modern optimizer.
Adam-style optimizers normalize by a running estimate of the gradient scale, so a uniformly smaller
gradient coefficient is divided back out — the reduced gradient scale from a large `α` gets compensated
away. To see the estimation error in the units the optimizer actually moves in, I should look at the L2
norm *normalized by the gradient scale*, i.e. divide the SKL estimator deviation by its `(1−α)`
coefficient and re-bound. Doing that, `E[|(1/(1−α))·(D_SKL^α(p^1_n,p^2_n) − D_SKL^α(p^1,p^2))|^2] ≤
c_1^*(α)/n^2 + c_2·log^2(α n)/((1−α)^2 n) + c_3·log^2(c_4 n)/(α^2 (1−α)^2 n)` with `c_1^*(α) =
min{1/(α^2 (1−α)^2), χ^2^2/(1−α)^4}`. Now the `α`-dependence is a real trade-off: the inverse-`α`
pieces want `α` larger, the inverse-`(1−α)` pieces want `α` smaller, and the normalized curve appears
convex over the useful range rather than giving a one-sided answer. There is a genuine sweet spot — small
enough that I have only nudged the original KL direction, large enough that the denominator floor and
the estimation-error reduction kick in. A mild value, `α = 0.1`, sits below the theorem's `α < 1/8`
range and gives 10% of the other distribution. In the reverse direction I care about, that means
`p̃ = 0.9·p + 0.1·q_θ`: enough to floor the denominator while leaving the reverse target mostly teacher.
So `α = 0.1`.

This also tells me precisely why skewing is *not* the same as the generalized JSD I could have reached
for. JSD interpolates by `D_JSD^β(p, q_θ) = β·D_SKL^β(p, q_θ) + (1−β)·D_SRKL^{1−β}(p, q_θ)` — it is a
*sum of two skewed KLs whose skew parameters are tied to the same `β`*. So the same `β` that sets the
skew of the forward leg sets `1−β` for the reverse leg: I cannot make *both* legs mildly skewed at once.
If I want `α = 0.1` on the reverse term, JSD forces `0.9` on the forward term, and vice versa. The
L2-vs-α analysis says I want a *mild* skew on the term I am using; JSD structurally cannot give a mild
skew to both legs at once. A single skewed (reverse) KL with a freely chosen mild `α` hits an operating
point that JSD's coupled parameter cannot reach. That is the analytical reason to prefer one skewed KL
over the interpolated JSD, not just a task-dependent comparison.

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
value, I would add that constant back; if I want code faithful to the canonical implementation, I leave it
out.

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

Let me trace the chain end to end. I wanted mode-seeking distillation on the student's own generations,
because reverse KL cures forward KL's mode-averaging and SGOs cure the off-policy training-inference
mismatch — but the combination trains badly. The gradient analysis says why: forward KL's coefficient
`r_{p,q_θ}` explodes where the student vanishes, reverse KL's `log r_{q_θ,p}` explodes where the teacher
vanishes, and on SGOs the teacher vanishes constantly, so my chosen configuration sits exactly in
reverse KL's blow-up regime. The blow-up is caused by a raw distribution going to zero in a denominator,
so I floor that denominator by skewing the KL against a mixture: skewed reverse KL `D_KL(q_θ, (1−α)p +
α q_θ)`, whose gradient coefficient `log r_{q_θ,p̃} + 1 − α·r_{q_θ,p̃}` cannot diverge on
teacher-vanishing samples because `p̃ ≥ α q_θ` and the extra `−α·r` term pulls it back. Skewing also
reduces the inverse-`α` contribution in the empirical L2 estimation bound, but normalizing by the
gradient scale introduces inverse-`(1−α)` terms and makes the useful curve appear convex, so I take the
mild point `α = 0.1` — still 90% teacher in the reverse target, but strong enough to floor the denominator.
This is structurally distinct from generalized JSD, whose single `β` ties the two legs' skews together
as `β` and `1−β` (with the official DistiLLM `js_distance(lam)` using `lam = 1−β` for that same
quantity); one freely tuned mild skewed KL hits the operating point JSD cannot reach. The
data-side scheduling (adaptive SGO probability, off-policy replay buffer) is a separate, orthogonal
system whose success rides on the skewed loss's fast early convergence; the loss itself is the masked,
vocabulary-summed skewed reverse KL at `α = 0.1`, keeping both legs of the reverse-KL gradient and
guarding `±inf` logits.
