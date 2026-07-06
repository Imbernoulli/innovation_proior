The objective is the whole task here — everything else is frozen — so the floor to start from is the
question of what is even wrong with the loss I already have, plain next-token cross-entropy, and which
of the three obvious handles to pull first. Let me write the loss down and actually stare at it instead
of reciting it. For one position with logits `z` over the vocabulary and target index `y`, the loss is
`-log softmax(z)_y = -z_y + log Σ_j exp(z_j)`. This is maximum likelihood and it is correct; that is not
in dispute. What bugs me is where its optimum sits. To drive this to zero I need `softmax(z)_y → 1`, and
that happens only as the gap `z_y - max_{j≠y} z_j` runs off to `+∞`. There is no finite `z` that
minimizes this loss. On data that is even close to separable at the margin — and token-level language
data effectively is, because for most contexts there genuinely is one overwhelmingly likely next token —
the loss is, quite literally, a standing instruction to grow the correct logit's lead without bound. The
gradient `∂ℓ/∂z_k = p_k - 1[k=y]` is bounded in `[-1, 1]`, so each step is gentle, but the target it is
chasing is at infinity, and over thirteen thousand iterations a bounded gradient pointed at infinity
walks the logits steadily upward. The fixed point of the gradient makes this exact: `p_k - 1[k=y] = 0`
requires `p = δ_{·,y}`, a one-hot, which as a softmax output demands infinite logit gaps. The loss has
no interior stationary point at all; the only place its gradient vanishes is at the boundary of logit
space, at infinity.

So what actually goes wrong from that runaway, and why is it worth fixing before I even reach for the
sharper instabilities? Two things, and they are the same thing seen from two sides. The model gets more
and more confident on its training targets — it learns to slam probability mass onto the observed next
token of every context, which is the textbook signature of over-confidence and, on the held-out
distribution, of overfitting: fitting the training labels says nothing about generalizing to held-out
ones. And the unbounded growth of the largest-minus-rest logit gap, against a bounded gradient, makes
the model rigid — once a huge gap is established the model is slow to revise it, because a bounded
gradient can only chip away at a gap that the loss keeps trying to make infinite. The unifying word for
both is over-confidence. This is exactly the thing the cheapest loss-layer modification should be built
to fix, and the diagnosis is clean: hard one-hot targets drive `z_y ≫ z_k`, which both overfits and
reduces adaptability, so the fix is to stop demanding probability one.

Before I pick a handle, I want to know which regime this run is even in, because how much any
over-confidence fix can buy depends on how hard the run is pushing into the over-confident corner, and
that is set by the budget, which I can read straight off the substrate. Micro-batch 64, gradient
accumulation 8, two GPUs — so `64·8·2 = 1024` sequences enter each optimizer step. The data is FineWeb
`sample-10BT`, about 7.1B tokens, consumed over 13,535 iterations, which is `7.1e9 / 13535 ≈ 524{,}600`
tokens per step; dividing by the 1024 sequences gives a sequence length of about 512, and `1024 · 512 =
524{,}288` tokens per step times 13,535 steps is `7.10e9`, so the arithmetic closes. The number that
matters is the ratio to parameters: `7.1e9 / 3.55e8 = 20.0` tokens per parameter, essentially one pass
over the corpus at almost exactly the compute-optimal token-to-parameter ratio. That reframes what
"over-confidence" can mean here. This is not a many-epoch run where the model sees the same examples
dozens of times and memorizes them; each token is seen about once, so classic overfitting-by-repetition
is muted, and the over-confidence I diagnosed is the milder, structural kind — the loss forever pushing
the correct-token gap wider — rather than the run pounding a handful of memorized sequences into
certainty. A single-epoch, compute-optimal run is precisely the setting where a heavy regularizer has
little long-run overfitting to prevent, and I should keep that in mind when I set the strength: the
payoff from smoothing the target at all is real but bounded, and the payoff from smoothing it *hard* is
smaller still.

There are three distinct places I could intervene, and the reason I take this one *first* is that it is
the cheapest, the most thoroughly understood, and the one whose failure mode will point cleanly at the
next rung. I could attack the absolute logit *level* — add a penalty on the log-partition so the
otherwise-free overall magnitude stops drifting. I could attack the logit *values* themselves —
structurally squash them through a bounded map before the softmax. Or I could attack the *target*: stop
asking for probability one in the first place. Each of the first two is a strictly larger intervention:
the level penalty adds an auxiliary term with a coefficient I would have to reason about, and the value
squash inserts a nonlinearity into the model's forward map that I would have to argue is a faithful
modeling change rather than a distortion. The target attack is the most surgical statement of "the
optimum is at infinity, so move the optimum to a finite place," it is a one-line change to the loss that
touches neither the model nor an auxiliary coefficient in any subtle way, and it is the right thing to
establish as the baseline because if even the gentlest, best-understood over-confidence fix already buys
something, then the harder numerical interventions have a real target to beat; if it does not, I have
learned that this short run is not over-confidence-limited and the next rung had better attack a
different handle. So target first, deliberately, as the conservative probe that calibrates the rest of
the ladder.

If the problem is that the target lives at infinity, the cure suggests itself: do not put the target at
infinity. Give every token a small floor of target probability so the correct logit has no incentive to
escape to `+∞`. Take the one-hot and bleed a little mass `ε` onto a fixed distribution `u` over the
vocabulary, `q'(k) = (1-ε)·δ_{k,y} + ε·u(k)`. The natural `u`, absent any prior knowledge, is uniform,
`u(k) = 1/V`, so `q'(k) = (1-ε)·δ_{k,y} + ε/V`. Now check that this actually kills the runaway. Every
entry of `q'` is at least `ε/V > 0`. If `z_y` tried to run off to `+∞`, then `p_y → 1` and `p_k → 0`
for `k ≠ y`, and the cross-entropy `-Σ_k q'(k) log p_k` would blow up on those wrong-class terms,
because `q'(k) = ε/V` is positive but `log p_k → -∞`. So an infinite logit gap is now infinitely
*expensive*, not free. That is label smoothing, and it is almost nothing to implement — but I want to
know exactly where it moved the optimum, not just that it moved it off infinity.

So I solve for the new fixed point. Cross-entropy `H(q', p)` against a fixed target `q'` is minimized,
over the free softmax output `p`, exactly at `p = q'` — that is the defining property of cross-entropy,
its minimum over `p` is the entropy `H(q')` and is attained at `p = q'`. So the optimal predicted
distribution is `p_y = 1-ε + ε/V` on the true token and `p_k = ε/V` on each of the others, and the
gradient of the smoothed loss, `p_k - q'(k)`, vanishes there — at a *finite* logit configuration, unlike
plain cross-entropy whose gradient vanished only at infinity. I can even read off how big the optimal
logit gap is now. At the optimum, `z_y - z_k = log(p_y / p_k) = log((1-ε + ε/V)/(ε/V)) ≈ log((1-ε)(V-1)/ε)`.
Plug in `ε = 0.05` and `V ≈ 50{,}257`: `log(0.95 · 50256 / 0.05) = log(954{,}864) = 13.77` nats. So
smoothing does not merely forbid the runaway in principle; it pins the optimal correct-versus-rest gap
at about 13.8 nats — a finite, computable ceiling in place of `+∞`. That is the whole mechanism made
concrete: the target moved from infinity to a point roughly fourteen nats out.

I can push the fixed-point picture one step further, from *where the optimum sits* to *whether the
optimizer actually feels a force toward it*, because those are not the same claim and the difference is
the whole reason a finite optimum helps. Under plain cross-entropy the gradient on the true-token logit
is `p_y - 1`, whose magnitude `1 - p_y` shrinks toward zero exactly as the model gives the loss what it
asks for — so the closer the model gets to its (infinite) optimum the *weaker* the pull, a vanishing
gradient forever chasing a target at infinity, which is why the gap creeps upward rather than snapping to
a value and never settles anywhere. Smoothing changes the sign structure. Its true-token gradient is
`p_y - q'(y) = p_y - (1 - ε + ε/V)`, and once the model overshoots and drives `p_y` past `0.95001` this
term turns *positive*, so descent now pushes the winning logit back *down*. That is a genuine two-sided
spring centered at `p_y ≈ 0.95`: below it the pull is up, above it the pull is down, and the strength
grows linearly in the displacement `p_y - q'(y)`. Put the number on it at a model that has run the
true-token probability up to `0.99`: the smoothing gradient there is `0.99 - 0.95001 = +0.040`, a
four-percent-of-full-scale downward force on the winning logit, whereas plain cross-entropy's gradient at
the same point is `0.99 - 1 = -0.010`, still pushing it *up*. So smoothing does not merely relocate the
optimum from infinity to 13.8 nats out; it converts a one-way monotone creep into a restored equilibrium
the optimizer can actually come to rest at, and the restoring force is exactly the `ε`-weighted uniform
pull the decomposition below will name.

Let me rewrite the loss to see its structure, because the structure is what tells me what I am really
doing. Cross-entropy is linear in the target, so
`H(q', p) = -Σ_k q'(k) log p_k = (1-ε)·(-Σ_k δ_{k,y} log p_k) + ε·(-Σ_k u(k) log p_k) = (1-ε)·H(q, p) + ε·H(u, p)`.
So smoothing is exactly the ordinary hard-label cross-entropy, downweighted by `(1-ε)`, plus an
`ε`-weighted term `H(u, p)` that pulls the prediction toward the prior `u`. And `H(u, p) = D_KL(u‖p) + H(u)`;
`H(u)` is a constant, so the second term is, up to a constant, a penalty on how far the prediction `p`
has drifted from uniform, with relative weight `ε/(1-ε)`. It is a regularizer that says "stay a bit
humble, do not get too far from the prior." With `u` uniform, `H(u, p) = Σ_k (1/V)(-log p_k) = mean_k(-log p_k)`,
so I never have to materialize the smoothed target vector — the second term is just `ε` times the mean
of the negative log-probabilities over the vocabulary. Let me sanity-check the decomposition on a tiny
case rather than trust the algebra blind: `V = 3`, `ε = 0.1`, target `y = 0`, and suppose the model
predicts `p = (0.5, 0.3, 0.2)`. Then `q' = (0.9 + 0.0333, 0.0333, 0.0333) = (0.9333, 0.0333, 0.0333)`,
and `H(q', p) = -(0.9333·log 0.5 + 0.0333·log 0.3 + 0.0333·log 0.2) = 0.7407`. The decomposition says
this should equal `(1-ε)·H(q, p) + ε·H(u, p) = 0.9·(-log 0.5) + 0.1·(mean of -log 0.5, -log 0.3, -log 0.2)
= 0.9·0.6931 + 0.1·1.1688 = 0.7407`. They agree to the digits I carried, so the split is real and the
gradient inherits it: `∂H(q',p)/∂z_k = p_k - q'(k) = (1-ε)·(p_k - δ_{k,y}) + ε·(p_k - 1/V)`, exactly
`(1-ε)` times the maximum-likelihood gradient plus `ε` times a uniform "flatten toward the prior" pull.
That decomposition is what I will lean on to reason about strength.

The structure also lets me place label smoothing among its target-side siblings and confirm I have the
right member, because "floor the target" is not the only way to say "stop demanding probability one." The
closest relative is a confidence penalty — add `-β·H(p) = β·Σ_k p_k log p_k` to the loss, rewarding the
model directly for keeping entropy in its prediction. Against the decomposition I just wrote these are
near mirror images: smoothing adds `ε·H(u,p) = ε·D_KL(u‖p) + const`, penalizing the KL *from* uniform to
the model, while the confidence penalty is `β·Σ p log p = β·D_KL(p‖u) + const`, the same divergence read
in the opposite direction. But they behave differently in ways that matter for a first rung. Differentiate
the confidence penalty and the per-logit pull works out to `β·p_k·(log p_k + H(p))`, weighted by `p_k`,
so it concentrates on the already-confident tokens and vanishes on the tail — where `p_k log p_k → 0` — 
never actively flooring the rare tokens, only bleeding mass off the peak, and it has no closed-form
optimum I can write down, since the balance point depends on `β` and the data together. Label smoothing's
uniform pull `ε·(p_k - 1/V)` is bounded, even-handed, and literally floors every coordinate toward `1/V`,
and it has the finite closed-form optimum `p = q'` I just solved for. A second sibling would smooth toward
the *unigram* token frequency instead of uniform, flooring common tokens more than rare ones — but that
bakes corpus statistics into the target, a data-dependent choice I would have to estimate and defend, and
it breaks the clean `H(u,p) = mean_k(-log p_k)` identity that lets me skip materializing the target
entirely. So among the target-side options, uniform label smoothing is the right member for the cheapest
rung: the same over-confidence cure, a bounded even-handed gradient, no extra coefficient beyond `ε`, no
data-dependent estimation, and an optimum I can write in closed form — exactly the surgical one-liner I
wanted to calibrate the ladder with.

Now the part that is specific to *this* task and is exactly where the harness diverges from the generic
recipe, and I have to get it right or I am cheating. The validation metric is the honest modeling
cross-entropy on FineWeb — plain `-log p_y`, no smoothing. Label smoothing deliberately fits the model
to a *softened* distribution rather than the data, so a model trained to put `1-ε` on the true token and
`ε/(V-1)` on the rest is a worse density estimator under the true one-hot likelihood than a model trained
on the data directly; that is a known and accepted trade in the original setting, where smoothing buys
calibration and downstream gains at the cost of raw likelihood. But here I am *graded* on that raw
likelihood. If I left the smoothing on during evaluation, the reported number would be the cross-entropy
against the softened target, not against the data, and that would be exactly the "do not lower the
reported loss by distorting the distribution" violation the contract forbids. So the smoothing has to be
applied **only during training**: when gradients are enabled I smooth, and under the no-grad evaluation
pass I fall back to standard cross-entropy, so `val_loss` is computed against the true targets and stays
comparable across every method on the ladder. PyTorch's `F.cross_entropy` exposes `label_smoothing`
directly, so the whole edit is one call with the smoothing coefficient gated on `torch.is_grad_enabled()`.
The full scaffold function is in the answer.

Two correctness points on that one-line call, because a library call still has to do exactly what I
reasoned about or the mechanism I just built is not the mechanism that runs. First, the gate itself:
`torch.is_grad_enabled()` returns `True` inside the training forward and `False` inside the
`@torch.no_grad()` evaluation forward, so `smoothing = 0.05 if torch.is_grad_enabled() else 0.0` yields
`ε = 0.05` while training and plain cross-entropy at eval, from one function body with no branching I have
to keep in sync — the same `compute_loss` serves both passes, which is precisely what makes the reported
`val_loss` an honest number I can line up against the rest of the ladder. Second, the ignore index has to
compose correctly with smoothing. `F.cross_entropy(..., ignore_index=-1, label_smoothing=0.05)` must not
bleed the `ε/V` floor onto the `-1` packed-boundary positions, and it does not: positions whose target is
`-1` are dropped from *both* the hard `H(q,p)` term and the uniform `H(u,p)` term before the mean, so the
smoothing floor is applied only where there is a real next token and the reduction denominator is the
count of valid positions. That matters quantitatively — if the library smoothed the ignored positions too,
it would be adding an `ε·H(u,p)` pull at boundaries the model should not be learning from, and averaging
over a larger denominator, which would silently rescale the effective `ε` away from the `0.05` I chose on
purpose. The masking is what keeps `ε` meaning what I set it to.

And I can put a number on what that training-time bias actually costs, which is the thing the eval split
protects the *reported* number from but cannot undo in the *trained model*. The extra term I am adding
to the training objective is `ε·H(u, p) = ε·mean_k(-log p_k)`. For a trained language model most of the
50k vocabulary entries carry tiny probability, so `-log p_k` for a typical tail token is on the order of
`log V ≈ 10.8` nats, and the uniform mean over the whole vocabulary is dominated by that tail; call it
order 10-to-11 nats. So at `ε = 0.05` the smoothing term contributes roughly `0.05·11 ≈ 0.5` nats to the
training loss, and its gradient `ε·(p - u)` is a steady pull flattening the prediction toward uniform.
That is not negligible — half a nat of off-data objective riding on top of a cross-entropy that is itself
only a couple of nats — and it is exactly the bias I am hoping the regularization repays. On a
single-epoch, compute-optimal run with muted overfitting, whether it repays is genuinely in doubt, which
is why I want this rung as the honest probe rather than as a method I am confident wins.

Let me also pin the floor that bias sits on, because it sharpens what "off-data" actually costs the
weights. The smoothed target `q'` carries its own entropy,
`H(q') = -q'(y) log q'(y) - (V-1)·(ε/V)·log(ε/V)`. With `q'(y) = 1 - ε + ε/V = 0.95001` the first term is
`-0.95001·log(0.95001) = 0.0487`; the tail carries `(V-1)·(ε/V) = 0.05·(50256/50257) ≈ 0.050` of mass at
per-token log-probability `log(ε/V) = log(9.95·10⁻⁷) = -13.82`, contributing `-0.050·(-13.82) = 0.691`;
so `H(q') ≈ 0.74` nats. That is the *irreducible* training cross-entropy the smoothing bakes in: even a
perfect predictor that outputs exactly `q'` scores `H(q', q') = H(q') ≈ 0.74` nats on the smoothed
objective, where against the true one-hot the floor is `0`. The eval split spares the *reported*
`val_loss` from that 0.74-nat floor — I grade against the one-hot — but the *weights* are still trained
toward a target whose best achievable point sits three-quarters of a nat off the data, and the eval split
cannot undo that in the model, only in the number. And netting the bias honestly against plain
cross-entropy rather than counting the added term alone: the smoothed objective is
`H(q,p) + ε·(H(u,p) - H(q,p))`, so the true excess over plain cross-entropy at a fixed prediction is
`ε·(H(u,p) - H(q,p)) ≈ 0.05·(10.8 - 2.3) = 0.43` nats — a hair under the crude `ε·H(u,p) ≈ 0.5` estimate
above, because the `(1-ε)` downweighting of the hard term gives a little of it back. Either accounting
lands in the same place: order half a nat of off-data pull riding on the objective, and that is exactly
the debt the regularization has to repay out of a single-epoch run where the overfitting it prevents is
already muted.

The second divergence from the canonical recipe is the coefficient, and now I can choose it from the
arithmetic rather than by taste. The original Inception/Transformer setting uses `ε = 0.1`. Compare the
two levers `ε` moves. The optimal-gap ceiling I computed is `log((1-ε)(V-1)/ε) = log(V-1) + log((1-ε)/ε)`,
and for a 50k vocabulary the `log(V-1) ≈ 10.8` term dominates: at `ε = 0.05` the ceiling is 13.77 nats,
at `ε = 0.1` it is `log(0.9·50256/0.1) = 13.02` nats — doubling `ε` tightens the confidence ceiling by
only 0.75 of a nat, because it moves only through the `log((1-ε)/ε)` term, `log 19 = 2.94` versus
`log 9 = 2.20`. So the choice of `ε` is *not* really a choice about where the confidence cap sits; both
values leave it near fourteen nats out. What `ε` actually controls is the off-data bias: the prior-pull
weight `ε/(1-ε)` goes from `0.05/0.95 = 0.053` to `0.1/0.9 = 0.111`, more than doubling, and the half-nat
smoothing term I just estimated becomes a full nat. On a short single-epoch run where the overfitting a
heavy regularizer would prevent barely develops, doubling the bias to buy three-quarters of a nat of
ceiling I do not need is a bad trade. So `ε = 0.05` is the task-local setting: enough of a floor to take
the sharpest edge off the runaway, light enough that the `(1-ε)` reweighting of the true-likelihood term
is barely perturbed and the prior-pull bias stays near half a nat. This rung is therefore not canonical
label smoothing — it is `ε = 0.05`, training-only, via the library's `label_smoothing` argument with the
`-1` ignore index the harness uses for packed boundaries.

Now reason about what this floor must do, because that is the entire point of running it first, and it is
also the crisp statement of what it *cannot* reach. Smoothing attacks the target, and through the target
it caps the *gap* between the true-class logit and the rest — the difference the softmax actually sees,
which I just pinned at about 13.8 nats. What it pointedly does *not* touch is the absolute *level* of the
logits. A model can sit at logits `(1000, 990, 990, …)` and have exactly the smoothed-optimal gaps while
being numerically enormous, because cross-entropy and its smoothed cousin are both invariant to a uniform
shift of all logits: adding a constant `c` to every logit multiplies numerator and denominator of the
softmax by `e^c` and changes nothing, so `q'` and `p` both see only differences and the whole level is a
free coordinate the smoothed loss says nothing about. Let me make that concrete rather than assert it —
`softmax(z + c·1)_k = e^{z_k+c}/Σ_j e^{z_j+c} = e^c·e^{z_k}/(e^c·Σ_j e^{z_j}) = softmax(z)_k`, so `p` is
literally the identical vector for every `c`, and therefore `H(q,p)`, `H(u,p)`, and their `ε`-blend are
all exactly constant along the whole ray `z + c·1`. Gradient descent on the smoothed loss feels precisely
zero force in that direction; the level is not merely unpenalized, it is invisible to this objective. Smoothing is a statement about differences and it
leaves the overall magnitude — the gauge — completely free. That matters because the run trains in
bfloat16, where large logit magnitudes are where the exponential in the softmax becomes unfaithful:
bfloat16 keeps float32's exponent range but far fewer mantissa bits, so once the logits are large the
rounding is coarse right where they enter the `exp`, and smoothing does nothing about any of it because
it never constrained the level in the first place. It also leaves a real question about whether trading
away true-likelihood for a milder gap is even the right trade when the *evaluation* is true likelihood: I
am training against a distribution offset by that half-nat prior-pull, and only the regularization
benefit, if any, can repay it.

So here is what I expect, stated against outcomes I can be wrong about — and I have no measured numbers
in front of me yet, so these are directions, not values. On the primary `val_loss`, light training-only
smoothing should land *close* to plain cross-entropy — within a few hundredths of a nat — and it is
genuinely unclear whether it lands above or below, because the regularization benefit on a single-epoch
run is small and it is fighting the half-nat bias I have put into the objective; my honest prior is that
the bias slightly wins and this comes in a touch *behind* plain cross-entropy, but I would not stake much
on the sign. The two perplexity metrics, WikiText-2 and LAMBADA, should track `val_loss` in the same
direction, since they are the same next-token likelihood on other corpora — and quantitatively, because
perplexity is `exp` of a per-token cross-entropy, a swing of a few hundredths of a nat in `val_loss` maps
to a few-percent swing in those perplexities, so if smoothing moves `val_loss` by `+0.0X` I should see
WikiText-2 and LAMBADA move by roughly `X` percent in the same direction rather than by some unrelated
amount; a perplexity move much larger than the `val_loss` move would be a sign something other than the
smoothing is in play. On the downstream accuracies
(ARC-Easy, HellaSwag, PIQA, WinoGrande), where a better-calibrated next-token distribution can help even
when likelihood does not, I would expect smoothing to be at worst neutral. If this rung comes in *behind*
a method that attacks the logit level or the logit values directly, the diagnosis is already written:
smoothing pulled on the gap, the gauge-invariant handle, and left untouched the absolute magnitude that
the bfloat16 softmax actually cares about — which says the next rung should stop attacking the target and
start attacking the level. This is the weakest rung on the ladder by design: it is the most conservative,
target-side over-confidence fix, and whatever it leaves on the table is precisely what the level-side and
value-side interventions are for.
