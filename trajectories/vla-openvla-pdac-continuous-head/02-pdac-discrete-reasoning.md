The diffusion rung landed where the budget said it would. On seed 42 it scored 0.1111 on goal_a, a
flat 0.0000 on goal_b, and 0.0833 on goal_c — geometric mean zero, because one empty subset zeroes the
product, and even the arithmetic mean is (0.1111 + 0 + 0.0833)/3 ≈ 0.0648, barely off the floor. It is
worth decoding those fractions into raw events, because they are almost too small to be rates. Each
subset is a handful of tasks at 3 trials: goal_a and goal_b are three tasks (9 rollouts each), goal_c is
four tasks (12 rollouts). So 0.1111 is exactly 1/9, 0.0000 is 0/9, and 0.0833 is 1/12 — the diffusion
policy completed one rollout out of nine on goal_a, none on goal_b, and one out of twelve on goal_c: two
successful rollouts out of thirty attempts, total. That is not a policy that half-works; it is a policy
at the noise floor that got lucky twice. The eval timers tell the rest of the story: 552, 567, 707
seconds per subset, which per rollout is 552/9 ≈ 61 s on goal_a and 707/12 ≈ 59 s on goal_c — call it a
minute of wall-clock per rollout, against the tens of seconds a single-pass head would take, exactly the
~50-trunk-passes-per-chunk tax of the reverse rollout. And `elapsed_train = 17306 s` is 4.8 hours, which
against the 5 H200h validity ceiling means the diffusion path spent essentially the entire budget just
training one seed, with the 50-step eval rollout piling ~1826 s more on top. This is precisely the
under-training I expected — the denoiser never learned, in 6000 steps, to walk noise back to coherent
action chunks, so the sampled actions are close to noise and the policy mostly fails. The lesson is
sharp: in this regime the deciding factor is not the *expressivity* of the head but whether the head can
be *fit* inside 6000 steps. Multimodality is worthless if the model that represents it is still noise.
So the next move is to stop modeling a distribution and ask for a head whose training signal is as
direct and data-efficient as possible — and the most direct, most data-grounded signal available on
this trunk is the model's *own* objective, the one it was pretrained with.

Step back to what the base trunk actually is. It is a web-pretrained vision-language model that
generalizes over novel objects, distractors, and rephrased instructions because it has seen trillions
of tokens, and its single native skill is next-token prediction under cross-entropy over a fixed
vocabulary. The diffusion head grafted a from-scratch denoiser — on the order of 200M randomly
initialized parameters — and a from-scratch MSE-on-noise objective onto that trunk; the discrete path
does the opposite — it keeps the trunk's mouth and its objective exactly, and only teaches it to *speak
actions* as tokens. The tension is one sentence: the model emits discrete tokens, and a robot action is
a vector of continuous reals. If the output must be a token and the action is a real number, the only
way through is to turn the real number into a token — give up the continuum and quantize.

How much do I lose by quantizing? The robot does not need infinite precision; it needs resolution below
the noise and tolerance the gripper and controller already carry. So discretize each of the 7 action
dimensions: chop a dimension's range into bins, and the action becomes "which bin," an integer, an
integer I can name with a token. Seven dimensions, seven integers, seven tokens; an action is a short
string of 7 tokens, and a chunk of K = 8 actions is 56 tokens. How many bins is a rate-distortion call —
too few and the control is chunky, too many and I ask a categorical head to discriminate among hundreds
of near-identical classes from a few hundred demonstrations.

Let me put numbers on that trade rather than reach for the default. On the normalized [−1,+1] scale the
range is 2.0, so with B bins the width of one bin is 2/B and the reconstruction error of a uniform
quantizer is bounded by half a bin, with root-mean-square error (bin width)/√12. At B = 256 that is a
bin width of 2/256 = 0.0078 and an RMS error of 0.0078/√12 ≈ 0.00226 per dimension — 256 levels is 8
bits per degree of freedom, finer than a teleoperated demonstrator's own repeatability, so the
quantization error is dominated by other error sources and effectively free. Halving to B = 128 (7 bits)
doubles the RMS error to ≈0.0045 and starts to bite on fine motions; doubling to B = 512 (9 bits) barely
improves distortion below the noise floor while doubling the number of near-identical classes the head
must separate from the same few hundred demonstrations, and — as I am about to rely on — it would need
512 spare tokens where I have room for a couple hundred. So 256 sits at the knee: 8 bits is a tame
256-way classification for a 7B model and one bin index fits one token cleanly, while the distortion is
already under the actuators' repeatability.

Where do I put the bins? Uniform bins from the raw min to max are wrong, because demonstration data has
heavy tails — an occasional teleop glitch way out at the edge — and a single outlier stretches the
interval so almost all 256 bins cover empty extremes and the dense bulk gets squeezed into a handful.
Concretely: if one glitched sample sits at 5.0 on the normalized scale while the real motion lives in
[−0.8, +0.8], uniform bins from −5 to +5 have width 10/256 ≈ 0.039, five times coarser than necessary,
and every bin outside [−0.8, 0.8] is dead. So set the interval by quantiles: clip to the 1st/99th
percentile of the actions on that axis (on the normalized [−1,+1] scale), lay 256 uniform bins across
the clipped interval, and let outliers clamp to the end bins. Uniform within the clipped interval keeps
encoding (a digitize) and decoding (a center lookup) trivial, and the clip restores the ≈0.0078 width
where the data actually is.

I should also ask whether *uniform* bins inside the clipped interval are the right shape, because there
is a tempting alternative: companding, μ-law-style non-uniform bins that pack more levels near zero where
small corrective motions live and spread them out at the extremes where only coarse reaches happen. That
would put resolution where fine manipulation needs it. But run the numbers before reaching for it — the
uniform grid already gives ≈0.00226 RMS error per dimension, which is under the demonstrator's own
repeatability, so companding would be buying precision below the noise floor of the data that supervises
it: no measurable gain, and it complicates decode from a single center lookup into an inverse-companding
curve, plus it makes the bin↔token arithmetic no longer a clean subtraction. The whole appeal of this
rung is that encode and decode are one line each; a non-uniform grid trades that away for a precision I
cannot even resolve. So uniform-within-clip it stays, and if precision is the true ceiling I will attack
it by removing the grid entirely rather than by curving it.

Now the hard part: make those integers be tokens the model *already has*. I could add 256 new
vocabulary entries, but the Llama tokenizer reserves only ~100 special slots and adding rows to the
embedding matrix means more randomly-initialized parameters — the very from-scratch cost I am trying to
avoid by staying in the trunk's native objective. And I can put a number on that cost: 256 new tokens
means 256 new embedding rows of width 4096 and 256 new output-projection columns of width 4096, so
2·256·4096 ≈ 2.1M freshly random parameters that the trunk has never seen a gradient for — small next to
the 7B trunk but exactly the kind of cold-start liability the diffusion rung punished. A BPE vocabulary
is ordered by frequency, and its tail is junk: token identities the pretrained model has barely ever
produced, carrying almost no learned meaning. So the cheapest tokens to overwrite are the last ones —
take the final 256 vocabulary entries and let them *be* the action bins. The model loses almost nothing
(it never used them), and I introduce zero new parameters: I reuse existing embedding rows and
output-projection columns, teaching them a new job during fine-tuning. I pin the index arithmetic to a
self-inverse map: bin index k of a dimension maps to token-id = vocab_size − k, so the action tokens
occupy the contiguous range [vocab_size − 256, vocab_size), and the bin↔token map is "subtract from
vocab_size," its own inverse — apply it twice, vocab_size − (vocab_size − k) = k, and I am back where I
started, so encode and decode share one line of arithmetic and cannot disagree.

Training is then *nothing new*, which is the entire point of starting from the trunk's own objective.
Every demonstration action becomes 7 action tokens appended to the prompt; the model is already a
teacher-forced next-token predictor with cross-entropy; I mask the loss to the action positions (the
prompt is set to the ignore index so I do not train the model to "predict" the given image and
instruction) and evaluate categorical cross-entropy on the action tokens only. No new loss function, no
denoiser, no sampler — the runtime even exposes this as `compute_discrete_objective`, because it is the
standard discrete path. And there is a real argument that cross-entropy over unordered bins is a *good*
control loss, not merely a convenient one: it treats the 256 bins as unordered classes, so it can put
mass on two separated bins at a fork and commit to one at decode — the categorical distribution
represents the same multimodality diffusion chased, natively, at the cost only of a smoothness prior the
giant model can learn from data anyway. The difference from the diffusion rung is decisive on the fit
axis: this multimodal representation rides the trunk's *pretrained* objective and adds zero new
parameters, so all 192,000 sample-gradient exposures (effective batch 32 × 6000 steps) land on the exact
cross-entropy the trunk already mastered, rather than being split across 50 noise levels feeding a
random-init denoiser. That is the fit-inside-6000-steps property I am after, bought by staying inside the
trunk's own competence.

There is a subtlety in how much of that multimodality actually survives to control, and I should not
oversell it. The categorical head *learns* a multimodal distribution — at a fork it can keep probability
on two separated bins — but at eval I decode greedily, argmax per position, which commits to the single
most-probable bin on each axis independently. That is the right call for a controller (I want one
decisive action, not a blurred average, so unlike the L2 point head I never emit the empty valley between
modes), but it does mean the benefit is "commit cleanly to the dominant mode," not "sample the full
joint." And the per-axis independence is a real limit: if two action dimensions are jointly bimodal —
reach-left-and-tilt-up versus reach-right-and-tilt-down — argmaxing each dimension on its own can splice
the left reach with the down tilt, a combination neither mode endorsed. Bidirectional attention over the
56 slots gives the logits a chance to correlate, but greedy per-position decode cannot guarantee a
coherent joint. So the honest read is that discrete inherits diffusion's *representational* multimodality
in training while spending it on decisive single-mode commitment at decode — good enough to clear a
floor, not the same thing as sampling a joint distribution.

But I should be honest about the flip side of "unordered classes," because it is what will cap this rung.
Cross-entropy does not know the bins are ordered. Suppose the true bin for some dimension is 131. A head
that instead puts its mass on bin 130 is off by one bin width, ≈0.0078 — a near-perfect action — while a
head that puts its mass on bin 12 is off by ≈0.93, a gross error. Under categorical cross-entropy over
unordered classes those two mistakes incur the *same* loss for the same predicted probability on the
wrong class: the objective sees "wrong class" and "wrong class," never "near miss" versus "disaster." So
the head gets no gradient pressure toward being *close* when it is wrong; it only learns right-class
versus wrong-class. On easy, decisive motions that is fine — the argmax is usually the right bin — but on
fine manipulation where the useful signal lives in the sub-bin gap between grid points, the loss is blind
to exactly the precision that matters. That blindness plus the 8-bit grid is the ceiling I am accepting
in exchange for the fast fit.

The masking deserves one more beat, because it is where "reuse the pretrained objective" becomes a
concrete gradient-routing decision. Teacher forcing feeds the whole sequence — image tokens, instruction
tokens, then the 56 action tokens — and the cross-entropy is computed at every position by default. If I
left it that way I would be spending gradient on teaching the model to "predict" the given image and
instruction, which are inputs, not outputs, and worse, the enormous prompt would swamp the 56 action
positions in the loss average. So the prompt positions are set to the ignore index and only the 56 action
tokens contribute, which is exactly what `compute_discrete_objective` does. That routing is what makes
the 192,000 exposures count: every one of them lands entirely on the action prediction the trunk must
actually perform at eval, undiluted by prompt-reconstruction terms. And there is a budget prediction that
falls out of this before I ever run it — since the head adds no parameters and the discrete forward is a
single trunk pass with a masked cross-entropy, training wall-clock is dominated by the 7B trunk's
forward/backward, not by anything the action method does. So `elapsed_train` should sit close to the
diffusion rung's ≈17300 s regardless of the head swap; what should move dramatically is *eval*, because
that is where diffusion paid its 50× rollout tax and discrete does not.

Decode is where the off-by-ones live, and this task's edit surface makes them explicit. At the action
positions the model gives a distribution over the *entire* vocabulary, but an action token is only ever
one of the last 256; a natural-language token at an action slot is meaningless as control. So I must not
argmax over the whole vocabulary — I slice the logits to the action sub-vocabulary
[vocab_size − num_bins, vocab_size) and argmax *within that slice*, then add the offset back to recover
the actual token-id. This is the one place the task's `decode_discrete_actions` differs from the naive
scaffold default, and the difference is load-bearing: the default example in the scaffold argmaxes over
all logits, which can select a non-action token and silently corrupt the decode; the official discrete
baseline restricts the argmax to the valid action block (and raises if the vocab bounds are
inconsistent). Greedy argmax, not sampling — for control I want the single most-likely bin per
dimension, deterministic.

Then invert the encoding with the same subtraction: bin_index = vocab_size − token_id. Now the
bin-vs-center care, and it is a genuine off-by-one I have to get right or every decoded action drifts by
a bin. 256 bin *edges* define only 255 intervals, so there are 255 bin *centers*, not 256. The digitize
returned indices in [1, 256]; subtract 1 to land in [0, 255], but I have only 255 centers (indices
[0, 254]), so I clip the top index down into the last real interval — clip(vocab_size − token_id − 1, 0,
num_centers − 1) with num_centers = bin_centers.shape[0] = 255. Trace the extreme case to be sure: an
action at the very top edge digitizes to 256, minus 1 is 255, which is one past the last center index
254 and would index out of bounds — the clip pulls it back to 254, the last real center, which is the
correct minimum-distortion reconstruction for anything in the top interval. The decoded value for a bin
is its *center*, the minimum-distortion reconstruction for a uniform quantizer, looked up from the
precomputed `runtime.model.bin_centers`; reshape to (batch, NUM_ACTIONS_CHUNK, ACTION_DIM) = (B, 8, 7)
and hand back normalized actions in [−1,+1] for the runtime to un-normalize. Let me confirm the decode
shapes close: `text_logits[:, start:end]` over the 56 action positions is (B, 56, vocab_size); slicing
the last 256 columns and argmaxing gives (B, 56) bin indices, and after the map-and-clip the center
lookup reshapes cleanly to (B, 8, 7), which is exactly the chunk shape the harness un-normalizes. The
dimensions line up.

Two scaffold details complete the wiring and both follow from the discrete path conditioning on no extra
action-side inputs: the action tokens are predicted from the prompt context, so I fill the
action-feature slots with zeros (`zero_action_token_features`, a (B, 56, llm_dim) zero tensor passed as
`action_token_features` to `runtime.forward`), and the method holds *no parameters of its own* — its
`__init__` is empty, because all learning is fine-tuning the trunk under its own objective. That empty
`__init__` is the quantitative heart of the fit argument: zero new parameters against the diffusion
rung's ~200M, so there is nothing cold to warm up. The discrete decode extends across the chunk
unchanged: the binning, the least-used-token map, the cross-entropy, and the center lookup are all
agnostic to how many action positions there are, so 56 slots are decoded exactly like one.

Now the falsifiable expectation against the diffusion numbers. The discrete head shares the diffusion
head's multimodality but pays none of its training cost: it reuses the pretrained next-token objective,
adds zero parameters, and decodes in a single trunk pass instead of 50. So I expect two things. First,
eval times collapse: where diffusion spent ~60 s per rollout on its 50-step reverse loop and 552/567/707
s per subset, a single decode pass should drop the per-subset eval back into the roughly 130–170 s range,
because there is no reverse loop — one trunk forward per chunk, not fifty. Second, success should clear
the diffusion floor: where diffusion managed 1/9, 0/9, 1/12 — two successful rollouts in thirty — the
discrete head, actually fit in 6000 steps because it never started from scratch, should register
non-trivial success on more than one subset, with an arithmetic mean comfortably above diffusion's
≈0.065. I do *not* expect it to be the ceiling, though, and the reason is the 256-bin grid together with
the ordinal blindness I traced above: cross-entropy over unordered bins throws away the fact that bin 130
is a near-miss when the truth is 131, and the 8-bit grid caps precision at ≈0.0078, exactly the level
fine manipulation lives below. So the prediction I am committing to is asymmetric: discrete should *beat
diffusion clearly* in this budget, yet still leave a precision gap traceable to the grid and the ordinal
blindness — a gap I cannot close without giving up the discretization itself, since it is baked into
every one of these choices. If discrete fails to clear the diffusion floor, my whole "fit-not-
expressivity is the bottleneck" diagnosis is wrong; if it clears it but does not dominate, then the
discretization ceiling — not the fit — is what is left to attack.
