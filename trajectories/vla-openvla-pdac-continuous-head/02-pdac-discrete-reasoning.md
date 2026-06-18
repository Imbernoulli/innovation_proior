The diffusion rung landed where the budget said it would. On seed 42 it scored 0.1111 on goal_a, a
flat 0.0000 on goal_b, and 0.0833 on goal_c — geometric mean zero, because one empty subset zeroes the
product, and even the arithmetic mean is ~0.065, barely off the floor. The eval timers tell the rest of
the story: 552, 567, 707 seconds per subset against the tens of seconds a single-pass head would take,
exactly the ~50-trunk-passes-per-chunk tax of the reverse rollout. This is precisely the under-training
I expected — the denoiser never learned, in 6000 steps, to walk noise back to coherent action chunks,
so the sampled actions are close to noise and the policy mostly fails. The lesson is sharp: in this
regime the deciding factor is not the *expressivity* of the head but whether the head can be *fit*
inside 6000 steps. Multimodality is worthless if the model that represents it is still noise. So the
next move is to stop modeling a distribution and ask for a head whose training signal is as direct and
data-efficient as possible — and the most direct, most data-grounded signal available on this trunk is
the model's *own* objective, the one it was pretrained with.

Step back to what the base trunk actually is. It is a web-pretrained vision-language model that
generalizes over novel objects, distractors, and rephrased instructions because it has seen trillions
of tokens, and its single native skill is next-token prediction under cross-entropy over a fixed
vocabulary. The diffusion head grafted a from-scratch denoiser and a from-scratch MSE-on-noise
objective onto that trunk; the discrete path does the opposite — it keeps the trunk's mouth and its
objective exactly, and only teaches it to *speak actions* as tokens. The tension is one sentence: the
model emits discrete tokens, and a robot action is a vector of continuous reals. If the output must be
a token and the action is a real number, the only way through is to turn the real number into a token —
give up the continuum and quantize.

How much do I lose by quantizing? The robot does not need infinite precision; it needs resolution below
the noise and tolerance the gripper and controller already carry. So discretize each of the 7 action
dimensions: chop a dimension's range into bins, and the action becomes "which bin," an integer, an
integer I can name with a token. Seven dimensions, seven integers, seven tokens; an action is a short
string. How many bins is a rate-distortion call — too few and the control is chunky, too many and I ask
a categorical head to discriminate among hundreds of near-identical classes from a few hundred
demonstrations. 256 bins per dimension is the established choice and it checks out: 256 levels is 8 bits
per degree of freedom, finer than the controller's own repeatability, so the quantization error is
dominated by other error sources and effectively free, while 256 classes is a tame classification
problem for a 7B model and one bin index fits one token cleanly.

Where do I put the bins? Uniform bins from the raw min to max are wrong, because demonstration data has
heavy tails — an occasional teleop glitch way out at the edge — and a single outlier stretches the
interval so almost all 256 bins cover empty extremes and the dense bulk gets squeezed into a handful. So
set the interval by quantiles: clip to the 1st/99th percentile of the actions on that axis (on the
normalized [−1,+1] scale), lay 256 uniform bins across the clipped interval, and let outliers clamp to
the end bins. Uniform within the clipped interval keeps encoding (a digitize) and decoding (a center
lookup) trivial.

Now the hard part: make those integers be tokens the model *already has*. I could add 256 new
vocabulary entries, but the Llama tokenizer reserves only ~100 special slots and adding rows to the
embedding matrix means more randomly-initialized parameters — the very from-scratch cost I am trying to
avoid by staying in the trunk's native objective. A BPE vocabulary is ordered by frequency, and its tail
is junk: token identities the pretrained model has barely ever produced, carrying almost no learned
meaning. So the cheapest tokens to overwrite are the last ones — take the final 256 vocabulary entries
and let them *be* the action bins. The model loses almost nothing (it never used them), and I introduce
zero new parameters: I reuse existing embedding rows and output-projection columns, teaching them a new
job during fine-tuning. I pin the index arithmetic to a self-inverse map: bin index k of a dimension
maps to token-id = vocab_size − k, so the action tokens occupy the contiguous range
[vocab_size − 256, vocab_size), and the bin↔token map is "subtract from vocab_size," its own inverse.

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
giant model can learn from data anyway. The difference from the diffusion rung is that this multimodal
representation rides the trunk's *pretrained* objective, so it has a head start instead of starting from
scratch — exactly the fit-inside-6000-steps property I am after.

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
bin-vs-center care: 256 bin *edges* define only 255 intervals, so there are 255 bin *centers*, not 256.
The digitize returned indices in [1, 256]; subtract 1 to land in [0, 255], but I have only 255 centers,
so I clip the top index down into the last real interval — clip(vocab_size − token_id − 1, 0,
num_centers − 1) with num_centers = bin_centers.shape[0]. The decoded value for a bin is its *center*,
the minimum-distortion reconstruction for a uniform quantizer, looked up from the precomputed
`runtime.model.bin_centers`; reshape to (batch, NUM_ACTIONS_CHUNK, ACTION_DIM) and hand back normalized
actions in [−1,+1] for the runtime to un-normalize.

Two scaffold details complete the wiring and both follow from the discrete path conditioning on no extra
action-side inputs: the action tokens are predicted from the prompt context, so I fill the
action-feature slots with zeros (`zero_action_token_features`, a (B, K·D, llm_dim) zero tensor passed as
`action_token_features` to `runtime.forward`), and the method holds *no parameters of its own* — its
`__init__` is empty, because all learning is fine-tuning the trunk under its own objective. The discrete
decode extends across the chunk unchanged: the binning, the least-used-token map, the cross-entropy, and
the center lookup are all agnostic to how many action positions there are, so K·D slots are decoded
exactly like one.

Now the falsifiable expectation against the diffusion numbers. The discrete head shares the diffusion
head's multimodality but pays none of its training cost: it reuses the pretrained next-token objective,
adds zero parameters, and decodes in a single trunk pass instead of 50. So I expect two things. First,
eval times collapse from the ~550-700 s/subset of the diffusion rollout back to the single-pass range
(roughly 130-170 s/subset on this protocol), because there is no reverse loop. Second, success should
clear the diffusion floor: where diffusion sat at 0.1111/0.0000/0.0833, the discrete head — actually
fit in 6000 steps because it never started from scratch — should register non-trivial success on more
than one subset, with an arithmetic mean comfortably above diffusion's ~0.065. I do *not* expect it to
be the ceiling, though, and the reason is the 256-bin grid: cross-entropy over unordered bins throws
away the ordinal structure of a continuous action (it does not know bin 130 is closer to the truth than
bin 12 when the answer is 131), and the 8-bit grid caps precision at exactly the level fine manipulation
lives below. So the prediction that sets up the next rung is asymmetric: discrete should *beat
diffusion clearly* in this budget, yet still leave a precision gap that a head regressing the continuous
action directly — no grid, no ordinal blindness — should be able to close. If discrete fails to clear
the diffusion floor, my whole "fit-not-expressivity is the bottleneck" diagnosis is wrong; if it clears
it but does not dominate, the discretization ceiling is the next thing to attack.
