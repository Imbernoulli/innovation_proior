The diffusion rung landed where the budget said it would. On seed 42 it scored 0.1111 on goal_a, 0.0000
on goal_b, and 0.0833 on goal_c — geometric mean zero, since one empty subset zeroes the product, and
even the arithmetic mean is ≈0.065. Decoding those fractions into raw events makes the picture sharp:
each subset is a handful of tasks at 3 trials, so goal_a and goal_b are 9 rollouts each and goal_c is 12,
which makes 0.1111 exactly 1/9, 0.0000 is 0/9, and 0.0833 is 1/12 — two successful rollouts out of
thirty, total. That is not a policy that half-works; it is one at the noise floor that got lucky twice.
The eval timers confirm the cost: 552/567/707 s per subset, ≈60 s per rollout against the tens of seconds
a single-pass head would take — the ~50-trunk-passes-per-chunk tax of the reverse rollout. And
`elapsed_train` = 17306 s is 4.8 hours, nearly the entire 5 H200h ceiling spent training one seed. This
is exactly the under-training I expected: the denoiser never learned in 6000 steps to walk noise back to
coherent chunks. The lesson is sharp — in this regime the deciding factor is not the head's *expressivity*
but whether it can be *fit* inside 6000 steps. Multimodality is worthless if the model representing it is
still noise. So the next move is to stop modeling a distribution and ask for a head whose training signal
is as direct as possible — and the most data-grounded signal on this trunk is the model's own pretrained
objective.

Step back to what the base trunk is: a web-pretrained vision-language model that generalizes over novel
objects, distractors, and rephrased instructions because it has seen trillions of tokens, and whose
single native skill is next-token prediction under cross-entropy over a fixed vocabulary. The diffusion
head grafted a from-scratch ~200M-parameter denoiser and a from-scratch MSE-on-noise objective onto that
trunk; the discrete path does the opposite — keep the trunk's mouth and its objective exactly, and only
teach it to *speak actions* as tokens. The tension is one sentence: the model emits discrete tokens, and
a robot action is a vector of continuous reals. The only way through is to turn the real number into a
token — give up the continuum and quantize.

How much do I lose by quantizing? The robot needs resolution below the noise the gripper and controller
already carry, not infinite precision. So discretize each of the 7 dimensions: chop the range into bins,
and the action becomes "which bin," an integer I can name with a token — 7 dimensions, 7 tokens, and a
chunk of K = 8 actions is 56 tokens. How many bins is a rate-distortion call. On the normalized [−1,+1]
scale the range is 2.0, so B bins give width 2/B and root-mean-square quantization error (width)/√12. At
B = 256 that is width 2/256 = 0.0078 and RMS ≈ 0.00226 per dimension — 8 bits per degree of freedom,
finer than a teleoperated demonstrator's own repeatability, so quantization error is dominated by other
sources and effectively free. Halving to 128 doubles the error and starts to bite on fine motions;
doubling to 512 barely improves distortion below the noise floor while doubling the near-identical
classes the head must separate from a few hundred demonstrations, and it would need 512 spare tokens
where I have room for a couple hundred. So 256 sits at the knee: a tame classification for a 7B model, one
bin index per token, distortion already under the actuators' repeatability.

Where do I put the bins? Uniform bins from raw min to max are wrong: demonstration data has heavy tails,
and a single teleop glitch out at the edge stretches the interval so almost all 256 bins cover empty
extremes while the dense bulk is squeezed into a handful — a glitch at 5.0 when the real motion lives in
[−0.8, +0.8] gives width ≈ 0.039, five times coarser than needed, and every bin outside the bulk is dead.
So set the interval by quantiles: clip to the 1st/99th percentile of the actions on that axis, lay 256
uniform bins across the clipped interval, and let outliers clamp to the end bins. Uniform-within-clip
keeps encoding (a digitize) and decoding (a center lookup) trivial and restores the ≈ 0.0078 width where
the data actually is. Companding — μ-law-style non-uniform bins packing resolution near zero — would buy
precision below the demonstrator's own repeatability for no measurable gain while complicating decode from
a single center lookup into an inverse curve, so uniform stays; if precision is the true ceiling I will
remove the grid rather than curve it.

Now make those integers be tokens the model *already has*. Adding 256 new vocabulary entries means 256
new embedding rows and output-projection columns of width 4096 — 2·256·4096 ≈ 2.1M freshly random
parameters that the trunk has never seen a gradient for, exactly the cold-start cost the diffusion rung
punished, and the Llama tokenizer reserves only ~100 special slots anyway. But a BPE vocabulary is ordered
by frequency and its tail is junk — token identities the pretrained model has barely produced, carrying
almost no learned meaning. So overwrite the last 256: let the final vocabulary entries *be* the action
bins. The model loses almost nothing, and I add zero parameters, reusing existing embedding rows and
output-projection columns and teaching them a new job during fine-tuning. The map is self-inverse — bin
index k of a dimension ↦ token-id = vocab_size − k — so the action tokens occupy the contiguous range
[vocab_size − 256, vocab_size) and encode and decode share one line of arithmetic that cannot disagree.

Training is then *nothing new*, which is the whole point of starting from the trunk's own objective.
Every demonstration action becomes 7 tokens appended to the prompt; the model is already a teacher-forced
next-token predictor with cross-entropy; I mask the loss to the action positions — the prompt is set to
the ignore index, so I do not train the model to "predict" its own image and instruction, and the 56
action tokens are not swamped by the enormous prompt in the loss average — and evaluate categorical
cross-entropy on those tokens only. The runtime exposes exactly this as `compute_discrete_objective`. And
cross-entropy over unordered bins is a *good* control loss, not merely a convenient one: it can put mass
on two separated bins at a fork and commit to one at decode, representing the same multimodality diffusion
chased, natively. The decisive difference from diffusion is on the fit axis: this rides the *pretrained*
objective and adds zero parameters, so all 192,000 sample-gradient exposures (effective batch 32 × 6000
steps) land on the exact cross-entropy the trunk already mastered, rather than being split across 50 noise
levels feeding a random-init denoiser. And since the head adds no parameters and the forward is a single
trunk pass with a masked cross-entropy, training wall-clock is dominated by the 7B trunk's
forward/backward, so `elapsed_train` should sit close to the diffusion rung's ≈17300 s; what should move
dramatically is *eval*.

One honesty check on how much of that multimodality survives to control: the head *learns* a multimodal
distribution, but at eval I decode greedily, argmax per position, committing to the single most-probable
bin on each axis. That is right for a controller — one decisive action, never the empty valley between
modes — but it means the benefit is "commit cleanly to the dominant mode," not "sample the joint." And
per-axis independence is a real limit: if two dimensions are jointly bimodal (reach-left-and-tilt-up vs
reach-right-and-tilt-down), argmaxing each alone can splice the left reach with the down tilt, a
combination neither mode endorsed. Bidirectional attention over the 56 slots lets the logits correlate,
but greedy per-position decode cannot guarantee a coherent joint.

The flip side of "unordered classes" is what caps this rung. Cross-entropy does not know the bins are
ordered: if the true bin for a dimension is 131, a head that puts its mass on bin 130 (off by one bin,
≈ 0.0078) and one that puts it on bin 12 (off by ≈ 0.93) incur the *same* loss for the same predicted
probability on the wrong class. So the head gets no gradient toward being *close* when it is wrong — it
only learns right-class versus wrong-class. On easy, decisive motions the argmax is usually the right bin,
but on fine manipulation, where the useful signal lives in the sub-bin gap between grid points, the loss
is blind to exactly the precision that matters. That blindness plus the 8-bit grid is the ceiling I accept
in exchange for the fast fit.

Decode is where the off-by-ones live. At the action positions the model gives a distribution over the
*entire* vocabulary, but a valid action token is only ever one of the last 256; a natural-language token
at an action slot is meaningless as control. So I slice the logits to [vocab_size − num_bins, vocab_size),
argmax *within* that slice, and add the offset back — this is the one place `decode_discrete_actions`
departs from the scaffold default, which argmaxes over all logits and can silently select a non-action
token, and I raise if the vocab bounds are inconsistent. Greedy argmax, not sampling — one decisive bin
per dimension. Then invert with the same subtraction, bin_index = vocab_size − token_id, and mind the last
off-by-one: 256 bin *edges* define only 255 intervals, so there are 255 centers, and clip(vocab_size −
token_id − 1, 0, num_centers − 1) with num_centers = bin_centers.shape[0] = 255 pulls a top-edge action
(which digitizes to 256, then 255, one past the last center index 254) back to center 254, the correct
minimum-distortion reconstruction for the top interval. Read the decoded value — the bin's center — off
the precomputed `runtime.model.bin_centers` and reshape to (B, 8, 7) for the runtime to un-normalize.

Two wiring details follow from the discrete path conditioning on no action-side input: the action tokens
are predicted from the prompt context, so I fill the action-feature slots with a zero (B, 56, llm_dim)
tensor passed as `action_token_features`, and the method holds *no parameters of its own* — its
`__init__` is empty. That empty `__init__` is the quantitative heart of the fit argument: zero new
parameters against the diffusion rung's ~200M, nothing cold to warm up. The decode is agnostic to chunk
length, so 56 slots decode exactly like one.

So the falsifiable expectation against the diffusion numbers. Discrete shares diffusion's multimodality
but pays none of its training cost — pretrained objective, zero parameters, one trunk pass instead of 50.
First, eval times should collapse out of the 552/567/707 rollout range back to the single-pass range,
several-fold cheaper, because there is no reverse loop. Second, success should clear the diffusion floor:
where diffusion managed two successful rollouts in thirty, the discrete head — actually fit in 6000 steps
because it never started from scratch — should register non-trivial success on more than one subset, with
an arithmetic mean comfortably above diffusion's ≈0.065. But not the ceiling: the 8-bit grid and the
ordinal blindness above cap precision at exactly the sub-bin level fine manipulation lives below. So the
prediction is asymmetric — discrete should *beat diffusion clearly* in this budget, yet still leave a
precision gap I cannot close without dropping the discretization itself. If discrete fails to clear the
diffusion floor, my whole "fit-not-expressivity is the bottleneck" diagnosis is wrong; if it clears it but
does not dominate, the discretization ceiling is what is left to attack.
