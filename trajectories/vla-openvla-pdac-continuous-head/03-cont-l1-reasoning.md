The discrete rung confirmed the diagnosis exactly. On seed 42 it scored 0.2222 on goal_a, 0.1111 on
goal_b, and 0.0000 on goal_c, and the eval timers collapsed to 190/126/165 seconds — back in the
single-pass range, no 50-step rollout, just as predicted once the head stopped denoising from scratch.
Decode those fractions the same way I read diffusion's: 0.2222 is 2/9, 0.1111 is 1/9, 0.0000 is 0/12, so
discrete completed three successful rollouts out of thirty where diffusion completed two — one more
success, but a real one, and spread so that two of three subsets came off the floor instead of one. The
arithmetic mean rose from diffusion's ≈0.0648 to (0.2222 + 0.1111 + 0)/3 ≈ 0.1111, a factor of ≈1.7 more
off-floor signal. The timing gap is even louder: discrete's eval summed to 190 + 126 + 165 = 481 s
against diffusion's 552 + 567 + 707 = 1826 s, a 3.8× collapse, and per rollout that is ≈21 s on goal_a
and ≈14 s on goal_c versus diffusion's ≈60 s — the 50-step reverse loop is simply gone. Notably it is
*not* a 50× wall-clock collapse even though the decode is 50× cheaper, because each rollout still pays a
fixed physics-simulation cost per control step that dominates once the neural decode is a single pass;
the head can only shave the decode, not the simulator. And `elapsed_train` barely moved — 16958 s versus
diffusion's 17306 s, a 348 s / ≈2% difference — confirming the prediction that training wall-clock is set
by the 7B trunk's forward/backward, not by the action method bolted on top. So the "fit, not
expressivity" reading was right: where the under-trained diffusion head sat at 0.1111/0.0000/0.0833, the
discrete head — riding the trunk's pretrained next-token objective, zero new parameters — registered
real success on two of three subsets. The diffusion rung lost because its denoiser never learned inside
6000 steps; the discrete rung won that comparison by reusing a loss the trunk already mastered.

But discrete did not break clear. Its geometric mean is still 0.0000, dragged to zero by the empty
goal_c, and even its best subset, 0.2222, is far from the regime where the policy actually completes
tasks. That is the second half of the prediction coming true, and it is exactly the asymmetry I set up:
discrete beats diffusion in this budget yet leaves a precision gap. The cause is structural and I named
it in advance — the 256-bin grid is an 8-bit quantizer over [−1,+1] with bin width 2/256 ≈ 0.0078 and
RMS reconstruction error ≈0.00226 per dimension, and fine manipulation lives in the gaps between grid
points, while cross-entropy over *unordered* bins throws away the ordinal structure of a continuous
action: it never learns that bin 130 is a near-miss when the truth is 131, it only learns "right class /
wrong class." Put those together over a chunk: even if every one of the 8×7 = 56 slots lands in the
correct bin, the reconstruction still carries ≈0.00226 of grid error per dimension, and where the head is
merely one bin off — invisible to the unordered loss — it carries a full ≈0.0078, all of it uncorrected
because the objective gave no gradient toward "closer." On goal_c, the hardest four-task subset (task ids
6,7,8,9, the hidden split, 12 rollouts), that coarseness and ordinal blindness apparently cost just
enough accumulated end-effector error per chunk that nothing completes — 0/12. The discretization that
made the action speakable as a token is now the ceiling. So the next move is forced: stop discretizing.
Let the head output the continuous action directly, train it with a loss that respects distance, and see
whether removing the grid is what unsticks goal_c.

Start from the trunk's gift. Parallel decoding already gives me, in a single forward pass, a hidden
state at each of the K·D = 56 action slots — shape (B, 56, hidden) — conditioned bidirectionally on the
full observation. The discrete path took those hidden states to bin-logits; instead I will map them
straight to real-valued normalized actions with a small regression head. No grid, no bottleneck; the
precision ceiling the 256 bins imposed simply disappears, and I keep every other property of the
scaffold — the single-pass decode, the chunking, the pretrained trunk.

Now the loss, and this is where I must be careful, because the loss choice interacts with the messy
reality of human demonstrations. The obvious default is squared error, L2: minimize the mean of
(predicted − ground_truth)². But demonstrations are noisy, and at a given observation a human might have
done slightly different things on different episodes, so the target is effectively a spread of plausible
actions, not a point. L2 penalizes large residuals quadratically, so it is pulled hard toward outliers
and its optimum is the *mean* of the conditional action distribution — and the mean of a spread of
distinct valid motions can be a blended action valid under none of them, the same mode-averaging failure
that motivated diffusion in the first place. L1, the mean absolute error, penalizes linearly, so it is
far less swayed by the occasional large deviation and its optimum is the *median*. Make it concrete: if
the logged actions on some axis at near-identical observations are {0.2, 0.2, 0.8}, the L2 optimum is the
mean 0.4 — a value that sits in the empty gap between the two behaviors, exactly the "into the obstacle"
midpoint — while the L1 optimum is the median 0.2, an action the expert actually took. For action
prediction the median is the better summary: robust to noisy demonstration outliers, and it lands on real
behavior rather than an average of behaviors. I briefly consider the Huber middle ground — quadratic
within a δ band, linear outside — which would keep L1's outlier robustness while smoothing the gradient
near zero. But Huber introduces δ as a hyperparameter I would have to tune inside a 6000-step budget with
no room to sweep, and its whole benefit (a gentler gradient on tiny residuals) is the opposite of what I
want here: I want a *constant*-magnitude gradient that keeps pressing even when the residual is small, so
the head is pushed to sub-grid precision rather than allowed to relax once it is "close enough." Since
the actions are already normalized to O(1), plain L1 needs no scale tuning at all. So the objective is
the mean L1 distance between predicted and ground-truth action chunks on the normalized [−1,+1] actions:
L = mean |predicted − ground_truth| over all K timesteps and all D dimensions, one `torch.nn.L1Loss`
call against `runtime.ground_truth_actions`. Because the actions are normalized, the loss is on a clean
comparable scale, which is part of why it converges fast and stably inside the budget.

I should sit with the alternative I already tried, because it sharpens *why* L1 is the right call here
and not merely a fallback. Diffusion's selling point is genuine — it represents multimodal action
distributions, so when there really are several distinct valid ways to act it can capture all of them
instead of collapsing to a median, an expressivity L1 lacks. But I watched diffusion lose this exact
protocol: its iterative sampling reintroduced the latency I want to avoid and, decisively, its
from-scratch denoiser could not be trained in 6000 steps. The bet L1 makes is that my trunk is a 7B model
with enormous capacity, and the usual reason simple regression loses to a generative head is
*under-capacity* — a small network forced to output one vector hedges by averaging modes. A
high-capacity model conditioned on a rich bidirectional observation can often pin down which mode the
current situation calls for and regress it cleanly, so the multimodality diffusion buys may be largely
redundant here. If that is right, L1 should *match or beat* diffusion's success while being dramatically
cheaper to fit and to run — and given that diffusion already lost to discrete on fit alone, L1 inherits
discrete's fit advantage and adds continuous precision on top. The honest limitation stands: if the
demonstrations are truly multimodal, where several genuinely different action sequences are all correct
for the same input, the median-seeking L1 head cannot represent that and diffusion would have the edge.
For the focused, consistent-strategy LIBERO-Goal demonstrations, the median is exactly what I want.

I want to be precise about *why* L1 fits when diffusion did not, because the glib answer — "smaller head"
— is wrong, and getting it wrong would mislead the next decision. The regression head I am about to build
is not small: an MLP-ResNet taking 7·4096 = 28672 features to a working width of 4096 has an input
projection of 28672·4096 ≈ 117M parameters, and with its residual blocks and output layer it lands near
180M — the same order as diffusion's ≈200M noise predictor. So the fit advantage cannot be parameter
count. It is the *objective*. Every one of the 192,000 sample-gradient exposures (effective batch 32 ×
6000 steps) drives a direct, convex L1 regression toward the exact action the head must reproduce at
eval, with the full-strength gradient on every example. Diffusion spent those same 192,000 exposures
split ≈50-fold across noise levels, on a noise-prediction proxy whose connection to the final action runs
through the whole reverse chain. Same-sized head, but L1's signal is undiluted and its target is the
thing I actually want; that, not size, is why it converges inside the budget where the denoiser could
not. Discrete had the extreme version of this — zero new parameters — but paid for it with the grid;
L1 keeps the undiluted-signal property while dropping the grid.

Now build the head and wire it to the action slots. After the forward pass I have a hidden state at each
of the K·D = 56 positions; I want one D-dimensional action per timestep, so K = 8 actions total. The
natural grouping is by timestep: the D=7 action positions belonging to timestep k together determine
that timestep's 7-dim action, so I gather those D hidden states for each timestep and feed them jointly —
reshape (B, 56, hidden) to (B, 8, 7·hidden) = (B, 8, 28672), each of the 8 rows the concatenation of that
timestep's 7 per-dimension hidden states, and map each row to D = 7 outputs. Let me confirm the widths
close: the head's input width is D·hidden = 7·4096 = 28672, which must equal input_dim·ACTION_DIM =
4096·7 in the constructor — it does — and its output width is action_dim = 7, applied across the 8
timesteps, giving (B, 8, 7), exactly the shape of `runtime.ground_truth_actions` the L1 compares against.
The reshape is well-posed and the loss is a plain elementwise absolute difference over B·8·7 numbers.

What should the head be? It sits on top of the LoRA-adapted trunk, reading high-dimensional features and
regressing a low-dimensional target on a few hundred demonstrations, so I want something expressive
enough to extract the action but stable and not prone to overfitting. A plain linear+ReLU stack is
finicky to train at this width; residual connections give gradients a clean path and layer
normalization stabilizes the activation scale, so an MLP-ResNet — a few residual feedforward blocks —
is the right shape, and I will pre-normalize (LayerNorm → Linear → ReLU, added to the input) because
pre-norm keeps the residual stream's gradients well-behaved, the same reasoning that favors pre-norm
feedforward blocks in transformers. Bracket the stack with an input projection that lifts D·hidden to
the working width, two such residual blocks, a final LayerNorm, and a linear projection to action_dim —
about four layers deep. It is tiny relative to the 7B trunk, so it adds negligible inference cost, which
is the whole point of choosing a head over a heavier generative decoder. The scaffold already provides
exactly this as `prismatic.models.action_heads.MLPResNet`, so I instantiate it with `num_blocks=2`,
`input_dim = input_dim * ACTION_DIM`, `hidden_dim`, `output_dim = action_dim`.

One thing the scaffold default tempts me into, and I should decline it deliberately. The starting-point
example wraps its regressor in a `temporal_mixer` — a LayerNorm → Linear → GELU → Linear block added
residually to the hidden states before the MLP-ResNet — the idea being to let timesteps talk to each
other before decoding. But look at where those hidden states come from: the trunk already ran
bidirectional attention over all 56 action slots, so every slot's representation is *already* a function
of every other timestep and dimension in the chunk. A pre-mixer bolted on top re-does, with a
few-hundred-demonstration budget and randomly initialized weights, the cross-slot mixing the 7B trunk
just performed for free — redundant capacity whose main effect on a small dataset is another surface to
overfit. So I drop the temporal mixer and feed the extracted hidden states straight into the MLP-ResNet.
For the same reason I keep the depth at `num_blocks=2` rather than stacking more: two residual blocks are
enough to turn a 28672-dim feature into a 7-vector, and every extra block is parameters I must fit and
regularize against a handful of episodes per task. The rule I am following is "let the trunk do the
representation work; keep the head just deep enough to read the answer off it," which is the whole
economic argument for a head over a heavier decoder.

The wiring at training time has a subtlety in extracting the action positions, and the task's
`extract_action_hidden_states` hook is exactly where it lives. I run the trunk forward with *zeroed*
action-token features — unlike diffusion, the L1 head injects nothing action-conditioned back into the
input; like discrete it hands a zero placeholder of shape (B, 56, llm_dim) to `runtime.forward`, since
it conditions only on the observation, not on any action-side input — under bidirectional attention. At
train time I have the ground-truth labels, so I build the boolean mask
`current_action_mask | next_actions_mask` over the response positions, select those hidden states, and
reshape to (B, 56, hidden); the runtime exposes both masks directly. At eval time there are no labels,
but the layout is known: the action positions start right after the prompt and run for 56 contiguous
slots, so I slice `[num_prompt_tokens : num_prompt_tokens + 56]` instead. Either way the head sees
(B, 56, hidden), regroups to (B, 8, 28672), and predicts (B, 8, 7). The loss is then the single L1 call
against the ground-truth chunk; `runtime.compute_action_l1_metrics` logs the auxiliary L1 diagnostics. At
inference, `predict_actions` returns the predicted normalized chunk and the action hidden states, and the
runtime un-normalizes.

The train/eval selection asymmetry is worth one sanity check, because if the two paths select different
positions the eval-time actions would be read off the wrong hidden states and the policy would silently
regress garbage. At train time the boolean mask `current_action_mask | next_actions_mask` is true on
exactly the response positions holding the chunk's actions, and there are 8×7 = 56 of them, so the
selected tensor reshapes to (B, 56, hidden). At eval time the mask is unavailable, but the sequence
layout is fixed: the prompt occupies the first `num_prompt_tokens` positions and the 56 action slots
follow contiguously, so `[num_prompt_tokens : num_prompt_tokens + 56]` selects the identical 56 positions
by construction. Both routes hand the head the same (B, 56, hidden) block in the same slot order, so the
head that learned to read timestep k's seven slots at training reads the same seven at eval. The
placeholder `action_token_features` are zeros in both phases, so nothing action-side leaks in to make the
two forwards differ. That consistency is what lets a head trained under teacher-forced masks be trusted
at label-free eval.

This is the same continuous-head reasoning the discrete rung's precision gap demanded — discrete's
unordered cross-entropy could not learn that a near-miss bin is nearly right; L1 on the continuous
output makes "nearly right" the loss itself, since |predicted − truth| shrinks smoothly as the
prediction approaches the target. Where discrete's grid floored the reconstruction at ≈0.0078 even on a
one-bin miss with no gradient to fix it, L1's error can drive continuously toward zero and its gradient
never stops pushing until it gets there. That ordinal-respecting loss is precisely what should recover
the sub-grid precision discrete threw away, and if my reading of goal_c is right — that accumulated
grid-and-ordinal error, not some other bottleneck, is what kept it at 0/12 — then removing the grid is
exactly the intervention that should move it.

The one place `CONFIG_OVERRIDES` lets me touch the schedule is warmup and decay, and the budget makes
both choices nearly forced. I set `lr_warmup_steps = 400`, which is 400/6000 ≈ 6.7% of the run — enough
to ramp the LoRA adapters and the head off their initialization without the optimizer taking a
destabilizing full-rate step into cold weights, but no more, because every warmup step is a step spent at
reduced learning rate that I cannot afford to waste when I only have 6000 of them. That is deliberately
less than the 800 I gave diffusion, and the reason is exactly the fit story: diffusion was warming a
~200M random-init denoiser on a hard noise-prediction proxy and needed the longer gentle ramp; the L1
head is a direct regression on a trunk that already speaks, so a shorter warmup suffices and the extra
400 steps are better spent at full rate. I set `num_steps_before_decay = 6000` so the learning rate holds
flat across the entire budget and only decays at the very end — with a run this short there is no reason
to spend a long tail annealing; I want maximum effective learning rate for as long as possible and just a
brief cooldown so the final checkpoint is not caught mid-oscillation. Both knobs push the same way: given
6000 steps, minimize time spent below full rate.

Now the falsifiable expectations against the discrete numbers. L1 keeps everything that made discrete
beat diffusion — single trunk pass, pretrained trunk, fits in 6000 steps — so its eval times should sit
in the same single-pass range as discrete (≈130–190 s/subset, the 190/126/165 band), not the diffusion
rollout range of 552/567/707, and `elapsed_train` should again land near ≈17000 s since the trunk still
dominates. The success claim is sharper. Removing the 256-bin grid and using a distance-respecting loss
should lift *every* subset, and critically it should unstick the one discrete left empty: I expect goal_c
to go non-zero, which alone would flip the geometric mean off zero where both discrete and diffusion
scored a flat 0.0000 — because a geometric mean is zero if and only if some subset is zero, so a single
positive goal_c is the whole difference between a trivial score and a real one. Concretely I expect L1 to
dominate discrete on all three subsets — goal_a and goal_b well above discrete's 0.2222/0.1111, and
goal_c clearly positive — yielding a geometric mean that is the first non-trivial score on the ladder.
The bar this rung must clear, then, is unambiguous: beat discrete on every subset and produce a positive
geometric mean. If L1 fails to lift goal_c off zero, my "discretization is the ceiling" diagnosis is
wrong and something else is capping goal_c; if it lifts goal_c and dominates the other two, then the
continuous regression head — fit-efficient like discrete, precise like no grid allows, and matching
diffusion's success without its training cost — is the strongest action method this budget supports, and
the ladder ends here.
