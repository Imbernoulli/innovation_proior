The discrete rung confirmed the diagnosis exactly. On seed 42 it scored 0.2222 on goal_a, 0.1111 on
goal_b, and 0.0000 on goal_c, and the eval timers collapsed to 190/126/165 s — back in the single-pass
range, no 50-step rollout, just as predicted once the head stopped denoising from scratch. Decoding the
fractions: 0.2222 is 2/9, 0.1111 is 1/9, 0.0000 is 0/12, so discrete completed three successful rollouts
out of thirty where diffusion completed two — one more, but spread so that two of three subsets came off
the floor instead of one, lifting the arithmetic mean from diffusion's ≈0.065 to ≈0.111. The timing gap
is louder: discrete's eval summed to 481 s against diffusion's 1826 s, a 3.8× collapse — though notably
*not* the 50× the decode alone would suggest, because each rollout still pays a fixed physics-simulation
cost per control step that dominates once the neural decode is a single pass; the head can shave the
decode, not the simulator. And `elapsed_train` barely moved, 16958 s versus 17306 s, confirming that
training wall-clock is set by the 7B trunk, not the head bolted on top. So "fit, not expressivity" was
right: the discrete head, riding the pretrained next-token objective with zero new parameters, registered
real success on two of three subsets where the under-trained denoiser could not.

But discrete did not break clear. Its geometric mean is still 0.0000, dragged to zero by the empty
goal_c, and even its best subset, 0.2222, is far from actually completing tasks — the second half of the
prediction, exactly the asymmetry I set up. The cause is the ceiling I named in advance: the 256-bin grid
is an 8-bit quantizer with bin width ≈ 0.0078, and cross-entropy over *unordered* bins gives no gradient
toward being close when wrong, so a one-bin miss carries its full ≈ 0.0078 uncorrected. On goal_c — the
hardest four-task subset (task ids 6,7,8,9, the hidden split, 12 rollouts) — that accumulated per-chunk
end-effector error apparently cost just enough that nothing completes, 0/12. The discretization that made
the action speakable as a token is now the ceiling. So the next move is forced: stop discretizing. Output
the continuous action directly, train with a loss that respects distance, and see whether removing the
grid is what unsticks goal_c.

Start from the trunk's gift. Parallel decoding already gives me, in a single forward pass, a hidden state
at each of the K·D = 56 action slots — shape (B, 56, hidden) — conditioned bidirectionally on the full
observation. The discrete path took those hidden states to bin-logits; instead I map them straight to
real-valued normalized actions with a small regression head. No grid, no bottleneck; the precision
ceiling the 256 bins imposed disappears, and I keep every other property of the scaffold — the
single-pass decode, the chunking, the pretrained trunk.

Now the loss, and it interacts with the messy reality of human demonstrations. Squared error, L2,
penalizes large residuals quadratically, so it is pulled hard toward outliers and its optimum is the
*mean* of the conditional action distribution — and the mean of a spread of distinct valid motions can be
a blended action valid under none of them, the same mode-averaging that motivated diffusion. L1, the mean
absolute error, penalizes linearly and its optimum is the *median*: if the logged actions on some axis at
near-identical observations are {0.2, 0.2, 0.8}, L2 lands on the mean 0.4, in the empty gap between
behaviors, while L1 lands on the median 0.2, an action the expert actually took. For action prediction
the median is the better summary — robust to noisy demonstration outliers, landing on real behavior. I
consider the Huber middle ground but decline it: it adds a δ I cannot afford to tune inside a 6000-step
budget, and its gentler gradient on small residuals is the opposite of what I want — I want a
constant-magnitude gradient that keeps pressing toward sub-grid precision even when the residual is small.
Since the normalized actions are O(1), plain L1 needs no scale tuning: one `torch.nn.L1Loss` over all K
timesteps and D dimensions against `runtime.ground_truth_actions`, on a clean comparable scale that is
part of why it converges fast and stably inside the budget.

It is worth sitting with the alternative I already tried, because it sharpens *why* L1 is the right call
and not merely a fallback. Diffusion's selling point is genuine — it represents multimodal action
distributions, capturing several distinct valid ways to act instead of collapsing to a median, an
expressivity L1 lacks. But the usual reason simple regression loses to a generative head is
*under-capacity*: a small network forced to output one vector hedges by averaging modes. My trunk is a 7B
model conditioned on a rich bidirectional observation, so it can often pin down which mode the current
situation calls for and regress it cleanly — the multimodality diffusion buys may be largely redundant
here. If so, L1 should *match or beat* diffusion's success while being dramatically cheaper to fit and
run; and since diffusion already lost to discrete on fit alone, L1 inherits discrete's fit advantage and
adds continuous precision on top. The honest limitation stands: if the demonstrations are truly
multimodal — several genuinely different sequences all correct for one input — the median-seeking L1 head
cannot represent that and diffusion would have the edge. For the focused, consistent-strategy LIBERO-Goal
demonstrations, the median is exactly what I want.

Why L1 fits when diffusion did not is *not* "smaller head" — that glib answer would mislead the next
decision. The MLP-ResNet I am about to build takes 7·4096 = 28672 features to a 4096 working width, an
input projection of ≈117M and near 180M all told, the same order as diffusion's ≈200M noise predictor. So
the fit advantage cannot be parameter count; it is the *objective*. Every one of the 192,000
sample-gradient exposures drives a direct, convex L1 regression toward the exact action the head must
reproduce at eval, at full-strength gradient on every example, where diffusion split those same exposures
≈50-fold across noise levels on a noise-prediction proxy whose connection to the final action runs through
the whole reverse chain. Same-sized head, undiluted signal — that, not size, is why it converges inside
the budget. Discrete had the extreme version (zero parameters) but paid for it with the grid; L1 keeps the
undiluted-signal property while dropping the grid.

Build the head and wire it to the action slots. I want one D-dimensional action per timestep, so I group
by timestep: the D = 7 slots belonging to timestep k together determine that timestep's 7-dim action, so
I reshape (B, 56, hidden) to (B, 8, 7·hidden) = (B, 8, 28672), each row a timestep's 7 per-dimension
hidden states concatenated, and map each row to 7 outputs — input width D·hidden = 4096·7 matches the
constructor's input_dim·ACTION_DIM, output (B, 8, 7), exactly `runtime.ground_truth_actions`' shape.

What should the head be? It sits on the LoRA-adapted trunk, regressing a low-dimensional target from
high-dimensional features on a few hundred demonstrations, so I want expressiveness with stability against
overfitting. A plain linear+ReLU stack is finicky at this width; residual connections give gradients a
clean path and layer normalization stabilizes activation scale, so a pre-norm MLP-ResNet (LayerNorm →
Linear → ReLU, added to the input) is the right shape — an input projection lifting D·hidden to the
working width, two residual blocks, a final LayerNorm, and a linear to action_dim. It is tiny relative to
the 7B trunk, so it adds negligible inference cost, which is the whole economic argument for a head over a
heavier generative decoder. The scaffold provides exactly this as `prismatic.models.action_heads.MLPResNet`,
instantiated `num_blocks=2`, `input_dim = input_dim * ACTION_DIM`, `hidden_dim`, `output_dim = action_dim`.

One thing the scaffold default tempts me into and I decline deliberately: it wraps its regressor in a
`temporal_mixer` — a LayerNorm → Linear → GELU → Linear block added residually before the MLP-ResNet — to
let timesteps talk before decoding. But the trunk already ran bidirectional attention over all 56 slots,
so every slot's representation is *already* a function of every other timestep and dimension in the chunk.
A pre-mixer would re-do, with randomly initialized weights and a few-hundred-demonstration budget, the
cross-slot mixing the 7B trunk just performed for free — redundant capacity whose main effect on a small
dataset is another surface to overfit. So I drop the mixer and feed the extracted hidden states straight
into the MLP-ResNet, and I keep `num_blocks=2` for the same reason: two residual blocks are enough to turn
a 28672-dim feature into a 7-vector, and every extra block is parameters I must fit and regularize against
a handful of episodes per task. Let the trunk do the representation work; keep the head just deep enough
to read the answer off it.

Wiring at training time: I forward with *zeroed* action-token features — unlike diffusion the L1 head
injects nothing action-conditioned back into the input, and like discrete it hands a zero placeholder of
shape (B, 56, llm_dim) to `runtime.forward`, since it conditions only on the observation. At train time I
have labels, so I select hidden states by the boolean mask `current_action_mask | next_actions_mask` over
the response positions and reshape to (B, 56, hidden); at eval time there are no labels, but the 56 action
slots follow the prompt contiguously, so `[num_prompt_tokens : num_prompt_tokens + 56]` selects the
identical positions in the same order. Both routes hand the head the same (B, 56, hidden) block, and the
zero placeholder in both phases means nothing action-side leaks in to make the two forwards differ — which
is what lets a head trained under teacher-forced masks be trusted at label-free eval. The loss is the
single L1 call; `runtime.compute_action_l1_metrics` logs the auxiliary L1 diagnostics; `predict_actions`
returns the predicted normalized chunk and the runtime un-normalizes.

This is the continuous head the discrete rung's precision gap demanded: where discrete's unordered
cross-entropy could not learn that a near-miss bin is nearly right, |predicted − truth| makes "nearly
right" the loss itself, shrinking smoothly toward zero with a gradient that never stops pushing until it
gets there. That ordinal-respecting loss is precisely what should recover the sub-grid precision discrete
threw away — and if my reading of goal_c is right, that accumulated grid-and-ordinal error and not some
other bottleneck is what kept it at 0/12, then removing the grid is exactly the intervention that should
move it.

The one schedule knob `CONFIG_OVERRIDES` exposes is warmup and decay, and the budget nearly forces both.
`lr_warmup_steps = 400` (≈6.7% of the run) ramps the LoRA adapters and the head off their initialization
without a destabilizing full-rate step into cold weights, but no more, because every warmup step is a step
at reduced rate I cannot afford to waste with only 6000 of them — deliberately less than the 800 I gave
diffusion, which was warming a ~200M random-init denoiser on a hard noise-prediction proxy, where a direct
regression on a trunk that already speaks needs less. `num_steps_before_decay = 6000` holds the rate flat
across the whole budget with only a brief final cooldown, since a run this short has no reason to spend a
long tail annealing; I want maximum effective rate for as long as possible and just enough cooldown that
the final checkpoint is not caught mid-oscillation. Both knobs push the same way: minimize time below full
rate.

Now the falsifiable expectations against the discrete numbers. L1 keeps everything that made discrete beat
diffusion — single trunk pass, pretrained trunk, fits in 6000 steps — so its eval times should sit in the
same single-pass range as discrete's 190/126/165 band, not the diffusion rollout range, and
`elapsed_train` should again land near ≈17000 s since the trunk still dominates. The success claim is
sharper. Removing the 256-bin grid and using a distance-respecting loss should lift *every* subset, and
critically it should unstick the one discrete left empty: a geometric mean is zero if and only if some
subset is zero, so a single positive goal_c is the whole difference between a trivial score and a real
one. I expect L1 to dominate discrete on all three subsets — goal_a and goal_b well above discrete's
0.2222/0.1111, and goal_c clearly positive — yielding the first non-trivial geometric mean on the ladder.
The bar this rung must clear is unambiguous: beat discrete on every subset and produce a positive
geometric mean. If L1 fails to lift goal_c off zero, my "discretization is the ceiling" diagnosis is wrong
and something else is capping goal_c; if it lifts goal_c and dominates the other two, then the continuous
regression head — fit-efficient like discrete, precise like no grid allows, and matching diffusion's
success without its training cost — is the strongest action method this budget supports, and the ladder
ends here.
