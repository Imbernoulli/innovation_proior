The discrete rung confirmed the diagnosis exactly. On seed 42 it scored 0.2222 on goal_a, 0.1111 on
goal_b, and 0.0000 on goal_c, and the eval timers collapsed to 190/126/165 seconds — back in the
single-pass range, no 50-step rollout, just as predicted once the head stopped denoising from scratch.
So the "fit, not expressivity" reading was right: where the under-trained diffusion head sat at
0.1111/0.0000/0.0833 with a ~0.065 arithmetic mean, the discrete head — riding the trunk's pretrained
next-token objective, zero new parameters — registered real success on two of three subsets and roughly
doubled the off-floor signal. The diffusion rung lost because its denoiser never learned inside 6000
steps; the discrete rung won that comparison by reusing a loss the trunk already mastered.

But discrete did not break clear. Its geometric mean is still 0.0000, dragged to zero by the empty
goal_c, and even its best subset, 0.2222, is far from the regime where the policy actually completes
tasks. That is the second half of the prediction coming true, and it is exactly the asymmetry I set up:
discrete beats diffusion in this budget yet leaves a precision gap. The cause is structural and I named
it in advance — the 256-bin grid is an 8-bit quantizer over [−1,+1], and fine manipulation lives in the
gaps between grid points, while cross-entropy over *unordered* bins throws away the ordinal structure of
a continuous action: it never learns that bin 130 is a near-miss when the truth is 131, it only learns
"right class / wrong class." On goal_c, the hardest four-task subset, that coarseness and ordinal
blindness apparently cost just enough accumulated end-effector error per chunk that nothing completes.
The discretization that made the action speakable as a token is now the ceiling. So the next move is
forced: stop discretizing. Let the head output the continuous action directly, train it with a loss that
respects distance, and see whether removing the grid is what unsticks goal_c.

Start from the trunk's gift. Parallel decoding already gives me, in a single forward pass, a hidden
state at each of the K·D action slots — shape (B, K·D, hidden) — conditioned bidirectionally on the
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
far less swayed by the occasional large deviation and its optimum is the *median*. For action prediction
the median is the better summary: robust to noisy demonstration outliers, and more precise. So the
objective is the mean L1 distance between predicted and ground-truth action chunks on the normalized
[−1,+1] actions: L = mean |predicted − ground_truth| over all K timesteps and all D dimensions, one
`torch.nn.L1Loss` call against `runtime.ground_truth_actions`. Because the actions are normalized, the
loss is on a clean comparable scale, which is part of why it converges fast and stably inside the budget.

I should sit with the alternative I already tried, because it sharpens *why* L1 is the right call here
and not merely a fallback. Diffusion's selling point is genuine — it represents multimodal action
distributions, so when there really are several distinct valid ways to act it can capture all of them
instead of collapsing to a median, an expressivity L1 lacks. But I watched diffusion lose this exact
protocol: its iterative sampling reintroduced the latency I want to avoid and, decisively, its from-
scratch denoiser could not be trained in 6000 steps. The bet L1 makes is that my trunk is a 7B model
with enormous capacity, and the usual reason simple regression loses to a generative head is *under-
capacity* — a small network forced to output one vector hedges by averaging modes. A high-capacity model
conditioned on a rich bidirectional observation can often pin down which mode the current situation
calls for and regress it cleanly, so the multimodality diffusion buys may be largely redundant here. If
that is right, L1 should *match or beat* diffusion's success while being dramatically cheaper to fit and
to run — and given that diffusion already lost to discrete on fit alone, L1 inherits discrete's
fit advantage and adds continuous precision on top. The honest limitation stands: if the demonstrations
are truly multimodal, where several genuinely different action sequences are all correct for the same
input, the median-seeking L1 head cannot represent that and diffusion would have the edge. For the
focused, consistent-strategy LIBERO-Goal demonstrations, the median is exactly what I want.

Now build the head and wire it to the action slots. After the forward pass I have a hidden state at each
of the K·D positions; I want one D-dimensional action per timestep, so K actions total. The natural
grouping is by timestep: the D=7 action positions belonging to timestep k together determine that
timestep's 7-dim action, so I gather those D hidden states for each timestep and feed them jointly —
reshape (B, K·D, hidden) to (B, K, D·hidden), each of the K rows the concatenation of that timestep's D
action-position hidden states, and map each row to D outputs. The head's input width is D·hidden and its
output width is action_dim, applied across the K timesteps.

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

The wiring at training time has a subtlety in extracting the action positions, and the task's
`extract_action_hidden_states` hook is exactly where it lives. I run the trunk forward with *zeroed*
action-token features — unlike diffusion, the L1 head injects nothing action-conditioned back into the
input; like discrete it hands a zero placeholder of shape (B, K·D, llm_dim) to `runtime.forward`, since
it conditions only on the observation, not on any action-side input — under bidirectional attention. At train time I have the ground-truth labels, so I build
the boolean mask `current_action_mask | next_actions_mask` over the response positions, select those
hidden states, and reshape to (B, K·D, hidden); the runtime exposes both masks directly. At eval time
there are no labels, but the layout is known: the action positions start right after the prompt and run
for K·D contiguous slots, so I slice `[num_prompt_tokens : num_prompt_tokens + K·D]` instead. Either way
the head sees (B, K·D, hidden), regroups to (B, K, D·hidden), and predicts (B, K, action_dim). The loss
is then the single L1 call against the ground-truth chunk; `runtime.compute_action_l1_metrics` logs the
auxiliary L1 diagnostics. At inference, `predict_actions` returns the predicted normalized chunk and the
action hidden states, and the runtime un-normalizes.

This is the same continuous-head reasoning the discrete rung's precision gap demanded — discrete's
unordered cross-entropy could not learn that a near-miss bin is nearly right; L1 on the continuous
output makes "nearly right" the loss itself, since |predicted − truth| shrinks smoothly as the
prediction approaches the target. That ordinal-respecting loss is precisely what should recover the
sub-grid precision discrete threw away.

Now the falsifiable expectations against the discrete numbers. L1 keeps everything that made discrete
beat diffusion — single trunk pass, pretrained trunk, fits in 6000 steps — so its eval times should sit
in the same single-pass range as discrete (130-180 s/subset), not the diffusion rollout range. The
success claim is sharper. Removing the 256-bin grid and using a distance-respecting loss should lift
*every* subset, and critically it should unstick the one discrete left empty: I expect goal_c to go
non-zero, which alone would flip the geometric mean off zero where both discrete and diffusion scored a
flat 0.0000. Concretely I expect L1 to dominate discrete on all three subsets — goal_a and goal_b well
above discrete's 0.2222/0.1111, and goal_c clearly positive — yielding a geometric mean that is the
first non-trivial score on the ladder. The bar this rung must clear, then, is unambiguous: beat discrete
on every subset and produce a positive geometric mean. If L1 fails to lift goal_c off zero, my
"discretization is the ceiling" diagnosis is wrong and something else is capping goal_c; if it lifts
goal_c and dominates the other two, then the continuous regression head — fit-efficient like discrete,
precise like no grid allows, and matching diffusion's success without its training cost — is the
strongest action method this budget supports, and the ladder ends here.
