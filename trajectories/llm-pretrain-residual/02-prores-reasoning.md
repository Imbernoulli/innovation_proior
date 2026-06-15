The vanilla run gives me one number to push against and it is exactly the kind of number that hides a
mechanism: validation loss 2.2763 on seed 42, WikiText-2 perplexity 44.28, LAMBADA 70.09, ARC-Easy
54.12, HellaSwag 33.82. Nothing crashed — the Pre-LN highway did its job, the run is healthy, the
losses are where a well-trained GPT-2 Medium on 7B FineWeb tokens should sit. So the failure here is not
a failure of *training*; it is a failure of *use*. The residual stream is a fixed unit-weight
accumulator, and the suspicion I wrote down at step 1 is that the deepest of the 24 layers are
contributing little — that the stream variance has climbed with depth, shrunk the deep blocks'
LayerNorm Jacobians toward identity, and left the top third of the stack doing near-nothing. I cannot
read layer-wise deadness off a single scalar loss, but the *shape* of the vanilla result is consistent
with it: a competent floor that has left capacity on the table. The 2.2763 is the number I have to
beat, and the lever has to be the depth flow, because that is the only thing I am allowed to touch and
the only thing the floor leaves rigid.

Let me make the deep-layer-death claim quantitative, because I want the fix to react to the mechanism,
not the symptom. Trace the Pre-LN recursion `x_{l+1} = x_l + F(LN(x_l))`. The normalization fixes the
scale of the *branch input* — what gets fed into attention or the MLP — but it does nothing to the
*stream* `x_l` I keep adding onto. So the stream variance accumulates with depth: under the usual
zero-mean weight assumptions the scale satisfies a product `σ²_{x_l} ∝ ∏_{k<l}(1 + 1/σ_{x_k})`, every
layer contributing a factor above one. The slow case is linear growth, `σ²_{x_L} ≳ Θ(L)`; the dangerous
case is exponential. Now write the block Jacobian: `∂(x + F(LN(x)))/∂x = I + (∂F/∂LN)·(∂LN/∂x)`, and
the second term carries a `1/σ_{x_l}` from the normalization dividing by the stream's own scale. When
the stream variance is large, that term is tiny, the block Jacobian collapses toward `I`, and a block
whose Jacobian is `≈ I` is becoming a local identity map — changing its input barely changes what it
adds. Chain that up the stack and the deepest layers are pushed toward doing nothing. That is the
mechanism behind the floor's left-on-the-table capacity: the variance explosion is not a cosmetic
numerical annoyance, it is the thing that turns deep layers into dead weight. So the question is sharp:
how do I keep those deep layers alive without touching attention, the MLP, or the norm — only the
residual flow?

One school says kill the variance growth and the deep layers come back. The cleanest version is to
down-weight deeper residuals statically: divide the branch input by `√l` at layer `l`, so the variance
curve flattens from exponential toward linear and the deep Jacobians stay off `I`. The math above tells
me exactly why that works — shrink the per-layer increment with depth and the product can't run away.
But sit with the *shape* of that fix: the factor `1/√l` depends only on depth and is *static*, baked in
for the whole run. Early on it is helpful, it is controlling the explosion. But late in training, when
everything has stabilized and I *want* those deep layers learning at full strength, the `1/√l` is still
throttling them. I would be trading "dead because the variance exploded" for "dead because I clamped
them." The cure is right but it is frozen in time, and that frozen-ness is the seam I want to pull.

The other school says the disease is instability of the *update*, not variance of the activation. The
bounded-update designs — Fixup, T-Fixup, DeepNorm — prove that what blows up a deep Transformer is the
model update moving the function by an unbounded amount, and they bound it with constants applied
*uniformly for all of training*, derived from the network's behavior *at initialization*. That is the
right protection for the worst instant — the chaotic early phase — but training is not one regime:
there is a chaotic warm-up, then a long stable phase of small gradual changes, then a decay (which is
exactly why people run warm-up-stable-decay LR schedules). A constraint sized for the worst init-time
instant and held fixed through the stable phase is over-conservative once I am in the stable phase. Same
frozen-in-time problem as `1/√l`, just from the update side.

And then the scalar-on-the-branch family — ReZero, SkipInit — which is the closest in form to what I
keep circling, and worth being precise about because the *next* rung is going to be exactly this family.
ReZero puts one learnable scalar on each branch, `x_{l+1} = x_l + α_l·F(LN(x_l))`, every `α_l`
initialized to 0. That zero init is lovely: at `t=0` the whole network is exactly the identity, the
input-output Jacobian is `I`, trivially well-conditioned, and the stream variance cannot blow up at init
because nothing is being added yet. But ReZero learns the `α_l` by gradient descent, independently per
layer, and *nothing says the shallow branches turn on before the deep ones*. The optimizer is free to
grow a deep `α` first if that locally lowers the loss — and given how entangled the stack is, that is
not hypothetical. A deep layer's input is the running output of every shallow layer beneath it; a
shallow layer's gradient is back-propagated through every deep layer above it. So when a deep branch
with freshly-random weights fires early, it does two bad things at once: it injects noise into the
representation that all the upper layers must now consume, and it corrupts the gradient signal the
shallow layers below are using to figure out what *they* should compute. The shallow layers are trying
to lay a stable foundation while the deep layers, firing early with garbage, eat that half-built
foundation and poison the feedback it is built from. A learned scalar gives me no control over this; its
only guarantee is "start at zero," and after step one it is the optimizer's call.

So let me name the common failure precisely, because all three families share it: every lever —
depth-aware init, DeepNorm's constant skip-scale, the static `1/√l`, the zero-init learnable scalar —
sets up the *first step* and then either freezes or hands the wheel to the optimizer. None is
*training-phase-aware*. But I have two facts that are explicitly about the *trajectory*, not the init.
One: training is staged — chaotic warm-up, then stable, then decay. Two: the layers converge unevenly —
shallow layers settle into their final representation earlier than deep ones. Put those next to the
dependency analysis and a different fix suggests itself. The problem is not only *how much* each branch
contributes — a magnitude question the static factors answer — it is *when*, and in *what order across
depth*, each branch is allowed to start contributing. That is a question about the whole trajectory, and
nobody's tool answers it because they all act at one instant.

What do I actually want? Each layer to start at the identity — keep ReZero's `α=0` clean start — and
then the contributions to *come on in order*: shallow first, deep last, the deep ones held back until
the shallow ones beneath them have stabilized. And I want to *impose* this, not hope the optimizer
discovers it, because the whole ReZero gap is that the optimizer does not respect the order. So the
scalar should not be a parameter at all. It should be a *predefined* function I set, ramping from 0 up
to 1 over training, slower for deeper layers. Write the residual write as
`x_{l+1} = x_l + α(l,t)·F(LN(x_l))`, where `α(l,t)` depends on layer index `l` and step `t` and is
fixed by me. At `t=0`, `α(l,0)=0` for all `l` — exact identity, the variance cannot blow up because
nothing is added yet. As `t→∞`, `α(l,t)→1` — the block becomes plain vanilla Pre-LN, so I lose nothing
in the limit; the deep layers end at full strength, which is exactly what static `1/√l` forbids. The
*schedule* of how `α` climbs is where I encode "shallow first."

Now pin the shape. Take the simplest monotone ramp from 0 to 1: linear in `t`, clipped at 1,
`α(l,t) = min(t/τ_l, 1)`, where `τ_l` is the step at which layer `l`'s scalar first reaches 1. The game
is in choosing `τ_l`: I want deeper layers to take longer, so `τ_l` increases with `l`. The cleanest
choice with a single knob is `τ_l = T·l`, where `T` is the warm-up length of the first layer. Then layer
1 finishes at `T`, layer `l` at `T·l`, and the deepest layer at `T·L`, giving `α(l,t) = min(t/(T·l), 1)`.
At any fixed `t` during the ramp the shallow layers have already reached `α=1` while the deep ones are
still partway up — the branches switch on in a wave that sweeps from shallow to deep — and after `T·L`
steps every `α=1` and the model runs exactly vanilla. That last property is the one the static methods
cannot have: the constraint is real early and *gone* late.

Check it against the three things I wanted, and make each fall out rather than be asserted. Identity at
init: `α(l,0)=min(0,1)=0`, so `x_{l+1}=x_l` exactly — ReZero's clean start for free, no init variance
blowup. Bounded update across time *and* depth: early on only shallow layers have nonzero `α`, so only a
few layers can move the function and they move it by a fraction `α<1`; the constraint relaxes itself
layer by layer as training proceeds, tight when updates are chaotic and fully released when they are
stable — the temporal awareness DeepNorm's constant lacks. The ordering: a deep layer's `α` stays near 0
while the shallow layers below it do their early large updates, so by the time it climbs enough to matter
the foundation beneath it has largely settled, and it is not injecting garbage back through the gradient
path while the shallow layers are still finding their footing. The variance explosion is cured *in time*
rather than statically: the deep `α` are near zero during the chaotic early phase when the stream is most
prone to running up the exponential branch, then the increments come on gradually as `α` ramps, so the
variance grows gently — the control of `1/√l` early, without its permanent late clamp, because my factor
goes to 1.

I should make sure the specific shape is load-bearing and not arbitrary. If every layer warmed up
*together* (`min(t/T,1)`, no `l`-dependence), I would delay the chaotic init updates but ignore the
order — all deep branches switch on simultaneously with the shallow ones, just postponed, so deep layers
still fire while shallow representations are unstable; weaker, and brittle at long warm-up because I am
scaling all residuals up at once after a delay. If I went the *wrong* way, deep-first (`τ_l = T·(L−l+1)`),
I would actively do the harmful thing — let the deep, randomly-initialized branches dominate early while
the shallow layers are starved, exactly the divergence pattern Post-LN is notorious for. So the *order*,
shallow before deep, is doing real work, and the linear-in-`l` stagger is the simplest schedule that
encodes it. And the static-vs-dynamic distinction is the whole thesis: freezing the constraint (hold
`α=1/√l` forever) keeps strangling the deep layers through the stable phase, while the matching schedule
that starts at that fraction and relaxes to 1 hands them back full capacity once they are past the
dangerous early regime. If my reading is right — bounded update is needed *at init*, not forever — the
relaxing version should win.

One free number, `T`. It sets the timescale: layer 1 warms over `T` steps, the whole model over `T·L`.
Two failure modes bracket it. Too small and I have barely delayed anything past the chaotic init — the
branches snap on immediately and I throw away the benefit. Too large and `T·L` eats a big fraction of my
budget, so the deep layers spend most of training artificially weakened and never get enough
full-strength steps to learn. So `T` wants to outlast the warm-up phase but stay a modest slice of
training. Here the run is 13,535 steps over 24 layers, so I do not even want a per-layer total `T·L`
that overruns it — I want most layers to reach full strength well inside the run. `T = 1000` is the
natural default: the first layer finishes in 1000 steps, the shallow half is at full strength within a
few thousand, and the wave has swept most of the stack before the cosine decay sets in. I set it once and
do not tune it; the only principled adjustment would be at extreme depth, keeping `T·L ≤` total steps so
every layer reaches full strength, which is not a concern at 24 layers.

A couple of choices I made without comment, now justified. I multiply only the *branch*
`F(LN(x_l))` by `α` and leave the skip `x_l` at weight 1 — the skip is the identity highway and the
gradient path; scaling *it* by `α` would attenuate the signal, and at `α=0` the network would collapse
to zero output instead of to the identity. Keeping the skip at 1 and dialing the branch from 0 to 1
means `α=0` is *exactly* identity and `α=1` is *exactly* vanilla, a clean interpolation between the two
endpoints I want. And it costs nothing: zero learnable parameters, so the optimizer and its decay /
no-decay groups are untouched — `configure_optimizers` and `CONFIG_OVERRIDES` stay at their defaults.

Now make it concrete in this task's edit surface, because there is one implementation wrinkle worth
getting right. The scalar `α(l,t)` depends on the global step `t`, which the `Block` does not see — and
the contract wants `Block.forward(x)→x` unchanged. So I keep the block vanilla and put the schedule in
the `GPT.forward` block loop, carrying the step as a non-trainable buffer on the model. There is a
second wrinkle: I want `x ← x + α·(branch contribution)`, but in this harness the block already folds
the skip in and returns `block_out = x + delta`, the full Pre-LN residual. Rather than reach inside the
block to scale only the branch, I recover the contribution as `delta = block_out − x` and write
`x ← x + α·(block_out − x)`, which scales exactly the branch and leaves the skip at weight 1; once a
layer's `α` has reached 1 I take `block_out` directly, which is bit-identical to vanilla. I read the
current step *before* the loop so the first training forward is truly `t=0` with every branch off, then
advance the buffer after this forward has used that step. Two extra lines of arithmetic, one integer
buffer, zero parameters. The distilled module and the literal scaffold fill are in the answer.

So the delta from step 1 is precise: where vanilla ran the bare loop `x = block(x)`, ProRes runs the
same blocks but writes each branch into the stream through a depth-staggered, time-ramping scalar that
holds deep layers near the identity early and releases them to full strength late. Here is what I expect
against the 2.2763 floor and where I am unsure. The mechanism predicts a *modest* validation-loss
improvement — this is a stability-and-conditioning fix, not a new representational capacity; I would be
disappointed by no movement and surprised by a large jump, so somewhere in the low-2.27s, a few thou
under vanilla, is the honest expectation. The clearest place to see it should be the perplexities, since
better-conditioned deep layers should sharpen the model's token predictions: I expect WikiText-2 and
especially LAMBADA to come in at or below vanilla's 44.28 / 70.09, with LAMBADA (the long-range
completion task that leans on the deep layers) the more sensitive of the two. Downstream accuracy at
this scale is noisy enough that I would not bet on a clean move on ARC-Easy or HellaSwag. And the honest
risk is the failure mode the static-vs-dynamic argument flagged in reverse: at only 24 layers and 13.5k
steps the variance explosion may be mild enough that vanilla was never badly conditioned, in which case
ProRes mostly delays harmless early updates and lands *near* the floor rather than clearly under it. If
the validation loss barely moves, the diagnosis for the next rung is already written: a fixed
*schedule* with no learned magnitude per layer is too blunt — I should let each layer *learn* its own
residual weight, and add a second, orthogonal route that the schedule never gives me, a direct path from
the token embedding into every depth.
