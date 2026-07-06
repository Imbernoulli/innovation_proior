Meta-SGD's numbers split exactly along the seam I flagged, and the split is the whole argument for the
next move. At 1-shot the learned per-parameter rate did what I predicted: mean 0.4760 against MAML's
0.4365 — a lift of $+0.0395$ — and, more telling, the seed spread collapsed from std 0.0147 to 0.0047,
a factor of $3.1$ tighter, the three seeds bunching at {0.4693, 0.4794, 0.4793}. Both halves of the
1-shot prediction landed: mean up, variance down, exactly the signature of removing the single-rate
knife-edge. But read the 5-shot columns and the other half of my worry landed too. miniImageNet 5-shot
*slipped* $-0.0142$, from 0.6379 to 0.6237, and CIFAR-FS 5-shot slipped further, $-0.0182$, from 0.7067
to 0.6885 — and CIFAR's seed spread actually *widened*, std 0.0027 to 0.0113, a factor of $4.2$, with
seed 123 sagging to 0.6726 while the other two sit at 0.6959 and 0.6971, a within-condition range of
$0.0245$ driven almost entirely by one seed. Now average the three benchmarks and the verdict is
sharp: MAML sits at $0.5937$, Meta-SGD at $0.5961$, a net move of $+0.0024$ — a wash. The per-parameter
rate did not raise the ceiling; it *moved accuracy around*, robbing the 5-shot benchmarks to pay 1-shot.
That is the signature of *meta-overfitting*: with five examples per class the rigid scalar step was
already near the ceiling, so the extra hundred-thousand-odd meta-parameters of $\alpha$ had little left
to fix and instead tuned themselves to the meta-training episodes in a way that did not transfer, and
the one seed that landed on a worse meta-optimum (0.6726) is what widened the spread. The lesson is
sharp: I have been adding *capacity to the step* — first a scalar, then a per-coordinate vector — and at
5-shot the ceiling is not the step's expressiveness at all. So I should stop pouring capacity into *how*
the step moves and ask a different question entirely: *which* parameters should the inner loop move at
all?

Here is what actually nags me about both rungs so far, and it is the same complaint from two sides.
The cost: the inner loop runs per task, inside every one of 60,000 meta-iterations, over every single
parameter of CNN4 — all $121\text{K}$ of them across four conv blocks plus BatchNorm plus the head — and
because the adapted parameters depend on $\theta$ both directly and through the inner gradient,
differentiating the meta-loss drags in the second-order term $(I-\alpha\nabla_\theta^2\mathcal{L}^{\text{
sup}})^\top\nabla_{\theta'}\mathcal{L}^{\text{qry}}$, a Hessian-vector product over the *whole* network,
one extra backward pass per task. Meta-SGD made this *worse*, not better — it added another $121\text{K}$
meta-parameters to the graph and a per-parameter rate to differentiate through. The opacity: I cannot
actually say what that full-network inner loop is *for*. Two stories, and I do not know which is true.
Story one, rapid learning: the meta-initialization is a cleverly conditioned launch point, and the few
inner steps make big, efficient changes to the network's internal representations, genuinely re-learning
each task. Story two, feature reuse: the meta-initialization already holds good, general conv features,
and the inner loop barely moves them — adaptation is almost a no-op on most of the network. These
predict completely different things about where the work happens, and Meta-SGD's 5-shot regression is a
hint: if the body were doing heavy per-task re-learning, a richer step should have helped at 5-shot too;
that it *hurt* suggests the body's adaptation was never the point, that the extra rates were free
capacity to overfit precisely because they had no real work to do.

Before I try to make the inner loop cheaper, I have to settle which story is true, because the answer
dictates what I am even allowed to cut. And the first thing to get straight is that CNN4 is not
homogeneous — there is a sharp structural difference between the head and everything before it, and it
matters here more than anywhere. Think about a five-way episode. The five output neurons get mapped to
an arbitrary, task-specific set of classes; in one task the five outputs mean (dog, cat, frog,
cupcake, phone), in the next, drawn from the same held-out pool, (airplane, frog, boat, car, pumpkin).
The number of distinct assignments is enormous — for any five classes there are $5! = 120$ ways to bind
them to the five output slots, times all the ways to choose the five classes from the pool — so there
is no fixed last layer that can be right for two tasks at once: the map from feature space to *these
particular five output slots* is a fresh, essentially random relabeling per task. So the head *must* be
free to move per task no matter which story is true; that is forced by the problem, not by any
diagnostic. The conv blocks, by contrast, compute class-agnostic features — edges, textures, parts —
and those *could* be shared across tasks without moving. So the rapid-learning-versus-feature-reuse
question is not really about the head at all; the head trivially must adapt. The whole question lives in
the body: does the inner loop substantively change the body's representations per task, or leave them
essentially fixed? And the body is where the parameters *are* — of the $121\text{K}$ total, the head is
only $8005$, about $6.6\%$; the convolutional body is $\sim113\text{K}$, fourteen times larger. So this
is also the question of what the overwhelming majority of the network's adaptation is doing.

How would I tell? Two ways. The blunt one: freeze a contiguous block of body layers at test time —
give them no inner update, force them to reuse the meta-initialization's features — and read off
accuracy. If rapid learning is the story, freezing should hurt; if feature reuse, it should barely
matter. The documented result on exactly this MAML/MiniImageNet setup is unambiguous: freezing layer 1
leaves 5-way-1-shot at 46.5 against 46.9 unfrozen; freezing layers 1–2, 46.4; 1–3, 46.3; freezing
*all four* conv layers, everything but the head, 46.3 ± 0.4; on 5-shot it slips only from 63.1 to 61.0.
Removing the inner-loop adaptation of the entire body costs about $0.6$ points at 1-shot — and here I
can calibrate that against my own MAML feedback, where the seed-to-seed std was $1.47$ points. A
$0.6$-point effect sitting *below* the $1.5$-point seed noise is statistically indistinguishable from
zero: whatever the body's inner loop contributes at 1-shot, it is smaller than the run-to-run scatter I
already tolerate. That tilts me hard toward feature reuse, but freezing is indirect. The sharper
measurement compares the body's representation before and after the inner loop directly — CCA/CKA of a
layer's activations pre versus post adaptation, 1 for identical functions. The established picture: the
body conv layers sit at CCA above 0.9, CKA near 1 — the inner loop induces almost no functional change
in the body — while the head sits below 0.5, moving a lot, as it must. The Euclidean weight movement
says the same even more starkly: the average distance between initialization and finetuned weights is
tiny for every layer except the last, *even though the body layers hold fourteen times more parameters
than the head*. More parameters, less movement. And this is not a quirk of the converged model — the
pattern holds from ten thousand iterations in. Feature reuse, not rapid learning. The body's good
features come from the *outer* loop accumulating across tasks; the inner loop, on the body, is along for
the ride. This also retro-explains Meta-SGD's wash: a learned per-coordinate rate on body parameters
that should barely move can only meta-overfit, which is exactly the 5-shot regression I just measured.

If the inner loop leaves the body's function essentially unchanged, then computing inner-loop updates
for the body is computing a correction that rounds to zero and then paying full second-order price to
backpropagate through it. The freezing experiment already showed I can drop body adaptation at *test*
time with no accuracy loss. The obvious next thought: why run it at *training* time either? I have to
be careful — freezing-at-test and not-adapting-at-train are different interventions. At test, freezing
the body of an already-trained model just says "do not bother adapting features that will not move." At
training, if I stop adapting the body in the inner loop, I change the *meta-objective* — the body
parameters are now learned purely by the outer loop, never task-adapted. Would the body still learn
good features that way? The timing result reassures me: the high body-similarity and freezing-robustness
are present from ten thousand iterations in, so the inner loop is inert on the body *throughout*
training, not just at the end. If it is not doing anything to the body during training, removing it
during training should not change what the body learns. So I can cut the body's inner loop at *both*
training and testing, and almost no inner loop survives — only the head, which genuinely has to adapt
for the per-task class alignment.

Where exactly do I make the cut, though — is head-versus-body the right partition, or should I keep the
last conv block adapting too? The freezing curve answers this in its own shape: 46.5 with layer 1
frozen, 46.4 with 1–2, 46.3 with 1–3, 46.3 with all four — the accuracy is essentially flat as I freeze
*more* of the body, and the CCA-above-0.9 verdict is uniform across all four conv layers, not just the
early ones. So there is no body layer whose adaptation I can point to as load-bearing; keeping any conv
block in the inner loop reintroduces exactly the meta-overfitting-prone rates I am trying to remove
while buying nothing the freezing curve can detect. The cleanest defensible cut is therefore all-body-
frozen, head-adapting. And the mirror-image partition — freeze the head, adapt the body — is not even
admissible here, because the relabeling argument makes the head the one thing that *provably* must move
per task; a frozen head cannot represent two different five-way class bindings, so "adapt body, freeze
head" fails on the problem's structure before I even measure it. Head-adapts / body-frozen is the only
partition consistent with both the feature-reuse evidence and the relabeling constraint.

Write the rule. Partition $\theta = (\theta_1,\dots,\theta_l)$, body $\theta_1,\dots,\theta_{l-1}$ and
head $\theta_l$. The inner loop updates only the head: $\theta_l$ descends the support loss at
$\alpha$, the body components are pinned at the meta-initialization across all inner steps. The outer
loop is untouched — same meta-loss, the query loss at the adapted parameters summed over the four-task
meta-batch — and the body parameters are still *learned*, just only by the outer loop now. That is the
whole move: keep the head's inner loop, delete the body's, train and test alike. And notice what this
does to cost, which was half my complaint: at evaluation the body is a single forward pass and only the
$8005$-parameter linear head iterates its ten steps, and the surviving Hessian-vector product is over
the head, not the $121\text{K}$-parameter whole — the inner-loop gradient shrinks by more than an order
of magnitude while, I will argue next, the accuracy does not.

Have I just reinvented first-order MAML? No, and the distinction is load-bearing, because it is why I
expect this to *match* MAML's accuracy rather than lag it like a first-order approximation would.
First-order MAML keeps the full inner loop (adapts every parameter) but *drops the Hessian*. I am doing
the opposite axis: I keep the second-order machinery and shrink *which* parameters get an inner loop
down to the head. I do not want to assert this — I want to see the surviving second-order term with my
own eyes on the smallest example that has a body and a head, so take scalars $\theta_1$ (body),
$\theta_2$ (head), prediction $f=\theta_2\theta_1 x$, MSE loss. Support point $(x_s,y_s)$: the
head's inner gradient is $\partial\mathcal{L}^{\text{sup}}/\partial\theta_2 = (\theta_2\theta_1 x_s -
y_s)\,\theta_1 x_s$, and the crucial thing is that this expression *contains $\theta_1$* — it was
computed on a forward pass through the body. One head-only step gives $\theta_2' = \theta_2 - \alpha
(\theta_2\theta_1 x_s - y_s)\theta_1 x_s$, with $\theta_1$ pinned. The adapted query prediction is
$f' = \theta_2'\,\theta_1\,x_q$, and now differentiate the query loss with respect to the *body*
$\theta_1$. Because $\theta_1$ sits in $\theta_2'$ (through that head gradient) as well as in the
explicit factor of $f'$, the derivative $\partial(\theta_2'\theta_1)/\partial\theta_1 = \theta_2 - 2
\alpha\,r_s\theta_1 x_s - \alpha\,\theta_1^2\theta_2 x_s^2$, where $r_s=\theta_2\theta_1 x_s-y_s$: a
leading $\theta_2$ that is the "first-order-looking" part, plus two $\alpha$-proportional terms that
are the curvature flowing through the head's inner step. Put numbers on it — $\theta_1=\theta_2=1$,
$x_s=x_q=1$, $y=0$, $\alpha=0.1$ — and it becomes concrete. Then $\theta_2'=1-0.1=0.9$, the whole
adapted output is $F(\theta_1)=\theta_1-0.1\theta_1^3$, so $F'(\theta_1)=1-0.3\theta_1^2 = 0.7$ at
$\theta_1=1$, and the body meta-gradient is $F\cdot F' = 0.9\times0.7 = 0.63$. First-order MAML, which
detaches $\theta_2'$ and keeps only the leading term, would report $0.9\times1 = 0.9$. The retained
second-order piece is the $-0.3\theta_1^2$ in $F'$ — a real $30\%$ of the body gradient, not a rounding
error. Removing the body's inner loop does *not* make this first-order; the curvature that flows
through the *head's* inner step is retained, and that is exactly the part of the second-order
information attached to the only thing still adapting. That is why I expect ANIL to match full MAML
where first-order MAML sometimes lags.

That surviving term also forces me to be honest about *where* the cost savings actually are, because
the two are the same fact seen twice. The body meta-gradient does not vanish — I just computed it is a
real $0.63$ — so the outer backward pass still flows into all $\sim113\text{K}$ body parameters, and
meta-training is *not* an order of magnitude cheaper end to end. What shrinks at training is the inner
loop specifically: `torch.autograd.grad` is taken over the $8005$-parameter head rather than the whole
network, so the per-step inner gradient and the Hessian-vector product it feeds are the small object,
even though the meta-backward that consumes them still touches the body. The clean, full order-of-
magnitude saving is at *evaluation*, where there is no outer backward at all: the body is a single
forward pass and only the tiny head iterates its ten steps. So the honest cost story is "much cheaper
inner loop, dramatically cheaper eval, meta-backward roughly unchanged" — which is exactly what I want,
since the meta-backward is where the retained curvature lives and I am keeping it on purpose.

Now ground it in *this task's* edit surface, because the implementation has a sharp gotcha and the
harness primitive forces a specific shape. The differentiable update I have is `l2l.update_module(model,
updates=[...])`, which wants a full-length list — one update per parameter, in order — and swaps each
$p$ for $p+u$. So I compute the support cross-entropy, take `torch.autograd.grad` with respect to *only
the head parameters* with `create_graph=True` (keeping the surviving second-order term I just verified),
and build the updates list with $-\alpha g$ for each head parameter and `torch.zeros_like(p)` for every
body parameter. The gotcha: `update_module` does *not* mutate in place — it *replaces* the parameter
objects with fresh tensors ($p+u$) every step. So if I grab references to the head parameters once
before the loop, after the first step those references are stale, my gradient is taken with respect to
detached tensors, and every subsequent update silently comes out zero — the method looks like it is
running while doing nothing, and worse, it would look *plausible*, since a zeros-everywhere body plus a
frozen-after-step-1 head still returns sane accuracy near the initialization. So I re-identify the head
parameters by name *inside* each inner step, from the current module: I record the head parameter
*names* once in `__init__` (names are stable across the object swaps even though the tensors are not — I
scan `named_parameters` for the ones containing "classifier"), and inside each step I re-collect the
current parameter objects matching those names, build an id-set, and route updates by id. The rule
carries no learnable optimizer state of its own — unlike Meta-SGD I add no meta-parameters — so
`meta_parameters()` returns `[]`, and the only things the outer Adam optimizes are the model's own
weights. The full scaffold module is in the answer.

There is a structural point worth naming, because it inverts the pattern of the ladder so far. Every
previous rung *added* to the inner loop's expressiveness — MAML fixed a scalar, Meta-SGD generalized it
to a diagonal, a strict superset. ANIL goes the other way: freezing the body is exactly MAML with the
body's inner rates constrained to zero, so ANIL's inner loop is a *restriction* of MAML's, a strictly
smaller hypothesis space of adaptations. On expressiveness alone it can only match MAML, never exceed
it. Yet I expect it to *beat* MAML at 5-shot, and the only way to square that is that the extra
expressiveness MAML had over ANIL was expressiveness to *overfit* — the body's freedom to adapt was
capacity spent fitting the support, not generalizing. So the move that helps is a constraint, not a
capacity, which is the precise opposite of Meta-SGD, where I added capacity and it hurt. Two rungs, two
directions, and the same underlying fact: at 5-shot the binding limit was never how expressive the step
is, it was how much room the inner loop has to overfit the body. ANIL removes that room.

So the delta from step 2 is a change of axis, not of capacity: Meta-SGD added a learnable
per-parameter rate over *all* parameters and meta-overfit at 5-shot; ANIL instead freezes the body's
inner loop entirely and adapts only the head, with no extra meta-parameters and a fraction of the
per-task cost — the inner-loop Hessian-vector product now runs over the $8005$-parameter head rather
than the $121\text{K}$-parameter whole, more than an order of magnitude smaller. Reading Meta-SGD's shape, the falsifiable claims are pointed at the two places it
failed. At **5-shot** ANIL should *recover* what Meta-SGD lost and likely edge past MAML, because the
meta-overfitting was caused by adapting body parameters that should not move — freeze them and the
regression should reverse: miniImageNet 5-shot back above 0.6379 and CIFAR-FS 5-shot back toward or
past 0.7067, with CIFAR's seed spread tightening from Meta-SGD's 0.0113 back toward MAML's 0.0027. At
**1-shot** ANIL should at least hold Meta-SGD's 0.4760 — the head still gets a genuine,
second-order-aware inner loop, which is the part that fixed 1-shot in the first place — and may improve
on it, since head-only adaptation from a single example is far less prone to the instability that forced
MAML's tiny rate (there is no full-network step to diverge, and the wobbling BatchNorm statistics no
longer feed an inner update). If instead 1-shot were to drop, that would falsify the feature-reuse story
for the thin-support regime and say the body's inner loop was load-bearing at 1-shot after all. The bet
is the opposite: feature reuse is the whole game, the body's inner loop was always along for the ride,
and removing it fixes 5-shot while holding 1-shot — which would make ANIL the strongest rung on every
benchmark at the lowest cost. Concretely the bar is the three-benchmark average: MAML sits at $0.5937$
and Meta-SGD at $0.5961$, so for ANIL to be the strongest rung rather than a third wash it has to clear
both — and if the feature-reuse account is right, it should clear them not by robbing one benchmark for
another the way Meta-SGD did, but by lifting the two 5-shot columns while at least holding 1-shot, a
uniform gain rather than a redistribution. The 5-shot columns are where I will know first whether
freezing the body was the right axis.
