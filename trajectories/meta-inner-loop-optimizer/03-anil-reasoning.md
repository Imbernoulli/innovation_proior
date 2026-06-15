Meta-SGD's numbers split exactly along the seam I flagged, and the split is the whole argument for the
next move. At 1-shot the learned per-parameter rate did what I predicted: mean 0.4760 against MAML's
0.4365 — a clear lift — and, more telling, the seed spread collapsed from std 0.0147 to 0.0047, the
three seeds bunching at {0.4693, 0.4794, 0.4793}. So the diagnosis was right: 1-shot's fragility was a
single-shared-rate problem, and giving each coordinate its own learnable rate removed the knife-edge.
But read the 5-shot columns and the other half of my worry landed too. miniImageNet 5-shot *slipped*,
0.6237 against MAML's 0.6379, and CIFAR-FS 5-shot slipped further, 0.6885 against 0.7067 — and CIFAR's
seed spread actually widened (std 0.0113, seed 123 sagging to 0.6726). So adding a per-parameter rate
helped exactly where the support set was thin and hurt where it was not. That is the signature of
*meta-overfitting*: with five examples per class the rigid scalar step was already near the ceiling, so
the extra hundred-thousand meta-parameters of $\alpha$ had little left to fix and instead tuned
themselves to the meta-training episodes in a way that did not transfer. The lesson is sharp: I have
been adding *capacity to the step* — first a scalar, then a per-coordinate vector — and at 5-shot the
ceiling is not the step's expressiveness at all. So I should stop pouring capacity into *how* the step
moves and ask a different question entirely: *which* parameters should the inner loop move at all?

Here is what actually nags me about both rungs so far, and it is the same complaint from two sides.
The cost: the inner loop runs per task, inside every one of 60,000 meta-iterations, over every single
parameter of CNN4 — all four conv blocks plus BatchNorm plus the head — and because the adapted
parameters depend on $\theta$ both directly and through the inner gradient, differentiating the
meta-loss drags in the second-order term $(I-\alpha\nabla_\theta^2\mathcal{L}^{\text{sup}})^\top
\nabla_{\theta'}\mathcal{L}^{\text{qry}}$, a Hessian-vector product over the *whole* network, one extra
backward pass per task. Meta-SGD made this *worse*, not better — it added $\alpha$ to the meta-graph
and a per-parameter rate to differentiate through. The opacity: I cannot actually say what that
full-network inner loop is *for*. Two stories, and I do not know which is true. Story one, rapid
learning: the meta-initialization is a cleverly conditioned launch point, and the few inner steps make
big, efficient changes to the network's internal representations, genuinely re-learning each task.
Story two, feature reuse: the meta-initialization already holds good, general conv features, and the
inner loop barely moves them — adaptation is almost a no-op on most of the network. These predict
completely different things about where the work happens, and Meta-SGD's 5-shot regression is a hint:
if the body were doing heavy per-task re-learning, a richer step should have helped at 5-shot too; that
it hurt suggests the body's adaptation was never the point.

Before I try to make the inner loop cheaper, I have to settle which story is true, because the answer
dictates what I am even allowed to cut. And the first thing to get straight is that CNN4 is not
homogeneous — there is a sharp structural difference between the head and everything before it, and it
matters here more than anywhere. Think about a five-way episode. The five output neurons get mapped to
an arbitrary, task-specific set of classes; in one task the five outputs mean (dog, cat, frog,
cupcake, phone), in the next, drawn from the same held-out pool, (airplane, frog, boat, car, pumpkin).
There is no fixed last layer that can be right for both at once — the assignment from feature space to
*these particular five output slots* is a fresh, essentially random relabeling per task. So the head
*must* be free to move per task no matter which story is true; that is forced by the problem. The conv
blocks compute class-agnostic features — edges, textures, parts — and those *could* be shared across
tasks without moving. So the rapid-learning-versus-feature-reuse question is not really about the head
at all; the head trivially must adapt. The whole question lives in the body: does the inner loop
substantively change the body's representations per task, or leave them essentially fixed?

How would I tell? Two ways. The blunt one: freeze a contiguous block of body layers at test time —
give them no inner update, force them to reuse the meta-initialization's features — and read off
accuracy. If rapid learning is the story, freezing should hurt; if feature reuse, it should barely
matter. The documented result on exactly this MAML/MiniImageNet setup is unambiguous: freezing layer 1
leaves 5-way-1-shot at 46.5 against 46.9 unfrozen; freezing layers 1–2, 46.4; 1–3, 46.3; freezing
*all four* conv layers, everything but the head, 46.3 ± 0.4; on 5-shot it slips only from 63.1 to 61.0.
Removing the inner-loop adaptation of the entire body costs essentially nothing. That tilts me hard
toward feature reuse, but freezing is indirect. The sharper measurement compares the body's
representation before and after the inner loop directly — CCA/CKA of a layer's activations pre versus
post adaptation, 1 for identical functions. The established picture: the body conv layers sit at CCA
above 0.9, CKA near 1 — the inner loop induces almost no functional change in the body — while the head
sits below 0.5, moving a lot, as it must. The Euclidean weight movement says the same even more
starkly: the average distance between initialization and finetuned weights is tiny for every layer
except the last, *even though the body layers have far more parameters than the head*. More
parameters, less movement. And this is not a quirk of the converged model — the pattern holds from
ten thousand iterations in. Feature reuse, not rapid learning. The body's good features come from the
*outer* loop accumulating across tasks; the inner loop, on the body, is along for the ride. This also
retro-explains Meta-SGD: a learned per-coordinate rate on body parameters that should barely move can
only meta-overfit, which is exactly the 5-shot regression I just saw.

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

Write the rule. Partition $\theta = (\theta_1,\dots,\theta_l)$, body $\theta_1,\dots,\theta_{l-1}$ and
head $\theta_l$. The inner loop updates only the head: $\theta_l$ descends the support loss at
$\alpha$, the body components are pinned at the meta-initialization across all inner steps. The outer
loop is untouched — same meta-loss, the query loss at the adapted parameters summed over the four-task
meta-batch — and the body parameters are still *learned*, just only by the outer loop now. That is the
whole move: keep the head's inner loop, delete the body's, train and test alike.

Have I just reinvented first-order MAML? No, and the distinction is load-bearing, because it is why I
expect this to *match* MAML's accuracy rather than lag it like a first-order approximation would.
First-order MAML keeps the full inner loop (adapts every parameter) but *drops the Hessian*. I am doing
the opposite axis: I keep the second-order machinery and shrink *which* parameters get an inner loop
down to the head. Take the smallest example with a body and a head, a two-layer net $\hat y =
\theta_2(\theta_1 x)$, one inner step. In my rule the body does not move, so the adapted prediction at
the query point is $[\theta_2 - \partial\mathcal{L}^{\text{sup}}/\partial\theta_2]\cdot\theta_1\cdot
x_2$. The *second* bracket collapsed to a bare $\theta_1$ — the body not adapting. But the *first*
bracket is still there and still contains $\theta_1$, because the head's inner-loop gradient
$\partial\mathcal{L}^{\text{sup}}/\partial\theta_2$ was computed on a forward pass through the body
$\theta_1$. So when I differentiate the query loss with respect to the body $\theta_1$, I still
differentiate *through* that head update, and a second-order term survives. Removing the body's inner
loop does *not* make this first-order — the curvature that flows through the *head's* inner step is
retained. That is exactly why I expect ANIL to match full MAML where first-order MAML sometimes lags: I
kept the part of the second-order information attached to the only thing still adapting.

Now ground it in *this task's* edit surface, because the implementation has a sharp gotcha and the
harness primitive forces a specific shape. The differentiable update I have is `l2l.update_module(model,
updates=[...])`, which wants a full-length list — one update per parameter, in order — and swaps each
$p$ for $p+u$. So I compute the support cross-entropy, take `torch.autograd.grad` with respect to *only
the head parameters* with `create_graph=True` (keeping the surviving second-order term), and build the
updates list with $-\alpha g$ for each head parameter and `torch.zeros_like(p)` for every body
parameter. The gotcha: `update_module` does *not* mutate in place — it *replaces* the parameter objects
with fresh tensors ($p+u$) every step. So if I grab references to the head parameters once before the
loop, after the first step those references are stale, my gradient is taken with respect to detached
tensors, and every subsequent update silently comes out zero — the method looks like it is running
while doing nothing. So I re-identify the head parameters by name *inside* each inner step, from the
current module: I record the head parameter *names* once in `__init__` (names are stable across the
object swaps even though the tensors are not — I scan `named_parameters` for the ones containing
"classifier"), and inside each step I re-collect the current parameter objects matching those names,
build an id-set, and route updates by id. The rule carries no learnable optimizer state of its own —
unlike Meta-SGD I add no meta-parameters — so `meta_parameters()` returns `[]`, and the only things the
outer Adam optimizes are the model's own weights. The full scaffold module is in the answer.

So the delta from step 2 is a change of axis, not of capacity: Meta-SGD added a learnable
per-parameter rate over *all* parameters and meta-overfit at 5-shot; ANIL instead freezes the body's
inner loop entirely and adapts only the head, with no extra meta-parameters and a fraction of the
per-task cost (at evaluation the body is a single forward pass and only the tiny linear head iterates
its ten steps). Reading Meta-SGD's shape, the falsifiable claims are pointed at the two places it
failed. At **5-shot** ANIL should *recover* what Meta-SGD lost and likely edge past MAML, because the
meta-overfitting was caused by adapting body parameters that should not move — freeze them and the
regression should reverse: miniImageNet 5-shot back above 0.6379 and CIFAR-FS 5-shot back toward or
past 0.7067, with CIFAR's seed spread tightening from Meta-SGD's 0.0113. At **1-shot** ANIL should at
least hold Meta-SGD's 0.4760 — the head still gets a genuine, second-order-aware inner loop, which is
the part that fixed 1-shot in the first place — and may improve on it, since head-only adaptation from
a single example is far less prone to the instability that forced MAML's tiny rate. If instead 1-shot
were to drop, that would falsify the feature-reuse story for the thin-support regime and say the body's
inner loop was load-bearing at 1-shot after all. The bet is the opposite: feature reuse is the whole
game, the body's inner loop was always along for the ride, and removing it fixes 5-shot while holding
1-shot — which would make ANIL the strongest rung on every benchmark at the lowest cost.
