I start where the scaffold starts, because the cheapest conceivable unlearning rule is also the right
thing to measure first: keep finetuning the trained model on the retained data and never show it the
forget set again. Before I add any active forgetting machinery, I want to know how far passive erosion
alone gets me, and exactly where it falls short — that failure is what every later rung will have to
out-do, so I need its number on the board.

The setup forced on me is specific, and the default fill has to live inside it. The harness has already
pretrained `f(·; w)` on the full dataset for 80 epochs of SGD with cosine annealing; class 0 is the
forget set; and now I get 20 epochs of an unlearning loop that hands me, on every step, one retain
minibatch `(retain_x, retain_y)` and one forget minibatch `(forget_x, forget_y)`, both already on the
device, plus a *fixed* `Adam(lr=0.001)` optimizer. I do not get to pick the optimizer, the learning
rate, the schedule, or the number of steps — the only thing I author is the body of `unlearn_step`. So
the design space here is genuinely just "what update do I compute from these two minibatches," and the
default chooses the most conservative possible answer: compute cross-entropy on the retain minibatch,
backpropagate, step; ignore `forget_x, forget_y` entirely. The `forget_weight = 0.0` field in the
scaffold is a placeholder advertising that the forget batch *could* be weighted in — the default leaves
it at zero.

The reason to expect this to preserve utility is direct. The retain loss is exactly the original
training objective restricted to `D_r`, run from weights that already minimize it well, with a small
Adam step size. There is no force in this loss that pushes the model off the function it already
computes on the retained classes — I am only reinforcing what it already does. So `retain_acc` should
stay high; the model keeps being good at the nine classes it is allowed to remember.

The reason to expect this to *fail at forgetting* is just as direct, and it is the crux. There is no
term anywhere in this objective that pushes *against* `D_f`. Catastrophic forgetting is the only thing
that could erode the forget-class knowledge — the hope that by training on the other classes long
enough, the representation drifts and the class-0 competence decays on its own. But a network that has
memorized class-0 images keeps classifying them correctly long after I stop training on them, because
nothing is actively unlearning them; the shared lower-layer features that recognize class 0 are also
load-bearing for visually adjacent retained classes, so retain finetuning has every incentive to *keep*
them. Twenty epochs of Adam on the retain set is nowhere near enough drift to erase a class the model
spent 80 epochs memorizing. I therefore expect the forget set to remain partially memorized, and — more
importantly for this task — for the membership-inference signal to survive.

That second point is what the metric is really measuring, and it is worth being precise about, because
the headline `forget_acc` will mislead me here. The forget class is class 0; the model still predicts
class 0 confidently on held-out class-0 images, so `forget_acc` on the test split could read near zero
*for the wrong reason* — but the real privacy leak is `forget_mia_auc`. The attack compares the model's
max-softmax confidence on forget-*train* images (members, seen during the original 80-epoch run) against
forget-*test* images (non-members), and reports the Mann-Whitney AUC. A model that genuinely never saw
`D_f` would be equally (un)confident on both, giving AUC ≈ 0.5; a model that *memorized* the forget
training images is systematically more confident on them, giving AUC well above 0.5. Retain finetuning
does nothing to close that train/test confidence gap on class 0 — it never touches class 0 — so I
expect `forget_mia_auc` to sit comfortably above 0.5, somewhere in the high-0.4-to-0.5 range, which is
the residual memorization the score punishes through the `(1 - forget_mia_auc)` term. The `unlearn_score`
will therefore be capped not by retain accuracy (which stays high) but by this un-erased membership
signal.

So this rung is deliberately the floor for the *forgetting* axis while being strong on the *utility*
axis. It tells me two things I will need. First, the achievable ceiling on `retain_acc` for this
harness — whatever passive finetuning preserves is roughly the best any later method can hope to hold,
because every active-forgetting term I add will only ever *cost* retain accuracy, never add it. Second,
the size of the membership signal I still have to kill: the gap between this rung's `forget_mia_auc` and
0.5 is the privacy work left undone. The next rung has to introduce an *active* forgetting pressure that
closes that gap — and the immediate danger, which I will diagnose against this rung's numbers, is that
the obvious active pressure (climbing the forget loss) is unbounded and will trade away exactly the
retain accuracy this rung shows is the thing worth protecting.

What I watch: `retain_acc` high and roughly at the harness ceiling on all three benchmarks;
`forget_mia_auc` stubbornly above 0.5, marking the un-erased memorization; `unlearn_score` middling —
good utility, weak forgetting. The full scaffold module is in the answer.
