The random run told me where the floor sits, and it told me in numbers that line up almost exactly with
the failure modes I predicted from the structure of the rule. On **spambase** — binary, balanced, a
budget that is a healthy fraction of the pool — random landed at 0.935 accuracy and 0.904 auc on seed 42,
the strongest of its three datasets, exactly as I expected: a representative i.i.d. sample is genuinely
hard to beat when the boundary is simple and the classes aren't pathological. On **letter** — 26 classes,
the largest pool, the budget spread thin — random sits at 0.835 accuracy / 0.767 auc on seed 42 but its
*mean* drops to 0.816 / 0.724, the widest gap between the lucky seed and the average of any dataset. That
is the decaying-`α` problem made visible: with 26 classes the region of uncertainty is large and random
keeps re-buying already-separated letters while the confusable pairs stay under-sampled, so the curve
rises slowly (low auc) and where it ends depends heavily on which letters the blind draw happened to
cover. **Splice** sits in between at 0.771 / 0.751 (seed 42), 0.789 / 0.736 (mean), the middle case I
guessed. So the diagnosis is the one I set up: random's weakness is *not* a learning problem, it is that
it spends budget representatively — on the bulk the model already handles — instead of on the contested
region. The fix is to stop drawing blindly and let the model say which unlabeled points it is least sure
of. The whole game is picking the informative ones, and random, by refusing to use the model, leaves the
entire `1/ε`-vs-`ln(1/ε)` adaptivity prize on the table.

So what does "informative" mean, concretely, for a rule I can actually run inside this scaffold? The
cleanest theoretical handle is the version space — the set of all hypotheses in my model class still
consistent with the labels I have seen. Each label I add must be consistent with the true concept, so it
carves the set down, and the question "which example is worth labeling" becomes "which example, once
labeled, carves the most off the version space." There is a clean object that pins this down: the region
of uncertainty `R = { x : two still-consistent hypotheses disagree on x }`. If `x` is outside `R`, every
consistent hypothesis already agrees on it, so labeling it can't contradict any of them — the version
space doesn't shrink, the label is wasted. If `x` is inside `R`, some consistent hypotheses say one thing
and others the opposite, so whatever the true label is, it kills the ones that guessed wrong. Only
examples inside `R` inform me. And its mass `α = Pr[x ∈ R]` is the very quantity I watched decay in
random's letter curve — random draws hit `R` with probability `α`, which heads to zero, which is why
random's auc on letter is so low. So in principle the fix is obvious: only ask about points inside `R`.
Then every query shrinks `R` and the per-label information stops decaying.

Can I just do that? Let me try the most direct things and watch them break, because the wall each hits
tells me what the rule actually has to look like. The first idea is membership queries: let the learner
*synthesize* the single point it is most confused about and hand it to the oracle. In concept-learning
theory this is enormously powerful — it is exactly what buys the `ln(1/ε)` rate. But a synthesized point
in feature space need not correspond to any real input; people who tried this on images got back
artificial hybrid characters with no recognizable symbol, gibberish a human can't classify. And in *this*
setting it is doubly dead: the harness gives me `query(n)` that must return *indices into `self.X`* — I am
choosing among real, fixed pool rows, I literally cannot fabricate a query. Wall. So don't synthesize,
filter: I have a giant pool of real unlabeled rows sitting in `self.X`; instead of constructing the ideal
query, draw real examples and only ask about the ones inside `R`. Every query is then a genuine row, and
the in-`R` guarantee still holds. This is the selective-sampling move, and it is right — it keeps the
efficiency argument while killing the gibberish problem.

But now I have to actually compute "is `x` in `R`," and `R` is defined through the *entire* version space.
To test membership I'd need to find two consistent hypotheses disagreeing on `x`, which means maintaining
the whole set of consistent hypotheses and re-deriving it after every label. For the small neural
classifier `self.clf` the harness trains, enumerating or even representing the version space is hopeless,
and I'd redo it every round. Worse, "in `R` or not" is a hard binary test, but the contract is a
*budget*: `query(n)` returns `n` indices, so I don't want yes/no, I want to *rank* the pool and take the
top `n`. The region of uncertainty is the right idea and an unusable object. Wall. So how do people
approximate `R` without the version space? The committee idea: draw two hypotheses at random from the
version space, run both on each candidate, and treat their *disagreement* as the signal "this point is
inside `R`" — because if two consistent hypotheses disagree on `x`, then `x` is by definition in `R`.
There is even real theory: a disagreement-querying learner can need only logarithmically many labels while
generalization error still falls. But to run it I'd need to draw hypotheses uniformly from the version
space, assume the data are clean enough that a perfectly-consistent classifier exists, and maintain a
*committee* of models rather than the single `self.clf` I am handed. On these tabular datasets — noisy
labels, no guarantee of a perfect classifier, and a harness that retrains exactly one network — all three
crack. Wall again, but an informative one: it tells me exactly what I need — a proxy for "x is in `R`"
that (i) uses only the single trained classifier the scaffold gives me, (ii) doesn't assume noise-free
realizability, and (iii) produces a *ranking*, a scalar I can sort to spend a budget.

Look hard at what a committee's disagreement is really measuring. Two hypotheses disagree on `x` precisely
when the evidence so far doesn't determine the label — where, if forced to commit, the learner would be
guessing. But the single trained `self.clf` already carries that information directly, because it is a
*probabilistic* classifier: it outputs a posterior, its own degree of belief about each label. Where two
consistent hypotheses would split, a well-trained probabilistic classifier sits near the fence — its
posterior is close to ambivalent. Where all consistent hypotheses agree, the posterior is confident. So
the classifier's own *uncertainty about its prediction* is a single-model, scalar surrogate for "x is in
the region of uncertainty," and it dodges all three cracks at once: one model (the one the scaffold
trains), no realizability assumption (a posterior is defined whether or not a perfect classifier exists),
and a number I can sort. And the harness hands me exactly this through `self.predict_prob(X, Y)`, which
returns the softmax `[len(X), n_classes]` — no committee, no dropout, just the forward pass I'd already do.

Make it concrete. The classifier estimates the posterior over classes; if forced to decide it picks
`ŷ = argmax_y p(y|x)` and acts with confidence `p(ŷ|x)`, the mass on its chosen class. When is it most
certain? When `p(ŷ|x)` is near 1 — the top class dominates. When least certain? When `p(ŷ|x)` is small —
even the best class barely wins, the mass is spread, the model is close to guessing. So the natural
uncertainty score is one minus the top-class probability, `1 − max_y p(y|x)`, and I query the examples
with the *smallest* maximum posterior. In the binary case the larger posterior is
`max(p, 1−p) = 0.5 + |p − 0.5|`, so minimizing the top-class probability is exactly "posterior nearest
0.5" — querying right at the decision boundary, where a label resolves the most. The multi-class form is
genuinely the same idea, not a heuristic bolted on. And `1 − max_y p(y|x)` has a reading I like even
better than "distance to the boundary": it is the model's own estimate of the probability that it will
*mislabel* `x`. If the model commits to `ŷ`, under its own posterior the probability it is wrong is exactly
`1 − p(ŷ|x)`. So I am spending labels on the examples the model believes it is most likely to get wrong —
its expected 0/1 loss, point by point — which are precisely the ones whose true labels would correct it.

Before I lock it in, what does this score throw away, and is a richer score worth it? `1 − max` looks only
at the single most probable class; it ignores how the rest of the mass is distributed. Two examples with
the same top probability of 0.5 differ if in one the other 0.5 sits on a single runner-up and in the other
it is smeared over many classes. So I could score by the margin `p(ŷ₁|x) − p(ŷ₂|x)` (uses the runner-up
the top-only score discards) or by the entropy `−Σ_y p(y|x) log p(y|x)` (uses every class). In the binary
case all three give the same ordering — top, margin, and entropy are all monotone in `|p − 0.5|` — and
they only differ once there are three or more classes, where entropy down-weights edge cases where one
class is merely unlikely while top-only and margin still value them. Entropy is natural if the loss I care
about is log-loss; top-only aims most directly at decision uncertainty under 0/1 error. For the *cleanest*
rule I want the minimal, most direct form — the one that scores nothing but the quantity `self.clf`
already exposes, its top-class probability, which equals the model's own misclassification probability.
That is least confidence. Margin and entropy are refinements I'd reach for if this leaves value on the
table; for now, label what the model is least sure of, full stop.

Now wrap it in the loop, because the selection is only half of it. The classifier scoring the pool is the
one trained on the labels I have so far; after I label this round's batch the harness retrains on the
enlarged set, and *that* classifier scores the next round. A deficient classifier this round picks
examples that, once labeled, tend to compensate for its deficiency next round — the exploratory feedback
that makes it work. One caveat carries straight over from random's numbers: the score is exactly as good
as the current classifier, and early on, when very few labels have been bought, the model is fit on almost
nothing and the uncertainty score can be nearly meaningless — which is precisely the regime where random's
*representative* draw was actually a sensible default. So least confidence is a bet that the model becomes
informative fast enough that chasing its boundary beats covering the pool.

So the rung-2 edit, against the literal scaffold: where random returned a permuted slice of the unlabeled
indices, I now take the unlabeled rows, call `self.predict_prob` for their posteriors, take the top-class
probability per row, sort ascending, and return the `n` smallest — the `n` the model is least confident
about, equivalently the `n` that maximize `1 − max_y p(y|x)`. The distilled rule and the literal scaffold
fill are in the answer.

Now the falsifiable expectations, stated against random's actual numbers. The sharpest prediction is on
**letter**: that is where random was weakest (auc mean 0.724, the lucky-seed gap widest), because the
decaying-`α` problem bites hardest with 26 classes — so chasing the boundary should open the largest gap
here, and I expect least confidence to lift letter accuracy clearly above random's 0.835 (seed 42) /
0.816 (mean). If it does *not* beat random on letter, the rule is failing exactly where it should win and
something is wrong with my premise that the model's boundary is informative. On **spambase** I expect the
*smallest* gain, possibly none worth noting: random already hit 0.935 / 0.904 there because the balanced
binary boundary is easy, and there is little contested region left to redirect budget toward — least
confidence on a near-solved binary problem can even *underperform* random by over-concentrating on a thin
boundary band and starving the rest, so I would not be surprised to see spambase auc dip slightly below
random's 0.904. On **splice** I expect a modest gain over random's 0.771 / 0.751. And one structural risk
I am explicitly watching: least confidence scores each point in isolation, with no term that looks at the
other chosen points, so the `n` least-confident rows can be near-duplicates clustered in one corner of the
boundary — I'd pay for `n` labels and learn roughly one's worth. If the gains are smaller than the
boundary argument predicts, that redundant-batch effect is the likely culprit, and the next rung will have
to make the uncertainty itself *distribution-aware* — to keep what least confidence sees while refusing to
chase noise it cannot fix.
