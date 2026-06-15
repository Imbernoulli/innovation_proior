Let me start from what actually happens when this model gets deployed, because the way I keep thinking about it — "make the classifier as accurate as possible" — is not the situation I'm in. The model sits in front of a radiologist, or a human moderator, or a clinician. It does not have to answer every case. For some inputs it can just say "I won't predict this one — send it to the expert." And the expert is often genuinely better on a chunk of inputs, because they see things I don't: the patient's history, the rest of the thread, a follow-up test. So the real object I'm optimizing is not the classifier's error; it's the error of the *combined* system, the model-plus-expert, where for each input something has to decide who answers. That decision is the thing I actually have to learn.

So let me write down the loss honestly. I have a target Y in {1,...,K}, covariates X, and an expert M who may look at extra context Z I never get to see. I want a predictor Ŷ: X → Y ∪ {⊥}, where ⊥ means "defer." If I don't defer, my classifier h pays its classification cost; if I defer, the expert pays whatever the expert's decision costs. The cleanest way to express it is to split Ŷ into two functions — a classifier h: X → Y and a rejector r: X → {0,1}, r=1 meaning defer — and then

  L(h,r) = E_{(x,y), m∼M|(x,y)} [ l(x,y,h(x))·I[r(x)=0] + l_exp(x,y,m)·I[r(x)=1] ].

I'll keep general costs around because they're free, but the case I really care about is plain misclassification on both sides:

  L_{0-1}(h,r) = E[ I[h(x)≠y]·I[r(x)=0] + I[m≠y]·I[r(x)=1] ].

Two things jump out immediately. First, this is a strict generalization of the older "learning with a reject option" problem: if the expert were just a constant penalty, l_exp = c for every input, then deferring always costs c and the rejector is choosing whether to pay c rather than risk a wrong prediction — that's exactly Chow's old error-versus-reject tradeoff, and the whole Cortes–DeSalvo–Mohri / Bartlett–Wegkamp line of reject surrogates. Here the "reject cost" is not a constant; it's `I[m≠y]`, which varies wildly across the input space — zero where the expert is great, one where the expert is hopeless. That instance-dependence is the entire reason this is harder, and I should keep it in view.

Second, and this is the wall I want to hit early so I understand why the obvious approaches are wrong: the natural thing everyone does is *train the classifier first*. Fit h to the target by minimizing classification error, get some confidence out of it, then bolt a deferral rule on top. Let me see what's optimal so I have a yardstick, then check whether that recipe can reach it.

What is the Bayes solution of L_{0-1}? Write η_y(x) = P(Y=y|X=x). Condition on x. If I've decided to predict (r=0), then my contribution is just the ordinary misclassification loss, and there's no reason to predict anything but the most likely label — the rejector's decision doesn't change what the best prediction is. So pointwise,

  h^B(x) = argmax_y η_y(x),

the standard classification Bayes rule. Now the rejector. I should defer exactly when handing it to the expert is cheaper in expectation than predicting with h^B. The expert's expected error is E[I[M≠Y]|x] = P(M≠Y|x). The classifier's expected error if it predicts is E[I[h^B(x)≠Y]|x] = 1 − max_y η_y(x). So defer iff

  P(M≠Y|x) ≤ 1 − max_y η_y(x)  ⇔  P(Y=M|x) ≥ max_y η_y(x),

i.e.

  r^B(x) = I[ max_y η_y(x) ≤ P(Y=M|X=x) ].

Stare at this for a second, because it's the lever for everything that follows. The optimal rejector compares the classifier's *confidence in its top class*, max_y η_y(x), against the probability that the expert *agrees with the target*, P(Y=M|x). Confidence on one side, expert-agreement on the other. Not entropy, not the expert's raw error in isolation — a head-to-head of two specific scalars. I'm going to come back and beat any candidate method against this exact form.

Now, back to the wall. Does "train h first, then decide" reach (h^B, r^B)? Over *all measurable functions* with infinite data — yes, trivially, because h^B is just the target's Bayes classifier and you can estimate P(Y=M|x) with a second model and compare. That's the confidence-comparison recipe (learn the task classifier, separately learn a model of whether the expert is right, defer to whoever's more confident), and it is consistent in that idealized sense. But I never have all measurable functions or infinite data. I have a *limited* hypothesis class and finite samples. And in that regime the recipe breaks in a way that's worth feeling concretely.

Picture two sub-populations. On group A the expert is excellent — say the expert uses side information, or computes a nonlinear boundary, that my linear classifier simply cannot represent. On group B only my model can help. Now I train a linear h on *all* the data to minimize target error. It tries to fit both groups at once; since group A isn't linearly separable for me, the best-on-average hyperplane separates neither group cleanly. The *right* system is obvious in hindsight: h should completely give up on group A, fit group B as well as it can, and r should route all of group A to the expert. But a classifier trained without any reference to the expert has no way to discover that it *ought to abandon* group A — nothing in "minimize target error over everything" tells it which region to sacrifice. So the gain from deferral isn't an afterthought you add to a finished classifier; it comes from letting the classifier *adapt* — specialize away from the expert's strong region — and that only happens if the classifier and the rejector are learned *together*, against the combined loss, not separately. That's the thing I have to engineer: joint learning, with one objective.

There's an extra cost to the two-model recipe I should note while I'm here: it fits two separate hypothesis classes (the task model and the expert model), so it pays the statistical price of both. When expert-labeled data is scarce — and it usually is, the expert is the expensive resource — that doubled complexity hurts. I'd prefer one model that does both jobs.

OK so I want a single convex differentiable loss, learnable with the ordinary deep-learning stack, whose minimizer is (h^B, r^B). The combined L_{0-1} is non-convex and combinatorial — products of indicators of h and r — so I need a surrogate. The reject-learning people already tried two-function surrogates. Cortes, DeSalvo and Mohri built convex upper bounds for the *binary* reject loss by taking two convex functions φ, ψ that upper-bound the step function and gluing them with the elementary inequality max(a,b) ≥ (a+b)/2, getting a max-hinge and a plus-hinge loss in (h, r). Clean, and it gave them generalization bounds and consistency — *for K=2*. The trouble is multiclass. Ni and collaborators tried to lift exactly those surrogates to K > 2 and proved it can't be done: those two-function constructions are provably *not* consistent for more than two classes, which is why everyone fell back to confidence thresholds in multiclass. So extending the hinge route is a dead end I shouldn't waste time on. I need a genuinely different surrogate.

Here's the reframing that gets me unstuck. I've been treating ⊥ as a special object glued onto a classification problem. What if I stop treating it as special and just call it *another class*? Make the label space Y ∪ {⊥} with K+1 elements, where class K+1 means "defer." Then the predict-or-defer decision is nothing but a (K+1)-way classification, and "which action minimizes expected cost" is just cost-sensitive classification over K+1 actions: assign costs c(1),...,c(K) for predicting each label and c(K+1) for deferring, and pick the action of least expected cost, argmin_i E[c(i)|x]. For my deferral problem the costs are c(i) = l(x,y,i) for i ∈ [K] and c(K+1) = l_exp(x,y,m).

I should pause, because the rejection-learning people *considered this exact move and rejected it*. Their objection: ⊥ "is not a class," there's no natural distribution over the augmented set, so the standard cost-sensitive machinery doesn't obviously apply. That objection is about wanting a generative story over the labels. But I don't need one — I only need a *loss whose minimizer picks the lowest-cost action*. So let me just try to build that loss directly and see if it's consistent, ignoring the philosophical worry about whether ⊥ "deserves" to be a class.

What's the most natural convex surrogate for cost-sensitive classification that I could actually train? The thing I know is plain cross-entropy: parameterize K+1 scores g_i(x), softmax them, and for an ordinary classification target y minimize −log(softmax_y). Cross-entropy is consistent for 0-1 classification: its minimizer's argmax is argmax_y η_y. I want the cost-sensitive analogue — something that reduces to cross-entropy when the costs are misclassification costs, but in general drives the argmax to the *cheapest* action. Cross-entropy puts all the weight on the correct class. Cost-sensitively, the "correct" thing isn't one class; it's "prefer cheaper actions." So weight each class's log-softmax term by how *good* that action is. The natural goodness of action i is "how much worse the worst action is than i," i.e. (max_j c(j) − c(i)) — large when i is cheap, zero for the most expensive action. So I'll try

  L̃_CE(g, x, c) = − Σ_{i=1}^{K+1} (max_j c(j) − c(i)) · log( exp(g_i(x)) / Σ_k exp(g_k(x)) ).

It's convex in g — a nonnegative combination of the convex terms −log softmax_i. Now the real test: is it consistent? Take the conditional expectation over the random costs, E_{c|x}, and minimize over the scores at that x. The inner objective is

  Σ_i E[max_j c(j) − c(i) | x] · ( −log softmax_i(g) ),

a convex function of g. Differentiate with respect to g_y and set to zero. Let p_i = softmax_i. Using ∂(−log p_i)/∂g_y = p_y − I[i=y] (the standard softmax-cross-entropy gradient), the derivative of the sum is Σ_i w_i (p_y − I[i=y]) where w_i = E[max_j c(j) − c(i)|x]. That's p_y·(Σ_i w_i) − w_y. Setting it to zero:

  softmax_y* = w_y / Σ_i w_i = E[max_j c(j) − c(y)|x] / Σ_i E[max_j c(j) − c(i)|x].

Now look at the argmax over y of the optimal scores. The denominator is a constant in y, so argmax_y softmax_y* = argmax_y E[max_j c(j) − c(y)|x] = argmax_y ( E[max_j c(j)|x] − E[c(y)|x] ). The first piece doesn't depend on y, so this is argmax_y ( −E[c(y)|x] ) = argmin_y E[c(y)|x]. The minimizer's predicted action is the lowest-expected-cost action. That's exactly cost-sensitive consistency. The surrogate works — and the rejection-learning worry that "⊥ isn't a class" was a non-issue, because I never needed a distribution over labels, only a per-action weight.

Sanity check that this really generalizes cross-entropy. Take misclassification costs c(i) = I[i ≠ y]. Then max_j c(j) = 1, and the weight on class i is (1 − I[i≠y]) = I[i=y], which is 1 for the true label and 0 elsewhere. So L̃_CE collapses to −log softmax_y — ordinary cross-entropy. Good, it's a strict generalization, which is a strong hint I'm on the right track rather than off in a corner.

Now specialize to my actual deferral problem under L_{0-1} and watch the costs collapse into something I can implement from a single label and a single expert decision. I have K+1 scores: g_1,...,g_K for the classes and g_⊥ for deferral. Define the classifier h(x) = argmax_{y∈Y} g_y(x) and the rejector r(x) = I[ max_{y∈Y} g_y(x) ≤ g_⊥(x) ] — defer when the deferral logit wins. Plug the deferral costs into the cost-sensitive surrogate and take the inner expectation E_{y|x}E_{m|x,y}. The classifier-side costs, being misclassification costs, give back the cross-entropy term toward the true label; the deferral-side cost, c(⊥) = I[m≠y], contributes a term proportional to the *expert agreement*. Let me just carry the per-example loss that the expectation is built from:

  L_CE(h,r,x,y,m) = − log( exp(g_y(x)) / Σ_{y'∈Y∪⊥} exp(g_{y'}(x)) ) − I[m=y] · log( exp(g_⊥(x)) / Σ_{y'∈Y∪⊥} exp(g_{y'}(x)) ).

Two terms, and both are interpretable. The first is ordinary cross-entropy of the (K+1)-way softmax toward the true label y — always on, pulling mass onto the right class. The second is *only active when the expert is correct* (m=y), and when it fires it pulls softmax mass onto the ⊥ logit — i.e. it teaches the model to defer exactly on the inputs where the expert agrees with the truth. That's a beautifully direct encoding of "learn to hand over the cases the expert gets right," and it never needs the expert's *label* — only the bit `expert-correct-or-not`.

I need to confirm three properties before I trust it: convex, an upper bound on L_{0-1}, and consistent (minimizer = (h^B, r^B)).

Convexity is immediate: both terms are −log of a softmax coordinate, which is convex in g, and I[m=y] ≥ 0 is just a nonnegative weight.

Upper bound. I want L_CE ≥ L_{0-1} pointwise, so I measure the cross-entropy in bits, matching the implementation's log2; with natural logs the same argument is multiplied by 1/log 2 and the minimizers do not change. Two cases on r. If r(x)=0 (predict) and the classifier is wrong, I[h(x)≠y]=1; but h(x)≠y means class y is not the argmax of the K+1 softmax, so softmax_y ≤ 1/2, hence −log2 softmax_y ≥ 1, and all the other terms on the right are ≥ 0. If r(x)=1 (defer), then by definition max_{y'∈Y} g_{y'} ≤ g_⊥, so the true-class logit g_y is below g_⊥, which means softmax_y ≤ 1/2 again, so −log2 softmax_y ≥ 1, and that single term already dominates the deferral cost I[m≠y] ≤ 1. Since L_{0-1} ≤ 1 in every case, L_CE ≥ L_{0-1} throughout. The cross-entropy-toward-y term is doing double duty: it's both the prediction loss and, when I defer, an automatic ≥1 floor that lets it upper-bound the deferral indicator.

Consistency is the one that matters, and I want to do it carefully, because this is where I confirm I've recovered the exact Bayes rule I derived at the start. Minimize over functions = minimize the conditional inner expectation at each x. Expand E_{y|x}E_{m|x,y}[L_CE]. The first term gives −Σ_y η_y(x) log softmax_y. The second term is −Σ_y η_y(x) Σ_m q_m(x,y) I[m=y] log softmax_⊥, where q_m(x,y)=P(M=m|x,y). The double sum collapses: Σ_y η_y(x) q_y(x,y) = Σ_y P(M=y, Y=y|x) = P(Y=M|x). So

  E[L_CE|x] = −Σ_{y∈Y} η_y(x) log softmax_y − P(Y=M|x) · log softmax_⊥.

This is convex in g; set partials to zero. For each i ∈ Y, the same softmax-cross-entropy gradient gives, at the optimum,

  softmax_i* = η_i(x) / (1 + P(Y=M|x)),

because the total "mass" of the targets is Σ_y η_y + P(Y=M|x) = 1 + P(Y=M|x). And differentiating with respect to g_⊥,

  softmax_⊥* = P(Y=M|x) / (1 + P(Y=M|x)).

The common denominator 1 + P(Y=M|x) is shared, so it cancels in every comparison. Therefore argmax_{y∈Y} g_y* = argmax_y η_y(x) = h^B(x) — the classifier head recovers the target's Bayes rule. And the rejector defers, r*(x)=1, exactly when softmax_⊥* ≥ max_{y∈Y} softmax_y*, i.e. when P(Y=M|x) ≥ max_y η_y(x). That is precisely r^B(x). The minimizer is (h^B, r^B). The surrogate is consistent. I've closed the loop: a single convex cross-entropy-style loss, trainable in one backward pass, whose optimum is the combined-system Bayes solution — and it settles the multiclass reject question that the hinge surrogates couldn't.

Now I want to come back and *prove*, not just assert, that the two earlier approaches are the wrong tool, because that's what justifies all this machinery instead of just using a confidence threshold. The mixture-of-experts route first. The deferral problem really does look like a hard mixture of two answerers, one of them fixed (the expert), so the tempting move is a soft-gate loss in the style of mixtures of experts: scores g_y for the classes giving the classifier, scores r_0, r_1 giving a soft gate r(x)=argmax_i r_i, and

  L_mix = −log softmax_y(g) · softmax(r_0) + I[m≠y] · softmax(r_1),

the gate softly mixing the classifier's cross-entropy and the expert's error. Is it consistent? Take the inner expectation. Differentiating with respect to g_i, the gate weight softmax(r_0) is a positive constant multiplying the classifier cross-entropy, so the optimal classifier satisfies softmax_i*(g) = η_i(x) — note the classifier's optimum *does not depend on r_0, r_1 at all*, which is already suspicious: this is the same independent classifier as the confidence recipe, fit to the target with no adaptation to the expert. Now plug that h^B back in and minimize over the gate. The classifier cross-entropy at its own optimum is the Shannon entropy H(h^B(x)) of the optimal posterior, so the gate is choosing between

  H(h^B(x)) · softmax(r_0)  +  P(Y≠M|x) · softmax(r_1),

and the optimal hard gate defers (r_1 wins) iff H(h^B(x)) ≥ P(Y≠M|x). Compare that to the Bayes rejector, which defers iff max_y η_y(x) ≤ P(Y=M|x), i.e. iff (1 − max_y η_y(x)) ≥ P(Y≠M|x). The soft-gate loss compares the classifier's *entropy* H(h^B(x)) to the expert's error, where the Bayes rule compares the classifier's *error* 1 − max_y η_y(x). Entropy and one-minus-max-probability are different functions of the posterior — they agree only in degenerate cases — so the mixture-of-experts minimizer is *not* the Bayes rejector. It's inconsistent, and now I know exactly why: it's measuring classifier uncertainty with the wrong scalar. (There's a matching empirical pathology that the analysis predicts: as the classifier trains, its cross-entropy term goes to zero on the training data while the expert-error term stays fixed, so "predict" becomes the uniformly cheaper branch and the gate collapses to never deferring. The inconsistency isn't academic; it makes the thing refuse to defer.)

Here's a subtlety worth pinning down, though, because it complicates the "consistency is everything" story. The mixture loss is still *realizable* (H,R)-consistent — meaning if there genuinely exists a pair in the hypothesis class with zero system error, the surrogate will find it. The argument is a scaling one: assume classes closed under scaling, take the zero-error pair (h*, r*), scale it by u and send u → ∞. On a point where r*=1 (defer) and the system has zero error, the expert must be right there, so I[m≠y]=0, and the hardened gate softmax(ur_1) → 1 kills that example's loss; on a point where r*=0 (predict), h* is correct by realizability so its scaled cross-entropy → 0 and softmax(ur_0) → 1 likewise. By monotone convergence the whole expected mixture loss → 0, so any near-minimizer has near-zero system loss. So realizable consistency holds even though classification consistency fails. Which of the two notions matters more isn't obvious a priori — but the entropy-vs-confidence mismatch above means the mixture loss learns the wrong deferral behavior whenever the realizable assumption doesn't hold, which is essentially always with real models. That tips me firmly toward demanding the stronger, classification consistency — which my L_CE has and the mixture loss doesn't. And as a bonus my loss is convex in g; the mixture loss is non-convex in (g, r) because of the gate product, so it's not even clean to optimize.

I also want the binary version worked out, partly to connect to the older reject-learning surrogates and partly because it exposes *why* the deferral cost being instance-dependent is the crux. In the binary case I can try to extend the Cortes max-hinge/plus-hinge directly. Let Y={−1,+1}, h, r: X→R, defer when r(x) ≤ 0, and take a slightly general expert cost l_exp = max(c, I[m≠y]) so pure rejection (constant c) is a special case. The Cortes construction upper-bounds the combined indicator loss by introducing two convex functions φ, ψ ≥ the step, using I[max{a,b}≤0] ≤ I[(a+b)/2 ≤ 0] (since max(a,b) ≥ (a+b)/2) and then φ, ψ on top, ending with the additive plus-hinge

  L_SH = exp( (α/2)(r(x) − h(x)y) ) + (c + I[m≠y]) · exp( −β·r(x) ),

with φ, ψ taken as exponentials. Now the crucial difference from constant-cost rejection: I'll show β can't be a constant. Take the inner expectation at x with η=P(Y=1|x) and q(x,y)=P(M=1|x,y). For η∈(0,1) the inner loss is convex in (u,v)=(h(x),r(x)); set ∂/∂u=0 first. The h-dependent terms are η·exp((α/2)(v−u)) and (1−η)·exp((α/2)(v+u)); their derivative in u vanishes at

  u* = (1/α) log( η/(1−η) ),

which has the sign of η − 1/2, the sign of the Bayes classifier h^B. Good — h matches. Now plug u* in and set ∂/∂v=0. Collecting the four mixture terms, the expert-side coefficient assembles into c(x) := c − c·P(M≠Y|x) + P(M≠Y|x) — an *instance-dependent* effective cost, because the expert's error rate varies with x. Carrying through (it's the same algebra as the Cortes–DeSalvo boosting analysis), the optimal v* has the sign of the Bayes rejector r^B if and only if

  β/α = sqrt( (1 − c(x)) / c(x) ).

For the normalized choice α=1 this is β = sqrt((1 − c(x))/c(x)). In the old constant-cost rejection problem c(x)≡c is constant, so a constant β worked. Here c(x) genuinely depends on x through P(M≠Y|x), so consistency *forces β to vary across the input* — a constant β is provably inconsistent. That's the binary fingerprint of the same fact that made the multiclass problem hard: the cost of deferring is not a number, it's a function of where you are. (And it's exactly the kind of awkwardness the K+1-class cross-entropy route sidesteps, since the cost-weighting there is handled per example by I[m=y] rather than by tuning a global β.)

Now the finite-sample side, because consistency is an infinite-data statement and I claimed this is *more* data-hungry than plain rejection. Go after the system loss directly: minimize the empirical L^S_{0-1}(h,r) = (1/n) Σ_i I[h(x_i)≠y_i] I[r(x_i)=0] + I[m_i≠y_i] I[r(x_i)=1] over h∈H, r∈R, and bound the gap to the population minimizer (h*, r*). The loss family L_{H,R} takes values in [0,1], so the standard Rademacher bound gives, w.p. ≥ 1−δ/2,

  L_{0-1}(ĥ*, r̂*) ≤ L^S_{0-1}(ĥ*, r̂*) + 2 R_n(L_{H,R}) + sqrt( log(2/δ)/(2n) ).

I need to relate R_n(L_{H,R}) to the individual classes and remember that the bound will multiply it by 2. The supremum of a sum is at most the sum of suprema, so the joint complexity splits into a classifier-indicator piece and an expert-indicator piece. For the classifier piece I[h(x)≠y]·I[r(x)=0] — a product of two indicators from two classes — there's a lemma (DeSalvo et al.) that the Rademacher complexity of a product of indicators from H and R is at most the sum of the complexities of the two classes; together with the standard fact that the complexity of indicators based on a class is half the complexity of the class, the final bound gets R_n(H) + R_n(R) from this part after that leading factor 2 is applied. The expert piece is where the extra cost shows up. The term Σ_i ε_i I[m_i≠y_i] I[r(x_i)=1] is supported *only* on the examples where the expert is wrong; let n_m^S = Σ_i I[m_i≠y_i] be that count. Conditioning on which examples those are, the inner sup is an empirical Rademacher complexity of R over n_m^S points, so the contribution has the form E[ (n_m^S/n) · R̂_{S_m}(R) ]. Now n_m^S ~ Binomial(n, P(M≠Y)) with mean nP(M≠Y). Split on whether n_m^S falls below half its mean: a Chernoff bound puts P(n_m^S < nP(M≠Y)/2) ≤ exp(−nP(M≠Y)/8), and on that low-count event the prefactor n_m^S/n is at most P(M≠Y)/2; otherwise n_m^S ≥ nP(M≠Y)/2 and the non-increasing sample-size behavior of Rademacher complexity gives R_{nP(M≠Y)/2}(R). After the indicator-class normalization and the leading factor in the uniform-convergence bound are accounted for, the expert-error contribution in the final inequality is

  (P(M≠Y)/2)·exp(−nP(M≠Y)/8) + R_{nP(M≠Y)/2}(R).

Collecting everything (and Hoeffding for L^S(h*,r*) ≤ L(h*,r*) + sqrt(log(2/δ)/(2n))),

  L_{0-1}(ĥ*, r̂*) ≤ L_{0-1}(h*, r*) + R_n(H) + R_n(R) + R_{nP(M≠Y)/2}(R)
                    + 2·sqrt( log(2/δ)/(2n) ) + (P(M≠Y)/2)·exp(−nP(M≠Y)/8).

Read it: the gap is controlled by the complexity of *both* the classifier and the rejector classes, plus a term that is the rejector's complexity evaluated at the effective sample size nP(M≠Y)/2 — the number of expert-mistake examples — plus an exponentially small slack. When the expert is perfect, P(M≠Y)=0, the extra rejector term and the exponential vanish and I recover the pure rejection-learning bound (Cortes' Theorem 1 with c=0). So deferral is *strictly* more sample-intensive than rejection: I pay an additional rejector-complexity term that scales with how often the expert errs, because the rejector can only learn where to defer from the examples where deferring actually matters. This is also the argument for not using two separate models: if h and r come from two independent networks I pay R_n(H)+R_n(R) in full as two separate trainings, whereas folding both into one shared backbone with a (K+1)-th output head shares representation and keeps the cost down — which is the design choice I'll make in code.

So let me make that design concrete. I take whatever backbone I'd normally use for the classifier and give it K+1 output units: the first K are the class logits g_1,...,g_K (the classifier h = argmax over those), and the (K+1)-th unit is g_⊥, the deferral logit. One softmax over all K+1. The rejector at inference is just "did the deferral unit win?" — argmax over the K+1 outputs equals the deferral index ⇔ defer; otherwise predict the argmax over the first K. No second network, no separate gate. Training is the L_CE loss, which needs only the target y and the single bit I[m=y].

But before I write it, one more knob the consistency analysis suggests and the deployment reality demands. L_CE as derived is consistent, which is great asymptotically, but recall the *adaptivity* point from the very beginning: under limited capacity I want the classifier to *give up* on the expert's strong region and spend its capacity where it's actually needed. The first term of L_CE keeps pulling the classifier toward fitting y *everywhere*, including on examples the expert already handles — wasting capacity. So introduce a weight α ≥ 0 on the target-fitting term that down-weights it exactly where the expert is correct:

  L_CE^α(h,r,x,y,m) = −( α·I[m=y] + I[m≠y] ) · log softmax_y − I[m=y] · log softmax_⊥.

When the expert is wrong (m≠y) the weight is 1 — fit the target hard, because here only the model can help. When the expert is right (m=y) the weight is α: with α=1 I get exactly L_CE (still consistent); with α<1 I tell the model "don't bother nailing y here, you can just defer," freeing capacity for the hard region. α≠1 is no longer consistent — it's a deliberate capacity/adaptivity trade — so I treat α as a hyperparameter to validate (a grid over, say, [0,10] is plenty), keeping α=1 available when consistency is what I want and pushing α down when adaptivity pays. That recovers the whole spectrum from "purely defer where the expert is right" to "always try to predict."

Let me write the loss exactly the way it goes into a training loop. For a batch I need, per example, the model's K+1 softmax probabilities, the target index, and the two scalar weights: `m = I[expert correct]` (1 if the expert's prediction equals the target, else 0) gating the deferral term, and `m2 = α·I[m=y] + I[m≠y]` weighting the target term. The deferral class is the last index, K. Then the per-example loss is `−m·log2(p[K]) − m2·log2(p[y])`, summed and averaged over the batch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeferralNet(nn.Module):
    """Backbone with K+1 outputs: units 0..K-1 are the class logits (classifier h),
    unit K is the deferral logit g_bot (the rejector). One shared network does both jobs."""

    def __init__(self, make_features, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.features = make_features()
        self.head = nn.Linear(self.features.out_dim, num_classes + 1)  # +1 deferral unit

    def forward(self, x):
        return self.head(self.features(x))                  # raw (K+1) logits


def deferral_loss(outputs, target, expert_label, num_classes, alpha):
    """Build the canonical weights m and m2, then apply reject_CrossEntropyLoss."""
    probs = F.softmax(outputs, dim=1)                       # softmax over Y union {defer}
    expert_correct = (expert_label == target)
    m = expert_correct.float()                              # I[m = y]
    m2 = torch.where(expert_correct, torch.full_like(m, alpha), torch.ones_like(m))
    return reject_CrossEntropyLoss(probs, m, target, m2, num_classes)


def reject_CrossEntropyLoss(outputs, m, labels, m2, n_classes):
    """Canonical batch loss on softmax probabilities.
       m  = I[expert prediction equals target]
       m2 = alpha*I[expert prediction equals target] + I[expert prediction differs]."""
    batch_size = outputs.size(0)
    rc = [n_classes] * batch_size                           # deferral index K
    eps = 1e-12
    loss = -m * torch.log2(outputs[range(batch_size), rc].clamp_min(eps)) \
           -m2 * torch.log2(outputs[range(batch_size), labels].clamp_min(eps))
    return torch.sum(loss) / batch_size


@torch.no_grad()
def predict_or_defer(outputs, num_classes):
    """Inference rule: defer iff the deferral unit wins the (K+1)-way argmax; otherwise
    predict argmax over the K class logits. (Equivalently r(x)=I[g_bot >= max_y g_y].)"""
    top = outputs.argmax(dim=1)
    defer = top == num_classes
    class_pred = outputs[:, :num_classes].argmax(dim=1)     # best label if we predict
    return defer, class_pred


def train(model, data_loader, optimizer, expert_fn, num_classes, alpha):
    model.train()
    for x, target in data_loader:                           # draw a minibatch
        outputs = model(x)                                  # forward: (K+1) logits
        expert_label = expert_fn(x, target)                # observed expert decisions m_i
        loss = deferral_loss(outputs, target, expert_label, num_classes, alpha)
        optimizer.zero_grad()
        loss.backward()                                     # backprop through the shared backbone
        optimizer.step()
```

For a coverage target instead of a free reject decision, I don't change training at all — I just threshold the deferral margin. Sort the test inputs by q(x) = g_⊥(x) − max_{y∈Y} g_y(x) (how strongly the model wants to defer), pick the threshold τ at the desired coverage quantile, and defer when q(x) ≥ τ. That gives any operating point from full automation to full deferral off a single trained model.

Let me trace the whole chain back so I'm sure it holds together. I started not with "make the classifier accurate" but with the combined model-plus-expert system, and saw that the gain from deferral comes from the *mismatch* between where the model is weak and the expert is strong — which means the classifier has to be allowed to adapt, to give up on the expert's region, which only happens if classifier and rejector are learned jointly against one combined loss. I wrote that combined loss and its Bayes solution: predict the most likely label, defer exactly when the classifier's top-class confidence falls below the expert's agreement probability. The classical reject surrogates (hinge-based, two-function) couldn't deliver a consistent multiclass version of this — proven impossible — so I reframed deferral as an ordinary (K+1)-class problem with ⊥ as just another class, brushing aside the old objection that "⊥ isn't a class," and built a cost-sensitive cross-entropy whose minimizer picks the cheapest action and which collapses to plain cross-entropy on misclassification costs. Specializing it to the deferral costs gave a two-term loss — cross-entropy toward the label, plus a defer-when-the-expert-is-right term gated by the single bit I[m=y] — that I verified is convex, upper-bounds the system 0-1 loss, and is consistent, recovering exactly the confidence-versus-agreement Bayes rejector. Checking the alternatives against that same Bayes rule showed the mixture-of-experts loss compares classifier *entropy* instead of *confidence* and so is inconsistent (and collapses to never deferring), while the confidence-comparison recipe trains the classifier without adaptation and pays for two models; the binary case revealed the deferral cost is intrinsically instance-dependent (forcing a non-constant β in the old surrogate), which the K+1 cross-entropy handles per example for free. The generalization bound showed deferral is strictly more sample-intensive than rejection — paying an extra rejector-complexity term scaled by the expert's error rate — which argues for one shared backbone over two networks. And the whole thing lands as a tiny change to the ordinary training stack: give the network one extra output unit for ⊥, swap cross-entropy for the two-term L_CE^α loss, and read off "defer iff the deferral unit wins."
