A learned model is rarely the last word in a high-stakes pipeline: a radiologist still reads the chest X-ray, a moderator still reviews the flagged post, a clinician still signs off on the risk score. In those deployments the model does not have to answer every instance — it can hand some of them to a downstream expert who often sees side information the model never does (the patient history, the full thread, a second test). So the thing I am actually optimizing is not the classifier's error; it is the error of the *combined* model-plus-expert system, where for each input something has to decide who answers. Writing the target as $Y\in\{1,\dots,K\}$, covariates $X$, and an expert $M$ who may consult extra context $Z$, I split the predictor into a classifier $h\colon X\to Y$ and a rejector $r\colon X\to\{0,1\}$ ($r=1$ meaning defer), and the honest objective for plain misclassification on both sides is $$L_{0\text{-}1}(h,r)=\mathbb{E}\big[\,\mathbb{I}[h(x)\neq y]\,\mathbb{I}[r(x)=0]+\mathbb{I}[m\neq y]\,\mathbb{I}[r(x)=1]\,\big].$$ This is a strict generalization of the old "learning with a reject option" problem: if the expert were just a constant penalty $l_{\exp}=c$, deferring always costs $c$ and we are back to Chow's error-versus-reject tradeoff and the Cortes–DeSalvo–Mohri / Bartlett–Wegkamp line of reject surrogates. Here the reject cost is not a constant; it is $\mathbb{I}[m\neq y]$, which is zero where the expert is great and one where the expert is hopeless. That instance-dependence is the entire reason deferral is harder, and it is the fact every candidate method must contend with.

Before proposing anything I want the yardstick. Conditioning on $x$ and writing $\eta_y(x)=P(Y=y\mid x)$, if I decide to predict there is no reason to predict anything but the most likely label, so $h^B(x)=\arg\max_y\eta_y(x)$. I should defer exactly when the expert is cheaper in expectation than predicting with $h^B$: the expert's expected error is $P(M\neq Y\mid x)$ and the classifier's is $1-\max_y\eta_y(x)$, so $$r^B(x)=\mathbb{I}\big[\,\max_y\eta_y(x)\le P(Y=M\mid x)\,\big].$$ The optimal rejector compares the classifier's *confidence in its top class* against the probability the expert *agrees with the target* — a head-to-head of two specific scalars, not entropy, not the expert's raw error in isolation. The existing options miss this in instructive ways. The confidence-comparison recipe — train $h$ on the task, separately train a model of $P(\text{expert correct}\mid x)$, defer to whoever is more confident — is consistent over all measurable functions, but it fits the classifier *ignoring the expert*. Under a limited hypothesis class that is fatal: picture group A where the expert is excellent (it uses side information, or a nonlinear boundary a linear $h$ cannot represent) and group B where only the model helps. A linear $h$ trained on all the data tries to fit both and separates neither; the right system has $h$ abandon group A entirely, specialize on B, and let $r$ route A to the expert — but a classifier trained without reference to the expert can never discover that it *should* give up on A. The gain from deferral comes from the classifier *adapting*, which only happens if $h$ and $r$ are learned jointly against one combined loss. That recipe also fits two hypothesis classes and pays the statistical cost of both, painful when expert labels are scarce. The mixture-of-experts soft gate learns one objective but, as I show below, optimizes the wrong scalar and collapses to never deferring. And the classical two-function hinge surrogates (Cortes–DeSalvo–Mohri, glued from two convex upper bounds via $\max(a,b)\ge(a+b)/2$) are consistent only for $K=2$; Ni et al. proved those constructions *cannot* be made consistent for $K>2$, which is exactly why multiclass abstention fell back to confidence thresholds. I need a single convex differentiable loss, trainable in the ordinary deep-learning stack, whose minimizer is $(h^B,r^B)$ — and the hinge route is a proven dead end.

I propose **Learning to Defer** through a consistent cost-sensitive surrogate, $L_{CE}^\alpha$. The reframing that unlocks it is to stop treating $\bot$ as a special object glued onto a classification problem and simply call it *another class*: make the label space $Y\cup\{\bot\}$ with $K+1$ elements and learn $K+1$ scores $g_1,\dots,g_K,g_\bot$ under one softmax. Then predict-or-defer is nothing but cost-sensitive $(K+1)$-way classification, $\arg\min_i\mathbb{E}[c(i)\mid x]$, with costs $c(i)=\mathbb{I}[i\neq y]$ for the classes and $c(\bot)=\mathbb{I}[m\neq y]$ for deferral. (The reject-learning literature considered and rejected this move, objecting that $\bot$ "is not a class" and there is no natural distribution over the augmented label set — but I never need a generative story over labels, only a loss whose minimizer picks the cheapest action.) The natural convex surrogate weights each class's log-softmax term by how *good* that action is, the goodness of action $i$ being how much worse the most expensive action is than $i$: $$\tilde L_{CE}(g,x,c)=-\sum_{i=1}^{K+1}\big(\max_j c(j)-c(i)\big)\,\log\mathrm{softmax}_i(g)(x).$$ It is convex in $g$ (a nonnegative combination of convex $-\log\mathrm{softmax}_i$ terms). To check consistency, take $\mathbb{E}_{c\mid x}$ and differentiate the inner objective with respect to $g_y$; using the standard softmax-cross-entropy gradient $\partial(-\log p_i)/\partial g_y=p_y-\mathbb{I}[i=y]$ and writing $w_i=\mathbb{E}[\max_j c(j)-c(i)\mid x]$, the optimum is $\mathrm{softmax}_y^\ast=w_y/\sum_i w_i$. The shared denominator cancels under the $\arg\max$, leaving $\arg\max_y(-\mathbb{E}[c(y)\mid x])=\arg\min_y\mathbb{E}[c(y)\mid x]$ — exactly the lowest-expected-cost action. And with misclassification costs $c(i)=\mathbb{I}[i\neq y]$, the weight on class $i$ is $1-\mathbb{I}[i\neq y]=\mathbb{I}[i=y]$, so $\tilde L_{CE}$ collapses to ordinary cross-entropy $-\log\mathrm{softmax}_y$: a strict generalization, a strong sign I am on the right track.

Specializing to the deferral costs and taking the inner expectation $\mathbb{E}_{y\mid x}\mathbb{E}_{m\mid x,y}$, the per-example loss collapses to two interpretable terms, with $h(x)=\arg\max_{y\in Y}g_y(x)$ and $r(x)=\mathbb{I}[\max_{y\in Y}g_y(x)\le g_\bot(x)]$: $$L_{CE}(h,r,x,y,m)=-\log\mathrm{softmax}_y(x)\;-\;\mathbb{I}[m=y]\cdot\log\mathrm{softmax}_\bot(x).$$ The first term is ordinary $(K+1)$-way cross-entropy toward the true label, always on. The second fires *only when the expert is correct* ($m=y$) and then pulls softmax mass onto the deferral logit — it teaches the model to defer exactly on the inputs the expert gets right, and it needs only the single bit $\mathbb{I}[m=y]$, never the expert's actual label. Three properties make me trust it. Convexity is immediate (both terms are $-\log$ of a softmax coordinate, weighted by the nonnegative $\mathbb{I}[m=y]$). It upper-bounds $L_{0\text{-}1}$ when measured in bits: if we predict and are wrong, $y$ is not the argmax so $\mathrm{softmax}_y\le 1/2$ and $-\log_2\mathrm{softmax}_y\ge 1$; if we defer, then $g_y\le g_\bot$ forces $\mathrm{softmax}_y\le 1/2$ again, so that one term already dominates the deferral cost $\mathbb{I}[m\neq y]\le 1$ — the cross-entropy term does double duty as prediction loss and as a $\ge 1$ floor under deferral. Consistency is the crux: expanding the conditional expectation, $\sum_y\eta_y(x)q_y(x,y)=P(Y=M\mid x)$ collapses the deferral term, giving $$\mathbb{E}[L_{CE}\mid x]=-\sum_{y\in Y}\eta_y(x)\log\mathrm{softmax}_y-P(Y=M\mid x)\log\mathrm{softmax}_\bot.$$ Setting partials to zero yields $\mathrm{softmax}_i^\ast=\eta_i(x)/(1+P(Y=M\mid x))$ and $\mathrm{softmax}_\bot^\ast=P(Y=M\mid x)/(1+P(Y=M\mid x))$; the common denominator $1+P(Y=M\mid x)$ cancels in every comparison, so $\arg\max_{y}g_y^\ast=h^B$ and the model defers iff $P(Y=M\mid x)\ge\max_y\eta_y(x)$, which is precisely $r^B$. The minimizer is the combined-system Bayes solution — one convex cross-entropy-style loss, one backward pass — and this settles the open multiclass reject question that the hinge surrogates provably could not.

This same Bayes form indicts the alternatives precisely. The mixture-of-experts soft gate $-\log\mathrm{softmax}_y(g)\cdot\mathrm{softmax}(r_0)+\mathbb{I}[m\neq y]\cdot\mathrm{softmax}(r_1)$ has an optimal classifier $\mathrm{softmax}_i^\ast(g)=\eta_i(x)$ that does not depend on the gate at all (the same un-adapted classifier as the confidence recipe), and plugging $h^B$ back in, the optimal hard gate defers iff $H(h^B(x))\ge P(Y\neq M\mid x)$ — it compares the classifier's *entropy* where the Bayes rule compares its *error* $1-\max_y\eta_y(x)$. Entropy and one-minus-max-probability agree only in degenerate cases, so the gate is inconsistent; the matching pathology is that as training drives the cross-entropy term to zero while the expert-error term stays fixed, "predict" becomes uniformly cheaper and the gate collapses to never deferring. (It is still *realizable* $(H,R)$-consistent by a scaling argument, but that is the weaker notion and it buys nothing once a zero-error pair does not exist, which is essentially always.) And it is non-convex in $(g,r)$ because of the gate product. The binary case exposes the deferral cost's instance-dependence from another angle: extending the plus-hinge $L_{SH}=\exp(\tfrac{\alpha}{2}(r-hy))+(c+\mathbb{I}[m\neq y])\exp(-\beta r)$, consistency requires $\beta/\alpha=\sqrt{(1-c(x))/c(x)}$ with effective cost $c(x)=c-c\,P(M\neq Y\mid x)+P(M\neq Y\mid x)$, so a constant $\beta$ — which sufficed for constant-cost rejection — is provably inconsistent, whereas the $(K+1)$-class cross-entropy handles the varying cost per example via $\mathbb{I}[m=y]$ for free. Finally, the generalization bound for empirical minimizers, $$L_{0\text{-}1}(\hat h^\ast,\hat r^\ast)\le L_{0\text{-}1}(h^\ast,r^\ast)+\mathfrak{R}_n(H)+\mathfrak{R}_n(R)+\mathfrak{R}_{nP(M\neq Y)/2}(R)+2\sqrt{\tfrac{\log(2/\delta)}{2n}}+\tfrac{P(M\neq Y)}{2}e^{-nP(M\neq Y)/8},$$ recovers the pure rejection-learning bound when $P(M\neq Y)=0$, so deferral is strictly more sample-intensive than rejection — the extra rejector-complexity term scales with the expert's error rate, because $r$ can only learn where to defer from the examples where the expert actually errs. That argues for folding $h$ and $r$ into one shared $(K+1)$-output backbone rather than paying $\mathfrak{R}_n(H)+\mathfrak{R}_n(R)$ as two separate trainings.

One last knob the analysis demands. $L_{CE}$ is consistent, but its first term keeps pulling the classifier to fit $y$ *everywhere*, including on inputs the expert already handles, wasting capacity. So I down-weight the target term where the expert is correct: $$L_{CE}^\alpha(h,r,x,y,m)=-\big(\alpha\,\mathbb{I}[m=y]+\mathbb{I}[m\neq y]\big)\log\mathrm{softmax}_y\;-\;\mathbb{I}[m=y]\log\mathrm{softmax}_\bot.$$ Where the expert is wrong the weight is $1$ — fit the target hard, only the model can help there. Where the expert is right the weight is $\alpha$: at $\alpha=1$ this is exactly $L_{CE}$ (still consistent), while $\alpha<1$ tells the model not to bother nailing $y$ there and frees capacity for the hard region. $\alpha\neq 1$ is a deliberate capacity/adaptivity trade rather than a consistent loss, so I validate it on a small grid (say $[0,10]$), keeping $\alpha=1$ when consistency is what I want. In code this lands as a tiny change to the ordinary stack: give the backbone $K+1$ output units (the first $K$ are the class logits, the last is $g_\bot$), train with $L_{CE}^\alpha$ using only the target and the bit $\mathbb{I}[m=y]$, and at inference defer iff the deferral unit wins the $(K+1)$-way argmax. For a coverage target nothing in training changes — rank test inputs by the defer margin $q(x)=g_\bot(x)-\max_{y\in Y}g_y(x)$ and threshold at the desired quantile, giving the whole coverage range off one trained model.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeferralNet(nn.Module):
    """Backbone with K+1 outputs: 0..K-1 are class logits (classifier h), K is the
    deferral logit g_bot (rejector r). One shared network does both jobs."""

    def __init__(self, make_features, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.features = make_features()
        self.head = nn.Linear(self.features.out_dim, num_classes + 1)   # +1 deferral unit

    def forward(self, x):
        return self.head(self.features(x))                              # raw (K+1) logits


def deferral_loss(outputs, target, expert_label, num_classes, alpha):
    """L_CE^alpha: build the canonical weights and call reject_CrossEntropyLoss.
       outputs:      (B, K+1) raw logits
       target:       (B,) labels y
       expert_label: (B,) expert decisions m
       alpha:        adaptivity knob (alpha=1 -> consistent L_CE)
    """
    probs = F.softmax(outputs, dim=1)                                  # softmax over Y union {defer}
    expert_correct = (expert_label == target)                         # I[m = y]  (gating bit)
    m  = expert_correct.float()                                        # weight on the deferral term
    m2 = torch.where(expert_correct,                                   # alpha*I[m=y] + I[m!=y]
                     torch.full_like(m, alpha), torch.ones_like(m))
    return reject_CrossEntropyLoss(probs, m, target, m2, num_classes)


def reject_CrossEntropyLoss(outputs, m, labels, m2, n_classes):
    """Canonical batch loss on softmax probabilities.
       m:  I[expert prediction equals target], weight on the deferral term
       m2: alpha*I[expert prediction equals target] + I[expert prediction differs]
    """
    batch_size = outputs.size(0)
    rc = [n_classes] * batch_size                                      # deferral index K
    eps = 1e-12
    loss = -m * torch.log2(outputs[range(batch_size), rc].clamp_min(eps)) \
           -m2 * torch.log2(outputs[range(batch_size), labels].clamp_min(eps))
    return torch.sum(loss) / batch_size


@torch.no_grad()
def predict_or_defer(outputs, num_classes):
    """Defer iff the deferral unit wins the (K+1)-way argmax; else predict argmax over the K
    class logits. Equivalent to r(x) = I[g_bot(x) >= max_y g_y(x)]."""
    top = outputs.argmax(dim=1)
    defer = top == num_classes
    class_pred = outputs[:, :num_classes].argmax(dim=1)
    return defer, class_pred


def coverage_threshold(outputs, num_classes, coverage):
    """For a coverage target: rank by the defer margin q(x) = g_bot - max_y g_y and threshold
    at the coverage quantile (single trained model gives the whole coverage range)."""
    q = outputs[:, num_classes] - outputs[:, :num_classes].max(dim=1).values
    tau = torch.quantile(q, coverage)             # top (1-coverage) by defer margin get deferred
    return q >= tau                                                    # True = defer


def train(model, data_loader, optimizer, expert_fn, num_classes, alpha):
    model.train()
    for x, target in data_loader:                                     # draw a minibatch
        outputs = model(x)                                            # (K+1) logits
        expert_label = expert_fn(x, target)                          # observed expert decisions
        loss = deferral_loss(outputs, target, expert_label, num_classes, alpha)
        optimizer.zero_grad()
        loss.backward()                                              # backprop through shared backbone
        optimizer.step()
```
