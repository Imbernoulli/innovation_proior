A network of fixed capacity sees contexts one after another — task 1, then task 2, then task 3 — and after each it must keep performing the earlier ones. Plain gradient descent on the new context overwrites exactly the weights that mattered to the old ones: catastrophic forgetting, the defining obstacle of continual learning. The standard cure is elastic weight consolidation: when a context $t$ finishes, freeze a snapshot $\theta^*_t$ of where the weights landed, measure how much each weight mattered with the diagonal Fisher information $F_t$, and while learning later contexts add a quadratic spring pulling each weight back toward $\theta^*_t$ with stiffness $F_{t,i}$, so the total loss is $L_{\text{new}}(\theta) + \sum_t \frac{\lambda}{2}\sum_i F_{t,i}(\theta_i-\theta^*_{t,i})^2$. Each term is stiff exactly where its context was sensitive and slack elsewhere, and it genuinely stops forgetting. But the sum has one term per past context, and each term carries a full parameter-sized anchor and a full parameter-sized Fisher. After a hundred contexts that is a hundred anchors and a hundred Fishers stored, and a hundred springs swept on every gradient step: both memory and per-step compute grow *linearly* in the number of contexts. For a genuine lifelong agent that linear growth is the thing that kills it — the bookkeeping outgrows the model. Worse, on a fixed parameter budget the growing stack of undamped springs eventually over-constrains the network so much that new contexts can no longer be learned at all. Synaptic Intelligence already keeps a single accumulated importance, but it estimates that importance as a path integral over the optimiser's trajectory rather than a local curvature, losing EWC's clean Bayesian reading, and its importance only ever grows, so old constraints still dominate. The real question is sharper than "how do I not forget": can I keep EWC's protection but pay a cost that is *constant* in the number of contexts, while leaving room to keep learning?

I propose Online EWC. It replaces EWC's per-context stack of springs with a *single* quadratic penalty, anchored at the most recent optimum, whose stiffness is a running, exponentially-decayed sum of the per-context diagonal Fishers — constant memory and constant per-step cost regardless of how many contexts have been seen. The first move comes from being honest about the Bayesian argument EWC itself rests on. The weights are random, and what is known after the data of contexts $1..k$ is the posterior $p(\theta\mid T_{1:k})$, which because the contexts are conditionally independent given $\theta$ builds up recursively, $p(\theta\mid T_{1:k}) \propto p(\theta\mid T_{1:k-1})\,p(T_k\mid\theta)$. Stare at that recursion: it never needs the individual context likelihoods lying around — the only object it carries forward is the single running posterior. EWC's linear growth came not from the math but from how it *approximated* the math: it Laplace-approximated each *likelihood* $p(T_t\mid\theta)$ separately into its own Gaussian $\mathcal N(\theta^*_t, F_t^{-1})$, and the product of those Gaussians is the product of springs. The consistent thing is to Laplace-approximate the *whole running posterior* instead. Run it forward and the difference shows from the third context. For context $C$ after $A,B$, the exact log posterior is $\log p(D_C\mid\theta) + \log p(\theta\mid D_A,D_B) + \text{const}$, but I have already discarded $D_A,D_B$, so I expand the *approximation* I built while learning $B$ — namely $\log p(D_B\mid\theta) - \frac12\sum_i F_{A,i}(\theta_i-\theta^*_{A,i})^2$ — around its mode, which is the optimum $\theta^*_B$ I already found (so the first-order term vanishes). Its curvature there has two pieces: the negative-log-likelihood of $B$ contributes $F_B$, and the spring around $\theta^*_A$ contributes its own coefficient $F_A$, since a quadratic's second derivative is its stiffness everywhere. So the consistent approximation of $p(\theta\mid D_A,D_B)$ is a *single* quadratic, and learning $C$ minimises

$$L_C(\theta) + \tfrac12\sum_i (F_{A,i}+F_{B,i})\,(\theta_i-\theta^*_{B,i})^2.$$

One spring, anchored at the *latest* optimum, with stiffness equal to the *sum* of past Fishers — the term count did not grow. And keeping EWC's old anchor at $\theta^*_A$ as well would be not merely redundant but wrong: $\theta^*_B$ was found while already being pulled toward $\theta^*_A$, so $A$'s information is baked into where $\theta^*_B$ sits; a separate surviving spring at $\theta^*_A$ imposes $A$'s constraint a *second* time, double-counting the earliest contexts and biasing the whole sequence toward them. Dropping the old anchors is therefore both cheaper and the consistent choice. The honest price is that Fisher is *local* — re-centring the accumulated stiffness at $\theta^*_{i-1}$ pretends the old contexts' curvature, measured at their own optima, still applies here — so the most distant contexts are remembered slightly less faithfully than under an explicit per-context anchor. That is a real, predictable trade: the recent past is favoured over the distant past.

The single re-centred spring already gives constant memory — one mean $\theta^*_{i-1}$ and one summed Fisher — but the undamped sum $\sum_t F_t$ only ever grows, and over a long sequence it re-rigidifies the fixed-capacity net until nothing new can be learned; and recurring contexts cannot be revised without storing per-context factors I have deliberately thrown away. Both problems need the same missing move: down-weight a context's earlier contribution to the shared summary without ever having stored it on its own. That is exactly what stochastic expectation propagation does — it keeps a *single shared* averaged factor standing in for all the per-context factors, and updates one by treating its contribution as a *fraction* of the shared factor: partially remove that fraction, refine, fold back. Transcribed into the Fisher accumulation, before folding in a new context I scale the shared summary down by a fraction $\gamma<1$ and then add the new Fisher,

$$F^*_i = \gamma\,F^*_{i-1} + F_i,\qquad 0\le\gamma\le 1.$$

This one change does both jobs. Unrolled it is a geometric sum, $F^*_i=\sum_{t\le i}\gamma^{\,i-t}F_t$, so a context $k$ boundaries ago contributes weight $\gamma^k$ inside the stored summary: the sum is bounded to an effective $\sim 1/(1-\gamma)$ recent contexts rather than climbing forever, which is precisely the room needed to keep learning on fixed capacity, and old contexts *fade* gracefully rather than being dropped catastrophically. For recurrence, scaling by $\gamma$ is the fractional partial-removal of a context's previous presentation before its fresh data is added — and because $\gamma$ is one scalar applied identically to every context, it needs no task *identities*, only that a boundary occurred. The same $\gamma$ also multiplies the penalty, because the object I regularise the new context against is the *already-down-weighted* shared factor, not the raw sum; under this cavity-style convention the applied stiffness is $\gamma F^*_{i-1}$ and the post-context update is $\gamma F^*_{i-1}+F_i$, with $\gamma$ kept in both places. The penalty for context $K>1$ is therefore

$$L_{\text{reg}} = \tfrac12\,\gamma\sum_i F^*_{K-1,i}\,(\theta_i-\theta^*_{K-1,i})^2,$$

added at every training step. The endpoints check the indexing: $\gamma=1$ recovers the strict undecayed single-penalty sum $\sum_t F_t$, while at $\gamma=0$ the loss-side factor zeroes the old-task penalty entirely; a middle value such as $\gamma=0.9$ keeps roughly the last ten contexts strongly represented and lets older ones decay. The same scalar is simultaneously the EP partial-removal fraction and the explicit forgetting knob, which is a sign it is the right scalar.

The importance $F_i$ itself is the diagonal Fisher at the finished context's optimum, the model's own expected squared score, $F_{i,j}=\mathbb E_x\,\mathbb E_{y\sim p_\theta(\cdot\mid x)}\big[(\partial\log p_\theta(y\mid x)/\partial\theta_j)^2\big]$. It is the estimator of choice because it equals the expected curvature of the negative log-likelihood near a minimum, is computable from first-order gradients alone, and is positive semidefinite. I estimate the outer expectation over $x$ by passing examples from the just-finished context through the trained network, and the inner expectation over $y$ by the model's own predicted class probabilities: per example, run a forward pass to get logits, softmax them to per-class weights $p_\theta(c\mid x)$, and for each class $c$ backprop $-\log p_\theta(c\mid x)$, square the gradient, and accumulate it weighted by $p_\theta(c\mid x)$. Summing the squared scores of every class weighted by its predicted probability is the Monte-Carlo estimate of the inner expectation — the true "all labels" Fisher rather than the cheaper empirical Fisher that would use only the observed label. This runs in eval mode so dropout does not perturb the curvature estimate, over a capped number of single-example passes for speed, normalised by the number of examples actually used. (When equal task weighting across very deterministic and very soft contexts is required, each context's Fisher can be normalised before accumulation so a large-norm Fisher does not dominate purely by scale; the minimal hook here keeps the standard all-label path.) Both pieces drop into the harness as two slots, and because the loop accumulates the hook's return *additively* into its stored importance, the estimator returns the *increment* relative to what is already stored — the full decayed-plus-new value $\gamma\cdot\text{existing}+F_{\text{new}}$ minus the existing value — so that "existing $+$ increment" lands exactly on $F^*=\gamma F^*_{\text{old}}+F_{\text{new}}$.

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def estimate_importance(model, dataset, prev_params, device):
    """Online EWC: diagonal Fisher with exponential-decay accumulation across contexts.

    Returns {param_name: increment} so the loop's additive accumulator ends up
    holding  F* = gamma * F*_old + F_new.
    """
    model.gamma = 0.9          # gamma < 1: EP partial-removal fraction == graceful-forgetting knob
    gamma = model.gamma        # gamma = 1 gives the strict undecayed single-penalty sum

    est_fisher = {}
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher[n] = p.detach().clone().zero_()

    mode = model.training
    model.eval()                                   # clean curvature estimate (no dropout noise)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    max_samples = min(len(data_loader), 200)
    n_samples = 0

    for idx, (x, y) in enumerate(data_loader):
        if idx >= max_samples:
            break
        x = x.to(device)
        output = model(x)
        with torch.no_grad():
            label_weights = F.softmax(output, dim=1)        # p_theta(y=c|x): inner-expectation weights
        for c in range(output.shape[1]):
            label = torch.LongTensor([c]).to(device)
            negloglikelihood = F.cross_entropy(output, label)
            model.zero_grad()
            negloglikelihood.backward(
                retain_graph=True if (c + 1) < output.shape[1] else False)
            for gen_params in model.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            est_fisher[n] += label_weights[0][c] * (p.grad.detach() ** 2)
        n_samples += 1

    est_fisher = {n: v / max(n_samples, 1) for n, v in est_fisher.items()}    # average over examples

    # F* = gamma * F*_old + F_new
    existing = getattr(model, '_custom_importance', {})
    for n in est_fisher:
        if n in existing:
            est_fisher[n] = gamma * existing[n] + est_fisher[n]

    # loop accumulates additively -> return increment to land on F* exactly
    result = {}
    for n in est_fisher:
        result[n] = est_fisher[n] - existing[n] if n in existing else est_fisher[n]

    model.train(mode=mode)
    return result


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """Online EWC penalty:  0.5 * gamma * sum( F* * (theta - theta*)^2 )."""
    gamma = getattr(model, 'gamma', 0.9)
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    fisher = importance_dict[n]               # F* (running, decayed sum)
                    prev = prev_params_dict[n]                # theta*_{i-1}: latest-boundary snapshot
                    losses.append((fisher * (p - prev) ** 2).sum())
    if losses:
        return 0.5 * gamma * sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)
```
