EWC erased SI's collapse: on Permuted-MNIST it jumped from 0.4474 to 0.8381, and the per-context line $\{0.93,0.93,0.92,0.90,0.87,0.85,0.81,0.75,0.71,0.72\}$ is no longer a slide to chance but a *gentle, monotone decay* — the bounded, PSD endpoint Fisher keeps all ten permutations alive. It near-tied SI on Split-CIFAR100 (0.5463) and came in just under SI on Split-MNIST (0.9577), as the endpoint Fisher under-weighting cleanly converged binary tasks predicts. But that gentle decay is itself the tell. The earliest contexts are remembered *worse* than the recent ones, and that is structural: the loop sums every context's Fisher into one buffer, $F_{\text{summed}}=\sum_t F_t$, which only ever grows, while re-anchoring to the latest boundary. On a fixed-capacity net the accumulating rigidity makes later permutations land lower *and* re-anchors the early contexts away from their own optima, un-protecting them. EWC has converted SI's explosion into a slow bleed — bounded per context, but unbounded in the *sum* over contexts. So the fix is no longer the estimator (the Fisher is the right curvature); it is *how importance accumulates across contexts*.

I propose **Online EWC**, and the route to it is redoing EWC's derivation without assuming the per-context stack of springs. After contexts $1{:}k$ the posterior builds purely recursively, $p(\theta\mid T_{1:k})\propto p(\theta\mid T_{1:k-1})\,p(T_k\mid\theta)$: the posterior after all $k$ contexts is the posterior after the first $k-1$ — used as the *prior* — times the likelihood of just the $k$-th. The only object this recursion ever needs is the *running posterior*; it never asks me to keep the individual context likelihoods around. That is the tell — EWC's stack of springs is nowhere in this equation. EWC's linear growth came not from the Bayesian math but from how it Laplace-approximated it: EWC applies the Gaussian to each context's *likelihood* $p(T_t\mid\theta)$ separately, one $\mathcal N(\theta^*_t,F_t^{-1})$ per context, and the product of those is the product of springs. But the recursion says the object to approximate is the whole running posterior.

Run the consistent version and the difference shows from the third context. For $A$ then $B$, Laplace-approximating $p(\theta\mid D_A)$ at $\theta^*_A$ gives the single spring $\tfrac12\sum_i F_{A,i}(\theta_i-\theta^*_{A,i})^2$, so learning $B$ minimizes $L_B+\tfrac12\sum_i F_{A,i}(\theta_i-\theta^*_{A,i})^2$ — identical to EWC. Now context $C$: the exact log posterior is $\log p(D_C\mid\theta)+\log p(\theta\mid D_A,D_B)+\text{const}$, and I must Laplace-approximate the running posterior $p(\theta\mid D_A,D_B)$. I no longer have $D_A,D_B$; what I have is the approximation built while learning $B$, namely $\log p(D_B\mid\theta)-\tfrac12\sum_i F_{A,i}(\theta_i-\theta^*_{A,i})^2$. Expanding *that* around its mode $\theta^*_B$ (where its first-order term vanishes), the curvature has two pieces — $B$'s NLL contributes $F_B$, and the spring around $\theta^*_A$ contributes its own constant stiffness $F_A$ — so the consistent Laplace approximation is a *single* quadratic $\tfrac12\sum_i(F_{A,i}+F_{B,i})(\theta_i-\theta^*_{B,i})^2$, anchored at the *latest* optimum with stiffness the *sum* of past Fishers. EWC at $C$ carries two springs, at $\theta^*_A$ and $\theta^*_B$; the consistent recursion carries one. And EWC's extra spring is not merely redundant but actively wrong: $\theta^*_B$ was found *while already being pulled toward $\theta^*_A$* by $A$'s spring, so $A$'s information is baked into where $\theta^*_B$ sits, and a spring at $\theta^*_B$ already inherits $A$'s pull. Keeping a separate spring still anchored at $\theta^*_A$ imposes $A$ a second time — a double-count that, over a long sequence, systematically over-asserts the earliest contexts. The fix is not to blend the old anchors (that preserves the double-counting) but to *drop* them and anchor only at the latest optimum, which already encodes the cumulative pull of everything before. The running state collapses to two parameter-sized objects: the latest mean and a running Fisher sum, $F^* \leftarrow F^*_{\text{old}}+F_{\text{new}}$. This is in fact exactly what the harness loop already does for EWC — which is why EWC already had constant memory and yet still bled.

So consistency alone is not the cure, because with $F^*=\sum_t F_t$ the stiffnesses still only ever add. Over ten Permuted contexts the summed Fisher grows in every direction any context cared about, and the fixed-parameter net becomes too rigid to learn later permutations — exactly the 0.93-to-0.72 decay. What I actually need is the ability to *remove*, or down-weight, a context's earlier contribution to the shared summary without ever having stored it separately. Expectation propagation refines a factor by dividing it out, but EP keeps one factor per context — the linear memory I am fleeing. *Stochastic* EP relaxes precisely this: keep a single shared averaged factor for all contexts, and treat any one factor as a *fraction* of the shared one; to update, down-weight the shared factor by $\gamma<1$, refine on new data, fold back in. Transcribed into the Fisher accumulation, EWC's $F^*\leftarrow F^*_{\text{old}}+F_{\text{new}}$ becomes
$$F^* \leftarrow \gamma\,F^*_{\text{old}} + F_{\text{new}},\qquad \text{penalty}\;=\;\tfrac12\,\gamma\sum_i F^*_i\,(\theta_i-\theta^*_i)^2 .$$
That single scalar does both jobs. For recurrence, scaling by $\gamma$ is the fractional removal of a context's previous presentation before fresh data is folded in — no per-context factor, no task identity, only that a boundary occurred. For capacity, $\gamma<1$ turns the unbounded sum into a geometric one: unrolling, $F^*_i=\sum_{t\le i}\gamma^{\,i-t}F_{t,i}$, so a context $k$ boundaries ago contributes with weight $\gamma^k$ and the summary settles to an effective $\sim 1/(1-\gamma)$ recent contexts instead of climbing forever. That is the room to keep learning on fixed capacity, and it fades old half-wrong constraints gracefully rather than dropping them. The endpoints check out: $\gamma=1$ recovers strict undecayed EWC, and with the loss scaled by $\gamma$ too, $\gamma\to0$ leaves no penalty. I set $\gamma=0.9$ — roughly the last ten contexts strongly represented, well matched to Permuted-MNIST's ten — overriding the framework default of $1.0$. The $\gamma$ sits in the penalty as well as the accumulation on purpose: the cavity-style reading says the object I regularize against is the already-down-weighted shared factor, so applied stiffness is $\gamma F^*_{i-1}$ and the post-context update is $\gamma F^*_{i-1}+F_i$, the two uses kept tied.

One harness-specific adapter is the only thing here not in a from-scratch Online EWC. The loop accumulates whatever I return *additively* into $\texttt{\_custom\_importance}$, computing $\text{existing}+\text{my\_return}$, but I want the buffer to land on $\gamma\cdot\text{existing}+F_{\text{new}}$. So inside $\texttt{estimate\_importance}$ I compute the full decayed-plus-new Fisher myself and then *subtract* the existing value before returning, so the loop's $\text{existing}+(\gamma\cdot\text{existing}+F_{\text{new}}-\text{existing})$ lands exactly on $\gamma\cdot\text{existing}+F_{\text{new}}$. The estimator itself is unchanged from EWC — the diagonal Fisher from softmax-weighted squared gradients over all classes, eval mode, capped sample. Against EWC's line I expect the geometric decay to relieve the accumulating rigidity: the later Permuted contexts should land higher than EWC's, the earliest may fade slightly (the honest cost of $\gamma<1$), and the average should edge above 0.8381; a near-tie or slight gain on Split-MNIST (five tasks well inside the $\sim10$-context effective memory); and essentially EWC's mid-0.5s on Split-CIFAR100, where the estimator is identical and ten contexts sits right at the decay window. The win, if it comes, is modest and concentrated exactly where the theory says — the long sequence whose accumulating springs were the last thing over-constraining the net — and it is the endpoint of this ladder: bounded, re-centered, decayed curvature at constant cost.

```python
# EDITABLE region of custom_regularization.py — step 3: Online EWC (decayed running Fisher)
def estimate_importance(model, dataset, prev_params, device):
    """Online EWC: Diagonal Fisher with exponential decay accumulation.

    When accumulating across contexts: fisher = gamma * fisher_old + fisher_new.
    Uses gamma=0.9 as the online Fisher decay for this benchmark.
    """
    # Explicitly set gamma on the model to override framework default (1.0).
    # With gamma=1.0, Online EWC reduces to standard EWC.
    model.gamma = 0.9
    gamma = model.gamma
    est_fisher = {}
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher[n] = p.detach().clone().zero_()

    mode = model.training
    model.eval()

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_samples = min(len(data_loader), 200)

    for idx, (x, y) in enumerate(data_loader):
        if idx >= n_samples:
            break
        x = x.to(device)
        output = model(x)
        with torch.no_grad():
            label_weights = F.softmax(output, dim=1)
        for label_index in range(output.shape[1]):
            label = torch.LongTensor([label_index]).to(device)
            negloglikelihood = F.cross_entropy(output, label)
            model.zero_grad()
            negloglikelihood.backward(
                retain_graph=True if (label_index + 1) < output.shape[1] else False
            )
            for gen_params in model.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            est_fisher[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)

    est_fisher = {n: v / max(n_samples, 1) for n, v in est_fisher.items()}

    # Apply decay to existing importance before adding new
    existing = getattr(model, '_custom_importance', {})
    for n in est_fisher:
        if n in existing:
            est_fisher[n] = gamma * existing[n] + est_fisher[n]

    # We return the full (decayed + new) Fisher, so the training loop
    # should replace (not add to) _custom_importance. To achieve this
    # with the accumulation logic in mid_edit, we subtract the existing
    # importance so that accumulation yields the correct result.
    result = {}
    for n in est_fisher:
        if n in existing:
            result[n] = est_fisher[n] - existing[n]
        else:
            result[n] = est_fisher[n]

    model.train(mode=mode)
    return result


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """Online EWC: 0.5 * gamma * sum(fisher * (param - prev_param)^2)."""
    gamma = getattr(model, 'gamma', 0.9)  # Already set to 0.9 in estimate_importance
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    fisher = importance_dict[n]
                    prev = prev_params_dict[n]
                    losses.append((fisher * (p - prev) ** 2).sum())
    if losses:
        return 0.5 * gamma * sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)
```
