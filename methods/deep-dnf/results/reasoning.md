OK, let me think this through from scratch. I have Boolean examples — `x ∈ {0,1}^n`, label `y ∈ {0,1}` — and the label is some unknown Boolean function of `x`. I want a neural model that learns it by gradient descent. But I want two things, and I should be honest that they're in tension. One is that it actually fits, including the functions that small nets are bad at. The other is that after training I can *read the function out* — say exactly which variables it depends on and how — as an explicit logical expression. The second want is the one that bites, because the obvious model fails it.

The obvious model is the additive perceptron stack. A single linear-threshold unit can already do AND or OR: for `x_1 ∧ x_2 ∧ x_3` I set three weights to 1, the rest to 0, the bias to `-2`, and pass `Σ w_i x_i + b` through a step. The sum is 3 only when all three are on, so `3 - 2 = 1 > 0` fires, and any missing variable drops the sum to 2, giving `2 - 2 = 0`, not positive — good, that's the AND. OR is the same with bias `-1`+... wait, OR of three is "fire if the count is at least 1", so threshold the sum at 1, bias `-1` plus a tiny offset, fine. And depth handles the rest: Minsky & Papert showed one such unit can't do XOR, but stack layers and any Boolean function is representable. Steinbach & Kohut spell this out — `η(Σ w_j x_j + b)` units, proper activations, enough layers, you can learn any Boolean function.

So why am I unhappy. Let me actually try to decode a trained one. Suppose a hidden unit is supposed to compute `x_1 ∧ x_2 ∧ x_3`. After training I have a vector of real weights and a real bias. To claim it's that AND I have to check: are exactly these three weights "large enough" and the rest "small enough" *relative to the bias threshold*? But the bias is the count threshold, it's continuous, it drifted during training, and there's a whole manifold of (weights, bias) tuples that compute the same gate — scale them all up, nudge the bias, same function. There's no canonical procedure to read "which variables, which polarity" off the unit. The logic is smeared across the weights and *hidden in the bias*. That's the legibility failure, and it's structural: the bias is doing the load-bearing work and the bias is exactly the part you can't read as logic.

And it's not just legibility. The additive design is empirically fragile in ways that point at the same culprit. Fit a random DNF over 10 bits: with fair-coin inputs `p=0.5` the MLP converges fine, but with skewed inputs `p=0.75` it keeps making test errors and never quite locks onto the exact function even after long training. And XOR over `n>30` — it sits at ~50% error, doesn't learn at all. The thing that has to be balanced against the variable contributions is the bias threshold, and when the input statistics shift, the right threshold shifts, and the optimizer chases it. So both problems — can't decode it, doesn't reliably converge — trace back to the same place: the bias.

Let me sit with that. What if I just refuse to have a bias? Drop the `+b` entirely. Then a single additive unit `η(Σ w_j x_j)` can't even do AND — without the `-2` there's no threshold, the sum 1, 2, 3 all map monotonically and I can't isolate "all three on." So removing the bias from the *additive* design breaks it. The bias isn't an accident there; the whole additive AND *needs* a threshold because addition can't tell "all of them" from "some of them." Addition only knows the count.

So the real problem is addition. AND is not about counting how many inputs are on; it's about *all* of them being on. Counting needs a threshold to recover "all"; the natural operation that already means "all" is multiplication. `x_1 · x_2 · x_3` is 1 exactly when every factor is 1, and 0 the instant any factor is 0 — no threshold, no bias, the gate is built into the algebra. So let me throw out the additive neuron and build a *multiplicative* one.

This is also the right move for the continuous relaxation I'll need for gradient descent. I have to extend truth values from `{0,1}` to `[0,1]` and pick smooth surrogates for the connectives that agree on the corners. The product family does exactly that: `NOT x = 1 - x`, `x ∧ y = x·y`, and by De Morgan `x ∨ y = 1 - (1-x)(1-y)`. Check the corners — `x·y` is the AND table; `1-(1-x)(1-y)` is 0 only when both are 0, the OR table — and in between they're smooth and differentiable. I'll work in `[0,1]` throughout. Good, the substrate is products of `[0,1]` values.

Now the actual design problem. A conjunction neuron should compute the AND over *some subset* of the input variables — I don't know which subset, I have to learn it. So I need a differentiable way to select a subset. The first idea that comes to mind is borrowed from pointer networks: put a softmax over the `n` inputs and let it attend to / select the variables in the clause. But that's wrong for two reasons. A softmax picks essentially *one* thing (it's a distribution that sums to 1), so to select `k` variables I'd need `k` separate softmaxes — which means I have to commit to the clause size `k` up front, and I don't know it. And even setting that aside, a softmax that has to concentrate sharply over a long input vector converges painfully slowly when `n` is large; I'd be fighting the optimizer. Selection-by-softmax is the wrong primitive here.

Let me back up and think about what "include or exclude variable `i` in this AND" really needs. It's a per-variable, independent binary decision — is `x_i` in the clause or not — not a competition among variables. So instead of one softmax over all inputs, I want one independent gate per input. Give each input `i` (for clause `j`) a *membership weight* `m_i ∈ [0,1]`: roughly 1 means "`x_i` is in this conjunction," roughly 0 means "it's not." `n` independent gates, no commitment to clause size, no winner-take-all.

How do I fold a membership into the product so that `m_i = 0` makes `x_i` disappear and `m_i = 1` makes it count? I need a per-variable factor `F_c(x_i, m_i)` to multiply into the conjunction, and I need it to behave by a truth table: when the variable is *excluded* (`m_i = 0`) the factor must be 1 (multiplying by 1 is the identity — the variable vanishes from the product, regardless of `x_i`); when it's *included* (`m_i = 1`) the factor must just be `x_i` (so the AND sees `x_i`). Let me write that table out: `(x_i=0, m_i=0) → 1`, `(x_i=1, m_i=0) → 1`, `(x_i=0, m_i=1) → 0`, `(x_i=1, m_i=1) → 1`. That's "1 unless `x_i=0` and `m_i=1`", i.e. it's the negation of `(¬x_i ∧ m_i)`. In the product algebra: `F_c = 1 - m_i(1 - x_i)`. Let me sanity-check it against the four rows — `m_i=0` gives `1 - 0 = 1` for both `x_i`; `m_i=1, x_i=0` gives `1 - 1·1 = 0`; `m_i=1, x_i=1` gives `1 - 1·0 = 1`. All four match. So

```
O_conj(x) = Π_i ( 1 - m_i (1 - x_i) ),   m_i ∈ [0,1].
```

That's beautiful — no bias anywhere, and I can read the clause straight off the `m_i`: the variables with `m_i ≈ 1` are exactly the literals in the conjunction. The legibility problem is *solved by construction*, not by post-hoc decoding.

I need `m_i` to stay in `[0,1]` and be a free parameter for gradient descent, so let `m_i = σ(c·w_i)` with `w_i ∈ ℝ` the trainable weight and `σ` the sigmoid. Why the constant `c > 1`? Because I want crisp logic at the end — memberships near 0 or near 1, not mushy 0.4's. Scaling the logit by `c` sharpens the sigmoid, so the optimizer is pushed toward the saturated ends and the final memberships are decisively in/out. (A hard relu-of-relu thresholding also squashes to `[0,1]`, but the sharpened sigmoid is more stable and lets me use a larger learning rate — the hard version has flat regions where the gradient dies.)

Let me check convergence, because the multiplicative form could have a nasty gradient. Differentiate the conjunction w.r.t. one membership: `∂O_conj/∂m_i = -(1 - x_i) · Π_{k≠i}(1 - m_k(1-x_k))`. The product over the *other* factors is `≥ 0`, so the sign of the gradient is the sign of `-(1 - x_i)`, i.e. `∂O_conj/∂m_i ∝ (x_i - 1)`. So to *increase* a membership toward 1, I need a training example where `x_i = 0` and yet the conjunction output is supposed to be 1 — a counterexample that says "this clause fired even though `x_i` was off, so `x_i` shouldn't be a required literal." If `x_i` is always 1 whenever the clause fires, there's nothing pushing `m_i` down and nothing pushing it up either; it's the counterexamples that move the memberships. So as long as each batch contains the relevant counterexamples and the learning rate is small enough, a single conjunction layer will converge to the right memberships. Good — the multiplicative design isn't just legible, it's trainable, and I can see *why* from the gradient.

Now the disjunction. By De Morgan, OR is the dual of AND: `x ∨ y = ¬(¬x ∧ ¬y) = 1 - (1-x)(1-y)`. So a disjunction neuron over a selected subset should be `1 - Π (1 - [contribution of selected x_i])`. I need the membership-gated per-variable contribution for OR: a factor that, when the variable is *excluded* (`m_i=0`), contributes nothing to the OR — meaning the `(1 - contribution)` factor must be 1, i.e. the contribution is 0 — and when *included* (`m_i=1`) contributes `x_i`. Truth table for the contribution `F_d(x_i, m_i)`: it should be 0 whenever `m_i=0` (variable not in the clause), and `x_i` when `m_i=1`. That's just `F_d = x_i · m_i`. Then

```
O_disj(x) = 1 - Π_i ( 1 - m_i x_i ),   m_i ∈ [0,1].
```

Check: `m_i = 0` makes that factor `1 - 0 = 1`, so it drops out of the product and `x_i` has no effect on the OR — right, it's excluded. `m_i = 1` makes the factor `1 - x_i`, so the product over the included ones is `Π(1-x_i)` and the OR is `1 - Π(1-x_i)` = "at least one of the included `x_i` is on." That's the noisy-OR form, the same `1 - Π(1-p)` that means "at least one independent cause fires." Same membership trick, same readability — the included literals are the ones with `m_i ≈ 1`.

So now I have two stackable layers, both bias-free, both decodable from their memberships, both differentiable. A *conjunction layer* of `m` neurons is `m` independent conjunction neurons each with its own membership vector — exactly like stacking `m` perceptrons into a layer, same parameter count, just multiplicative instead of additive. Same for a disjunction layer.

How do I assemble them into a model of an arbitrary Boolean function? Here's the lever: every Boolean function has a disjunctive normal form — it's an OR of ANDs of literals. So if I cascade a conjunction layer (producing a bank of candidate conjunctions of the inputs) into a single disjunction neuron (OR-ing those conjunctions together), I have a differentiable DNF:

```
DNF(x) = DISJ( CONJ(x) ).
```

The conjunction layer's outputs are the soft truth values of `m` candidate terms; the disjunction layer ORs the ones it selects. Train end to end and the conjunction memberships tell me the literals of each term, the disjunction memberships tell me which terms are in the function — the whole formula falls out of the weights. (Symmetrically, `CONJ(DISJ(x))` gives a CNF model, for problems whose natural form is a product of clauses; and since some functions are short in one form but exponential in the other — e.g. `(x_1∨x_2)∧(x_3∨x_4)∧…` is `2^{n/2}` terms in DNF — I could even OR a CNF and a DNF branch, `1-(1-CNF)(1-DNF)`, to let the model pick whichever normal form is compact. But for learning a DNF target the DNF cascade is the matched hypothesis class, so I'll build that.)

Now the practical wall. The conjunction is a product of `n` factors each in `[0,1]`. For large `n`, even when memberships are settling correctly, that product is a product of many sub-1 numbers and it underflows toward zero — and a near-zero output with a near-flat product surface means tiny gradients. The fix is to compute the product in the log domain: `O_conj = exp( Σ_i log(ε + 1 - m_i(1 - x_i)) )`, with a small `ε` for stability. This converts the product to a sum (numerically stable, no underflow) and is always valid because every factor is in `[0,1]`. It costs an extra log and exp per neuron but keeps the same multiply/add complexity otherwise. Same trick for the disjunction's product.

There's a second large-`n` wall: initialization. If I start all memberships at random middling values, then for big `n` the conjunction is a product of many factors that are neither 0 nor 1 — so the output is some tiny number and, worse, `∂O_conj/∂m_i = -(1-x_i)·Π_{k≠i}(…)` has that product of many sub-1 terms multiplying it, so *every* membership's gradient is crushed. The model can't get off the ground. The cure is to start *sparse*: initialize most memberships near 0 (so most factors are ≈ 1 and don't shrink the product or the gradient) and only a small subset near 1. Concretely, initialize the membership logits with a strongly negative mean — equivalently fill them with a large negative constant plus a little noise — so `σ(c·w)` starts near 0 for almost everything, and the few variables that matter get pulled up by the counterexample gradients. A conjunction that starts as "almost no literals" can *grow* the literals it needs; one that starts as "all literals, all half-on" is stuck in a vanishing-gradient swamp.

Let me now make the literal selection a bit richer, because real DNF terms use both polarities — a term can require `x_i` *or* require `¬x_i`. My membership `m_i` as defined only chooses include-`x_i`-or-not; it has no way to say "include the *negation* of `x_i`." I could add a second membership for the negated literal, but then a variable could be told to include both `x_i` and `¬x_i` at once, which is contradictory. The clean way is to make the per-variable decision a single *categorical* choice over three mutually exclusive options: use the positive literal `x_i`, use the negative literal `1 - x_i`, or skip the variable entirely. Three options that must sum to 1 — that's a softmax over three logits per (term, variable). Call the three probabilities `pos, neg, skip`. The per-variable factor going into the conjunction is then their mixture:

```
literal = pos · x + neg · (1 - x) + skip · 1.
```

Read it against the cases: if `skip ≈ 1` the factor is ≈ 1 and the variable drops out (this is the old `m_i ≈ 0` exclusion). If `pos ≈ 1` the factor is `x` — the positive literal must be on for the AND. If `neg ≈ 1` the factor is `1 - x` — the negated literal, on when `x = 0`. So this is exactly the membership idea generalized from "in/out of the positive literal" to "positive / negated / out," which is what mixed-polarity DNF needs, and it's still one normalized categorical per variable so it can't pick two contradictory literals. I'll clamp `literal` into `[ε, 1]` for the log, then `conj = exp(Σ log literal)`, clamped at 1.

Each term also wants its own on/off knob — a learnable gate saying "is this candidate conjunction actually part of the function." So scale each term's truth value by `σ(g_j)` with `g_j` a per-term logit: `term_prob_j = σ(g_j) · conj_j`. Initialize the gates negative (terms start mostly off, consistent with the sparse-start logic), and let useful terms turn on.

Then OR the terms with the noisy-OR disjunction, in the log domain for the same stability reason: `prob = 1 - Π_j (1 - term_prob_j) = 1 - exp( Σ_j log(1 - term_prob_j) )`, using `log1p(-term_prob)` for accuracy near small `term_prob`, and clamping `term_prob` just below 1 so the log doesn't blow up. Finally I want to train with binary cross-entropy on a logit, so convert the probability to a logit: `logit = log(prob) - log(1 - prob)` (plus a small output bias for calibration), and feed it to `BCEWithLogitsLoss`.

Two more things to make the learned formula a *clean* DNF rather than a soggy approximation. The DNF inductive bias is "few terms, each short" — sparse literals and sparse terms — and nothing so far enforces that; the model could spread a little membership mass over many variables and many terms and still fit the training set. So add two gentle penalties. First, literal sparsity per term: define each term's *usage* as the total non-skip mass, `usage_j = Σ_i (pos_{j,i} + neg_{j,i})` — that's the expected number of literals in term `j` — and penalize it for *exceeding* a target width `w` (the known term width), `Σ_j (usage_j - w)_+^2`, one-sided so a term may be shorter than `w` for free but pays for being bloated. Second, term sparsity: penalize the mean gate `mean_j σ(g_j)` so the model prefers to leave terms off unless they earn their place. Both with tiny coefficients — they're a nudge toward crisp, short, few-term logic, not a hard constraint that would distort the fit.

Let me also fix the layer width. I'm learning an `s`-term DNF but I don't know `s` exactly and the optimizer benefits from slack — extra candidate terms it can leave gated off — so set the number of conjunction neurons to a few times the expected term count with a floor (e.g. `max(4s, 32)`). Over-provisioning terms is cheap because the term-sparsity penalty and the gates will switch the unused ones off.

Now I can write the whole thing. The conjunction layer is the 3-way-softmax membership product in log space; the disjunction is the noisy-OR in log space; the output is the logit; the loss is BCE plus the two sparsity nudges; the optimizer is Adam-style with a small learning rate, which matches the "small learning rate so counterexamples drive convergence" analysis. Each block traces directly back to a step above: products because AND means "all," membership softmax because literal selection is a per-variable include/negate/skip choice with no bias, noisy-OR because OR is the De Morgan dual, log-domain because the long products underflow, sparse init because a dense start kills the gradient, and the penalties because a DNF is short and few-termed.

```python
import torch
from torch import nn


class NeuralDNF(nn.Module):
    """Differentiable DNF: a bank of soft conjunctions (membership-gated products
    of literals) OR'd together by a noisy-OR. The parameters ARE the formula —
    each term's literals are read off its per-variable categorical (pos/neg/skip),
    each term's presence off its gate."""

    def __init__(self, n_features: int, n_terms: int):
        super().__init__()
        # per (term, variable): logits over {use x_i, use ¬x_i, skip}
        self.literal_logits = nn.Parameter(torch.empty(n_terms, n_features, 3))
        # per term: on/off gate logit (the disjunction's membership)
        self.term_logits = nn.Parameter(torch.empty(n_terms))
        self.out_bias = nn.Parameter(torch.zeros(1))
        self.n_features = n_features
        self.n_terms = n_terms
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # sparse start: pos/neg literals OFF, skip ON, so every term begins as
        # "almost no literals" (factors ~1) and grows literals via counterexample
        # gradients — a dense half-on start would vanish the product gradient.
        with torch.no_grad():
            self.literal_logits[..., 0].fill_(-4.0)   # positive literal: off
            self.literal_logits[..., 1].fill_(-4.0)   # negative literal: off
            self.literal_logits[..., 2].fill_(4.0)    # skip: on
            self.literal_logits.add_(0.03 * torch.randn_like(self.literal_logits))
            self.term_logits.fill_(-3.0)              # terms start gated off
            self.out_bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # per-variable literal factor: mixture over use-x / use-(1-x) / skip.
        probs = torch.softmax(self.literal_logits, dim=-1)   # categorical membership
        pos, neg, skip = probs[..., 0], probs[..., 1], probs[..., 2]
        literal = (
            pos.unsqueeze(0) * x.unsqueeze(1)               # positive literal: x
            + neg.unsqueeze(0) * (1.0 - x.unsqueeze(1))     # negative literal: 1-x
            + skip.unsqueeze(0)                              # skip: factor 1 (drop var)
        ).clamp(min=1e-6, max=1.0)
        # conjunction = product of literal factors, in log space (avoids underflow)
        conj = torch.exp(torch.log(literal).sum(dim=-1)).clamp(max=1.0)
        # gate each term, then noisy-OR over terms (also in log space)
        term_prob = torch.sigmoid(self.term_logits).unsqueeze(0) * conj
        log_not = torch.log1p(-term_prob.clamp(max=1.0 - 1e-6)).sum(dim=-1)
        prob = (1.0 - torch.exp(log_not)).clamp(min=1e-6, max=1.0 - 1e-6)
        # to a logit for BCEWithLogits
        return torch.log(prob) - torch.log1p(-prob) + self.out_bias

    def literal_usage(self) -> torch.Tensor:
        # expected #literals per term = total non-skip mass; drives literal sparsity
        probs = torch.softmax(self.literal_logits, dim=-1)
        return (probs[..., 0] + probs[..., 1]).sum(dim=-1)


def fit_and_predict(model, train_x, train_y, test_x, term_width, seed,
                    lr=2e-3, epochs=30, batch_size=512):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    n = train_x.shape[0]
    target_width = float(term_width)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(train_x[idx]).view(-1)
            loss = criterion(logits, train_y[idx])
            # literal sparsity: one-sided penalty for terms wider than target w
            usage = model.literal_usage()
            width_penalty = ((usage - target_width).clamp(min=0.0) ** 2).mean()
            # term sparsity: prefer few active terms
            gate_penalty = torch.sigmoid(model.term_logits).mean()
            total = loss + 1e-4 * width_penalty + 1e-4 * gate_penalty
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return (model(test_x).view(-1) >= 0.0).long()
```

The causal chain, end to end: AND means "all of them," and addition can only count, so the gate needs a bias threshold I can neither read nor stabilize — switch to multiplication, where "all" is built into the product and the bias disappears. Selecting which variables go into each AND wants a per-variable include/exclude decision, not a winner-take-all softmax over inputs, so attach an independent membership to each variable; gate it into the product as `1 - m(1-x)` (and generalize to a pos/neg/skip categorical so terms can use either polarity). OR is the De Morgan dual, so a disjunction layer is the noisy-OR `1 - Π(1 - m·x)` of the conjunctions. Cascade conjunction-then-disjunction because every Boolean function is a DNF. The long products underflow and vanish the gradient, so compute them in the log domain and initialize sparse. And because a DNF is short and few-termed, nudge literal-usage and term-gates toward sparsity. The parameters that come out *are* the learned formula.
