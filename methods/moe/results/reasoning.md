OK, let me think this through from scratch. The thing I keep running into is a single, blunt fact: how much a network can know is bounded by how many parameters it has. On the big text corpora — a billion words, tens of billions of words — quality just keeps climbing as I add parameters. Jozefowicz and colleagues took LSTM language models up past a hundred million parameters and the perplexity kept dropping. So the data clearly has more to teach than my models can currently hold.

But there's a wall. In an ordinary network every parameter fires on every example. So if I double the parameters, I double the work per example, and if I'm also training on more examples — which I have to, because more capacity needs more signal — the cost goes as roughly the product of the two. It's a quadratic blow-up in total training cost as I scale both axes together. Hardware isn't keeping up with that. I can't just keep making the dense model bigger.

So I want to break the link between parameter count and compute-per-example. I want a model that *holds* an enormous number of parameters but only *touches* a tiny fraction of them for any given example. Route each example to a few specialist subnetworks; leave the rest dormant. Then capacity is the total bank of subnetworks, but compute is just the handful that ran.

This isn't a new wish. People have been talking about "conditional computation" for years — turn parts of the network on and off per example. The trouble is that nobody has actually cashed it out at scale, and when I look at why, the reasons are concrete and worth holding in mind, because they're going to constrain everything I design.

First, GPUs are far faster at dense arithmetic than at branching. So a per-example decision can't gate a single neuron — the branching overhead would swamp any savings. Whatever I switch on or off has to be a big enough chunk of computation that the decision pays for itself. That argues for coarse units: whole subnetworks, not individual units.

Second, large batches are what make these devices efficient — you amortize the cost of loading and updating parameters over many examples. But the moment I route examples conditionally, each conditionally-active chunk sees only the examples routed to it, which is a *smaller* batch. So conditional computation is in tension with the batching that makes hardware fast. I'll come back to this — it's a real problem, not a footnote.

Third, bandwidth. A GPU cluster can have thousands of times more aggregate compute than inter-device bandwidth. Anything that has to ship parameters or activations across the network for each example — an embedding lookup is the classic case — becomes bandwidth-bound and the compute sits idle. So if I shard subnetworks across devices, I have to make sure the ratio of compute-per-subnetwork to bytes-shipped is high.

Fourth, the people who tried hard binary gates needed extra loss terms to get the sparsity and the balance they wanted, and training the gates was awkward. Bengio and colleagues used boolean gates trained with a REINFORCE-style estimator and three auxiliary losses. REINFORCE means a high-variance gradient estimate for the gating decision — you sample a hard choice and reward it. That's a lot of variance to fight through to learn a good router.

Fifth, and this one is easy to overlook: the existing demonstrations were on small data — image sets of a few hundred thousand examples. It is hard to believe those labels carry enough information to train millions, let alone billions, of parameters. If I want to actually *use* huge capacity, I need a domain where the corpus is genuinely enormous. Language is exactly that: there's effectively unlimited text. So I'll build this for language modeling and translation, where capacity is known to matter and the data is there.

Now, what's the right shape for "a bank of specialist subnetworks plus a router"? I don't have to invent the shape from nothing — there's an old idea here. Adaptive mixtures of local experts, from Jacobs, Jordan, Nowlan and Hinton back in '91, later the hierarchical version by Jordan and Jacobs. You have a set of expert networks `E_1 … E_n`, and a gating network `G` that produces a weight for each expert, and the output is the gate-weighted sum of the expert outputs:

    y = Σ_i G(x)_i E_i(x)

The gate is a softmax: `G(x) = Softmax(x · W_g)`. Train the whole thing jointly and the experts specialize — different experts come to handle different regions of the input. That's almost exactly the object I want. Each `E_i` is a specialist; `G` is the router.

But classically the mixture *is* the entire model, and crucially the gate is *dense*: the softmax gives every expert a nonzero weight, so I still evaluate all `n` experts for every example. That's no savings at all — capacity and compute rise together, same wall as before. Eigen, Ranzato and Sutskever took a good step by using mixtures as *components* inside a deeper network, each with its own gate, instead of as the whole model — that's the right granularity for "a layer I can drop in." They even noted you could make the gate sparse. But they didn't actually do it; they kept it dense.

So let me stare at the equation and ask where a saving could possibly come from:

    y = Σ_i G(x)_i E_i(x)

Wherever `G(x)_i = 0`, the term contributes nothing — and, the part that matters, I never have to evaluate `E_i(x)` at all. So the question becomes: can I make `G(x)` itself *sparse* — mostly zeros, only a few nonzero entries — without breaking the joint training? If I can force `G(x)` to have exactly `k` nonzero entries, then I run exactly `k` experts per example regardless of how large `n` is. The parameter bank is `n` experts; the per-example compute is `k` experts. Let me sanity-check the arithmetic that this is actually the decoupling I wanted: with `n = 4096` and `k = 4`, the layer holds `4096` experts' worth of weights but every example touches `4/4096 ≈ 0.1%` of them — capacity up by three orders of magnitude, compute essentially flat. So the sparsity of the gate *is* the conditional computation; that's the lever. Everything now hinges on producing a gate that is sparse, trainable, and — given a balancing problem I can already smell coming — not collapsed onto a favored few.

How do I make a softmax output sparse? The crudest thing that could work: compute the gating logits, keep only the top `k` of them, and force the rest to zero. Concretely, take the logits, set everything outside the top `k` to `−∞`, then softmax. The `−∞` entries exponentiate to zero, so the softmax is supported on exactly `k` experts and still sums to one over them:

    G(x) = Softmax(KeepTopK(H(x), k))

    KeepTopK(v, k)_i = v_i if v_i is among the top k of v, else −∞

where for the moment `H(x) = x · W_g`. Top-k is a discontinuous operation — as `x` moves, the identity of the surviving `k` can flip, and `G` jumps there. That's theoretically uncomfortable, and I should be honest that I haven't proven it's harmless. The set where the membership flips has measure zero (it's where two logits are exactly tied), and on either side of it the softmax over the surviving `k` is smooth. I'll proceed on the bet that the gate spends almost no time on that boundary, but I'm flagging it as an assumption, not a result.

Can I train this gate by plain backprop, the way the rest of the network trains? The choice of `k` matters here, and rather than assert it let me actually compute the gradient for small `k`. Take logits `[2.0, 1.0, 0.5, −1.0]` and look at what the softmax-over-survivors does:

- `k = 1`: only the top logit survives, `Softmax([2.0]) = [1.0]`. It's a constant. I differentiate a downstream use of that gate value back to the logits and the gradient is exactly `[0, 0, 0, 0]` — there is literally no signal to `W_g` from this example. Dead.
- `k = 2`: survivors `[2.0, 1.0]`, `Softmax = [0.7311, 0.2689]`. Now differentiate the first gate value: the gradient to the logits comes out `[0.5898, −0.5898, 0, 0]` — genuinely nonzero on both surviving experts. The gate gets a real signal to up-weight the one that helped and down-weight the one that didn't.

So `k = 1` truly kills gate learning and `k ≥ 2` truly revives it; that's not a hunch, the numbers show it. This is the occasionally-sensitive behavior of noisy rectifiers: a unit is off for many inputs but, when it's on, it passes gradient. That's exactly why I want `k > 1` rather than `k = 1`. And notice what I've avoided: I don't need REINFORCE. The selected experts are differentiable; I get low-variance backprop gradients straight through the gate, not a high-variance policy-gradient estimate of a hard choice. That's a clean win over the boolean-gate line.

Now the problem I could already smell — the one that quietly kills naive trainable gating. I train `G` and the experts together, and I think through what happens dynamically: the gate starts, by some small initial accident, to prefer a few experts. Those experts therefore get more examples, so they get more gradient, so they get better, so the gate prefers them even more. It's a self-reinforcing collapse — the gate converges to a state where it always picks the same small clique, and the other experts, never selected, never train, never improve, and can never get picked. The capacity I paid for is wasted, and the gradient can't recover the dead experts because they're never on. Eigen and colleagues saw this and used a hard constraint at the start of training; Bengio and colleagues put a soft constraint on the batch-average of each gate. So I need some force that pushes the gate toward spreading load across all experts.

Let me think about what I'd even want to equalize. Over a batch `X`, define the *importance* of expert `i` as the total gate weight it receives:

    Importance(X)_i = Σ_{x ∈ X} G(x)_i

If the gate is balanced, these are all about equal; if it's collapsed, a few are huge and the rest are near zero. I want a penalty that is large when these values are spread out and zero when they're equal — and it should be scale-free, because the absolute magnitude of importance depends on batch size and on `k`. The natural scale-free measure of spread is the coefficient of variation, `CV = std/mean`. Squaring it gives `CV² = variance / mean²`, a smooth differentiable function. Let me confirm it has the property I want at the endpoints: for `[1, 1, 1, 1]`, variance is `0` so `CV² = 0`; for something collapsed like `[3, 0.01, 0.01, 0.01]` the variance is large and the mean is small, so `CV²` is large and positive. Zero exactly when all entries are equal, growing as they spread — that's the shape I want. So:

    L_importance(X) = w_importance · CV(Importance(X))²

with a hand-tuned weight `w_importance` setting how hard I push. This term is added to the model's loss; backprop through it nudges `W_g` toward giving every expert comparable total weight, which breaks the symmetry that lets a clique run away. Good — that handles the collapse of *importance*.

But wait — I should check whether equal importance is actually the property the hardware cares about. Importance is a sum of gate *weights*. One expert could get a few examples each with a large gate value; another could get many examples each with a small gate value; the two could have identical importance while receiving wildly different example *counts*. Why do I care about counts as well as weights? Because of the batching and the hardware. If I'm sharding experts across devices, an expert that receives far more examples than its peers becomes a straggler — it needs more memory for its activations and it's the bottleneck of the step. Balanced *importance* doesn't guarantee balanced *load*, and load is what the distributed system actually feels.

So I want a second penalty on the number of examples each expert receives. The obvious thing — count how many examples have `G(x)_i > 0` and equalize the counts — runs straight into a wall: a count is a discrete integer. It has zero gradient almost everywhere. I can't backprop through "how many examples landed in the top-k for expert `i`." I need a *smooth* surrogate for the count that I can differentiate.

Here's where I get to reuse something. What is the count, really? It's a sum over examples of an indicator: did expert `i` make the top-k for example `x`? `Σ_x 1[G(x)_i > 0]`. The indicator is the discontinuity. The standard trick to smooth an indicator is to replace it with the *probability* of the event under some noise. And I have a reason to *want* noise in the gate anyway.

Let me motivate the noise on its own terms first, because it does double duty. If my gate is deterministic, the collapse dynamic has no escape hatch: an expert that's currently disfavored is never tried, so the importance loss is pushing against a gate that can only nudge logits, not actually sample alternatives. If instead I inject randomness into the gating logits, then on any given example a normally-disfavored expert occasionally gets bumped into the top-k, gets some examples, gets some gradient — a little exploration that keeps the dead experts alive long enough for the balancing loss to do its work. So I'll add tunable Gaussian noise to each logit *before* the top-k:

    H(x)_i = (x · W_g)_i + StandardNormal() · Softplus((x · W_noise)_i)

The clean signal is `(x · W_g)_i`. To it I add standard-normal noise scaled by a per-expert, per-input standard deviation, `Softplus((x · W_noise)_i)`, with a second trainable matrix `W_noise` controlling how much noise each component gets. Why Softplus on the noise scale? Because a standard deviation must be nonnegative, and Softplus maps any real to a positive number smoothly (unlike a hard ReLU clamp it stays differentiable everywhere and doesn't kill the gradient at zero). The network can learn to be noisy where exploration helps and quiet where it's confident. Then `KeepTopK` and the softmax run on this noisy `H(x)`.

Now the smoothing should fall out. With noise in `H`, whether expert `i` lands in the top-k is a random event, and it has a *probability* — a function of the clean logits and the noise scale. Define `P(x, i)` as the probability that `G(x)_i` is nonzero, taking a fresh draw of the noise on component `i` while holding the already-sampled noise on the other components fixed. When is `G(x)_i` nonzero? Exactly when `H(x)_i` is large enough to be in the top `k` — that is, when `H(x)_i` exceeds the `k`-th greatest of the *other* components of `H(x)`. Call that threshold `kth_excluding(H(x), k, i)`: the `k`-th highest entry of `H(x)` ignoring component `i` itself. So

    P(x, i) = Pr( (x·W_g)_i + StandardNormal()·Softplus((x·W_noise)_i)
                  > kth_excluding(H(x), k, i) )

The only random quantity is the standard normal. Rearranging, the event is

    StandardNormal() > ( kth_excluding(H(x), k, i) − (x·W_g)_i ) / Softplus((x·W_noise)_i)

and the probability that a standard normal exceeds some value `t` is `1 − Φ(t) = Φ(−t)`. So

    P(x, i) = Φ( ( (x·W_g)_i − kth_excluding(H(x), k, i) ) / Softplus((x·W_noise)_i) )

with `Φ` the standard normal CDF. It's smooth and differentiable in `W_g` and `W_noise`, and it slides from 0 (clean logit far below the threshold) to 1 (far above). The noise standard deviation in the denominator is what makes this a soft step instead of a hard one; with zero noise it collapses back to the indicator and loses its gradient, which is precisely why the noise had to be there for the load estimate to be differentiable.

Before I trust this, I should pin down the one part I keep hand-waving: the threshold `kth_excluding`. It depends on whether expert `i` is *currently* in the top-k. Let me work a concrete example with `n = 4`, `k = 2`, one input. Clean logits `[2.0, 1.0, 0.5, −1.0]`, noise stddevs `[0.5, 0.5, 0.5, 0.5]`, and one noise draw `[0.3, −0.2, 1.2, 0.1]`, giving noisy logits `H = [2.15, 0.90, 1.10, −0.95]`. The top `k+1 = 3` noisy values, descending, are `[2.15, 1.10, 0.90]`, so the selected top-2 are experts `0` and `2`.

My first instinct for the two thresholds: if `i` is in the top-k it had to clear the `k`-th best overall, and if it's out it has to clear the `(k+1)`-th best. So for an *in* expert use `top[k−1] = 1.10`, for an *out* expert use `top[k] = 0.90`. Let me test that against ground truth. I'll Monte-Carlo the actual event for each `i`: resample only `i`'s noise, hold the other three noisy logits fixed at `H`, and measure how often `i` lands in the top-2. The honest bar for `i` is the `k`-th largest among the *other* three components. For expert `0` (in), the other three are `[0.90, 1.10, −0.95]`, whose 2nd-largest is `0.90`; the brute-force probability comes out `≈ 0.986`. But my formula with `top[k−1] = 1.10` gives `Φ((2.0 − 1.10)/0.5) = 0.964` — close but the bar I used (`1.10`) doesn't even match the true bar (`0.90`). For expert `1` (out) the formula gives `0.579` against a brute-force `0.421`, and they're on opposite sides of `0.5`. So my threshold assignment is *wrong* — I had the two cases swapped.

Working out why: if `i` is currently *in* the top-k, then removing it from contention drops the bar, because the `k`-th best of the remaining `n−1` is the element just below the current top-k — that's the `(k+1)`-th best overall, `top[k] = 0.90`. If `i` is currently *out*, the `k`-th best of the others is just the `k`-th best overall, `top[k−1] = 1.10`. So it's the opposite of what I wrote: *in* → `top[k]`, *out* → `top[k−1]`. Re-running the check with this correction, for all four experts the formula matches the Monte-Carlo probability to three decimals: expert 0 (in) `0.986` vs `0.986`, expert 1 (out) `0.421` vs `0.421`, expert 2 (in) `0.212` vs `0.212`, expert 3 (out) `0.000` vs `0.000`. Now I believe both the `Φ`-formula and the threshold bookkeeping. That's also why the code grabs `k+1` values rather than `k`, and why in 0-based indexing the "if I'm in" threshold is at position `k` (the `(k+1)`-th value) and the "if I'm out" threshold is at position `k−1` (the `k`-th value). Getting that index off by one would have silently poisoned the load estimate, so I'm glad I traced it.

Now define the smooth load and its penalty, mirroring importance:

    Load(X)_i = Σ_{x ∈ X} P(x, i)
    L_load(X) = w_load · CV(Load(X))²

So I have two balancing losses with the same `CV²` shape: `L_importance` equalizes total gate weight, `L_load` equalizes the (smoothly-estimated) number of examples. Do I need both? Let me reason about the corner cases. Importance alone leaves the count-imbalance I described — fine for quality, bad for the distributed load. Load alone equalizes counts but could let the gate weights be lopsided. I'd expect either one alone to already get quality to roughly the same place (both attack the same collapse), with no balancing loss at all being much worse — the gate collapses, the CV of importance blows up, and one expert ends up with many times the average load. I haven't run the full training to confirm the quality is *identical* across the two single-loss choices, so I'll treat that as a claim to check empirically; what I'm confident of from the construction is that using both, with modest weights, targets both balanced weights and balanced counts. They're cheap insurance, so I'll keep both and tune the two `w`'s together.

One more thing about the noise I almost missed. At the very start of training, before either balancing loss has had time to act, a random initial imbalance could already overload one expert and blow past its memory budget — an out-of-memory crash before the soft constraints even engage. The fix is in the initialization: set both `W_g` and `W_noise` to zero. Then at step zero the clean logits are all zero and the only thing distinguishing experts is the noise — pure symmetric randomness — so every expert gets roughly equal load in expectation from the outset, and the balancing losses take over from a safe starting point.

Now back to the performance wall I deferred — the shrinking batch. If the gate sends `k` of `n` experts per example, then over a batch of `b` examples each expert sees only about `kb/n` examples, which for large `n` is `≪ b`. The experts are starved of the big batches that make the hardware efficient. Making `n` bigger makes this worse. So I need each expert's batch to be large in absolute terms even though it's a small fraction.

The trick is to combine two kinds of parallelism. In ordinary distributed training I'd run `d` copies of the model on `d` devices, each chewing its own batch of `b`, synchronizing through parameter servers — pure data parallelism. Here I keep the standard layers and the gating network data-parallel as usual, but I keep only *one* shared copy of each expert, sharded across the devices — the experts are *model*-parallel. The `d` data-parallel batches run synchronously, and for the MoE layer I combine them: each expert receives the relevant examples gathered from *all* `d` input batches. So instead of `kb/n` examples, each expert now gets about `k·b·d/n` — a factor of `d` larger. The same devices act as data-parallel replicas for the standard layers and as model-parallel shards for the experts. And this is what lets the whole thing scale gracefully: to add experts (parameters), I add devices in proportion; the per-expert batch stays constant, the per-device memory and bandwidth stay constant, the step time stays constant. Capacity grows with the cluster at fixed cost per device.

Two more batch-size boosts. Since I apply the same MoE to every timestep of the sequence, I can wait for the previous layer to finish and then hand the MoE *all* the timesteps at once as one big batch — a free multiplicative factor equal to the number of unrolled steps, just from the convolutional structure. (If I ever made the MoE *recurrent* — replacing an LSTM's weight matrices — this trick breaks, because each step's input depends on the previous step's MoE output; then I'd lean on activation-recomputation to claw back the memory for a big batch.)

And the bandwidth wall: because the experts are stationary on their shards, the only thing crossing the network is each expert's inputs and outputs, not its parameters. For this to stay compute-bound rather than bandwidth-bound, the ratio of an expert's compute to the size of its input-plus-output must exceed the device's compute-to-bandwidth ratio — thousands to one on a GPU. Let me actually compute that ratio for a one-hidden-layer feedforward expert. Per example the multiply-adds are `input_size · hidden_size` in the first matrix plus `hidden_size · output_size` in the second; the bytes that cross the network are the input and output vectors, `input_size + output_size` elements. The ratio is `hidden_size · (input_size + output_size) / (input_size + output_size) = hidden_size`. So with `input = output = 512` and `hidden = 1024`, compute is `512·1024 + 1024·512 = 1{,}048{,}576` and IO is `512 + 512 = 1024`, ratio `1024` — exactly the hidden size; and with `hidden = 4000` the ratio is `4000`. So the compute-to-io ratio *is* the hidden size, which means I can buy back efficiency simply by making the experts' hidden layer wider. Convenient: the same knob that adds capacity also raises the arithmetic intensity past the device's compute-to-bandwidth threshold.

If `n` gets really large, even the top-k over `n` logits and the branching factor get unwieldy. I can fold the experts into a two-level hierarchy: a primary gate picks among `a` groups, and each group is itself a little mixture with its own secondary gate over `b` experts. The output multiplies the two gate levels:

    y_H = Σ_i Σ_j G_primary(x)_i · G_i(x)_j · E_{i,j}(x)

Importance generalizes the obvious way — `Importance_H(X)_{i,j} = Σ_x G_primary(x)_i · G_i(x)_j`. The load is the one place I have to be careful. The tempting definition is "the load on expert `(i,j)` is just the secondary load within group `i`," `Load_i(X^{(i)})_j` over the examples routed to group `i`. But that quantity has no gradient with respect to the *primary* gate — the primary gate decides *which* examples reach group `i`, and I'd be ignoring that dependence, so the primary gate would never get a balancing signal and the groups themselves could collapse. So instead I weight the secondary load by the primary load and normalize by the number of examples that reached the group:

    Load_H(X)_{i,j} = Load_primary(X)_i · Load_i(X^{(i)})_j / |X^{(i)}|

so the gradient flows back to the primary gate too. With `k = 2` at each level I keep the per-level branching small while reaching thousands of experts.

Let me now put the layer together in code. The skeleton is: a gating method that returns the sparse gate matrix and the smooth load vector; a way to route examples to only their selected experts and recombine; and the two balancing losses. I'll write the routing with a small dispatcher that, given the gate matrix, builds a compact per-expert batch from the nonzero entries, runs each expert once on its batch, and scatter-adds the weighted outputs back. This mirrors a real implementation rather than evaluating every expert on every example (which would defeat the entire purpose).

```python
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class SparseDispatcher:
    """Builds a compact input batch per expert from the nonzero gate entries,
    and recombines expert outputs weighted by the gates. Only the examples with
    gates[b, i] > 0 are ever sent to expert i — this is where the compute saving
    is actually realized."""

    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        # Tensor2Tensor orders pairs from transpose(gates): expert-major,
        # then batch-major within each expert.
        where = torch.nonzero(gates.t() > 0, as_tuple=False)
        self._expert_index = where[:, 0]
        self._batch_index = where[:, 1]
        self._part_sizes = (gates > 0).sum(0).tolist()      # examples per expert
        self._nonzero_gates = gates[self._batch_index, self._expert_index].unsqueeze(1)

    def dispatch(self, inp):
        # gather each expert's slice; split into one tensor per expert
        inp_exp = inp[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:                                # weight by G(x)_i
            stitched = stitched.mul(self._nonzero_gates)
        zeros = stitched.new_zeros(self._gates.size(0), stitched.size(1))
        # y[b] = sum over its selected experts of G(x)_i * E_i(x)
        return zeros.index_add(0, self._batch_index, stitched.float())

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class Expert(nn.Module):
    """One feed-forward expert: a wide ReLU hidden layer. Widening hidden_size
    is the knob that both adds capacity and raises arithmetic intensity."""

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size,
                 noisy_gating=True, k=4):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList(
            [Expert(input_size, output_size, hidden_size) for _ in range(num_experts)])
        # zero-init both gates: at step 0 only the noise distinguishes experts,
        # so the load starts balanced and nothing overflows before the losses act
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        # squared coefficient of variation = var / mean^2; zero iff all equal
        eps = 1e-10
        if x.numel() <= 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        x = x.float()
        return x.var(unbiased=False) / (x.mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)                            # true discrete count

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        # smooth P(x, i): probability the noisy logit clears the top-k threshold,
        # with two thresholds depending on whether i is currently in the top-k.
        # 'if in' uses position k (the (k+1)-th value); 'if out' uses k-1 (the k-th)
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)   # Phi((clean - thr)/std)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate                       # (x . W_g)_i
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits                            # H(x): clean + N(0,1)*softplus
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        # KeepTopK: zero everywhere except the k selected experts
        gates = torch.zeros_like(clean_logits).scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev,
                                       top_logits).sum(0)    # smooth Load(X)_i
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)                            # Importance(X)_i
        loss = self.cv_squared(importance) + self.cv_squared(load)  # both balancing losses
        loss *= loss_coef                                    # w_importance = w_load = loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)               # only k*b/n examples per expert
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)               # y = sum_i G(x)_i E_i(x)
        return y, loss
```

So the chain is: capacity is bounded by parameters but dense compute is bounded by the same parameters, so to grow one without the other I route each example to only `k` of `n` experts and skip the rest — `y = Σ_i G(x)_i E_i(x)` with a sparse `G`. To get sparsity I keep the top `k` logits and `−∞` the rest before the softmax, and I take `k > 1` because the `k = 1` softmax is constant and passes no gradient while `k ≥ 2` does (I checked the gradient explicitly), so the gate trains by plain backprop instead of a high-variance policy gradient. Trained naively the gate collapses onto a favored clique, so I add `CV²` penalties on per-expert importance (total gate weight) and on load (number of examples); the example count is discrete, so I make it differentiable by injecting Gaussian noise into the logits and using the probability `Φ((clean − threshold)/stddev)` that a noisy logit clears the top-k bar — a formula I checked against a Monte-Carlo estimate on a small example, which also caught me assigning the in/out thresholds backwards. The same noise gives the exploration that keeps disfavored experts alive, and zero-initializing the gates keeps load balanced before the penalties engage. Finally, to feed the experts the big batches the hardware needs, I make the standard layers data-parallel and the experts model-parallel so each expert pools examples across all devices, and I widen the experts to keep them compute-bound rather than bandwidth-bound — the compute-to-io ratio works out to exactly the hidden size — so capacity scales with the cluster at fixed cost per device.
