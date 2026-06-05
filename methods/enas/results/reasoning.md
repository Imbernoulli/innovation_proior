Let me start from what actually hurts about searching for neural architectures. I have a controller — an LSTM with parameters θ — that defines a distribution π(m; θ) over candidate architectures m. It samples one, I build that child network, I train it, I measure how good it is, and I push θ toward the architectures that scored well. The reinforcement-learning version of this works: the architectures it finds are competitive with the best ones experts design by hand, on image classification and on language modeling. So the formulation is right. The problem is the price tag. To score *one* candidate, I initialize a fresh network with random weights, train it to convergence, read off a single number — its validation accuracy — and then I throw all those trained weights in the trash. Then the controller samples the next candidate and I do it all again from scratch. Thousands of candidates, each a full from-scratch training run. That's how a single search ends up eating 450 GPUs for three or four days — tens of thousands of GPU-hours — and if I try to economize by training each child for fewer epochs or sampling fewer candidates, the architectures I find get noticeably worse. So I can't just turn the budget knob down.

Where exactly is the money going? Not the controller — the controller is a small LSTM, sampling a sequence of tokens is cheap, the REINFORCE update is cheap. It's the inner loop: "train this child network to convergence." And the part of that which feels most wasteful is the *from scratch*. Every single child rediscovers, from random initialization, how to do convolution, how to represent edges and textures, the whole low-level substrate — only to be measured once and discarded. I am paying full price for weights I keep for exactly one evaluation. If I could stop re-learning weights from scratch for every candidate, the whole thing would collapse in cost. That's the lever.

So let me poke at the assumption that each child needs its own weights. Why do I believe that? Reflex says different architectures are different functions, so of course they need different parameters. But two things I know push back. Transfer learning: weights trained for one model on one task transfer to other models on other tasks with little or no modification — parameters are not as architecture-specific as the reflex assumes. And weight inheritance in neuro-evolution: when you mutate an architecture, you don't reinitialize the child, you copy the parent's weights and fine-tune briefly, and it works. So weights *can* be reused across architectures. The evolutionary version reuses them only locally — child must be a small mutation of its parent. What if I went all the way and let *every* architecture in the search space share *one* set of weights?

Let me see if that even makes sense, or if it's nonsense. I have an enormous space of candidate networks. The standard cell-search trick is illuminating here: instead of designing a whole network, design a small cell — pick, for each of a handful of nodes, which earlier nodes feed it and which operation to apply — then stack copies of the cell. So a candidate is really a set of *choices over a fixed menu of components*: node 3 takes input from node 1 and applies a 5×5 separable conv; node 4 takes input from node 2 and applies max-pool; and so on. Now here's the thing I keep circling back to. Across all the candidates in the space, the menu of components is the *same*. "5×5 separable conv applied to the output of node 1" is a component that appears, identically, in a huge number of different candidate architectures. They differ in *which* components they wire together, not in *what the components are*.

Stare at that. If I draw every possible component as an edge in one big graph — a node for each computation slot, an edge for "the output of node i, transformed by operation o, feeds node j" — then the entire search space is a single big directed acyclic graph, and every candidate architecture is just a *subgraph* of it: a particular choice of which edges are active. The controller's job, recast, is to pick a subgraph. And now the weight-sharing idea isn't a leap, it's almost forced: give each edge of the big graph its own parameters — the weight matrix for "op o from node i to node j" lives on that edge — and a candidate architecture, by selecting which edges it activates, automatically selects *which* of these shared parameters it uses. Two different architectures that both happen to use the 5×5-separable-conv-from-node-1 edge use the *same* weight matrix for it. The superposition of all child models over one DAG, with parameters living on the edges and shared by every child that activates that edge. Call the shared child-model weights ω, kept once, for the whole search.

The cost picture changes completely. I no longer train one network per candidate. I keep one big pool of weights ω. When the controller samples an architecture m, I don't reinitialize anything — m just tells me which subset of ω to use, and that subset is *already partially trained* from every previous candidate that used those same edges. No from-scratch training per candidate. That's the 1000× I'm after, if it works.

Now I have to actually pin down two things: how the controller samples a subgraph, and how I train two coupled sets of parameters — the controller's θ and the shared weights ω.

Take the recurrent-cell case first because it's the cleanest. I have a DAG with N nodes. I want the controller to decide both the *topology* (who connects to whom) and the *operations* (which nonlinearity at each node) — both, not just the operations on a fixed tree, because fixing the topology throws away half the design freedom. So at node 1, the input is the cell's input x_t together with the previous hidden state h_{t-1}; the controller samples an activation function, say tanh, and node 1 computes h_1 = tanh(x_t·W^(x) + h_{t-1}·W_1^(h)). At node ℓ > 1, the controller samples a previous index j < ℓ and an activation function f; node ℓ computes h_ℓ = f(h_j · W_{ℓ,j}^(h)). The crucial detail for sharing: for every ordered pair j < ℓ there is an *independent* matrix W_{ℓ,j}^(h), and by picking the previous index j, the controller is also picking *which matrix gets used*. So the parameters live on the edges (ℓ, j), and a cell's choice of edges is a choice of which W's it touches. For the cell's output, I average all the loose ends — the nodes nobody used as an input — so nothing computed is wasted. With four activation choices and N nodes there are 4^N · N! configurations; at N = 12 that's about 10^15 cells, all sharing the one pool of W's.

The controller itself: an LSTM, small, say 100 hidden units. It emits the decisions autoregressively — softmax over choices at each step, and the decision it just made is fed back in as the embedding for the next step, so later decisions are conditioned on earlier ones (which input you picked should inform which activation, etc.). At the first step there's no previous decision, so it gets an empty/zero input embedding.

Now the training. Two parameter sets, θ (controller) and ω (shared child weights), and I'll alternate two phases.

Phase for ω: fix the controller's policy π(m; θ) and do SGD on ω to reduce the expected loss over architectures the controller currently likes,

  minimize over ω of  E_{m ~ π(m;θ)} [ L(m; ω) ],

where L(m; ω) is the ordinary cross-entropy (or LM loss) of child m, computed on a training minibatch, using the shared weights ω restricted to m's active edges. I can't differentiate an expectation over a discrete distribution directly, but I don't need to — I'm differentiating with respect to ω, not θ, and for a *fixed* sampled m the loss L(m; ω) is perfectly differentiable in ω. So I get a Monte-Carlo estimate:

  ∇_ω E_{m~π}[ L(m; ω) ] ≈ (1/M) Σ_{i=1}^M ∇_ω L(m_i; ω),  m_i ~ π(m; θ).

This is unbiased. How many samples M do I need per ω-update? My instinct says many, because each sampled architecture only exercises a sliver of ω and the per-sample gradient is high-variance compared to ordinary fixed-architecture SGD. Let me just try M = 1 — update ω using the gradient from a *single* architecture sampled fresh each minibatch. That should be too noisy... and yet it's fine. It works. Which, once I see it, makes sense: over a whole epoch I draw a different architecture every minibatch, so across the epoch ω gets gradient signal from a broad spread of subgraphs, and the per-minibatch architecture-noise averages out the same way minibatch-sampling noise already does. So I don't need the expensive M; one sample per step, and I train ω over a full pass through the training data.

Phase for θ: now fix ω and update the controller to prefer architectures that score well. The score is a reward R(m, ω), and I want to maximize the expected reward

  maximize over θ of  E_{m ~ π(m;θ)} [ R(m, ω) ].

Here the dependence on θ *is* through the discrete sampling, so I can't backprop through it — this is exactly the REINFORCE situation. The score-function estimator gives

  ∇_θ E_{m~π}[ R ] = E_{m~π}[ R(m, ω) · ∇_θ log π(m; θ) ],

which needs only that I can sample m and evaluate the scalar R — R need not be differentiable, which is good because it won't be. This estimator is high-variance, so I subtract a baseline b: use (R − b)·∇_θ log π, with b an exponential moving average of recent rewards. The baseline doesn't bias anything, because E[ b · ∇_θ log π ] = b · ∇_θ E[1] = b·∇_θ Σ_m π(m;θ) = b·∇_θ 1 = 0 — subtracting any constant-in-m quantity leaves the gradient's expectation unchanged while shrinking its variance. Concretely I keep a scalar baseline and slide it toward each new reward, b ← b − (1 − decay)·(b − R), and the controller's surrogate loss is −log π(m;θ)·(R − b), whose gradient is the negative of the REINFORCE gradient I want to ascend. I'll use Adam for θ.

What should the reward R be, and crucially — measured on which data? If I reward training-set performance, the controller will happily discover architectures that overfit the training split. I want architectures that *generalize*. So R is measured on the *validation* set, held out from the data ω trains on. For language modeling, perplexity is what I care about but it's a "lower is better" quantity, so I turn it into a reward with c / valid_ppl; for image classification, accuracy on a validation minibatch is already "higher is better," so R is just that. This split — ω learns on the training data, θ is rewarded on validation data — is what keeps the search honest.

So the loop is: train ω for a whole pass over the training data (one sampled architecture per minibatch, M = 1), then freeze ω and train θ for a couple thousand steps (each step: sample an architecture, evaluate its validation reward, REINFORCE-update θ), then alternate.

One thing I should worry about: the controller collapsing too early onto a narrow set of architectures, before ω has had a chance to make a broad range of edges competent — a rich-get-richer trap where the first few architectures that look good get all the training and everything else stays untrained and therefore looks bad forever. I want to keep the policy exploratory for a while. Two standard knobs. Add the controller's sample entropy to the reward, weighted by a small coefficient, so the policy is rewarded for staying uncertain — fights premature collapse. And tame the logits before sampling: divide them by a temperature (> 1, flatter distribution, more exploration) and squash them with a scaled tanh, logit ← c·tanh(logit), so no single choice's probability can run away to near-1 early. These keep the sampling spread out long enough for ω to become a fair judge of many subgraphs.

Now the convolutional spaces. Two flavors.

Macro: search the *entire* network, layer by layer. At layer k the controller makes two decisions. First, which previous layers to connect to — it can pick any subset of the k−1 earlier layers, and their outputs get concatenated along the channel dimension and fed to layer k. That's the skip-connection mechanism, and it gives 2^{k-1} possible connection patterns at layer k. Second, which operation this layer computes, from a menu of six: 3×3 and 5×5 convolutions, 3×3 and 5×5 depthwise-separable convolutions, and 3×3 max-pool and 3×3 average-pool. Each operation at each layer carries its own parameters, again shared across all child networks that activate it. With L layers, that's 6^L · 2^{L(L−1)/2} networks; at L = 12, about 1.6×10^29. Because picking subsets of skip connections freely can leave the network sparsely or densely connected somewhat at random, I add a gentle pressure toward a target connection density: a KL term between the controller's per-pair skip probability and a chosen prior ρ (around 0.4), added to the reward — so the search prefers a sane amount of skip structure rather than degenerate all-or-nothing wiring.

Micro: don't search the whole network — search a small *cell* and stack it, the way the cell-search line does, which shrinks the space enormously. The DAG has B nodes. Nodes 1 and 2 are the cell's inputs (the outputs of the two preceding cells in the stacked network). For each remaining node, the controller samples *two* previous nodes and *two* operations (from a menu of five: identity, 3×3 and 5×5 separable conv, 3×3 max-pool, 3×3 avg-pool), applies each operation to its chosen input, and *adds* the two results — that's the node's output. Loose ends are concatenated to form the cell's output. A reduction cell — which halves spatial resolution — is realized from the same space by applying every operation with stride 2; following the cell-search convention I sample the reduction cell conditioned on the conv cell, so the controller runs for 2(B−2) blocks total. Counting: at node i the controller picks 2 of i−1 inputs and 2 of 5 ops, and doing this for the conv cell and (independently) the reduction cell gives (5 × (B−2)!)^4 cells; at B = 7, about 1.3×10^11 — far smaller than the macro space, which is the point of cells.

Deriving the final architecture once search is done: sample several architectures from the trained π(m; θ), score each on a single validation minibatch, and take the single best one to re-train from scratch with full settings. I could instead train *all* the sampled candidates from scratch and pick the best on a separate validation set — that squeezes out a bit more — but it reintroduces exactly the from-scratch cost I was trying to kill, and taking just the top-1-by-shared-weights candidate gets nearly the same result far more cheaply. So: sample a few, pick the best by shared-weight reward, retrain that one.

One refinement on the recurrent cell that I want, because the bare h_ℓ = f(h_j·W) transitions are shallow. Borrow the highway-gate idea from the recurrent-highway-network line: instead of h_2 = ReLU(h_1·W), use a gated mix h_2 = c_2 ⊙ ReLU(h_1·W) + (1 − c_2) ⊙ h_1 with gate c_2 = sigmoid(h_1·W^(c)). The gate lets the cell carry a value through unchanged or transform it, which stabilizes the deeper recurrent transitions the search wants to build and gives gradients a clean path — the same reason highway/residual connections help everywhere.

Let me write the controller as the autoregressive sampler it is. The shape that matters: at each decision it runs the LSTM, forms a softmax over the current choice set, samples, records the log-probability and the entropy of that choice, and feeds the sampled choice's embedding in as the next input. Topology decisions (which previous index, in the conv macro space realized as a per-previous-layer binary "skip or not") and operation decisions are emitted in exactly this interleaved way.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(nn.Module):
    """Autoregressive LSTM policy pi(architecture; theta). Samples discrete
    decisions one at a time, feeding each decision back in as the next input.
    Returns the architecture, the summed log-prob of its decisions, and the
    summed entropy (used to keep exploration alive)."""

    def __init__(self, num_ops, num_nodes, hidden=100,
                 temperature=5.0, tanh_constant=2.5):
        super().__init__()
        self.hidden, self.num_ops, self.num_nodes = hidden, num_ops, num_nodes
        self.temperature, self.tanh_constant = temperature, tanh_constant
        self.lstm = nn.LSTMCell(hidden, hidden)
        self.g_emb = nn.Parameter(torch.zeros(1, hidden))   # empty first input
        self.op_emb = nn.Embedding(num_ops, hidden)         # feed chosen op back
        self.op_soft = nn.Linear(hidden, num_ops)           # op softmax head
        # attention-style head over previous nodes' hidden states picks the input edge
        self.w_prev = nn.Linear(hidden, hidden, bias=False)
        self.w_curr = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)

    def _sample_logit(self, logit):
        logit = logit / self.temperature
        logit = self.tanh_constant * torch.tanh(logit)      # keep early probs tame
        probs = F.softmax(logit, dim=-1)
        dist = torch.distributions.Categorical(probs)
        choice = dist.sample()
        return choice, dist.log_prob(choice), dist.entropy()

    def sample(self):
        h = torch.zeros(1, self.hidden); c = torch.zeros(1, self.hidden)
        x = self.g_emb
        arc, log_probs, entropies, anchors = [], [], [], []
        for node in range(self.num_nodes):
            # 1) which previous node feeds this one (an edge => which shared W)
            h, c = self.lstm(x, (h, c))
            if anchors:                                     # node 0 has no predecessors
                query = torch.tanh(torch.cat(anchors, 0) + self.w_curr(h))
                logit = self.v(query).transpose(0, 1)       # score each prev node
                prev, lp, ent = self._sample_logit(logit)
                arc.append(int(prev)); log_probs.append(lp); entropies.append(ent)
                x = anchors[int(prev)]
            anchors.append(self.w_prev(h))
            # 2) which operation at this node
            h, c = self.lstm(x, (h, c))
            op, lp, ent = self._sample_logit(self.op_soft(h))
            arc.append(int(op)); log_probs.append(lp); entropies.append(ent)
            x = self.op_emb(op)
        return arc, torch.stack(log_probs).sum(), torch.stack(entropies).sum()


class SharedChild(nn.Module):
    """The single DAG holding ALL child models in superposition. Every edge
    (op o, predecessor j -> node) owns its own parameters; an architecture
    selects WHICH of these shared parameters it activates. No per-candidate
    weights, no from-scratch training."""

    def __init__(self, num_ops, num_nodes, channels):
        super().__init__()
        # one parameter set per (node, predecessor, op) edge -- shared across
        # every architecture that activates that edge
        self.edges = nn.ModuleDict()
        for node in range(num_nodes):
            for prev in range(node):
                for op in range(num_ops):
                    self.edges[f"{node}_{prev}_{op}"] = make_op(op, channels)

    def forward(self, x, arc):
        # arc lists, per node, (predecessor, op); activate only those edges
        h = [x]
        for node, (prev, op) in enumerate(parse(arc), start=1):
            h.append(self.edges[f"{node}_{prev}_{op}"](h[prev]))
        loose = loose_ends(arc, len(h))                     # nodes used by nobody
        return sum(h[i] for i in loose) / len(loose)        # average the loose ends


def train_shared_weights(controller, child, train_loader, opt_omega):
    """Phase 1: fix theta, SGD on the shared weights omega. One architecture
    sampled per minibatch (M = 1 is enough -- the epoch averages over many)."""
    for inputs, targets in train_loader:
        with torch.no_grad():
            arc, _, _ = controller.sample()
        opt_omega.zero_grad()
        loss = F.cross_entropy(child(inputs, arc), targets)  # L(m; omega)
        loss.backward()
        opt_omega.step()


def train_controller(controller, child, valid_loader, opt_theta,
                     baseline, entropy_weight=1e-4, bl_dec=0.99):
    """Phase 2: fix omega, REINFORCE on theta with a moving-average baseline.
    Reward is measured on the VALIDATION split to select for generalization."""
    for _ in range(2000):
        arc, log_prob, entropy = controller.sample()
        inputs, targets = next(iter(valid_loader))
        with torch.no_grad():
            reward = accuracy(child(inputs, arc), targets)   # R(m, omega), val data
        baseline = baseline - (1 - bl_dec) * (baseline - reward)
        loss = -log_prob * (reward - baseline) - entropy_weight * entropy
        opt_theta.zero_grad(); loss.backward(); opt_theta.step()
    return baseline


def search(controller, child, train_loader, valid_loader, opt_omega, opt_theta):
    baseline = 0.0
    for epoch in range(num_epochs):
        train_shared_weights(controller, child, train_loader, opt_omega)
        baseline = train_controller(controller, child, valid_loader,
                                    opt_theta, baseline)
    # derive: sample a few architectures, keep the best by validation reward,
    # retrain that single one from scratch with full settings
    cands = [controller.sample()[0] for _ in range(num_samples)]
    return max(cands, key=lambda a: reward_on_minibatch(child, a, valid_loader))
```

The causal chain: the cost of architecture search lived entirely in training every candidate from scratch and discarding its weights; transfer learning and weight inheritance said weights aren't as architecture-bound as assumed; recasting the search space as one big DAG made every candidate a *subgraph* of a single graph, so putting parameters on the edges lets all children share one pool of weights ω and a candidate's choice of edges automatically selects its weights; training then splits into alternating phases — SGD on ω over the training data using one sampled architecture per minibatch, and REINFORCE on the controller θ with a moving-average baseline, rewarded on the *validation* split so it selects for generalization, with entropy and tempered logits to keep exploration alive — and at the end the best subgraph is retrained from scratch, all on a single GPU in under a day instead of hundreds of GPUs for days.
