OK, let me think this through from scratch. The fact I keep circling back to is the shape of the scaling curves. Quality climbs as a power law in three things — parameters, data, compute — over many orders of magnitude, and when I stare at the parameter axis specifically, it looks like it contributes to quality in a way that's separable from the raw arithmetic I do per token. That's a strange and tantalizing thing to notice, because in the dense Transformer I work with every day, those two are the same knob. Every parameter fires on every token. So if I want more parameters, I pay more FLOPs per token, and at the scales I care about that FLOP bill — energy, wall-clock, memory per accelerator — is exactly the thing that's pinning me down. I can't just keep widening the model.

So the goal crystallizes: unweld parameter count from per-token compute. I want a Transformer that *holds* an enormous number of parameters but only *touches* a tiny, input-dependent slice of them for any given token. Then capacity is the whole bank; the FLOPs are just the slice that ran.

I'm not starting from nothing here — there's a known object that already does this, the Mixture-of-Experts layer. A set of expert sub-networks, each just an ordinary feed-forward net, sitting behind a little router. For a token \(x\), the router emits logits \(h(x)=W_r x\), softmaxes them into a gate \(p_i(x)=e^{h(x)_i}/\sum_j e^{h(x)_j}\), and you evaluate the top-\(k\) experts and combine them weighted by their gate values: \(y=\sum_{i\in\mathcal{T}}p_i(x)E_i(x)\). With \(k\ll N\), the per-token FLOPs are those of \(k\) experts, flat in the total expert count \(N\). That's precisely the decoupling I want. It works — it's given state-of-the-art translation and language modeling. And yet almost nobody uses it. So before I build anything, I want to be honest about *why* it hasn't caught on, because those reasons are going to constrain every choice I make. Three things: it's complex, it's communication-heavy, and it's unstable to train. Let me take them seriously one at a time, and let the simplifications fall out of the pain.

Start with the routing rule itself, the top-\(k\). The accepted wisdom — stated outright when this layer was introduced — is that \(k\) has to be at least 2. The argument was that with a single expert the router gets no useful gradient: that to *learn* to route you need to compare at least two experts, see which did better, and shift mass accordingly. And a later study reinforced this, finding higher \(k\) in lower layers mattered when you stack many routing layers. So the field has settled on \(k=2\) as a kind of floor.

But step back and ask what \(k=2\) actually costs me, given my three pains. Two experts per token means: two expert FFNs evaluated per token, so more FLOPs than the minimum. Two selections to compute in the router. And — this is the one that really bites — each token has to be physically *shipped* to two experts, which may live on two different accelerators, and its two outputs gathered back. Routing on a distributed mesh is an all-to-all shuffle, and top-2 doubles the traffic of that shuffle relative to sending each token to just one place. Communication is one of my three named enemies, and \(k=2\) is paying double into it for reasons I haven't actually verified.

So let me poke at the justification. Is \(k\ge 2\) really necessary for a trainable router? The claim is that with \(k=1\) the router gradient vanishes. Let me write down what \(k=1\) would even be. Token \(x\), pick the single argmax expert \(i\), output

    y = p_i(x) · E_i(x),    i = argmax_j p_j(x).

Now where does the gradient to the router \(W_r\) come from? The selection — the argmax — is discontinuous, sure, no gradient flows through *which* expert was chosen. But look at the output: I didn't just route to \(E_i\), I multiplied its output by the gate value \(p_i(x)\). And \(p_i(x)\) is a perfectly smooth softmax of \(W_r x\). So \(\partial y/\partial W_r\) flows through that scalar \(p_i(x)\) multiplying the expert's contribution. The loss can push \(p_i(x)\) up or down, and that's a real, nonzero gradient on the router weights, exactly as it is in the top-\(k\) case — the gate value sits in the graph whether \(k\) is 1 or 2. The "need to compare two experts" intuition is about exploring *alternatives*, not about whether a gradient exists. The gradient exists with \(k=1\). The conjecture conflated "the router can't improve" with "the router can't get a gradient," and only the second would actually block training.

That feels almost too easy, so let me make sure I'm not fooling myself about exploration. With \(k=1\) I only ever see the chosen expert; I get no counterfactual signal about how the *other* experts would have done on this token. That's a genuine exploration problem — it's the structure of a contextual bandit, really: I pick an arm, I see only that arm's outcome. But it's a problem I can attack separately, with a little exploration noise, not a reason the whole thing can't learn. I'll come back to the noise. The headline is: \(k=1\) is differentiable. If that holds up empirically, then routing to a single expert simplifies the implementation, halves the router's selection work, lets me make each expert's buffer smaller, and — the prize — halves the all-to-all communication. Every one of my three pains eases at once. So I'll commit to it and call this single-expert layer a Switch. Maximal sparsity: one expert per token.

Now the second pain made concrete, which is really a hardware fact I can't wish away. On a TPU the tensor shapes must be known at compile time. But routing is *dynamic* — how many tokens land on a given expert depends on the data and on the current router, and it will be uneven. I can't have a `[variable]`-sized expert batch. So I have to fix, ahead of time, how many tokens each expert will process. Call it the expert capacity. The natural baseline is "everybody gets an equal share": if there are \(T\) tokens in the batch and \(N\) experts, the even split is \(T/N\) tokens each. But routing is never perfectly even, so if I set capacity to exactly \(T/N\), the moment one expert is even slightly over-subscribed I'm in trouble. So I pad it with a buffer — a *capacity factor* \(\ge 1\):

    expert_capacity = (tokens_per_batch / num_experts) × capacity_factor.

A capacity factor above 1 gives slack to absorb imbalance. But it's not free: every slot I reserve and don't fill is padded, wasted compute and wasted memory. So there's a knob with a real trade-off — too small and I overflow constantly, too big and I burn compute on empty pad.

And what do I do when an expert *does* overflow — more tokens routed to it than its capacity? I can't grow the buffer at runtime. The clean answer: drop the overflow tokens *from this layer*. Concretely, "drop" means I skip the expert computation for those tokens and let their representation pass straight to the next layer through the residual connection. That's not a hack; it's mathematically exact — a skipped token's output is just the identity, which is what the residual already provides. So an overflowed token isn't corrupted, it's merely un-improved by this one layer. And here's where the \(k=1\) choice pays off a second time: with top-1, each token claims one slot instead of two, so for the same drop rate I need only half the capacity — the per-expert buffer halves, which means less pad compute, less memory, and less to shuffle. Top-1 and small capacity are natural allies. In fact I'd expect the Switch to *prefer* low capacity factors, which is exactly the regime I care about at huge scale where memory is scarce.

But dropping tokens is only tolerable if drops are *rare*, and drops are only rare if the load is actually balanced. Which drags me to the real, structural problem with this whole family: collapse. There's nothing in the loss so far that stops the router from funneling most tokens into a few favorite experts. Worse, it's self-reinforcing — an expert that gets more tokens trains faster, gets better, and so gets chosen even more, a rich-get-richer spiral. If that happens, a handful of experts overflow (mass drops, mass identity-passthrough) while the rest sit idle (wasted parameters, wasted devices). I need to actively push the router toward spreading tokens evenly across all \(N\) experts.

So I need an auxiliary loss that I add to the main cross-entropy, whose minimum is uniform routing. Let me try to construct it from what I want and see what's forced. What I *actually* want equal is the **counts**: the fraction of tokens that get sent to each expert,

    f_i = (1/T) Σ_x 1{argmax_j p_j(x) = i},

the empirical load on expert \(i\). I want every \(f_i = 1/N\). The trouble is immediate: \(f_i\) is built from an argmax indicator. It's a hard count. Its gradient with respect to the router is zero almost everywhere — there's nothing to push on. So I can't just penalize the deviation of \(f\) from uniform directly; it's not differentiable.

I need a smooth stand-in for "how much the router wants expert \(i\)." The obvious differentiable quantity is the router's own probability mass, averaged over the batch:

    P_i = (1/T) Σ_x p_i(x),

the mean gate probability assigned to expert \(i\). This *is* differentiable in \(W_r\) — it's an average of softmaxes. Under perfectly uniform routing I'd want \(P_i = 1/N\) too. So now I have two length-\(N\) vectors, \(f\) (the hard counts, what I care about, no gradient) and \(P\) (the soft probabilities, has a gradient), and I want to drive both to uniform.

How do I combine them into a single scalar whose minimum is uniformity and whose gradient does something sensible? My first instinct is something like \(\sum_i (P_i - 1/N)^2\) — penalize \(P\)'s deviation from uniform on its own. But that ignores \(f\) entirely; it would equalize the *probabilities* without regard to where the tokens actually went, and the thing I truly want balanced is the counts. I want the loss to *know* which experts are currently overloaded and push their probabilities down preferentially. So let me couple them: weight each expert's soft probability by its observed hard load. That is, take the dot product

    loss ∝ Σ_i f_i · P_i.

Let me check this has the right minimum. \(f\) is a fixed (this-batch) distribution summing to 1; \(P\) is a distribution summing to 1 that I get to move. I'm minimizing \(\sum_i f_i P_i\) over the simplex for \(P\). A weighted average like \(\sum_i f_i P_i\) is minimized by putting all of \(P\)'s mass on the expert with the *smallest* \(f_i\) — i.e. the loss tells the router: stop sending probability to the experts that are already overloaded, send it to the underused ones. That's exactly the corrective pressure I want. And jointly, as both vectors respond, the only fixed point where neither can be improved is \(f_i = P_i = 1/N\) for all \(i\): the symmetric, uniform allocation. Let me verify the value there. With everything at \(1/N\),

    Σ_i f_i P_i = Σ_i (1/N)(1/N) = N · (1/N²) = 1/N.

Good — it bottoms out at \(1/N\) under uniformity. The gradient flows only through \(P\) (since \(f\) is the non-differentiable count), so in practice \(f\) acts as a per-expert *weight* on the differentiable \(P_i\): the more tokens an expert hoarded this batch, the harder its probability gets pushed down. Differentiable, and pointed the right way.

One nuisance: the value at the optimum, \(1/N\), depends on \(N\). If I sweep the number of experts, the loss magnitude drifts, and a single coefficient \(\alpha\) would mean different things at different scales. Fix it by multiplying by \(N\):

    loss = α · N · Σ_{i=1}^N f_i · P_i.

Now at uniform routing the loss is \(\alpha \cdot N \cdot (1/N) = \alpha\), constant regardless of how many experts I have. Clean. So \(\alpha\) is a pure, scale-invariant strength knob, and I can pick one value and trust it across model sizes. I'll need to find that value — I want it large enough that the router actually balances early in training, but small enough that it doesn't drown out the language-modeling loss I actually care about. Sweeping it across powers of ten, something around \(10^{-2}\) sits in that sweet spot: it balances the load within the first few thousand steps without distorting the primary objective.

I notice I've collapsed something here. The older design carried *two* separate auxiliary losses — one for load (the counts) and one for importance (the gate mass) — bolted together with two coefficients. But the \(f\cdot P\) construction folds both concerns into a single term: \(f\) is the load, \(P\) is the importance, and their coupled product is minimized exactly at the balanced state I want. One loss, one coefficient. That's a real simplification, and simplicity was the whole point.

Let me make sure I can actually compute this cheaply on the mesh. Per group of tokens on a core: I take the router probabilities `[tokens, N]`, average over tokens to get the \(P\) vector; I take the one-hot of the argmax, `[tokens, N]`, average over tokens to get the \(f\) vector; element-wise multiply and sum over experts. If I implement the "sum over experts" as a *mean* over the \(N\) experts (dividing by \(N\)) and then multiply by \(N^2\), I get \(\frac{1}{N}\sum_i f_i P_i \cdot N^2 = N \sum_i f_i P_i\), the same thing up to the \(\alpha\) I apply at the top level. Either way it's a couple of reductions — negligible next to the experts themselves.

Now the third pain: stability, and specifically precision. I want to train this in 16-bit (bfloat16) for the speed and the memory, because at trillion-parameter scale that matters enormously. The earlier sparse models gave up and ran *everything* in 32-bit to stay stable — which is safe but doubles the bytes in exactly the all-to-all routing traffic I worked so hard to halve. So full fp32 is self-defeating for me. Where does the instability actually come from? The router's softmax. Exponentiating logits in bfloat16 is numerically nasty — small mantissa, and `exp` blows the dynamic range — and a wobble there gets amplified by the hard switching downstream. So the fragile operation is *localized*: it's the router's logits-to-softmax computation, a tiny tensor on each device, not the big expert matmuls or the data crossing the network.

That localization is the whole opening. I don't need fp32 *everywhere*; I need it *only inside the router*, only for that softmax. So: cast the router input to float32, compute the logits and the softmax in float32 — where exp is well-behaved — make the routing decision, build the dispatch and combine tensors, and then cast those back down to bfloat16 *before* they leave the function and hit the all-to-all. The float32 only ever lives on-device, local to the router; nothing fp32-sized is ever broadcast across the mesh. I get the numerical stability of fp32 where it's needed and pay the bf16 communication cost everywhere it matters. Selective precision, not blanket precision.

There's another stability lever, and it's the initialization. Hard-switching layers seem to amplify whatever noise is present early in training — a slightly mis-scaled weight perturbs the routing decision, which perturbs everything downstream discontinuously. The default Transformer initialization draws weights from a truncated normal with standard deviation \(\sigma=\sqrt{s/n}\), \(n\) the fan-in, scale \(s=1\). If I just shrink that scale — divide \(s\) by ten, down to \(0.1\) — the early activations and routing logits start smaller and gentler. When I picture the spread of outcomes across random seeds, the big-init runs are all over the place and some destabilize; the small-init runs cluster tightly and train better on average. So: reduce the init scale by 10×. Cheap, and it markedly cuts both the mean loss and the seed-to-seed variance early on.

Now back to that exploration worry I parked. With \(k=1\) the router only ever sees the chosen expert — no counterfactual on the others — which is the contextual-bandit structure, pure exploitation if I always take the argmax. I want a little exploration so the router occasionally tries a runner-up and discovers it's good. The lightest possible intervention: jitter the router's *input* a touch — multiply it by uniform noise around 1, say in \([1-\epsilon, 1+\epsilon]\), during training only. That randomly perturbs the logits enough to occasionally flip a borderline argmax, giving non-favored experts a chance, without changing the architecture or adding a loss. Weighed against the alternatives — deterministic argmax (no exploration), sampling from the softmax (too much noise, hurts), dropout on the router input — the multiplicative input jitter is the one that helps without destabilizing. So I'll add input jitter in training.

There's also the regularization story for *after* pre-training. My model has far more parameters than the FLOP-matched dense baseline, so when I fine-tune on a small downstream task it overfits harder. The naive move — crank dropout everywhere — actually makes things worse: the shared, non-expert layers can't take that much dropout. But the *experts* are where the excess capacity lives. So apply the regularization where the parameters are: keep a modest dropout (around 0.1) on the non-expert layers and a much larger dropout (around 0.4) *inside* the experts. Targeted expert dropout, regularizing the high-capacity part without crippling the backbone.

Let me also chase two branches to their conclusions so I know what I'm leaving out. One: do I really have to drop overflow tokens? I could instead iteratively reroute an overflowed token to its *second*-choice expert, then third, and so on, guaranteeing almost nothing gets dropped. It's appealing — "leave no token behind." But when I reason it through, once the router has learned a token↔expert association, forcibly sending that token to its second-best expert is likely to *hurt* the representation more than a clean residual passthrough would. And indeed, when I weigh it, the rerouting buys no benefit over simple drop-to-residual; the extra machinery isn't worth it. Keep the simple drop. Two: do experts have to live only in the FFN slot? I could also replace the attention's query/key/value projections with Switch layers. That shows quality gains — but it's markedly more prone to diverge in bfloat16, and stability is a hill I'm not willing to die on right now. So experts stay in the FFN slot for the main design, attention-experts noted as a future direction.

Let me also keep the distributed picture straight, because it's where the communication savings actually land. On the mesh I lay out cores along a data dimension and (optionally) a model dimension, and for the experts I put one expert per core along the data dimension. For each token, the router locally computes a one-hot assignment and a slot index within the target expert's capacity — a binary *dispatch* tensor of shape `[cores, tokens_per_core, experts, capacity]`, and a float *combine* tensor of the same shape holding the gate value at the chosen `(expert, slot)`. The slot index comes from a cumulative count: as I scan the tokens routed to an expert, the running total is each token's position in that expert's buffer, and I keep only the tokens whose position is within capacity — that single comparison *is* the overflow drop. Then an einsum of inputs against the dispatch tensor gathers the per-expert batches, an all-to-all reshards them from "split by core" to "split by expert," each expert runs its FFN, an all-to-all sends the outputs back, and a final einsum against the combine tensor scales each output by its gate value and scatters it home. The bytes in those all-to-alls scale as experts × capacity × \(d_{model}\) — and top-1 halving the capacity is exactly what halves that bill.

Let me write the layer, grounded in how this actually runs. I'll keep the router and the loss explicit, because those are the contribution; the expert is just the ordinary FFN.

```python
import torch
import torch.nn as nn


class Expert(nn.Module):
    """An ordinary per-token FFN — this is just the dense sub-layer, replicated
    N times with independent weights. The contribution is the routing, not this."""
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # "expert dropout": larger when fine-tuning

    def forward(self, x):
        return self.wo(self.dropout(self.act(self.wi(x))))


class SwitchTop1Router(nn.Module):
    """Top-1 router. One linear projection to experts, softmax in float32 for
    stability (selective precision), argmax to pick a single expert, keep the
    gate value to scale the expert output (this is what keeps k=1 differentiable),
    and enforce a fixed per-expert capacity by dropping overflow."""
    def __init__(self, d_model, num_experts, capacity_factor, jitter=1e-2,
                 router_dtype=torch.float32):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.jitter = jitter
        self.router_dtype = router_dtype
        self.classifier = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):                       # x: [tokens, d_model]
        in_dtype = x.dtype
        x = x.to(self.router_dtype)             # selective precision: fp32 locally
        if self.training and self.jitter > 0:   # input jitter for exploration
            x = x * torch.empty_like(x).uniform_(1.0 - self.jitter, 1.0 + self.jitter)

        logits = self.classifier(x)             # [tokens, N]
        probs = torch.softmax(logits, dim=-1)   # fp32 softmax — the fragile op
        gate, expert_index = probs.max(dim=-1)  # top-1: argmax expert + its prob

        # fixed capacity: tokens per batch / experts, padded by capacity_factor
        tokens = x.shape[0]
        capacity = int((tokens / self.num_experts) * self.capacity_factor)

        one_hot = nn.functional.one_hot(expert_index, self.num_experts)  # [tokens, N]
        # position of each token within its chosen expert's buffer (cumulative count)
        position = (torch.cumsum(one_hot, dim=0) - 1)
        position = (position * one_hot).sum(dim=-1)                      # [tokens]
        # keep only tokens whose position is within capacity — this IS the drop
        keep = position < capacity
        gate = gate * keep                       # dropped tokens get gate 0 -> residual

        return (gate.to(in_dtype),               # scalar gate per kept token
                expert_index, keep, probs.to(in_dtype))


def switch_load_balancing_loss(probs, expert_index, num_experts):
    """loss = N * sum_i f_i * P_i.
    f_i = fraction of tokens whose argmax == i   (hard count, NOT differentiable)
    P_i = mean router probability to expert i     (differentiable -> gradient path)
    Minimized at uniform routing (f_i = P_i = 1/N -> value 1/N -> N*1/N = 1)."""
    f = nn.functional.one_hot(expert_index, num_experts).float().mean(dim=0)  # [N]
    P = probs.mean(dim=0)                                                     # [N]
    return num_experts * torch.sum(f * P)


class SwitchFFN(nn.Module):
    """Drop-in replacement for the dense FFN: a bank of N expert FFNs + a top-1
    router. Routes each token to its single argmax expert within capacity, drops
    overflow to the residual, recombines scaled by the gate value, and returns
    the auxiliary balancing loss to be added to the model loss with weight alpha."""
    def __init__(self, d_model, d_ff, num_experts, capacity_factor=1.25,
                 expert_dropout=0.1, alpha=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.router = SwitchTop1Router(d_model, num_experts, capacity_factor)
        self.experts = nn.ModuleList(
            Expert(d_model, d_ff, expert_dropout) for _ in range(num_experts))

    def forward(self, x):                        # x: [batch, seq, d_model]
        b, s, d = x.shape
        x = x.reshape(-1, d)                      # [tokens, d_model]
        gate, expert_index, keep, probs = self.router(x)

        y = torch.zeros_like(x)                   # un-routed tokens stay 0 here;
        for i in range(self.num_experts):         # the residual add outside is identity
            sel = keep & (expert_index == i)
            if sel.any():
                # scale each expert's output by its gate value -> the k=1 gradient path
                y[sel] = self.experts[i](x[sel]) * gate[sel].unsqueeze(-1)

        aux = self.alpha * switch_load_balancing_loss(probs, expert_index,
                                                      self.num_experts)
        return y.reshape(b, s, d), aux
```

So the chain, start to end: the scaling curves say parameters matter on their own axis, but in a dense model parameters and per-token FLOPs are the same knob, so I route each token to a sparse subset of experts to break them apart. The existing top-2 routing pays double in compute, capacity, and — worst — communication, justified by a conjecture that a single expert gives no router gradient; but writing \(y=p_i(x)E_i(x)\) shows the gate value keeps the router differentiable at \(k=1\), so I take the maximal-sparsity Switch: one expert per token, halving selection, capacity, and the all-to-all. Static shapes force a fixed per-expert capacity \((T/N)\times\) a capacity factor, with overflow dropped cleanly to the residual — and top-1 means I need only half the buffer. To stop the router collapsing onto a few experts, I want equal token counts \(f_i\), but counts have no gradient, so I couple them to the differentiable mean probabilities \(P_i\) in the single loss \(\alpha N\sum_i f_i P_i\), which is minimized exactly at uniform routing and made scale-invariant by the factor \(N\). The instability that kept these models in fp32 lives in the router's softmax, so I cast only that local computation to float32 and recast to bf16 before the network shuffle — fp32 stability at bf16 communication cost — and I shrink the init scale 10× to tame the hard-switch amplification of early noise. A touch of input jitter buys exploration in the bandit-like routing, and targeted expert dropout regularizes the high-capacity experts at fine-tuning time. The result is one expert per token, one balancing loss, fp32 only where it's fragile: a sparse FFN that scales parameters far past the dense baseline at matched FLOPs.
