# Context

## Research question

The quality of a large neural language model rises predictably with three
quantities — the number of parameters, the size of the training corpus, and the
amount of compute spent — and the relationship is a power law over many orders of
magnitude (Kaplan et al., 2020). One reading of those scaling curves is striking:
*parameter count* contributes to quality in a way that is, to first order,
separable from the floating-point operations (FLOPs) actually performed per
token. In an ordinary dense Transformer the two are welded together — every
parameter participates in every token, so doubling parameters doubles the
arithmetic per token, and at the scales that matter this becomes the binding
constraint on energy, wall-clock, and accelerator memory.

The precise goal is to *unweld* them: build a Transformer whose parameter count
can be scaled up by one to three orders of magnitude while the FLOPs applied to
each token stay essentially fixed. A practical solution must (i) make each token
activate only a small, input-dependent subset of the parameters; (ii) keep that
selection trainable end-to-end; (iii) run efficiently on accelerators (TPUs,
GPUs) that are built for dense matrix multiplies and require statically declared
tensor shapes; (iv) keep the cross-device communication induced by routing
within budget; and (v) train *stably* — including in reduced (16-bit) precision —
at the largest scales, where prior sparse models have been observed to diverge.

## Background

**Conditional computation and the cost wall.** The idea that only part of a
network should fire on any given input is the standard route to "capacity without
proportional compute." The obstacles that have kept it from scaling are concrete:
accelerators are far faster at dense arithmetic than at branching, so a routing
decision must switch a *large* block of computation on or off to pay for itself;
large batches amortize parameter loads, but routing splits a batch into smaller
per-block batches; and inter-device bandwidth is small relative to aggregate
compute, so any design that ships activations across the network per token risks
becoming bandwidth-bound.

**Mixture-of-Experts layers.** The dominant realization of conditional
computation in modern sequence models is the Mixture-of-Experts (MoE) layer
(Jacobs et al., 1991; Jordan & Jacobs, 1994; Shazeer et al., 2017). A set of
\(N\) expert sub-networks \(\{E_i\}_{i=1}^N\) — typically each an ordinary
feed-forward network — sits behind a small trainable *router*. For a token
representation \(x\), the router produces logits \(h(x)=W_r x\) and a softmax
gate
\[
  p_i(x)=\frac{e^{h(x)_i}}{\sum_j e^{h(x)_j}}.
\]
Only the top-\(k\) experts (by gate value) are evaluated, and the layer output is
their gate-weighted combination,
\[
  y=\sum_{i\in\mathcal{T}} p_i(x)\,E_i(x),
\]
where \(\mathcal{T}\) is the set of selected indices. With \(k\ll N\), the FLOPs
per token are those of \(k\) experts, independent of the total \(N\) — so capacity
(all \(N\) experts' parameters) and compute (\(k\) experts) are decoupled. The
prevailing wisdom is that \(k\) must be at least 2: it was conjectured that with
a single selected expert the router receives no useful gradient — that learning
to route requires *comparing* at least two experts — and a follow-up study found
that larger \(k\) in lower layers helped models with many routing layers
(Ramachandran & Le, 2018).

**Two persistent difficulties of MoE.** First, *router collapse / load
imbalance*: nothing stops the router from sending most tokens to a handful of
experts, which is self-reinforcing (favored experts train more, so they are
selected more) and wrecks the even per-device load that efficient parallelism
needs. The standard remedy is one or more differentiable auxiliary losses added
to the training objective to push the router toward uniform usage. Second,
*static shapes vs. dynamic routing*: accelerators (especially TPUs) require
tensor shapes fixed at compilation, but the number of tokens an expert receives
is data-dependent and uneven. The accepted fix is to give each expert a fixed
*capacity* — a maximum number of tokens it will process — and to handle overflow
by skipping the expert for the excess tokens, letting their representation pass
through unchanged via the residual connection (Lepikhin et al., 2020).

**Observed instabilities and precision.** Sparse expert models have been
empirically harder to train than dense ones, an instability attributed to the
hard, near-discontinuous routing decisions at each layer. The softmax inside the
router is a particular culprit in low precision: in 16-bit floating point the
exponentials in the gate are numerically fragile. The reported practice has been
to train the entire MoE in 32-bit precision to stay stable (Lepikhin et al.,
2020), at the cost of doubling the bytes moved in the cross-device communication
that routing requires.

**Distributed primitives.** Mesh-TensorFlow (Shazeer et al., 2018) abstracts the
physical accelerators into a logical mesh and shards tensors along *named*
dimensions, with statically determined shapes. It supports data-parallel,
model-parallel, and expert-parallel layouts and the all-reduce / all-to-all
collectives they require; it is the substrate on which experts can be placed one
(or several) per core and tokens shuffled to them. The same library reintroduced
MoE layers into the Transformer's feed-forward slot.

## Baselines

**Dense Transformer (Vaswani et al., 2017).** The model to beat and to match on
FLOPs. Its position-wise feed-forward sub-layer maps each token independently:
with input \(x\in\mathbb{R}^{d_{model}}\), an intermediate
\(h=xW_{in}\in\mathbb{R}^{d_{ff}}\) (with \(d_{ff}\) several times \(d_{model}\)),
and output \(y=\mathrm{ReLU}(h)W_{out}\). Scaling a dense model means growing
\(d_{model}\)/\(d_{ff}\)/depth in tandem, which raises parameters *and* FLOPs
together and is ultimately bounded by per-accelerator memory; beyond that point
one resorts to model-parallel sharding, which adds an all-reduce on every forward
and backward pass. The gap it leaves: no way to add parameters without adding
proportional per-token compute.

**Sparsely-gated MoE Transformer (Shazeer et al., 2017; in the Transformer FFN
slot via Shazeer et al., 2018; scaled by Lepikhin et al., 2020).** Replaces the
FFN with the top-\(k\) expert layer above (typically \(k=2\)). It is the first
design to decouple capacity from compute at scale and the direct point of
comparison. Its mechanics and the specific gaps it leaves:
- **Top-2 routing** evaluates two experts per token, so per-token expert FLOPs,
  router selection work, and the per-expert capacity (buffer batch size) all
  scale with the number of experts evaluated per token. The choice \(k=2\) rests
  on the conjecture that \(k\ge 2\) is *necessary* for a trainable router (above);
  the cost of that choice along every one of these axes is borne without that
  conjecture having been put to the test.
- **Two auxiliary losses.** The original design carried separate
  load-balancing and importance-weighting terms (later partially simplified to a
  single load term in Shazeer et al., 2018; Lepikhin et al., 2020), adding
  hyper-parameters and conceptual overhead.
- **Communication.** Routing each token to two experts on (potentially) two
  different cores means the all-to-all traffic of dispatching tokens and
  gathering their outputs scales with the number of experts each token is sent to.
- **Precision/stability.** Trained in 32-bit precision throughout for
  stability, which inflates the bytes moved in the routing collectives and forgoes
  the speed of 16-bit training.
- **Capacity overflow** handled by dropping overflow tokens to the residual; the
  rates and the choice of capacity factor interact with quality in ways that
  larger \(k\) makes more buffer-hungry.

Taken together, these are the axes along which the existing sparse design carries
cost: complexity (multiple experts per token, multiple auxiliary terms),
communication and capacity overhead, and the instability that forces 32-bit
training — all without an established account of which of them are actually
forced by the requirement that the router remain trainable, and which are
incidental.

## Evaluation settings

The natural yardstick is a strong dense Transformer trained on a large,
deduplicated web corpus ("Colossal Clean Crawled Corpus," C4; Raffel et al.,
2019), under a masked / span-corruption language-modeling pre-training objective
(15% of tokens corrupted, replaced by sentinels), with quality reported as
negative log perplexity (in nats). FLOP-matching — applying the same compute per
token as the dense baseline — is the protocol that isolates the parameter axis.
Comparisons are made on three bases: per training step, per wall-clock second (to
account for routing/communication overhead), and against a *larger dense* model
that spends its budget on FLOPs instead of parameters.

Downstream, the established transfer protocol is pre-train then fine-tune on
smaller tasks: the GLUE and SuperGLUE suites (handled as token-proportional
mixtures), summarization (CNN/DailyMail, XSum; Rouge-2), extractive and
closed-book question answering (SQuAD; Natural Questions, Web Questions, TriviaQA;
exact match), and reasoning/commonsense sets (ARC, ANLI, Winogrande; accuracy). A
multilingual variant of the corpus (mC4, 101 languages) is the setting for
multi-task multilingual pre-training. Compression of large models is measured by
distillation into a small dense student.

## Code framework

The pieces that already exist: a Transformer with a per-token feed-forward
sub-layer, the optimizer and span-corruption training loop, and a distributed
tensor library that shards along named dimensions under static shapes and
provides the all-reduce / all-to-all collectives. The dense FFN is the slot a
sparse alternative will occupy. What does *not* yet exist is the routing rule, the
capacity/overflow handling, and the balancing objective — those are the empty
stubs below.

```python
import torch
import torch.nn as nn

# --- existing dense sub-layer ------------------------------------------------

class FeedForward(nn.Module):
    """Per-token Transformer FFN: x -> ReLU(x W_in) W_out. Applied identically
    to every token. This is the dense sub-layer a sparse layer will replace."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.wo(self.act(self.wi(x)))


# --- empty slots the contribution will fill in ------------------------------

class Router(nn.Module):
    """Map each token to one or more experts and a scalar gate value, under the
    constraints of the problem statement (static shapes, trainability, bounded
    communication, stability in reduced precision)."""
    def __init__(self, d_model, num_experts):
        super().__init__()
        # TODO
        pass

    def forward(self, x):
        # TODO
        raise NotImplementedError


def balancing_loss(*args, **kwargs):
    """Auxiliary objective that discourages the router from collapsing onto a
    few experts. Must be differentiable through whatever the router exposes,
    and minimized when the load is uniform across experts."""
    # TODO
    raise NotImplementedError


class SparseFFN(nn.Module):
    """Drop-in replacement for FeedForward: a bank of expert FFNs plus a router.
    Capacity, overflow handling, and the gate-weighted recombination go here."""
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.experts = nn.ModuleList(FeedForward(d_model, d_ff)
                                     for _ in range(num_experts))
        self.router = Router(d_model, num_experts)
        # TODO: capacity_factor and the dispatch/combine machinery

    def forward(self, x):
        # TODO: route each token, run only its expert(s) within capacity,
        #       drop overflow to the residual, recombine scaled by the gate,
        #       and also return the auxiliary balancing loss
        raise NotImplementedError
```
