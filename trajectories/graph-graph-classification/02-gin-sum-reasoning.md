The stripped DiffPool came back and its numbers say exactly what I suspected when I built it: the
learned soft pool, with no link or entropy auxiliary to lean on, behaves like a noisy mean and pools
*less* informatively than it should. Read the rows. On PROTEINS it sits at 70.98 mean accuracy with a
band so tight across seeds — {70.98, 71.44, 70.52} — that it is plainly *not* learning a
seed-distinguishing partition; it is collapsing to roughly the same diffuse, mean-like answer every run,
the signature of an assignment softmax that never committed to crisp clusters. PROTEINS is also the
*lowest* of the three datasets for it, which is telling: PROTEINS has the larger graphs where a real
hierarchy would help most, and the stripped pool gets the least out of them. On NCI1 it lands at 78.25,
respectable but unremarkable. MUTAG is the giveaway on variance: {85.64, 78.71, 79.77}, a six-point
spread, exactly the coin-flip I predicted on a 188-graph set where the soft assignment has almost no
data to fit. So the diagnosis is clean and it is not a tuning problem. The one idea I kept — a learned
soft clustering pooled as Sᵀ X — needed the auxiliaries the harness would not let me wire in, and
without them it discards counts (it is mean-like) and, worse, makes no use of the per-layer node
embeddings the scaffold hands me in `layer_outputs`. I threw away two things: the *injective* reduction
that keeps multiplicities, and the *multi-scale* signal sitting unused in the layer stack.

So let me stop trying to learn a hierarchy with no support for it and go back to first principles on the
*flat* reduction, because that is what I actually have a clean theory for. The question is narrow: among
permutation-invariant reductions over the multiset of node embeddings, which one keeps the most
information? DiffPool's failure mode was that it became a mean, and a mean has a precise defect. Take a
multiset of node features and an "inflated" copy where every multiplicity is scaled by the same integer
k: the mean is identical for both, because `(1/(k·n))·k·Σf = (1/n)·Σf`. So a mean captures only the
*distribution* — the proportions of neighbor features — and is blind to absolute counts. Max is worse
still: `max_x f(x)` depends only on which *distinct* elements are present, so it sees only the *support*
and loses both counts and proportions. The sum, by contrast, is *injective* on bounded multisets: with a
suitable per-element map f, `Σ_x N^{-Z(x)}` positionally encodes the exact multiplicity profile in base
N, so distinct multisets give distinct sums. Among {sum, mean, max} there is a strict pecking order in
discriminative power — sum ⊐ mean ⊐ max — and DiffPool, by drifting to a mean, sat below the top. The
sum is the one reduction that keeps everything, and the GIN backbone in front of me was built precisely
so that an injective sum readout makes the whole network as discriminating as the Weisfeiler–Lehman test
— the ceiling for any message-passing GNN. Using anything weaker than a sum at the readout *wastes the
backbone's expressivity*. That settles the aggregator: sum, not mean, not max.

But a single sum over the final layer is still flat in the other sense DiffPool was reaching for, and
here I can finally do something about it without needing a clustering at all. The scaffold hands me
`layer_outputs`, a list of the per-layer node embeddings, one `[N_total, hidden_dim]` tensor for each of
the five GIN layers — and the DiffPool rung used *none* of it. Why does that list matter? Because a
node's embedding after k rounds is a learned summary of its rooted subtree of height k: the height-1
embedding sees immediate neighbors, the height-5 embedding sees five hops out. As depth grows, node
representations get more global and more discriminative — I want enough depth for power. But the deepest
features are also the most *specialized*; on small datasets like MUTAG the shallower, more local features
often generalize better, and over-smoothing in deep GNNs can wash the last layer out. Rather than gamble
on one depth — and reading only the last layer is exactly the gamble — I should use *all* of them. This
is the jumping-knowledge idea: let the final graph representation reach back to every depth instead of
trusting the single deepest layer. The multi-scale signal DiffPool wanted from a learned hierarchy is
already present, for free, in the layer stack; I just have to read it.

So the readout writes itself: sum-pool each layer's node embeddings *independently* into a per-layer
graph vector, then *concatenate* across layers. `h_G = CONCAT( Σ_{v∈G} h_v^{(k)} : k = 1..K )`. Each
per-layer sum is injective on its own multiset of node features; concatenation keeps all K of them side
by side rather than mixing them, so the classifier downstream can weight depths as it likes. And there
is a clean reading of *what* this computes: a node's height-k embedding is a learned encoding of a
height-k rooted subtree, so summing them is the learnable analogue of *counting subtrees* — exactly what
the WL subtree kernel does by hand, except the subtrees are embedded in a continuous space, so *similar*
subtrees land near each other (something one-hot WL labels can never do). The readout therefore
generalizes both WL and the WL subtree kernel. The output width is `hidden_dim × num_layers` —
5×64 = 320 here — with no projection bottleneck, so every depth's full signal reaches the classifier.

There is one wrinkle I have to handle, and it is the place this task's readout departs from the bare
textbook JK-sum, so let me reason it out rather than copy it. Concatenating per-layer *sum* pools means
concatenating vectors at very *different scales*: the deeper GIN layers, after several rounds of
neighbor summation plus the (1+ε) self-weighting inside GINConv, can have systematically larger
magnitudes than the shallow ones, and a sum over a 100-node graph is ~100× the per-node scale while a
sum over a 17-node MUTAG graph is ~17× — so the same concatenated vector mixes wildly different
magnitudes across both layers and graph sizes. Feed that straight into the classifier and the
large-magnitude coordinates dominate the first linear layer's gradient, the small ones are effectively
ignored, and some folds simply fail to converge — the optimization stalls on the scale mismatch, not on
anything about the graphs. The fix that keeps the injectivity intact is to **batch-normalize each
layer's graph-level embedding before concatenating**: a `BatchNorm1d(hidden_dim)` per layer, applied to
the `[B, hidden_dim]` pooled vectors. BatchNorm is an affine, invertible (at fixed statistics)
re-scaling, so it does not collapse the multiset distinctions the sum encoded — it just puts every
layer's graph embedding on a common, well-conditioned scale before they are stacked, so the classifier
sees all five depths on equal footing and every fold trains. This per-layer graph-BN is the one piece I
add beyond "sum each layer and concatenate," and it is there for optimization stability, not
expressivity.

Made concrete in the scaffold: the readout holds a `nn.ModuleList` of `num_layers` `BatchNorm1d(hidden_dim)`
modules; in `forward` it loops over `layer_outputs`, does `g = global_add_pool(h, batch)` for each, runs
`g` through that layer's BN, collects the results, and returns `torch.cat(graph_embs, dim=-1)`. I set
`self.output_dim = hidden_dim * num_layers` so the fixed classifier head expects the full concatenated
width. Note I read from `layer_outputs` and ignore `x` (the final-layer embedding) — `x` is just
`layer_outputs[-1]`, already included. No `edge_index`, no dense batching, no clusters: the whole readout
is a per-layer sum, a per-layer BN, and a concat. The full scaffold module is in the answer.

So the delta from the DiffPool rung is precise and points in the opposite direction from what I tried
first. DiffPool spent its capacity trying to *learn* structure (a soft clustering) with none of the
support that makes that pay, and ended up mean-like — discarding counts and ignoring depth. This rung
spends *no* learned capacity on the pooling itself: it takes the provably-injective sum, applies it at
every depth, and lets the downstream classifier — not a fragile assignment softmax — decide how to weigh
the scales. The only learned parameters in the readout are the per-layer BNs, and they exist for
conditioning, not for clustering.

Reading DiffPool's shape, here is what I expect and where it is falsifiable against its numbers. PROTEINS
is the cleanest test: DiffPool's diffuse mean-like pool flattened to 70.98 with a near-zero seed spread;
if the diagnosis is right — that a mean discards counts and a single depth misses scale — then summing
over every layer should *lift* PROTEINS clearly above ~71 and, because the readout is now deterministic
rather than a learned-and-stuck softmax, I expect it to still be tight but at a higher level. NCI1, the
largest set where the injective-sum-feeds-WL-power argument bites hardest, should also rise above
DiffPool's 78.25 — this is the dataset where keeping counts and reading every depth should matter most.
MUTAG is the one I am least sure about: the deterministic readout removes the assignment-softmax
coin-flip, so I expect its seed variance to *shrink* relative to DiffPool's six-point spread, but on 188
graphs noise will remain, and the mean could land near or modestly above DiffPool's 81.37 without the
gap being decisive. The falsifiable claim is the ranking: if reading every layer with an injective sum is
the right correction, gin-sum should beat DiffPool on the *aggregate* and most clearly on PROTEINS and
NCI1 — the datasets where counts and depth, not luck, decide. If PROTEINS does *not* move, my whole
"DiffPool became a mean" story is wrong and I would look elsewhere.

The causal chain in one breath: DiffPool's stripped soft pool became a noisy *mean* — flat at 70.98 on
PROTEINS with no seed spread, six-point variance on MUTAG — discarding counts and ignoring the per-layer
embeddings entirely → so go back to the flat reduction with a clean theory: among {sum, mean, max} only
the *sum* is injective on the multiset of node features (mean keeps only the distribution, max only the
support), and an injective sum makes the GIN backbone as powerful as WL → recover the missing scale not
by learning a hierarchy but by reading the one the backbone already built: sum-pool *each layer*
(`layer_outputs`) and concatenate (jumping knowledge, the learnable WL-subtree-kernel) → batch-normalize
each layer's graph embedding before concatenating so the different per-layer/per-size magnitudes don't
stall optimization → drop it into `GraphReadout` with `output_dim = hidden_dim × num_layers`, expecting
PROTEINS and NCI1 to clear DiffPool's 70.98 and 78.25 and MUTAG's seed variance to tighten.
