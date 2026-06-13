The representation is the whole point, but it bolts onto a frozen pipeline, and the floor I start from
is just "learn *any* molecular encoder on these scaffold-split tasks at all." So before reaching for
anything clever I want the simplest learned graph encoder that I can argue, from first principles, is
not throwing away structure for free — and then watch where it breaks, because where it breaks tells me
what the next rung has to fix. The scaffold hands me an atom/bond graph per molecule (atoms with the
136-dim feature vector, bonds with the 9-dim one, in the sparse `x / edge_index / edge_attr / batch_idx`
format) and asks for one vector per graph that a small head turns into per-task logits. That is exactly
the message-passing setting, so the design question is narrow: what aggregator, what per-layer
transform, what readout — and which of those choices actually decides how much the encoder can tell two
molecules apart.

Let me write down what every graph net in this family actually is, stripped to its skeleton, because the
differences I care about are smaller than they look. Each atom starts at `h_v^{(0)} = x_v`; then for `k`
rounds, `a_v^{(k)} = AGGREGATE({h_u^{(k-1)} : u∈N(v)})` and `h_v^{(k)} = COMBINE(h_v^{(k-1)}, a_v^{(k)})`,
and a graph vector is `h_G = READOUT({h_v^{(K)}})`. GCN is this with a mean aggregator and a single
linear+ReLU; GraphSAGE's pooling variant is an element-wise max of a transformed neighborhood. The only
things that vary are the function that squashes the bag of neighbors and the transform around it. So my
real question is: what does `h_v^{(k)}` represent, and when are two atoms (or two molecules) *forced* to
the same vector — because that collapse is the ceiling on what any such encoder can distinguish, and on
a scaffold split, distinguishing genuinely-novel structure is the entire game.

Trace what `h_v^{(k)}` captures. Round one, an atom sees its bonded neighbors. Round two, each neighbor
has already absorbed *its* neighbors, so the atom sees two bonds out. After `k` rounds `h_v^{(k)}` is a
summary of the rooted subtree of height `k` hanging off `v`. Two atoms should collapse to the same
vector only if those rooted subtrees are genuinely identical. And I recognize this loop — relabel a node
by hashing (its own label, the multiset of neighbor labels), iterate — it is the Weisfeiler-Lehman
color-refinement test for graph isomorphism. WL's hash is *injective*: different (color, neighbor-
multiset) inputs always get a brand-new distinct color, so WL never throws away a distinction once it
has found one. A graph net runs everything through continuous, lossy functions — a mean, a max, a linear
layer — that can squash two different inputs to the same output. That comparison is the right ruler: WL
is a strong, almost-complete stand-in for isomorphism that has the *same shape* as my encoder, so I can
ask whether the encoder reaches it.

The bound goes both ways and it is worth knowing exactly. If WL cannot separate two graphs, the two have
the identical multiset of (label, neighbor-multiset) pairs at every round — otherwise WL's injective
hash would have separated them at the next round. By induction within a graph, equal WL labels force
equal encoder features (same input, same shared AGGREGATE/COMBINE applied to equal inputs), so the
encoder's feature-multisets march in lockstep too, and a permutation-invariant readout returns the same
value: **no message-passing GNN beats WL.** There is the ceiling. And it is *reached* exactly when the
neighbor aggregation, the combine, and the readout are all injective on multisets — because then the
encoder's features are a faithful (injective) recoding of WL's labels, and whatever WL separates, the
encoder separates. Injectivity on multisets is the whole game.

Now the practical payoff: which aggregator is injective? Over bounded multisets, **sum** can be made
injective — pick a per-element code and sum, and the total is a positional encoding of the multiset's
multiplicity profile, so distinct multisets give distinct sums. **Mean** keeps only the *distribution*:
it cannot tell a multiset from any inflated copy of itself (scale every multiplicity by `k` and the mean
is unchanged), so `{green,red}` and `{green,green,red,red}` collapse. **Max** keeps only the *support*:
`{green,red}` and `{green,red,red}` collapse, because the second red is invisible. So the discriminative
ranking is strict: sum ⊐ mean ⊐ max. And the transform around the sum must be a real MLP, not a single
linear+ReLU — with all-nonnegative inputs a bias-free linear+ReLU degenerates into "sum first, then one
linear map," which cannot separate `{1,1,1,1,1}` from `{2,3}` (both sum to 5). One more: if I merge the
center atom into the neighbor bag and sum the flat multiset, I lose which element was the root — the
middle of `a–b–b` and of `b–a–b` both become `{a,b,b}` — so I tag the center, `(1+ε)·h_v + Σ_u h_u`,
which stays injective for irrational `ε` and is a learnable scalar in practice. The maximally expressive
fill is therefore: sum the neighbors, add `(1+ε)` times the center, push through an MLP, and read out by
summing (ideally over every layer, jumping-knowledge style).

So the theory points hard at *sum* pooling and a sum readout. But I am building the **scaffold's starter
fill**, and it deliberately departs from the theoretical maximum in three ways, each of which I'll keep,
because the point of rung one is a deliberately weak, honest floor — and I want the departures named so
the next rung can react to them rather than to a paper. First, the per-layer message folds in the **bond
features**: `msg = h_v[src] + edge_proj(e_vw)`, then summed into the destination atom. The clean WL-
maximal GIN ignores edges entirely; here the edges (single/double/aromatic, conjugated, in-ring) are
chemically load-bearing, so the message is edge-aware — an additive bond bias before the sum. Second,
the per-layer aggregation over neighbors is a sum (good, injective), and the center is tagged by
`(1+ε)`, so each `GINConv` is `MLP((1+ε)·x + Σ_{N(v)}(x[u]+edge))` with a BatchNorm inside the MLP —
that part is faithful to the expressive recipe. Third — and this is the load-bearing weakness — the
*graph readout is **mean** pooling*, `Σ_v h_v / |V|`, off the **last** layer only, not a sum and not
concatenated across depths. By the ranking above, mean readout throws away exactly the multiplicity
information sum keeps; on an unlabeled-degree sense it cannot tell a molecule from a scaled copy of its
substructure distribution. The starter trades the provably-maximal readout for the simplest one. It also
wraps the four layers in a residual stack (`x ← x + dropout(relu(norm(conv(x)))`)) so gradients flow and
the encoder doesn't over-smooth, with hidden width 256, four layers, dropout 0.1, and a two-layer FFN
head to the `num_tasks` logits.

Let me reason about what this floor must do, because that is the entire reason to run it. The encoder is
local: four rounds reach atoms four bonds away, and a drug-like molecule's diameter is often larger, so
the pooled vector is a sum-then-*average* of local views and can miss anything genuinely global about the
molecule. It is also data-hungry — it learns its whole representation from scratch against a few hundred
to a few thousand labels — and it carries *no* external chemical prior, no pretrained weights, nothing
the fixed-descriptor camp would hand it for free. On a *random* split a learned encoder this plain can
still score by recognizing scaffolds it memorized; but the evaluation here is a **scaffold** split, so
test molecules are structurally novel, and memorization buys nothing. Worse, the mean readout actively
hurts on the scaffold split: averaging discards the size/count signal that often correlates with the
property, and on a single-task binary target with a hard distribution shift between train and test
scaffolds, a representation that only sees the *proportions* of local environments has very little to
hold onto. I therefore expect this rung to be the weakest thing I can run by construction — a learned
encoder with the lossiest readout in the family, no prior, and a receptive field smaller than the
molecule.

I expect the three datasets to split on how much the property is decided by *local* chemistry that mean-
pooling of four-hop neighborhoods can still capture under a hard scaffold shift. Tox21 is twelve assays
with a lot of molecules; even a weak local encoder gets enough signal that multi-task averaging should
keep it well above chance — I'd expect it to be the best of the three for this rung. BACE is a single,
fairly structured enzyme-inhibition target where local substructure matters a lot, so a local encoder
should get meaningful traction. BBBP is the danger: a single binary target with a severe scaffold shift,
where the property depends on global, whole-molecule physicochemistry (lipophilicity, polar surface
area, size) that a four-hop *mean*-pooled GNN with no descriptors essentially cannot see — I would not be
surprised if this rung lands near *chance* (ROC-AUC ≈ 0.5) on BBBP, because the encoder has been handed
the one task its weaknesses (locality, lossy mean readout, no global prior) bite hardest, on the split
that punishes memorization. If BBBP does collapse toward 0.5 while BACE and Tox21 stay well above it,
the diagnosis is already written and points two ways at once: the readout is throwing away count
information (swap mean for sum, and reach back to the raw bond identity instead of letting four tied-ish
rounds wash it out), and there is a cheap global prior — molecule-level descriptors — sitting unused that
would patch exactly the locality/no-prior failure on a task like BBBP. That is the next rung.

So at rung one the encoder is settled and my edit is the *default* one: the scaffold's starter GIN —
edge-aware sum-of-neighbors message with a `(1+ε)` center tag and an MLP, four residual layers with
BatchNorm, **mean** pooling off the last layer, a two-layer head (the distilled module is in the
answer). It is the floor by construction: maximally expressive in the message, deliberately lossy in the
readout, local in receptive field, and prior-free — and on a scaffold split, on the task most decided by
global physicochemistry, it should show me exactly how much a plain learned encoder leaves on the table.

The causal chain in one breath: I want the simplest learned graph encoder whose representational power I
can actually reason about, so I measure it against the WL test → no message-passing GNN beats WL, and it
*reaches* WL only when aggregation, combine, and readout are injective, which forces sum over mean over
max and an MLP over a single linear layer, with a `(1+ε)` center tag → but the scaffold's starter fill
deliberately keeps the injective sum *message* (edge-aware) while using the lossy **mean** *readout* off
the last layer, no jumping-knowledge, no external prior → so this floor is local (four hops < molecule
diameter), data-hungry, and prior-free, and on a scaffold split it should be best on Tox21, decent on
BACE, and near chance on BBBP — and that collapse is what forces, at rung two, a sum readout that keeps
counts plus a global descriptor prior to patch the locality.
