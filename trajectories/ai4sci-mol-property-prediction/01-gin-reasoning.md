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

Let me be honest about the space of "floors" I could set here, because the choice is not innocent. I
could make the floor a fixed-descriptor classifier — hash each molecule to an ECFP bit-vector or a panel
of RDKit descriptors and run a random forest — which needs no representation learning at all and would
give a number, but it teaches me nothing about whether a *learned* graph encoder transfers across
scaffolds, which is the question the whole ladder exists to answer. I could jump straight to the
provably-maximal graph net (sum readout, jumping-knowledge across all depths, a real MLP everywhere) and
squeeze the 2D graph for all it is worth on rung one. But then if the number is good I will not know
*which* ingredient bought it — the sum readout, the depthwise concatenation, the expressive combine — and
if it is bad I will not know which one to blame. And I could leap to the pretrained 3D transformer the
strongest baseline uses, but a first rung that already spends the biggest hammer leaves me no controlled
story about what the geometry and the pretraining each contributed. The discipline I want is one variable
per rung: establish the plainest learned encoder whose ceiling I can *prove*, name the exact ways it is
deliberately lossy, and let each later rung turn exactly one of those knobs so the feedback table
attributes the win. That points at the scaffold's own starter fill — a small GIN with mean pooling — not
as a default I inherit passively but as the one floor whose failure modes I can read straight off its
design.

So let me write down what every graph net in this family actually is, stripped to its skeleton, because
the differences I care about are smaller than they look. Each atom starts at `h_v^{(0)} = x_v`; then for
`k` rounds, `a_v^{(k)} = AGGREGATE({h_u^{(k-1)} : u∈N(v)})` and `h_v^{(k)} = COMBINE(h_v^{(k-1)}, a_v^{(k)})`,
and a graph vector is `h_G = READOUT({h_v^{(K)}})`. GCN is this with a mean aggregator and a single
linear+ReLU; GraphSAGE's pooling variant is an element-wise max of a transformed neighborhood. The only
things that vary are the function that squashes the bag of neighbors and the transform around it. So my
real question is: what does `h_v^{(k)}` represent, and when are two atoms (or two molecules) *forced*
to the same vector — because that collapse is the ceiling on what any such encoder can distinguish, and
on a scaffold split, distinguishing genuinely-novel structure is the entire game.

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

The bound goes both ways and it is worth grinding out exactly, because the two directions are the whole
justification for the design. First direction, the ceiling: suppose WL cannot separate graphs `G` and
`G'`. Then at every round the two have the identical multiset of (label, neighbor-multiset) pairs —
otherwise WL's injective hash would have minted a new color somewhere and separated them at the next
round. Now run the encoder on both and induct on `k`. At `k=0` the atom features are the initial WL
labels, so equal WL labels mean equal encoder features. Assume it holds at `k-1`: two atoms with equal
WL labels have, by the WL-indistinguishability, equal neighbor-label-multisets, hence — feeding equal
inputs through the *same* shared AGGREGATE and COMBINE — equal `h^{(k)}`. So the encoder's per-atom
feature multiset marches in lockstep with WL's color multiset at every round, and a permutation-invariant
readout, seeing equal multisets, returns equal graph vectors. **No message-passing GNN separates what WL
cannot** — there is the ceiling, and it is a hard one. Second direction, the reachability: the encoder
*attains* that ceiling exactly when AGGREGATE, COMBINE, and READOUT are each injective on multisets,
because then the encoder's features are a faithful (injective) recoding of WL's labels and whatever WL
separates the encoder separates. So the entire design problem collapses to one question — is each of my
three operations injective on multisets — and injectivity on multisets is the whole game.

Now the practical payoff: which aggregator is injective? Let me make the three collapses concrete with
actual numbers rather than adjectives. Represent a neighbor bag by its multiplicity vector over element
types; take two colors, green and red. **Sum** over one-hot codes turns the bag `{green, red}` into
`(1,1)` and `{green,green,red,red}` into `(2,2)` — distinct, kept apart, and in general the summed code
is a positional encoding of the multiplicity profile, so distinct bounded multisets give distinct sums.
**Mean** turns both of those into `(0.5, 0.5)`: it keeps only the *distribution* and cannot tell a bag
from any inflated copy of itself, so `{green,red}` and `{green,green,red,red}` collapse to the identical
vector. **Max** turns `{green,red}` into `(1,1)` and `{green,red,red}` into `(1,1)` as well: it keeps
only the *support*, so the second red is invisible. The discriminative ranking is therefore strict,
sum ⊐ mean ⊐ max, and it is not a matter of taste — it is which multiset statistic each operation
preserves, computed above on two-element examples.

The transform around the sum has to be a genuine MLP, not a single linear+ReLU, and here too I can show
the collapse in numbers rather than assert it. Suppose I transform each neighbor by a bias-free
linear+ReLU and then sum. On all-nonnegative inputs `ReLU(w·x) = w·x`, so summing gives `w·Σx` — a linear
functional of the raw sum. Feed it the scalar bags `{1,1,1,1,1}` and `{2,3}`: both have `Σx = 5`, so both
map to `w·5`, identical, and no linear layer downstream can ever pull them apart because the collision
happened at the sum. Put a real two-layer MLP in the way — even something as crude as a soft threshold
`φ(x) = ReLU(x − 1.5)` applied before summing — and the first bag maps to `5·φ(1) = 0` while the second
maps to `φ(2)+φ(3) = 0.5 + 1.5 = 2`. Separated. Chemistry lives exactly in these conjunctions of counts —
"an aromatic nitrogen *and* a nearby carbonyl," a threshold on how many hydrogen-bond donors — which are
nonlinear functions of the neighbor-count vector, so the combine must be nonlinear or it cannot express a
pharmacophore at all. One more injectivity leak to plug: if I fold the center atom into the neighbor bag
and sum the flat multiset, I lose which element was the root. The middle atom of the path `a–b–b` sees
the flat bag `{a,b,b}`, and so does the middle of `b–a–b` — same multiset, different molecules — so I tag
the center, `(1+ε)·h_v + Σ_u h_u`, which stays injective for irrational `ε` (the center's contribution
lands on an incommensurable scale that the integer neighbor-counts can never reproduce) and is a
learnable scalar in practice. The maximally expressive fill is therefore: sum the neighbors, add
`(1+ε)` times the center, push through an MLP, and read out by summing — ideally over every layer,
jumping-knowledge style, so no depth's distinctions are discarded.

So the theory points hard at *sum* pooling and a sum readout. But I am building the **scaffold's starter
fill**, and it deliberately departs from the theoretical maximum in three ways, each of which I'll keep,
because the point of rung one is a deliberately weak, honest floor — and I want the departures named so
the next rung can react to them rather than to a paper. First, the per-layer message folds in the **bond
features**: `msg = h_v[src] + edge_proj(e_vw)`, then summed into the destination atom. The clean WL-
maximal GIN ignores edges entirely; here the edges (single/double/aromatic, conjugated, in-ring) are
chemically load-bearing, so the message is edge-aware — an additive bond bias before the sum. Second, the
per-layer aggregation over neighbors is a sum (good, injective), and the center is tagged by `(1+ε)`, so
each `GINConv` is `MLP((1+ε)·x + Σ_{N(v)}(x[u]+edge))` with a BatchNorm inside the MLP — that part is
faithful to the expressive recipe. Third — and this is the load-bearing weakness — the *graph readout is
**mean** pooling*, `Σ_v h_v / |V|`, off the **last** layer only, not a sum and not concatenated across
depths. By the ranking I just computed, mean readout throws away exactly the multiplicity information sum
keeps; it hands the head only the *proportions* of local environments, blind to a molecule versus a scaled
copy of its substructure distribution. The starter trades the provably-maximal readout for the simplest
one, and it is a deliberate choice, not an oversight — it is the single knob I most want to have measured
before I turn it.

It is worth grounding that mean-readout weakness in a molecule rather than leaving it as an abstract
statistic, because the collapse it causes is exactly the kind BBBP will punish. Take a molecule `M` and
imagine a larger homolog `M'` that repeats each of `M`'s local environments — a symmetric dimer, or a
chain extended by a chemically-identical repeat unit. Every rooted subtree that appears in `M` appears in
`M'`, just with doubled multiplicity, so the *multiset* of atom vectors in `M'` is (to first order) two
copies of `M`'s. Mean pooling divides by atom count and returns the **same** graph vector for `M` and
`M'`; sum pooling returns `M`'s vector doubled. The property, though, is not scale-invariant: blood-brain
penetration falls off sharply once a molecule grows past roughly 450 daltons and its polar surface area
climbs, so `M` and `M'` can sit on opposite sides of the barrier while the mean-pooled encoder hands the
head one identical vector for both. This is the same fact as "mean keeps only the distribution," but now
it says something operational — the encoder is structurally blind to the *size* axis that a global,
extensive property like BBB penetration turns on. And it connects to why sum is the graph-native fix:
polar surface area itself is literally a *sum* over per-atom polar contributions, an extensive quantity, so
a readout that sums atom vectors can in principle recover it while a readout that averages has divided the
signal away before the head ever sees it. That is the precise sense in which mean is the load-bearing
weakness, and it is deliberate.

The edge-aware message deserves the same scrutiny, since it is the one place the starter is *more* than a
textbook GIN. Folding the bond features in additively — `x[src] + edge_proj(e_vw)` before the sum — is a
low bias way to let the same pair of atoms send a different message across a single bond than across a
double or aromatic one, which matters because chemistry that a bare topology hides lives in the bond
labels: a carbonyl versus a hydroxyl, an aromatic ring versus a saturated one, a bond inside a ring versus
a freely-rotating one. It is additive rather than, say, a gated or bilinear interaction because at rung one
I want the cheapest edge-awareness that keeps the sum injective (adding a bond code to each neighbor code
before summing preserves distinctness of the summed multiset), and richer bond-atom coupling is a knob I
would rather leave for a rung that has a reason to turn it. So the message is expressive and honest; it is
only the readout that I am holding lossy on purpose.

It wraps the four layers in a residual stack, `x ← x + dropout(relu(norm(conv(x))))`, and this is worth a
second of thought because it is what keeps the floor from being *even weaker* than intended. Repeated
neighbor-averaging is a smoothing operator, and iterating a smoothing operator drives every node's feature
toward the same fixed point — the leading eigenvector of the propagation — so after enough naive rounds a
deep GNN's atoms all look alike and the encoder loses the very distinctions WL would keep. Four rounds is
already deep enough for this over-smoothing to start biting; the residual connection lets each layer add a
*correction* to the previous representation rather than overwrite it, so an atom's identity survives the
stack instead of washing into the graph mean, and BatchNorm keeps the per-layer activations at a scale
where gradients actually flow. Hidden width 256, four layers, dropout 0.1, a two-layer FFN head to the
`num_tasks` logits.

Before I run it I want to size it against the data, because the scaffold split is going to punish anything
that has to learn its whole story from a handful of molecules. Count the parameters: the atom embedding is
`136×256 ≈ 35k`; each `GINConv` carries two `256×256` linears (~66k each), a `9×256` edge projection, and
a BatchNorm, about `135k` per layer, so four layers are `~540k`; the head adds another `~66k`. The whole
encoder is on the order of `6.4×10^5` trainable weights. BBBP's scaffold split leaves roughly `1.6k`
training molecules, so this floor carries about **400 parameters for every training example it will ever
see** — wildly over-parameterized, entirely dependent on the inductive bias of message passing plus
BatchNorm/dropout to not simply memorize. On a *random* split that memorization would even score, because
the test scaffolds would rhyme with training ones; on a *scaffold* split it buys nothing, so the 400:1
ratio is not a strength waiting to be unlocked, it is a warning that this encoder has almost no data to
pin down a global story and will lean on whatever local regularity it can find.

And "local" is the second structural limit, which I can also make concrete. Four rounds reach atoms four
bonds away; two atoms co-influence a single atom's final vector only if one sits inside the other's 4-hop
ball, and a pharmacophore whose two ends are, say, eight bonds apart is never jointly resolved in *any*
atom's representation. Take the degenerate case of a linear chain of `L` heavy atoms: its graph diameter
is `L−1`, so the two ends exchange information only when `L−1 ≤ 4`, i.e. `L ≤ 5`. Drug-like molecules
routinely have a longest path well past five atoms, so the pooled vector is a sum-then-*average* of local
views that never saw the molecule whole. The encoder is, by construction, local (receptive field smaller
than the molecule), data-hungry (400:1), and prior-free (no pretrained weights, none of the
physicochemistry the fixed-descriptor camp gets for free) — and the mean readout then discards the size
and count signal on top of all that.

Let me reason about what this floor should do, dataset by dataset, because the whole reason to run it is to
watch *which* task the weaknesses bite. I expect the three to split on how much the property is decided by
*local* chemistry that a mean-pooled bank of four-hop neighborhoods can still capture under a hard scaffold
shift. Tox21 is twelve assays with the most molecules; even a weak local encoder gets enough signal, and
multi-task averaging over twelve heads stabilizes it, so I'd expect it to be the best of the three at this
rung. BACE is a single, fairly structured enzyme-inhibition target where the label is carried by a local
binding-pocket substructure, so a local encoder should get real traction. BBBP is the danger, and the
reasoning above says exactly why: it is a single binary target whose answer is *global* whole-molecule
physicochemistry — lipophilicity, polar surface area, molecular size — precisely the extensive, whole-
molecule quantities that a four-hop encoder cannot assemble and that a mean readout then normalizes away.
Handed the one task its three weaknesses all bite at once, on the split that forbids memorization, I would
not be surprised to see this rung land near *chance*, ROC-AUC ≈ 0.5, on BBBP. I am not certain of the
exact number — the residual stack and edge-aware message might scrape a little signal — but I expect BBBP
to be far the weakest of the three, and if it does collapse toward 0.5 while BACE and Tox21 stay clearly
above it, the diagnosis is already written and points two ways at once: the readout is throwing away count
information (swap mean for sum, reach back to raw bond identity instead of letting four tied-ish rounds
wash it out), and there is a cheap global prior — molecule-level descriptors — sitting entirely unused
that would patch exactly the locality/no-prior failure on a task like BBBP. That is the next rung, and the
BBBP number is the falsifiable pivot: a near-0.5 there confirms "lossy readout + no global prior on a
global task," whereas a BBBP that comes in respectable would tell me the mean-pool floor is stronger than
I think and I should look elsewhere.

So at rung one the encoder is settled and my edit is the *default* one: the scaffold's starter GIN —
edge-aware sum-of-neighbors message with a `(1+ε)` center tag and an MLP, four residual layers with
BatchNorm, **mean** pooling off the last layer, a two-layer head (the distilled module is in the answer).
It is the floor by construction: maximally expressive in the message, deliberately lossy in the readout,
local in receptive field, over-parameterized against the labels, and prior-free — and on a scaffold split,
on the task most decided by global physicochemistry, it should show me exactly how much a plain learned
encoder leaves on the table.

The causal chain in one breath: I want the simplest learned graph encoder whose representational power I
can actually reason about *and* whose failures I can attribute one knob at a time, so I measure the family
against the WL test → the bound is two-sided: no message-passing GNN separates what WL cannot, and it
*reaches* WL only when aggregation, combine, and readout are injective, which forces sum over mean over max
(shown on `(1,1)` vs `(2,2)`), an MLP over a single linear layer (shown on `{1,1,1,1,1}` vs `{2,3}`), and
a `(1+ε)` center tag → but the scaffold's starter fill deliberately keeps the injective sum *message*
(edge-aware) while using the lossy **mean** *readout* off the last layer, no jumping-knowledge, no external
prior → so this floor is local (four hops < molecule diameter), data-hungry (~400 params per training
molecule), and prior-free, and on a scaffold split it should be best on Tox21, decent on BACE, and near
chance on BBBP — and that collapse is what would force, at rung two, a sum readout that keeps counts plus a
global descriptor prior to patch the locality.
