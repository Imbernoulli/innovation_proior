The starter GIN told me exactly what's missing, and it told me in numbers. BBBP came in at **0.5099** —
that is *chance*; a coin flip would score the same. BACE held at 0.726 and Tox21 at 0.747, both clearly
above chance, so the encoder can learn *something* — it is not broken across the board. The split is
sharp and it is informative: the two tasks where local substructure carries the label (an enzyme-
inhibition pocket in BACE, twelve toxicity assays in Tox21 with enough molecules that multi-task
averaging stabilizes it) survive a weak local encoder; the single binary target whose answer is decided
by *global* whole-molecule physicochemistry — does this thing cross the blood-brain barrier, which is
mostly a story about lipophilicity, polar surface area, size — collapses to a coin flip on the scaffold
split. So this is not a learning-rate problem and not a credit-assignment problem. It is two structural
problems at once, both of which I can read straight off the GIN design: the readout *mean*-pools and
throws away the count information sum keeps, and the four-hop receptive field with no external prior
simply cannot see the global property that BBBP turns on. The 0.51 is the tell — the encoder had nothing
global to hold onto and nothing to memorize on a scaffold split, so it guessed. I have to fix the message
passing *and* hand the model the global prior it lacked.

Take the message passing first, because there is a flaw in the GIN-style update deeper than the readout.
The starter aggregates, for each atom, a sum over *all* its neighbors — including, at the next step, the
atom it just sent a message to. Trace one edge between atoms 1 and 2. At step `t` atom 2 forms its
message by summing over its neighbors, which includes atom 1, so `h_2^{t+1}` contains a piece of `h_1`.
Next step atom 1 sums over *its* neighbors, which includes atom 2, so atom 1 pulls back `h_2^{t+1}` —
the piece of `h_1` it just sent over. The message atom 1 sent out one bond has bounced right back along
the same bond. Walks that go out and immediately back — `v→w→v`, the pattern `v_i = v_{i+2}` — are
called *totters*; they bring in no new information, they re-mix what the atom already had, and they do it
on every edge, every step. The representation gets steadily polluted with echoes of itself. That is not a
corner case; it is the structural disease of *atom-centered* message passing, and it is one reason a
plain GIN's deep features wash out into a uniform blob (which the residual stack only partly hides).

I half-recognize the cure, because I've seen it in belief propagation: the message from node `v` to a
neighbor `w` is built from everything `v` has heard *except* what came from `w`. That exclusion is the
whole reason BP doesn't double-count. The atom-centered update violates it — it sums over all of `N(v)`
with no exclusion, so it tells atom 1 what atom 1 just said. But to drop the offending term the update
has to know which neighbor the message is *going to*, and a single hidden vector on an undirected atom is
direction-blind. The state is in the wrong place. So move it onto **directed bonds**: keep `h_vw` and
`h_wv` distinct. Now "the message heading from `v` out to `w`" is a first-class object, and I build it
from the *other* directed bonds flowing into `v`, skipping the reverse:
`m_vw = Σ_{k∈N(v)\w} h_kv`. The reverse message `h_wv` drops out by construction — exactly the term that
caused the totter. This is the directed-edge form of the loopy-BP embedding; the atom-centered GIN was
the *mean-field* embedding, the one with the totter. The choice between them is the choice between
embedding mean-field and embedding loopy BP, and loopy BP is the one that respects the exclusion.

Make the rest concrete. Initialize each directed bond from the atom it leaves and the bond's own
features, `h_vw^0 = ReLU(W_i·[x_v ; e_vw])` — mixing source atom and bond features inside one matrix so
their interaction survives (a double bond *in a ring* is different from a double bond *not* in a ring).
The message function is trivial — the message along `k→v` is just `h_kv` — and all the learning sits in
a tied update: `h_vw^{t+1} = ReLU(h_vw^0 + W_m·m_vw)`, the *same* `W_m` every round so depth is a free
hyperparameter, and a skip back to `h_vw^0` at every step so the raw bond identity never washes out under
the tied recurrence (the exact failure mode I just watched bite GIN's deep features). After `T` rounds I
return to atoms by summing the directed bonds that *end* at each atom, `m_v = Σ_{w∈N(v)} h_wv`, re-inject
the atom features, `h_v = ReLU(W_a·[x_v ; m_v])`, and — this is the second fix to GIN's readout — pool
the atoms by **sum**, not mean. Sum is the injective multiset aggregator; mean keeps only the
distribution, and the distribution is exactly what was insufficient on BBBP. The molecule vector `Σ_v h_v`
keeps the counts.

The exclusion looks expensive — for every directed bond, sum over the source atom's neighbors minus one —
but it factors. The sum over `N(v)\w` is the sum over *all* of `N(v)` minus the one excluded term, and
the sum over all incoming bonds at `v` doesn't depend on `w`: compute `a_v = Σ_{k∈N(v)} h_kv` once per
atom, then for each outgoing bond `v→w` the message is `a_v − h_wv`. One subtraction per bond. To find
the reverse bond instantly, store bonds in adjacent forward/reverse pairs so the reverse of bond `e` is
`e XOR 1` — and the scaffold's `edge_index` already lays bonds out this way, so the directed scheme costs
essentially the same as a plain atom aggregation. The architectural advantage is free.

Now the part that directly attacks BBBP's 0.51. Even fixed, this is a *local, data-hungry, prior-free*
encoder — `T≈3` hops is still smaller than a drug-molecule's diameter, and BBBP's answer lives in global
physicochemistry the message passing cannot reach in three hops, on a few hundred training molecules.
There is a cheap external source of exactly that global, prior-laden chemical knowledge sitting unused:
molecule-level RDKit 2D descriptors. The fixed-descriptor camp would hand the model molecular weight,
LogP, topological polar surface area, H-bond donor/acceptor counts, rotatable bonds, aromatic and
aliphatic ring counts, fraction of sp3 carbons, heteroatom count, molar refractivity, Labute ASA — a
compact set that *is* the lipophilicity/size/polarity story BBBP turns on. So compute a fixed-length
RDKit descriptor vector per molecule and **concatenate it to the graph vector before the head**:
`ŷ = head([h ; h_f])`. This is a hybrid — the learned message-passed `h` supplies task-specific, locally
resolved structure; the fixed `h_f` supplies a global chemical prior that needs neither a large `T` nor
much data. On a tiny single-target task like BBBP, `h_f` is the regularizer that hands the model a view
of chemistry it could never learn from a few hundred scaffolds, and it reaches across the molecule where
three-hop message passing cannot. This is the patch for precisely the failure I measured.

The descriptors have wildly different scales — a molecular weight in the hundreds next to a fraction in
`[0,1]` next to an integer ring count — so I must standardize them before they meet the head, or the
large-range features drown the small ones. The clean version would map each through its CDF (a percentile
in `[0,1]`, identical in meaning across features, immune to outliers and to non-normal count
distributions), fit on a large background once. In *this* task's edit surface I approximate that with a
BatchNorm-style running normalizer on the descriptor branch: a per-feature running mean/std updated as
batches arrive, so train and test see the same scaling — the streaming standardization, not a precomputed
CDF table. One honest limitation I'll respect rather than hide: the harness only reliably exposes SMILES
to this branch through `batch._smiles`; when that attribute is absent the descriptor branch falls back to
a zero vector, and after the running normalizer that contributes nothing, leaving the pure GNN branch — so
the descriptor lift is real where SMILES are available and gracefully absent where they are not. I also
keep the descriptor set deliberately compact (the seventeen above) rather than the full few-hundred CDF-
normalized panel, because the compact set is the chemically load-bearing subset and is robust to compute
in-loop.

The remaining knobs follow from something, not a hat. Depth `T = 3`: enough hops to build a useful local
environment around each atom without the tied recurrence over-smoothing or, even with the skip,
washing out distinctions — and the cost is linear in `T`. Hidden width 300, the usual sweet spot for
these molecule sizes. **Sum** pooling for the graph readout (keeps counts), with the head a two-layer FFN
over `[h ; h_f]`. Dropout I'll leave near zero by default and let the driver lift it per dataset where
overfitting bites — the small single-target sets (BACE, BBBP) want a little, the regression-style targets
more. The loss is the fixed pipeline's masked BCE for the multi-task Tox21 (absent assays contribute no
gradient), and the scaffold split stays — the architecture and the evaluation are answering the same
question (does the representation transfer to new chemistry) from two ends.

So the delta from rung one is concrete: where GIN summed over *all* neighbors (tottering) and mean-pooled
(losing counts) with no global prior, I move the state onto directed bonds and exclude the reverse
message (BP-style, no totter), add the tied-update-plus-skip so depth is free and bond identity survives,
**sum**-pool to keep counts, and concatenate a running-normalized RDKit descriptor vector so the global
physicochemistry BBBP needs is handed in directly. Reading GIN's shape, here is what I expect and where
I'm exposed. BBBP is the falsifiable claim: GIN sat at chance (0.51); if the descriptor branch is really
supplying the global prior BBBP turns on, BBBP should move *clearly off chance* — it is the metric that
most directly tests the diagnosis, and if BBBP stays near 0.5 my story about "global prior fixes it" is
wrong. BACE should at least hold and likely improve as the directed message passing kills the totter
that was muddying its local substructure. Tox21 I expect to hold around GIN's 0.75 or edge up — it had
the most data and was never the bottleneck. The risk I can already name: the descriptor branch only helps
where `batch._smiles` is exposed, and if it is *not* exposed at finetune the model is just a directed-bond
MPNN with sum pooling — better than GIN's tottering mean-pool encoder, but without the global lift, in
which case BBBP improves only modestly rather than dramatically. If *that* is what I see, the diagnosis
for the next rung is already written: the cure isn't more 2D graph machinery at all, it's the *3D
geometry* the property actually depends on, plus pretraining to escape the data-hunger — feed the model
atoms-in-space instead of a 2D graph.

The causal chain in one breath: GIN's measured failure is structural — chance (0.51) on BBBP because a
*mean*-pooled, four-hop, prior-free encoder cannot see global physicochemistry, while BACE (0.73) and
Tox21 (0.75) survive on local substructure → fix the message passing by moving state onto directed bonds
and excluding the reverse term (`m_vw = a_v − h_wv`), killing the totter, with a tied update plus skip so
depth is free and bond identity survives → fix the readout by **sum**-pooling to keep the counts mean
threw away → patch the locality/no-prior gap that BBBP exposed by concatenating a running-normalized RDKit
2D descriptor vector (the global lipophilicity/polarity/size prior), dropping gracefully to the pure GNN
where SMILES aren't exposed → expecting BBBP to move clearly off chance, BACE to hold/improve, Tox21 to
hold — and if BBBP barely moves, the next rung must abandon 2D graphs for the 3D geometry plus pretraining.
