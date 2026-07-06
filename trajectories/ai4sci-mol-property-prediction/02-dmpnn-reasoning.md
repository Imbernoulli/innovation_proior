The starter GIN told me exactly what's missing, and it told me in numbers. BBBP came in at **0.5099** —
that is *chance*; a coin flip would score the same, and the 0.0099 above 0.5 is inside the noise I'd get
from reshuffling a few borderline predictions. BACE held at **0.7261** and Tox21 at **0.7470**, both
clearly above chance, so the encoder can learn *something* — it is not broken across the board. Let me
read the split as a spacing, because the spacing is the diagnosis. BBBP sits `0.216` below BACE and
`0.237` below Tox21; those are not the kind of gaps that random seed variation opens up between three
tasks trained by the same pipeline, they are a structural cliff. The two tasks where local substructure
carries the label — an enzyme-inhibition pocket in BACE, twelve toxicity assays in Tox21 with enough
molecules that multi-task averaging stabilizes a weak encoder — survive; the single binary target whose
answer is decided by *global* whole-molecule physicochemistry — does this thing cross the blood-brain
barrier, which is mostly a story about lipophilicity, polar surface area, size — falls all the way to the
coin flip on the scaffold split. So this is not a learning-rate problem and not a credit-assignment
problem; if it were, all three would sag together, not one cliff-edge while two hold. It is two structural
problems at once, both of which I predicted and can now read straight off the GIN design: the readout
*mean*-pools and throws away the count information sum keeps, and the four-hop receptive field with no
external prior simply cannot see the global property BBBP turns on. The 0.51 is the tell — the encoder
had nothing global to hold onto and nothing to memorize on a scaffold split, so it guessed. I have to fix
the message passing *and* hand the model the global prior it lacked, and I should be honest that those are
two independent moves aimed at two independent failures.

Before I add anything, take the message passing itself, because there is a flaw in the GIN-style update
deeper than the readout, and I want to see it in a concrete molecule rather than in the abstract. Take the
three-atom path `a–b–c`. At step 1 the atom-centered update forms `h_b^1 = COMBINE(h_b, h_a + h_c)` and
`h_a^1 = COMBINE(h_a, h_b)` — so `h_b^1` now carries a piece of `h_a`, the message `a` sent one bond over.
At step 2, `h_a^2 = COMBINE(h_a^1, h_b^1)`, and `h_b^1` still contains that piece of `h_a`. So `h_a^2`
pulls back the very message `a` emitted two steps ago: it has traveled `a→b→a` and returned. Walks that go
out and immediately back — `v→w→v`, the pattern `v_i = v_{i+2}` — are called *totters*; they carry no new
information, they re-mix what the atom already had, and they happen on every edge at every step. The
representation is steadily polluted with echoes of itself. And the damage is directional, not just
redundant: each atom keeps re-injecting a scaled copy of its *own* earlier state, so the update nudges
`h_a` back toward the subspace it already occupies rather than toward genuinely new two-hop structure.
Across the graph the layer-to-layer change comes to be dominated by these self-echoes, the atom features
grow more correlated with their own history than with their surroundings, and the amount of *new* structure
each round contributes shrinks. That is not a corner case; it is the structural disease of *atom-centered*
message passing, and it is one reason a plain GIN's deep features wash out into a uniform blob — the same
over-smoothing I flagged at rung one, now with a named mechanism — which the residual stack only partly
hides.

I half-recognize the cure, because I've seen it in belief propagation: the message from node `v` to a
neighbor `w` is built from everything `v` has heard *except* what came from `w`. That exclusion is the
whole reason BP doesn't double-count. The atom-centered update violates it — it sums over all of `N(v)`
with no exclusion, so it tells atom `a` what atom `a` just said. But to drop the offending term the update
has to know which neighbor the message is *going to*, and a single hidden vector sitting on an undirected
atom is direction-blind: `h_a` cannot simultaneously be "what `a` says to `b`" and "what `a` says to
everyone else." The state is in the wrong place. So move it onto **directed bonds**: keep `h_vw` and
`h_wv` as distinct objects. Now "the message heading from `v` out to `w`" is a first-class thing, and I
build it from the *other* directed bonds flowing into `v`, skipping the reverse:
`m_vw = Σ_{k∈N(v)\w} h_kv`. On the `a–b–c` path, the message forming the directed bond `b→c` is
`Σ_{k∈N(b)\c} h_kb = h_ab` — it excludes `h_cb`, so the `c→b→c` echo never forms. The reverse message
drops out by construction, exactly the term that caused the totter. This is the directed-edge form of the
loopy-BP embedding; the atom-centered GIN was the mean-field embedding, the one with the totter, and the
choice between them is the choice between embedding mean-field and embedding loopy BP — loopy BP is the
one that respects the exclusion.

Make the rest concrete. Initialize each directed bond from the atom it leaves and the bond's own features,
`h_vw^0 = ReLU(W_i·[x_v ; e_vw])` — mixing source atom and bond features inside one matrix so their
interaction survives, because a double bond *in a ring* is a different chemical object than a double bond
*not* in a ring and I want that coupling present from step zero. The message function is trivial — the
message along `k→v` is just `h_kv` — and all the learning sits in a tied update,
`h_vw^{t+1} = ReLU(h_vw^0 + W_m·m_vw)`, the *same* `W_m` every round so depth is a free hyperparameter, and
a skip back to `h_vw^0` at every step so the raw bond identity never washes out under the tied recurrence
(the exact failure mode I just watched bite GIN's deep features). After `T` rounds I return to atoms by
summing the directed bonds that *end* at each atom, `m_v = Σ_{w∈N(v)} h_wv`, re-inject the atom features,
`h_v = ReLU(W_a·[x_v ; m_v])`, and — this is the second fix to GIN's readout — pool the atoms by **sum**,
not mean. Sum is the injective multiset aggregator; mean keeps only the distribution, and the distribution
is exactly what was insufficient on BBBP, where the label rides on extensive, whole-molecule quantities
that a per-atom average has divided away. The molecule vector `Σ_v h_v` keeps the counts.

The exclusion looks expensive — for every directed bond, a sum over the source atom's neighbors minus one
— but it factors, and it is worth doing the arithmetic so I know the architectural upgrade is actually
free. The sum over `N(v)\w` is the sum over *all* of `N(v)` minus the one excluded term, and the full sum
over incoming bonds at `v` does not depend on `w`: compute `a_v = Σ_{k∈N(v)} h_kv` once per atom, then for
each outgoing bond `v→w` the message is `a_v − h_wv`, one subtraction. The naive version costs, per atom,
`deg(v)·(deg(v)−1)` message terms — quadratic in degree — so the total is `O(Σ_v deg(v)^2)`; the factored
version is one length-`deg(v)` sum per atom plus one subtraction per directed bond, `O(Σ_v deg(v)) = O(E)`.
Concretely a degree-4 atom (a quaternary carbon) drops from `4·3 = 12` message terms to a `4`-term sum plus
`4` subtractions, and the gap widens as degree grows. To find the reverse bond instantly, store bonds in
adjacent forward/reverse pairs so the reverse of bond `e` is `e XOR 1` — check it: pair `(a→b)=0,(b→a)=1`
gives `0^1=1` and `1^1=0`; pair `(b→c)=2,(c→b)=3` gives `2^1=3` and `3^1=2`; the involution is exact. The
scaffold's `edge_index` already lays bonds out this way, so the directed scheme costs essentially the same
as a plain atom aggregation. The architectural advantage is free.

It is worth taking the exclusion to its limiting case to be sure it does what I claim. On an *acyclic*
fragment — a tree, which is what most drug side-chains and linkers are — belief propagation with the
reverse-message exclusion is *exact*: every message travels each edge once in each direction and never
revisits, so there is provably no double counting and no totter, and the directed encoder is echo-free on
those substructures by construction. The approximation only re-enters inside *rings*, where a message can
loop back around the cycle rather than straight back along its own bond, and there the scheme becomes loopy
BP rather than exact BP. That is the honest scope of the fix: the reverse-exclusion kills the length-2
`v→w→v` totters everywhere, exactly and for free, and leaves only the longer ring-length walks as residual
mixing — a strict improvement over the atom-centered update that tottered on every bond, acyclic or not.

Now I have to decide how much to change at once, and there is a real design choice hiding here. I could
make the minimal edit — swap GIN's mean readout for a sum and change nothing else — which cleanly tests the
readout in isolation, but it leaves the totter untouched and, more to the point, does nothing for BBBP's
*global-prior* gap, so the single-target task would stay near chance and I'd have learned little. I could
make the other minimal edit — bolt molecule-level descriptors onto the existing GIN — which tests the prior
in isolation but keeps the tottering mean-pool encoder muddying the local tasks. Or I fix both diagnosed
failures in one rung: move to directed bonds (kills the totter), sum-pool (keeps counts), and concatenate a
descriptor prior (patches the global gap). Changing two things at once carries the usual confound — if the
aggregate moves I can't attribute it to one cause — but the two changes target *disjoint* task-failures,
which is what saves the experiment: the descriptor branch is aimed squarely at BBBP's global-physicochemistry
gap, while the directed-bond-plus-sum rework is aimed at message quality on all three, so the per-task split
in the feedback will attribute for me. I hold off on jumping to a 3D or pretrained encoder deliberately —
I have not yet exhausted what the 2D bond graph can do, and if I leap the whole ladder now I will never be
able to say whether 2D message passing was truly capped or just clumsily arranged. So this rung stays inside
the 2D graph and spends its budget on the directed reformulation plus the prior.

Now the part that directly attacks BBBP's 0.51. Even fixed, this is a *local, data-hungry, prior-free*
encoder — `T≈3` hops is still smaller than a drug-molecule's diameter, and BBBP's answer lives in global
physicochemistry the message passing cannot reach in three hops, learned from a few hundred training
molecules. There is a cheap external source of exactly that global, prior-laden chemical knowledge sitting
unused: molecule-level RDKit 2D descriptors. And the mapping to BBBP is not vague — the medicinal-chemistry
rules of thumb for blood-brain penetration are almost a checklist over these very descriptors: molecular
weight under ~450, topological polar surface area under ~90 Å², computed LogP in roughly the 1–3 band,
fewer than ~3 hydrogen-bond donors, a modest count of rotatable bonds. Every one of those axes is a single
RDKit call — `MolWt`, `TPSA`, `MolLogP`, `NumHDonors`, `NumRotatableBonds` — so a compact panel of them
(molecular weight, LogP, TPSA, H-bond donor/acceptor counts, rotatable bonds, aromatic and aliphatic ring
counts, fraction of sp3 carbons, heteroatom count, molar refractivity, Labute ASA, the saturated- and
heterocycle counts) *is* the lipophilicity/size/polarity story BBBP turns on, handed to the model for free.
So compute a fixed-length RDKit descriptor vector per molecule and **concatenate it to the graph vector
before the head**: `ŷ = head([h ; h_f])`. This is a hybrid — the learned message-passed `h` supplies
task-specific, locally resolved structure; the fixed `h_f` supplies a global chemical prior that needs
neither a large `T` nor much data. On a tiny single-target task like BBBP, `h_f` is the regularizer that
hands the model a view of chemistry it could never learn from a few hundred scaffolds, and it reaches
across the whole molecule where three-hop message passing cannot. This is the patch for precisely the
failure I measured.

It matters that the prior I bolt on is *global physicochemistry* specifically, and not just any fixed
featurization, so let me rule out the tempting alternatives on the axis they'd actually add. An ECFP/Morgan
fingerprint is also a fixed, zero-training prior, but it is a bag of *local* circular substructures — the
same kind of information the message passing already extracts, just hashed — so it would reinforce the axis
the encoder is already good at while doing nothing for the *whole-molecule* quantities BBBP needs; wrong
axis. Simply stacking more graph layers extends the receptive field, but it stays local, stays
data-hungry, and cannot invent lipophilicity or polar surface area out of a few hundred labels; wrong cure.
A learned global-attention pool could in principle assemble a molecule-level summary, but it is *learned*,
so on a single-target set of a few hundred scaffolds it inherits the very data-hunger I'm trying to route
around. The RDKit descriptor panel is the one option that adds a genuinely *new, global, and prior-laden*
axis at zero data cost — molecular weight, LogP, TPSA, donor/acceptor counts are computed from cheminformatics
rules distilled over decades, not fit to my 1.6k training molecules — which is exactly why it is the right
concatenation for the failure I measured rather than a reflex toward "add more features."

The descriptors have wildly different scales — a molecular weight in the hundreds next to a fraction in
`[0,1]` next to an integer ring count — so I must standardize them before they meet the head, or the
large-range features drown the small ones and the head effectively only ever sees molecular weight. The
clean version would map each descriptor through its CDF (a percentile in `[0,1]`, identical in meaning
across features, immune to outliers and to non-normal count distributions), fit once on a large background.
In *this* task's edit surface I approximate that with a BatchNorm-style running normalizer on the
descriptor branch: a per-feature running mean/std updated as batches arrive, so train and test see the
same scaling — the streaming standardization, not a precomputed CDF table. One honest limitation I'll
respect rather than hide: the harness only reliably exposes SMILES to this branch through `batch._smiles`;
when that attribute is absent the descriptor branch falls back to a zero vector. Let me check that the
fallback is genuinely inert rather than injecting noise: an all-zeros batch through the running normalizer
maps to `(0 − running_mean)/running_std`, which is a *constant* vector across every molecule in the batch —
not literally zero, but per-molecule uninformative — and a constant fed into the linear head is absorbed
into that head's bias term, so it shifts the intercept and carries no molecule-specific signal. The pure
GNN branch then supplies all the per-molecule variation, which is the behavior I want: the descriptor lift
is real where SMILES are available and harmlessly degenerate where they are not. I also keep the descriptor
set deliberately compact (the seventeen above) rather than the full few-hundred CDF-normalized panel,
because the compact set is the chemically load-bearing subset and is robust to compute in-loop.

The remaining knobs follow from something, not a hat. Depth `T = 3`: enough hops to build a useful local
environment around each atom without the tied recurrence over-smoothing or, even with the skip, washing
out distinctions — and, because `W_m` is tied across rounds, depth costs no extra parameters, so `T=3` is
as cheap as `T=1`. That tying is worth pricing out: the encoder's weight is one `W_i` on `[atom;bond]→300`
(~44k), one shared `W_m` of `300×300` (~90k) reused at every step, and one `W_o` on `[atom;msg]→300`
(~131k); with the head that is on the order of `3.6×10^5` trainable weights — *fewer* than rung one's GIN,
even though depth-3 directed message passing is strictly more expressive, precisely because tying `W_m`
means depth-3 does not triple that `90k` the way an untied stack would (it would have cost `~180k` more).
Hidden width 300, the usual sweet spot for these molecule sizes. **Sum** pooling for the graph readout
(keeps counts), with the head a two-layer FFN over `[h ; h_f]`. Dropout I'll leave near zero by default and
let the driver lift it per dataset where overfitting bites — the small single-target sets want a little,
the regression-style targets more. The loss is the fixed pipeline's masked BCE for the multi-task Tox21
(absent assays contribute no gradient), and the scaffold split stays — the architecture and the evaluation
are answering the same question (does the representation transfer to new chemistry) from two ends.

So the delta from rung one is concrete: where GIN summed over *all* neighbors (tottering) and mean-pooled
(losing counts) with no global prior, I move the state onto directed bonds and exclude the reverse message
(BP-style, no totter), add the tied-update-plus-skip so depth is free and bond identity survives,
**sum**-pool to keep counts, and concatenate a running-normalized RDKit descriptor vector so the global
physicochemistry BBBP needs is handed in directly. Reading GIN's shape, here is what I expect and where I'm
exposed. BBBP is the falsifiable claim: GIN sat at chance (0.51); if the descriptor branch is really
supplying the global prior BBBP turns on, BBBP should move *clearly off chance* — it is the metric that
most directly tests the diagnosis, and if BBBP stays near 0.5 my story about "global prior fixes it" is
wrong. BACE should at least hold and likely improve as the directed message passing kills the totter that
was muddying its local substructure; Tox21 I expect to hold around GIN's 0.75 or edge up, since it had the
most data and was never the bottleneck. The honest risk I can already name cuts against those last two:
the directed-bond rework is motivated by a totter that may simply not have been *their* bottleneck — they
were already healthy under GIN — so it is entirely possible the rework buys them nothing measurable and the
descriptor branch is the only real mover, in which case BACE and Tox21 sit roughly where they were. And a
sharper exposure: the descriptor branch only helps where `batch._smiles` is exposed at finetune; if it is
not, the model is just a directed-bond MPNN with sum pooling — better-formed than GIN's tottering mean-pool
encoder, but without the global lift, in which case BBBP improves only modestly rather than dramatically.
If *that* is what I see — BBBP creeping rather than leaping — the diagnosis for the next rung is already
written: the cure isn't more 2D graph machinery at all, it's the *3D geometry* the property actually
depends on, plus pretraining to escape the data-hunger — feed the model atoms-in-space instead of a 2D
graph.

The causal chain in one breath: GIN's measured failure is structural — chance (0.51) on BBBP, a full
`0.22–0.24` below BACE (0.73) and Tox21 (0.75), because a *mean*-pooled, four-hop, prior-free encoder
cannot see global physicochemistry, while the two local-substructure tasks survive → fix the message
passing by moving state onto directed bonds and excluding the reverse term (`m_vw = a_v − h_wv`, factored to
`O(E)`, reverse via `e XOR 1`), killing the totter, with a tied update plus skip so depth is free and bond
identity survives → fix the readout by **sum**-pooling to keep the counts mean threw away → patch the
locality/no-prior gap that BBBP exposed by concatenating a running-normalized RDKit 2D descriptor vector
whose axes *are* the BBB checklist (MW, TPSA, LogP, HBD), dropping gracefully to the pure GNN where SMILES
aren't exposed → expecting BBBP to move clearly off chance, BACE to hold/improve, Tox21 to hold — and if
BBBP barely moves, the next rung must abandon 2D graphs for the 3D geometry plus pretraining.
