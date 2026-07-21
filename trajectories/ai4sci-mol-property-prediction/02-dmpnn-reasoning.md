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
— but it factors. The sum over `N(v)\w` is the full sum over `N(v)` minus the one excluded term, and the
full sum does not depend on `w`: compute `a_v = Σ_{k∈N(v)} h_kv` once per atom, then for each outgoing
bond `v→w` the message is `a_v − h_wv`, one subtraction. The naive version costs `deg(v)·(deg(v)−1)`
message terms per atom — quadratic in degree, `O(Σ_v deg(v)^2)` total; the factored version is one
length-`deg(v)` sum per atom plus one subtraction per bond, `O(E)`. A degree-4 quaternary carbon drops
from `4·3 = 12` terms to a 4-term sum plus 4 subtractions. To find the reverse bond instantly, store bonds
in adjacent forward/reverse pairs so the reverse of `e` is `e XOR 1`; `edge_index` already lays them out
this way, so the directed scheme costs essentially the same as a plain atom aggregation.

The limiting case fixes the scope of the fix. On an *acyclic* fragment — a tree, which is what most drug
side-chains and linkers are — belief propagation with the reverse-message exclusion is *exact*: every
message travels each edge once in each direction and never revisits, so the directed encoder is echo-free
there by construction. The approximation only re-enters inside *rings*, where a message can loop around
the cycle rather than straight back along its own bond, and there the scheme becomes loopy BP. So
reverse-exclusion kills the length-2 `v→w→v` totters everywhere, and leaves only the longer ring-length
walks as residual mixing — a strict improvement over the atom-centered update that tottered on every bond.

I fix both diagnosed failures at once: move to directed bonds (kills the totter), sum-pool (keeps counts),
and concatenate a descriptor prior (patches the global gap). Changing two things carries the usual
confound, but the two changes target *disjoint* task-failures, which is what saves the experiment — the
descriptor branch is aimed squarely at BBBP's global-physicochemistry gap, the directed-bond-plus-sum
rework at message quality on all three, so the per-task split in the feedback will attribute for me. I
hold off on jumping to a 3D or pretrained encoder deliberately: I have not yet exhausted what the 2D bond
graph can do, and if I leap now I will never be able to say whether 2D message passing was truly capped or
just clumsily arranged. So this step stays inside the 2D graph and spends its budget on the directed
reformulation plus the prior.

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

It matters that the prior is *global physicochemistry* specifically, not just any fixed featurization. An
ECFP/Morgan fingerprint is also a zero-training prior, but it is a bag of *local* circular substructures —
what the message passing already extracts, just hashed — so it reinforces the axis the encoder is already
good at and does nothing for the whole-molecule quantities BBBP needs. Stacking more graph layers stays
local, stays data-hungry, and cannot invent lipophilicity or polar surface area from a few hundred labels.
A learned global-attention pool could assemble a molecule-level summary but is itself *learned*, so it
inherits the data-hunger I'm trying to route around. The RDKit descriptor panel is the one option adding a
genuinely new, global, prior-laden axis at zero data cost — MW, LogP, TPSA, donor/acceptor counts come
from cheminformatics rules distilled over decades, not fit to my 1.6k molecules.

The descriptors have wildly different scales — a molecular weight in the hundreds next to a fraction in
`[0,1]` next to an integer ring count — so I must standardize them before they meet the head, or the
large-range features drown the small ones and the head effectively only ever sees molecular weight. The
clean version would map each descriptor through its CDF (a percentile in `[0,1]`, identical in meaning
across features, immune to outliers and to non-normal count distributions), fit once on a large background.
In *this* task's edit surface I approximate that with a BatchNorm-style running normalizer on the
descriptor branch: a per-feature running mean/std updated as batches arrive, so train and test see the
same scaling — the streaming standardization, not a precomputed CDF table. One honest limitation:
the pipeline only reliably exposes SMILES to this branch through `batch._smiles`, and when that attribute
is absent the descriptor branch falls back to a zero vector. That fallback is inert rather than noisy: an
all-zeros batch through the running normalizer maps to `(0 − running_mean)/running_std`, a *constant*
vector across the batch — per-molecule uninformative — and a constant into the linear head is absorbed
into its bias, shifting the intercept with no molecule-specific signal. The pure GNN branch then supplies
all the per-molecule variation: the descriptor lift is real where SMILES are available and harmlessly
degenerate where they are not. I keep the descriptor set compact (the seventeen above) rather than the
full few-hundred CDF-normalized panel, because the compact set is the chemically load-bearing subset and
robust to compute in-loop.

The remaining knobs follow from something, not a hat. Depth `T = 3`: enough hops to build a useful local
environment without the tied recurrence over-smoothing distinctions away, and because `W_m` is tied across
rounds depth costs no extra parameters, so `T=3` is as cheap as `T=1`. Counting weights: one `W_i` on
`[atom;bond]→300` (~44k), one shared `W_m` of `300×300` (~90k) reused every step, one `W_o` on
`[atom;msg]→300` (~131k); with the head, ~`3.6×10^5` — *fewer* than the GIN's `6.4×10^5` even though
depth-3 directed message passing is strictly more expressive, because tying `W_m` means depth-3 does not
triple that `90k` the way an untied stack would. Hidden width 300. **Sum** pooling (keeps counts), head a
two-layer FFN over `[h ; h_f]`. Dropout near zero by default, lifted per dataset by the driver where
overfitting bites. The loss stays the pipeline's masked BCE for multi-task Tox21 (absent assays contribute
no gradient), and the scaffold split stays.

So the delta from the starter GIN: where it summed over *all* neighbors (tottering) and mean-pooled
(losing counts) with no global prior, I move the state onto directed bonds and exclude the reverse message
(no totter), add a tied update with a skip so depth is free and bond identity survives, **sum**-pool to
keep counts, and concatenate a running-normalized RDKit descriptor vector so the global physicochemistry
BBBP needs is handed in directly.

Here is what I expect and where I'm exposed. BBBP is the falsifiable claim: GIN sat at chance; if the
descriptor branch really supplies the global prior BBBP turns on, BBBP should move *clearly off chance* —
and if it stays near 0.5 my story is wrong. BACE should at least hold and likely improve as the directed
message passing kills the totter muddying its local substructure; Tox21, which had the most data and was
never the bottleneck, I expect to hold or edge up. The honest risk cuts against those last two: the totter
may simply not have been *their* bottleneck — they were already healthy under GIN — so the rework may buy
them nothing measurable and leave the descriptor branch the only real mover. And a sharper exposure: the
descriptor branch only helps where `batch._smiles` is exposed at finetune; if it is not, this is just a
directed-bond MPNN with sum pooling — better-formed than GIN's tottering mean-pool, but without the global
lift, so BBBP creeps rather than leaps. If that is what I see, the next move is not more 2D machinery at
all — it is the *3D geometry* the property actually depends on, plus pretraining to escape the
data-hunger.
