The vanilla CLIP run told me exactly where it breaks, and it told me in numbers. On DUD-E it is already
strong — AUROC 0.895, BEDROC 0.703, EF@0.5% 51.3 — because DUD-E's decoys are property-matched ZINC
molecules and a clean binder/non-binder cosine boundary separates them well; in-batch contrastive
learning is doing its job there. DEKOIS holds up too (AUROC 0.892, BEDROC 0.711). But LIT-PCBA is the
tell: AUROC collapses to 0.576 — barely above chance — and BEDROC craters to 0.065 with EF@0.5% at 7.27,
a seventh of the DUD-E number. That split is not noise; it is structural. LIT-PCBA is the benchmark
built to *remove* the artificial-decoy crutch — confirmed actives versus confirmed inactives at
realistic ratios, with graded potencies crowded together — and that is precisely the regime a pure
contrastive separation cannot rank, because it has no notion of "more active." Two actives of the same
target both get pulled toward it with equal force; nothing distinguishes a 10 nM binder from a 10 µM
one. The metric is top-heavy and LIT-PCBA's top ranks are decided by *ordering* graded actives, which
vanilla CLIP simply does not do. So the failure that bites is sharp and it is an *ordering* problem
sitting on top of the *separation* I already have. And I left two things on the table at rung one that
are exactly the levers for it: the per-ligand pIC50 in `act_list`, and the ESM-2 sequence tower the loss
never touched.

Let me start from what the data actually is, because that is where both fixes live. Screening data is
organized by assay: one target, a block of tested ligands each with an active/inactive label and, for
the actives, an affinity — pIC50. And affinities are only comparable *inside* one assay; across assays
the pH, temperature, cofactors, readout all differ, so a pIC50 of 7 here and a 7 there do not mean the
same thing. Whatever I learn has to be relative-within-assay, never absolute-across-assays. That single
constraint shapes everything that follows. The harness already exposes the assay structure I need:
`batch_list[i]=(s,e)` is the contiguous ligand span of pocket $i$, and `act_list[i]` is the sorted
activities of those ligands. So the question is how to turn graded within-assay activity into the loss.

First, keep what works. The symmetric in-batch contrastive shape and the normalize-and-dot similarity
stay — they gave me the DUD-E strength I do not want to lose, and the false-negative mask stays too,
because on a sub-0.1% distribution a true binder treated as a negative is the one corruption I cannot
afford. But the contrastive core has no preference among binders, and the metric most rewards getting
the *strongest* one to the very top. So I need a *ranking* signal inside the assay, and I need it without
re-introducing the false-negative problem the contrastive term already guards against.

The textbook move is a listwise loss — sort the assay's ligands by affinity and fit a Plackett-Luce
model, maximizing the probability of the affinity-sorted order, with a position decay so early ranks
matter more. Let me try to just adopt that and watch where it fails. A full Plackett-Luce loss commits
to a single *total order* over every ligand in the assay. But the within-assay affinities are noisy and
only loosely comparable — measurement error alone can flip two nearby ligands. A total order would
penalize the model for failing to separate two ligands whose measured affinities differ by less than
experimental error; I would be fitting the noise. This is exactly the kind of overconfidence that would
*hurt* LIT-PCBA, where the actives crowd together: forcing spurious order among near-ties is how you
scramble the very top of the list. The ordering I actually trust is coarse. In med-chem the rule of
thumb for a "real" potency difference is roughly threefold in IC50 — anything inside ~3× is a wash. So a
strict total order is the wrong object; I want to enforce order *only between pairs clearly separated in
affinity* and stay silent inside a ~3× band.

So reshape the ranking loss around that tolerance. Threefold in IC50 is $\log_{10}(3)\approx0.477$ in
log-affinity units. Sort each pocket's ligands strongest-first, position 0 the strongest. For each
ligand as an *anchor* at position `k_rel`, I want it to outscore the ligands clearly weaker than it and
ignore the ones within noise. So build a softmax over the in-pocket ligands but *mask out* every other
ligand whose activity is not clearly below the anchor's — keep `idx` in the denominator only if
`acts[k_rel] - log10(3) > acts[idx]`, i.e. strictly more than threefold weaker, and always keep the
anchor. The loss is $-\log$ of the anchor's softmax probability against that pruned set of clearly-weaker
rivals: if it is genuinely the strongest among them it wins and the loss is small; if a clearly-weaker
ligand outscores it, it pays. Because I only ever compare against ligands more than 3× weaker, I never
punish ordering two near-ties — the noise tolerance I wanted. I run this only when $L_i>2$; with one or
two ligands there is no meaningful within-assay order. And the weighting: the metric is top-heavy, so
ranking the *strongest* ligand above everything matters far more than the 20th. That is the DCG
intuition — discount each anchor by $1/\log(\text{rank}+2)$, so anchor 0 (the strongest) carries the
largest weight $1/\log 2\approx1.44$, decaying for deeper anchors. The "+2" keeps the first anchor's log
finite rather than $\log 1 = 0$. This spends the ranking gradient on the head of the list, which is what
BEDROC and EF reward — exactly the LIT-PCBA failure mode.

Now I revisit the contrastive term itself, because graded data changes how I should treat the
ligand-to-pocket direction. At rung one the row-local term matched every ligand to its pocket
unconditionally. But a weak or inactive ligand is *noisy evidence* for a matched pair — pulling a 10 µM
ligand toward its pocket as hard as a 10 nM one teaches the space to call marginal binders confident
positives, which is exactly wrong when the metric lives at the top. So I gate it: for ligand $k$ in a
multi-ligand pocket, if `acts[k-s] < 5` (below the conventional ~10 µM "active" cutoff in pIC50), skip
it as a contrastive positive. When $L_i=1$ I keep the single ligand regardless, because otherwise that
pocket contributes nothing to this term. I do *not* gate the column term the same way: the column term
says which query owns the tested ligand column, and that ownership is part of the assay structure even
when the ligand is weak. So the per-pathway loss is the column term (multi-positive ownership), the gated
row-local selected-pair term, and the new activity-gated ranking term — three pieces, each
$\sqrt{L_i}$-normalized so big assays count more but not linearly, summed.

That is the ordering fix. The second lever is the sequence tower, and it is nearly free. The pocket is
the primary signal, but it is a single sampled conformation; the ESM-2 protein sequence is a
complementary, structure-free view of the same target that helps when the pocket is noisy or ambiguous —
which is exactly the LIT-PCBA condition. The whole three-term computation is a function of one query
embedding against the shared ligand embeddings, so I factor it into a helper and call it *twice*: once
with the pocket embedding as query, once with the protein-sequence embedding as query, against the same
ligand tower. The molecule embeddings are shared between the two pathways, so the ligand tower learns
from both views at once. With both pathway weights at 1, the total training loss is just
pathway(pocket) + pathway(sequence). This is the step where the `prot_emb` I projected but ignored at
rung one finally enters the loss — and where the model wrapper's third tower starts earning its keep.

Now I have to be precise about what *this task's* edit surface does, because it differs from the generic
contrastive-ranking story in one detail that matters and that I should not import wrongly. In the generic
formulation it is natural to carry a leading reserved coordinate in the embedding and score only the
learned slice with `[:, 1:]` — a bookkeeping convention that becomes load-bearing once embeddings live on
a manifold. But this rung is still **Euclidean**, and the scaffold here projects each feature with a
plain NonLinearHead, L2-normalizes, and scores the *full* 128-d vector: `logits = emb_poc @ emb_mol.T *
logit_scale`, no slice. There is no reserved coordinate to drop. So I keep the embeddings exactly as the
default fill produced them — three NonLinearHeads, L2-normalize, no leading coordinate — and run the
three-term helper on the full dot product. Getting this right keeps rung two an honest minimal delta over
rung one: same projection geometry, same masking, same $\sqrt{L_i}$ discipline, same detached log-scale
(init $\log 13$, detached inside the helper so the objective cannot trivially sharpen itself by cranking
the scale). The *only* changes are the activity gate on the row term, the activity-aware ranking term,
and the second sequence pathway.

The scale point deserves one line because it threads through both pathways. After L2-normalization the
dot product lives in $[-1,1]$, far too compressed for a peaked softmax, so I multiply logits by
$\exp(\text{logit\_scale})$ — the inverse temperature — and I `.detach()` it inside the helper. Without
the detach the ranking and contrastive softmaxes could lower themselves by sharpening every distribution
via the scale alone, never improving an embedding; detaching forces the gradient into the directions.

At evaluation nothing changes in shape from rung one except that the sequence view now votes. I embed the
target's pockets, the target's sequence, and every candidate ligand; score by the max over the pocket
conformers of `pocket · ligand` (a molecule binding *any* conformation should rank high), add the
analogous max over the sequence view, and rank by the sum — the same Euclidean dot product the loss
optimized, so train and test agree. (The full scaffold module is in the answer.)

So the delta from rung one is concrete: keep the in-batch contrastive separation that won DUD-E; gate the
row-local positive term so weak ligands stop being pulled in as confident binders; add an activity-aware
ranking term that pushes strong binders above clearly-weaker ones with a threefold-IC50 noise tolerance
and a DCG head-of-list discount; and turn on the sequence pathway as a complementary target view. Reading
vanilla CLIP's numbers, here is what I expect and where I am unsure. DUD-E should hold or improve
slightly — the ranking term cannot hurt a benchmark already separated well, and the sequence view adds a
second target signal; but DUD-E's ceiling is high, so I do not expect a large jump, and it is even
possible the extra ranking pressure shifts a metric or two by a hair in either direction. The real test is
LIT-PCBA: its AUROC and BEDROC should *climb*, because the ranking term and the sequence view directly
attack the graded-actives failure that left it at 0.576 / 0.065. DEKOIS should hold near its strong rung-one
values. The falsifiable claim is sharp: if the ordering diagnosis is right, the largest relative gain on the
ladder should appear on LIT-PCBA's AUROC and BEDROC, *not* on DUD-E — and if instead LIT-PCBA barely moves,
then the problem was never ordering but the Euclidean geometry itself collapsing near-identical ligands,
which would be the diagnosis that forces the next rung.
