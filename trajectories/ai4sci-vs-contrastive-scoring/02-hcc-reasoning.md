The vanilla CLIP run told me exactly where it breaks, and it told me in numbers, so let me read them as
mechanism before I touch anything. On DUD-E it is already strong — AUROC 0.895, BEDROC 0.703, EF@0.5% 51.3
— because DUD-E's decoys are property-matched ZINC molecules and a clean binder/non-binder cosine boundary
separates them well; in-batch contrastive learning is doing its job. DEKOIS holds up too, AUROC 0.892,
BEDROC 0.711, and the fact that DEKOIS's BEDROC (0.711) sits essentially on top of DUD-E's (0.703) tells me
the *separation* machinery transfers cleanly across two differently-built artificial-decoy benchmarks. Then
LIT-PCBA: AUROC 0.576 — barely above the 0.5 of a coin — BEDROC 0.065, EF@0.5% 7.27. Put the gaps in ratio
form, because the ratios are the diagnosis. DUD-E's EF@0.5% over LIT-PCBA's is $51.3/7.27=7.1\times$; the
BEDROC ratio is $0.703/0.065=10.8\times$. So the collapse is worst on precisely the most top-heavy metric,
and it widens as I look higher up the list — an order of magnitude on BEDROC, seven-fold on EF@0.5%, against
"only" a 0.32 absolute drop on the whole-curve AUROC. That is not a uniform difficulty offset; it is a
signature. Two facts have to be held together honestly: the AUROC at 0.576 says the model can barely tell
LIT-PCBA actives from inactives *at all* across the bulk of the list, and the even-worse BEDROC/EF say that
whatever weak separation exists is not concentrated at the top. Read the AUROC literally — it is the
probability that a random active outranks a random inactive — and 0.576 is only $0.076$ above the $0.5$ of a
coin, where DUD-E's 0.895 is $0.395$ above it. So the pocket–ligand separation on realistic decoys is
running at roughly a fifth of its DUD-E strength, which is a large enough shortfall that I should not assume
a single lever fixes it; both the ordering and the separation are weak, and I want the two fixes I have to
hit both. LIT-PCBA is the benchmark built to remove the
artificial-decoy crutch — confirmed actives versus confirmed inactives at realistic ratios, graded potencies
crowded together — and a pure contrastive separation has no notion of "more active": two actives of the same
target are pulled toward it with equal force, so nothing distinguishes a 10 nM binder from a 10 µM one. The
metric is top-heavy and LIT-PCBA's top ranks are decided by *ordering* graded actives, which vanilla CLIP
simply does not do. So the failure that bites is an *ordering* problem sitting on top of a *separation*
problem — and the two levers I left on the table at rung one, the per-ligand pIC50 in `act_list` and the
ESM-2 sequence tower the loss never touched, are exactly the two that attack them.

Let me start from what the data actually is, because both fixes live there. Screening data is organized by
assay: one target, a block of tested ligands each with an active/inactive label and, for the actives, an
affinity — a pIC50. And there is a constraint on those affinities I have to respect or I will poison the
loss: they are only comparable *inside* one assay. Across assays the pH, temperature, cofactors, and readout
all differ, so a pIC50 of 7 measured here and a 7 measured there do not mean the same thing. Whatever I
learn from activity has to be relative-within-assay, never absolute-across-assays. That single constraint
shapes the entire ordering term. The harness already exposes the structure I need: `batch_list[i]=(s,e)` is
the contiguous ligand span of pocket $i$, and `act_list[i]` its ligands' activities. So the question is how
to turn graded within-assay activity into a loss term without breaking the separation that already won DUD-E.

First, keep what works, and be explicit about why. The symmetric in-batch contrastive shape and the
normalize-and-dot similarity stay — they gave me the DUD-E and DEKOIS strength I do not want to spend — and
the false-negative mask stays too, because on a sub-0.1% distribution a true binder treated as a negative is
still the one corruption I cannot afford, graded data or not. What the contrastive core lacks is any
preference *among* binders, and the metric most rewards getting the strongest one to the very top. So I need
a ranking signal inside the assay, and I need it without re-introducing the false-negative problem the
contrastive term already guards against.

There are three shapes the activity signal could take, and I should walk them rather than reach for the
first. The most direct is an activity *regression* head: project the pocket–ligand pair to a scalar and fit
it to the pIC50 with an MSE. I reject it on the constraint I just established — affinities are not comparable
across assays, so there is no single global target to regress to; a pIC50 of 7 is strong in one assay and
weak in another, and an MSE to an absolute number would be fitting an incoherent label. It also drags back
in exactly the two problems the contrastive reframing was built to escape: it needs the scarce, calibrated
affinity labels, and it optimizes squared error uniformly across the whole affinity range, which is the
opposite of the top-heavy ranking the metric rewards. Regression is answering the affinity-prediction
question again, and screening does not ask it. The second shape is a purely *ranking* signal — order the
ligands within an assay — which respects the within-assay constraint (it only ever compares ligands measured
under the same conditions) and never touches an absolute number. That is the right family; the only
remaining question is which ranking loss. The textbook move is a listwise loss: sort the assay's ligands by
affinity and fit a Plackett–Luce model,
maximizing the probability of the affinity-sorted total order, with a position decay so early ranks matter
more. Let me try to adopt that wholesale and watch precisely where it fails, because the failure mode tells
me the shape of the fix. A full Plackett–Luce loss commits to a single *total order* over every ligand in
the assay. But the within-assay affinities are noisy and only loosely comparable — measurement error alone
can flip two nearby ligands — and a total order would penalize the model for failing to separate two ligands
whose measured affinities differ by less than the experimental error. I would be fitting the noise, and
worse, I would be fitting it *at the top of the list*, forcing spurious order among near-ties, which is
exactly how you scramble the very ranks LIT-PCBA is scored on. Quantify the tolerance I actually trust: in
medicinal chemistry the rule of thumb for a "real" potency difference is roughly threefold in IC50, which in
log-affinity units is $\log_{10}(3)\approx0.477$; anything inside that band is a wash. So the object I want
is not a total order at all — it is order enforced *only between pairs clearly separated in affinity*, with
silence inside the $\sim3\times$ band. A pair differing by 0.2 in pIC50 (a factor of $10^{0.2}\approx1.6$,
well inside noise) should generate no gradient; a pair differing by 1.0 (tenfold, unambiguous) should.

So reshape the ranking loss around that threshold. Sort each pocket's ligands strongest-first, position 0 the
strongest. For each ligand as an *anchor* at relative position `k_rel`, I want it to outscore the ligands
clearly weaker than it and ignore the ones within noise, so I build a softmax over the in-pocket ligands but
*mask out* every rival whose activity is not clearly below the anchor's — keep `idx` in the denominator only
if `acts[k_rel] - log10(3) > acts[idx]`, i.e. strictly more than threefold weaker, and always keep the anchor
itself. The loss is $-\log$ of the anchor's softmax probability against that pruned set of clearly-weaker
rivals: if it is genuinely strongest among them it wins and the loss is small; if a clearly-weaker ligand
outscores it, it pays. Let me trace one assay to be sure the masking does what I intend rather than trust the
inequality. Take three ligands with pIC50 $[8.0,\,6.0,\,5.5]$, already strongest-first. Anchor 0 ($8.0$):
threshold $8.0-0.477=7.523$; ligand 1 at $6.0<7.523$ is a rival, ligand 2 at $5.5<7.523$ is a rival, so
anchor 0 competes against both — correct, an 8.0 should sit above a 6.0 and a 5.5. Anchor 1 ($6.0$):
threshold $6.0-0.477=5.523$; ligand 0 at $8.0\ge5.523$ is masked (it is *stronger*, not a should-rank-below),
ligand 2 at $5.5<5.523$ survives — barely, since $6.0-5.5=0.5>0.477$ — so anchor 1 competes against ligand 2
only. Now perturb: if ligand 2 were $5.6$ instead, $6.0-5.6=0.4<0.477$, it would be masked as a near-tie and
anchor 1 would have no rival at all, contributing essentially zero loss. That is exactly the behavior I
wanted — the tolerance silences near-ties and the loss only fires on unambiguous inversions — and I could
only confirm it by walking the boundary case. I run this term only when $L_i>2$; with one or two ligands
there is no meaningful within-assay order to enforce.

I could have written the clearly-separated ordering as a sum of independent pairwise hinges — one
$\max(0,\,m-(s_{\text{anchor}}-s_{\text{rival}}))$ per clearly-separated pair — instead of a masked softmax,
and it is worth being explicit about why I did not. The pairwise form is $O(L_i^2)$ per assay, treats every
pair in isolation, and gives me no natural place to put the head-of-list emphasis the metric demands: I would
have to bolt on a per-pair position weight by hand. The masked softmax couples all of an anchor's
clearly-weaker rivals in one denominator — so the anchor is pushed above its whole weaker cohort at once, not
pair by pair — and it composes cleanly with the same $\log$-softmax machinery the contrastive terms already
use, which lets the anchor weighting carry the position discount in a single factor. Same tolerance, same
head emphasis, one coherent object instead of a quadratic pile of independent hinges.

The weighting of the anchors matters as much as the masking, because the metric lives at the head of the
list. Ranking the *strongest* ligand above everything is worth far more than ranking the twentieth, so I
discount each anchor by the DCG-style $1/\log(\text{rank}+2)$. Compute the profile to see it is a head
emphasis and not a cliff: anchor 0 carries $1/\log 2=1.44$, anchor 1 carries $1/\log 3=0.91$, anchor 2
carries $1/\log 4=0.72$, anchor 5 carries $1/\log 7=0.51$. The strongest anchor is weighted about $2.8\times$
the fifth, so the head is emphasized, but the decay is *gentle* — a natural log, not $1/\text{rank}$ which
would fall as $1,\,0.5,\,0.33,\dots$ and starve the deeper anchors that still carry real signal. The "+2"
keeps the first anchor's log finite rather than $\log 1=0$. This spends the ranking gradient on the top of
the list, which is where BEDROC and EF are decided — precisely the LIT-PCBA failure mode.

Graded data also changes how I should treat the contrastive ligand-to-pocket direction, and I would be
inconsistent to leave it untouched. At rung one the row-local term matched every ligand to its pocket
unconditionally. But a weak or inactive ligand is *noisy evidence* for a matched pair — pulling a 10 µM
ligand (pIC50 5) toward its pocket as hard as a 10 nM one (pIC50 8), a thousandfold difference in affinity
treated identically, teaches the space to call marginal binders confident positives, which is exactly wrong
when the metric lives at the top. So I gate it: for ligand $k$ in a multi-ligand pocket, if `acts[k-s] < 5`
(below the conventional $\sim10$ µM "active" cutoff in pIC50), I skip it as a contrastive positive. When
$L_i=1$ I keep the single ligand regardless, because otherwise that pocket contributes nothing to this term
at all. I deliberately do *not* gate the multi-positive column term the same way: the column term encodes
*which query owns* the tested ligand column, and that ownership is part of the assay structure even when the
ligand is weak — a weak binder is still this target's ligand and not some other's. So each pathway is three
pieces now — the column term (multi-positive ownership), the activity-gated row-local selected-pair NLL, and
the new activity-gated ranking term — each $\sqrt{L_i}$-normalized with the same sub-linear discipline from
rung one, and summed.

The gate threshold deserves a second's arithmetic, because the two obvious cutoffs behave very differently.
I set it at pIC50 $5$, the conventional $\sim10$ µM active/inactive line; below it the "binder" label itself
is unreliable, so gating a sub-5 ligand out as a confident positive is discarding noise, not signal. Moving
the cutoff up to $6$ (1 µM) would look stricter and "cleaner," but it would delete an entire decade of
genuine weak-but-real binders from the positive set — every 10 µM–1 µM ligand — shrinking the already-scarce
positives on exactly the assays where they are most crowded. So $5$ is the honest floor: strict enough to
drop unreliable positives, loose enough to keep real ones. And the gate and the ranking term are consistent
with each other rather than at odds: a gated sub-5 ligand is removed as a *contrastive positive* but it still
appears in `act_list`, so it can be masked *in* as a clearly-weaker *rival* for a stronger anchor in the
ranking term. A weak binder is therefore demoted, not deleted — it teaches "rank below the strong ones"
without teaching "sit confidently at your pocket," which is precisely the asymmetric treatment a noisy weak
binder deserves.

That is the ordering fix; the second lever is the sequence tower, and it is nearly free. The pocket is the
primary target signal, but it is a single sampled conformation, and where that conformation is noisy or
ambiguous the pocket view alone has nothing to fall back on — which, reading LIT-PCBA's near-chance AUROC, is
plausibly part of why the whole-curve separation is failing there, not only the top. The ESM-2 protein
sequence is a complementary, structure-free view of the same target that does not depend on which pocket
snapshot I drew. The whole three-term computation is a function of one query embedding against the shared
ligand embeddings, so I factor it into a helper and call it *twice*: once with the pocket embedding as query,
once with the protein-sequence embedding as query, against the *same* ligand tower. Because the molecule
embeddings are shared between the two pathways, the ligand tower learns from both views at once, and the two
target views vote into the same space. With both pathway weights at 1, the total training loss is just
pathway(pocket) + pathway(sequence). This is the step where the `prot_emb` I projected but ignored at rung
one finally enters the loss.

I want to be sure "both weights at 1" is a defensible default and not a hidden knob I am ducking. Count what
the loss now sums: three terms per pathway, two pathways, so up to six components. Every one of them is a
$-\log$ of a softmax probability — the column NLL, the row-local NLL, the masked ranking NLL — so each
per-element contribution is $O(1)$ (a well-behaved softmax over an assay of a handful to a few dozen ligands
sits around $\log L_i$, single digits), and each is $\sqrt{L_i}$-normalized identically. So the six
components are already on the same scale by construction; none silently dominates, and there is no calibration
gap that a hand-tuned weight would have to paper over. That is exactly the condition under which a flat
weight of 1 is principled rather than lazy: I only introduce a weight where the terms are on genuinely
different scales, and here they are not. The pocket pathway and the sequence pathway are the same three
$O(1)$ terms computed against the same ligand tower, so weighting them equally says "trust the two target
views equally until the numbers tell me otherwise," which is the right prior for a first pass.

Now I have to be precise about what *this* edit surface does, because the generic contrastive-ranking story
differs from the scaffold in one detail I must not import wrongly. Some contrastive reference code carries a
leading reserved coordinate in the embedding and scores only the trailing slice with `[:, 1:]`, and it would
be easy to copy that convention reflexively. But nothing in *this* fill sets such a coordinate aside: the
scaffold projects each feature with a plain NonLinearHead, L2-normalizes, and scores the *full* 128-d vector,
`logits = emb_poc @ emb_mol.T * logit_scale`, with no slice. There is no reserved coordinate here, and
pretending there were one — slicing off index 0 — would silently discard a real dimension of the embedding
that my heads are actively using. So I keep the embeddings exactly as the default fill produced them — three
NonLinearHeads, L2-normalize, no leading coordinate — and run the three-term helper on the full dot product.
Getting this right keeps rung two an honest minimal delta over rung one: same projection geometry, same
masking, same $\sqrt{L_i}$ discipline, same detached $\log 13$ inverse temperature (detached inside the
helper so neither the ranking nor the contrastive softmax can lower itself by sharpening every distribution
via the scale alone rather than improving an embedding). The *only* changes are the activity gate on the row
term, the activity-aware ranking term, and the second sequence pathway. I am spending exactly two of the
levers I named at rung one and nothing else, so that whatever the numbers do, they attribute cleanly.

At evaluation nothing changes in shape from rung one except that the sequence view now votes. I embed the
target's pockets, the target's sequence, and every candidate ligand; score by the max over the pocket
conformers of `pocket · ligand` — a molecule binding *any* conformation should rank high — add the analogous
max over the sequence view, and rank by the sum. It is the same Euclidean dot product the loss optimized, so
train and test agree, and the sequence contribution enters test scoring the same way it entered training.

So the delta from rung one is concrete: keep the in-batch contrastive separation that won DUD-E and DEKOIS;
gate the row-local positive term so weak ligands stop being pulled in as confident binders; add an
activity-aware ranking term that pushes strong binders above clearly-weaker ones with a threefold-IC50 noise
tolerance and a DCG head-of-list discount; and turn on the sequence pathway as a complementary target view.
Reading vanilla CLIP's numbers, here is what I expect and where I am honestly unsure. DUD-E should hold or
improve slightly — the ranking term cannot hurt a benchmark already separated well, and the sequence view
adds a second target signal; but DUD-E's ceiling is high (0.895 AUROC, 0.703 BEDROC already), so I do not
expect a large jump, and it is even possible the extra ranking pressure nudges a top-heavy metric or two by a
hair in either direction as the space reorganizes to satisfy the new term. The real test is LIT-PCBA: its
AUROC and BEDROC should *climb*, because the ranking term and the sequence view directly attack the
graded-actives failure that left it at 0.576 / 0.065 — the ranking term working on the BEDROC/EF side by
ordering the strong actives up, the sequence view plausibly on the AUROC side by supplying separation where
the pocket snapshot was ambiguous. DEKOIS should hold near its strong rung-one values (0.892 / 0.711),
since its decoys, like DUD-E's, are already well-separated and the new terms have little there to fix. The
falsifiable claim is sharp and I will hold myself to it: if the ordering diagnosis is right, the largest
relative gain on the ladder should appear on LIT-PCBA's AUROC and BEDROC, *not* on DUD-E. And the negative
branch is just as informative — if instead LIT-PCBA barely moves, then the problem was never ordering but
something the ranking term structurally cannot reach: the Euclidean geometry itself collapsing
near-identical ligands onto near-identical vectors, so that no reweighting of a dot-product softmax can pry
them apart. That would be the diagnosis that forces the next rung, and the LIT-PCBA numbers are what will
decide between the two. (The full scaffold module is in the answer.)
