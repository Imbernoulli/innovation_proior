The vanilla CLIP run told me exactly where it breaks, and it told me in numbers, so let me read them as
mechanism. On DUD-E it is already strong — AUROC 0.895, BEDROC 0.703, EF@0.5% 51.3 — because DUD-E's
decoys are property-matched ZINC molecules and a clean binder/non-binder cosine boundary separates them
well. DEKOIS holds too, AUROC 0.892, BEDROC 0.711, and the fact that DEKOIS's BEDROC sits essentially on
top of DUD-E's tells me the *separation* machinery transfers cleanly across two differently-built
artificial-decoy benchmarks. Then LIT-PCBA: AUROC 0.576 — barely above the coin's 0.5 — BEDROC 0.065,
EF@0.5% 7.27. The ratios are the diagnosis. DUD-E's EF@0.5% over LIT-PCBA's is $51.3/7.27=7.1\times$; the
BEDROC ratio is $0.703/0.065=10.8\times$. So the collapse is worst on precisely the most top-heavy metric
and it widens as I look higher up the list — an order of magnitude on BEDROC, seven-fold on EF@0.5%,
against "only" a 0.32 absolute drop on the whole-curve AUROC. That is not a uniform difficulty offset; it
is a signature. And two facts have to sit together: 0.576 AUROC ($0.076$ above chance, where DUD-E's is
$0.395$ above) says the model can barely tell LIT-PCBA actives from inactives across the bulk of the list
*at all*, and the even-worse BEDROC/EF say the weak separation that does exist is not concentrated at the
top. LIT-PCBA is the benchmark built to remove the artificial-decoy crutch — confirmed actives versus
confirmed inactives, graded potencies crowded together — and pure contrastive separation has no notion of
"more active": two actives of the same target are pulled toward it with equal force, so nothing
distinguishes a 10 nM binder from a 10 µM one. So the failure that bites is an *ordering* problem sitting
on a *separation* problem — and the two levers I left on the table, the per-ligand pIC50 in `act_list` and
the ESM-2 sequence tower the loss never touched, are exactly the two that attack them. The shortfall is
large enough (LIT-PCBA separation running at roughly a fifth of DUD-E strength) that I should not assume
one lever fixes it; I want both to hit both problems.

Start from what the data actually is, because both fixes live there. It is organized by assay: one target,
a block of tested ligands each with an active/inactive label and, for the actives, a pIC50. And there is a
constraint on those affinities I have to respect or I will poison the loss: they are only comparable
*inside* one assay. Across assays the pH, temperature, cofactors, and readout all differ, so a pIC50 of 7
here and a 7 there do not mean the same thing. Whatever I learn from activity has to be
relative-within-assay, never absolute-across-assays — that single constraint shapes the entire ordering
term. The harness already exposes the structure: `batch_list[i]=(s,e)` is the contiguous ligand span of
pocket $i$, `act_list[i]` its ligands' activities.

Keep what works, explicitly. The symmetric in-batch contrastive shape and the normalize-and-dot similarity
stay — they gave me the DUD-E and DEKOIS strength I do not want to spend — and the false-negative mask
stays too, because on a sub-0.1% distribution a true binder treated as a negative is still the one
corruption I cannot afford, graded data or not. What the contrastive core lacks is any preference *among*
binders, which is what the metric most rewards. So I need a ranking signal inside the assay, without
re-introducing the false-negative problem the contrastive term already guards against.

The most direct shape for that signal is an activity *regression* head — project the pocket–ligand pair to
a scalar and fit the pIC50 with an MSE — and I reject it on the constraint I just established: affinities
are not comparable across assays, so there is no single global target to regress to, and an MSE to an
absolute number would be fitting an incoherent label. It also drags back the two problems the contrastive
reframing was built to escape — the scarce calibrated labels, and squared error spread uniformly across the
affinity range, the opposite of the top-heavy ranking the metric rewards. Regression answers the
affinity-prediction question again, and screening does not ask it. A purely *ranking* signal — order the
ligands within an assay — respects the within-assay constraint and never touches an absolute number. The
textbook move is a listwise Plackett–Luce loss over the affinity-sorted total order, and it is worth
watching precisely where that fails, because the failure shapes the fix. A full total order penalizes the
model for failing to separate ligands whose measured affinities differ by less than experimental error —
and it penalizes it *at the top of the list*, forcing spurious order among near-ties, which is exactly how
you scramble the very ranks LIT-PCBA is scored on. The medicinal-chemistry rule of thumb for a "real"
potency difference is roughly threefold in IC50, i.e. $\log_{10}(3)\approx0.477$ in log-affinity; anything
inside that band is a wash. So the object I want is not a total order — it is order enforced *only between
pairs clearly separated in affinity*, with silence inside the $\sim3\times$ band: a pair differing by 0.2
in pIC50 (a factor $\approx1.6$, inside noise) generates no gradient; a pair differing by 1.0 (tenfold,
unambiguous) does.

Reshape the ranking loss around that threshold. Sort each pocket's ligands strongest-first, position 0 the
strongest. For each ligand as an *anchor* at relative position `k_rel`, build a softmax over the in-pocket
ligands but *mask out* every rival whose activity is not clearly below the anchor's — keep `idx` in the
denominator only if `acts[k_rel] - log10(3) > acts[idx]`, i.e. strictly more than threefold weaker, and
always keep the anchor itself. The loss is $-\log$ of the anchor's softmax probability against that pruned
set of clearly-weaker rivals. Walking the boundary confirms the mask does what I intend: for ligands at
pIC50 $[8.0,6.0,5.5]$, anchor 1 ($6.0$, threshold $5.523$) masks ligand 0 at $8.0$ (stronger, not a
should-rank-below) and keeps ligand 2 at $5.5$ — but only barely, since $6.0-5.5=0.5>0.477$; nudge ligand 2
to $5.6$ and $6.0-5.6=0.4<0.477$ masks it as a near-tie, leaving anchor 1 with no rival and essentially
zero loss. The tolerance silences near-ties and the loss fires only on unambiguous inversions. I run this
term only when $L_i>2$; with one or two ligands there is no meaningful within-assay order to enforce. (I
use a masked softmax rather than a pile of independent pairwise hinges because it couples all of an
anchor's clearly-weaker rivals in one denominator — the anchor is pushed above its whole weaker cohort at
once — and composes with the same log-softmax machinery the contrastive terms already use, so the anchor
weight can carry the head-of-list discount in a single factor.)

That anchor weight matters as much as the masking, because the metric lives at the head of the list.
Ranking the *strongest* ligand above everything is worth far more than ranking the twentieth, so I discount
each anchor by the DCG-style $1/\log(\text{rank}+2)$: anchor 0 carries $1/\log2=1.44$, anchor 1 $1/\log3=
0.91$, anchor 5 $1/\log7=0.51$ — the strongest weighted about $2.8\times$ the fifth, so the head is
emphasized but the decay is gentle, a natural log rather than $1/\text{rank}$, which would fall as
$1,0.5,0.33,\dots$ and starve the deeper anchors that still carry real signal. The "+2" keeps the first
anchor's log finite. This spends the ranking gradient where BEDROC and EF are decided.

Graded data also changes how I treat the contrastive ligand-to-pocket direction, and I would be
inconsistent to leave it untouched. Vanilla CLIP matched every ligand to its pocket unconditionally. But a
weak ligand is *noisy evidence* for a matched pair — pulling a 10 µM ligand (pIC50 5) toward its pocket as
hard as a 10 nM one (pIC50 8) teaches the space to call marginal binders confident positives, exactly wrong
when the metric lives at the top. So I gate it: for ligand $k$ in a multi-ligand pocket, if `acts[k-s] < 5`
(below the conventional $\sim10$ µM "active" cutoff in pIC50), I skip it as a contrastive positive. When
$L_i=1$ I keep the single ligand regardless, or that pocket contributes nothing. I deliberately do *not*
gate the multi-positive column term the same way: it encodes *which query owns* the tested ligand column,
and that ownership is part of the assay structure even when the ligand is weak — a weak binder is still
this target's ligand and not some other's. So each pathway is three pieces now — the column term
(multi-positive ownership), the activity-gated row-local NLL, and the new activity-gated ranking term —
each $\sqrt{L_i}$-normalized with the same sub-linear discipline, and summed.

The gate cutoff is pIC50 $5$, and the two obvious choices behave very differently. At $5$, the $\sim10$ µM
active/inactive line, the "binder" label itself is unreliable below it, so gating a sub-5 ligand out as a
confident positive discards noise, not signal. Moving to $6$ (1 µM) would look cleaner but delete an entire
decade of genuine weak-but-real binders, shrinking the already-scarce positives on exactly the assays where
they are most crowded. And the gate and the ranking term stay consistent: a gated sub-5 ligand is removed
as a *positive* but still appears in `act_list`, so it can be masked *in* as a clearly-weaker *rival* for a
stronger anchor. A weak binder is therefore demoted, not deleted — it teaches "rank below the strong ones"
without teaching "sit confidently at your pocket," the asymmetric treatment a noisy weak binder deserves.

That is the ordering fix; the second lever is the sequence tower, and it is nearly free. The pocket is a
single sampled conformation, and where that snapshot is noisy or ambiguous the pocket view alone has
nothing to fall back on — plausibly part of why LIT-PCBA's whole-curve separation is failing, not only its
top. The ESM-2 protein sequence is a complementary, structure-free view of the same target that does not
depend on which pocket snapshot I drew. The whole three-term computation is a function of one query
embedding against the shared ligand embeddings, so I factor it into a helper and call it *twice*: once with
the pocket embedding as query, once with the protein-sequence embedding, against the *same* ligand tower.
Because the molecule embeddings are shared, the ligand tower learns from both views at once and the two
target views vote into the same space. This is where the `prot_emb` I projected but ignored finally enters
the loss.

The two pathway weights I set to 1, and that is principled rather than a knob I am ducking: the loss now
sums up to six components — three terms, two pathways — but every one is a $-\log$ of a softmax probability
over an assay of a handful to a few dozen ligands, so each per-element contribution is $O(1)$ ($\approx\log
L_i$, single digits), and each is $\sqrt{L_i}$-normalized identically. The six are already on the same scale
by construction; none silently dominates, so there is no calibration gap a hand-tuned weight would have to
paper over. Equal weights say "trust the two target views equally until the numbers tell me otherwise,"
the right prior for a first pass.

One scaffold detail I must not import wrongly. Some contrastive reference code carries a leading reserved
coordinate and scores only the trailing slice with `[:, 1:]`, and it would be easy to copy that reflexively.
But nothing in *this* fill sets such a coordinate aside: it projects with a plain NonLinearHead,
L2-normalizes, and scores the *full* 128-d vector, `emb_poc @ emb_mol.T * logit_scale`, no slice. Slicing
off index 0 here would silently discard a real dimension the heads are actively using. So I keep the
embeddings exactly as before — three NonLinearHeads, L2-normalize, no leading coordinate — and run the
three-term helper on the full dot product, same detached $\log13$ inverse temperature (detached inside the
helper so neither the ranking nor the contrastive softmax can lower itself by sharpening every distribution
via the scale alone). The *only* changes are the activity gate on the row term, the activity-aware ranking
term, and the second sequence pathway, so whatever the numbers do, they attribute cleanly.

At evaluation nothing changes in shape except that the sequence view now votes: embed the target's pockets,
the target's sequence, and every candidate ligand; score by the max over pocket conformers of $\text{pocket}
\cdot\text{ligand}$, add the analogous max over the sequence view, rank by the sum. Same Euclidean dot
product the loss optimized, so train and test agree and the sequence contribution enters scoring the same
way it entered training.

Reading vanilla CLIP's numbers, here is what I expect and where I am unsure. DUD-E should hold or improve
slightly — the ranking term cannot hurt an already well-separated benchmark and the sequence view adds a
second signal, though its ceiling is high (0.895 / 0.703) so I expect no large jump, and the extra ranking
pressure could nudge a top-heavy metric by a hair either way as the space reorganizes. The real test is
LIT-PCBA: its AUROC and BEDROC should *climb*, the ranking term ordering strong actives up on the BEDROC/EF
side and the sequence view supplying separation where the pocket snapshot was ambiguous on the AUROC side.
DEKOIS should hold near its strong values, its decoys already well-separated. The a-priori claim I will hold
myself to: if the ordering diagnosis is right, the largest relative gain on the ladder appears on LIT-PCBA,
not on DUD-E. And the negative branch is just as informative — if LIT-PCBA barely moves, the problem was
never ordering but the Euclidean geometry itself, collapsing near-identical ligands onto near-identical
vectors so that no reweighting of a dot-product softmax can pry them apart. That would be the diagnosis that
forces the next step. The full module is in the answer.
