The HCC numbers confirm the ordering diagnosis and, in the same breath, mark the limit the ranking term
cannot reach — so let me read them carefully, because the residual is what dictates this rung. LIT-PCBA moved
exactly where I predicted: AUROC climbed 0.576 → 0.628, BEDROC 0.065 → 0.087, and EF@0.5% nearly doubled,
7.27 → 11.15 ($\times1.53$). The relative gains *are* concentrated on the realistic benchmark, just as the
falsifiable claim said they would be if the bottleneck were ordering. DUD-E even firmed its AUROC, 0.895 →
0.920, and DEKOIS held and firmed, 0.892 → 0.922. So activity-aware ranking plus the sequence view did the
work I designed them for. But two things in the same table say the ranking term has run out of room. First,
read LIT-PCBA in absolute terms rather than as a delta: 0.628 AUROC is still soft — only $0.128$ above the
$0.5$ of a coin, against DUD-E's $0.420$ above — and 0.087 BEDROC is still an order of magnitude below the
DUD-E number; $0.693/0.087=8.0\times$. The lever helped and *stopped* helping while the realistic benchmark
is still mostly unsolved. Second, look at what the ranking term cost on the easy benchmark: DUD-E's BEDROC
slipped 0.703 → 0.693 and EF@0.5% slipped 51.3 → 48.8 even as its AUROC rose. That is a small, honest
tradeoff — the extra ranking pressure reorganized the space to satisfy the within-assay order and gave back a
hair of the very-top enrichment where DUD-E was already near ceiling. Both readings point the same way: I
have spent the ordering lever, it delivered a bounded gain, and the thing still failing on LIT-PCBA is not
ordering. It is the alternative I flagged at rung two — if LIT-PCBA only partly moved, the residual is not the
loss but the *geometry*, the Euclidean space itself collapsing the cases that matter most. So now I have to
stare at the geometry.

Where does Euclidean space fail, specifically, and can I make the failure quantitative rather than a slogan?
Activity cliffs. Two ligands that are almost the same molecule — one substituent removed — that bind tens of
fold differently. A structure encoder, doing its job, maps near-identical molecules to near-identical
vectors, and in a flat, normalized space that is fatal, because the dot-product score is a smooth function of
the angle between vectors and dies *quadratically* as that angle shrinks. Put the pocket and two unit-norm
ligands separated by a small angle $\theta$; the score gap between the two ligands is
$\cos\theta-\cos 0\approx-\theta^2/2$ to leading order. For a structurally tiny $\theta=0.05$ rad that is a
gap of $1-\cos(0.05)=0.00125$ — three orders of magnitude below anything I could distinguish from noise. So
the Euclidean model is not merely bad at cliffs; the geometry *forces* the two scores together, and it gets
worse the closer the pair is, since the gap shrinks as $\theta^2$. The ranking term I just added cannot fix
this: it asks the score to order the pair, but the two embeddings are so close that there is no room to
*place* the order without forcing structurally similar molecules to large distance — which would fight the
encoder's smoothness everywhere and wreck the genuinely-similar-and-similar-affinity pairs too. And this is
exactly LIT-PCBA's regime, where realistic actives include near-identical analogues, so the residual
0.628/0.087 is the model failing on precisely the pairs where function diverges but structure does not. The
ranking loss ran out of room because the *space* has no room.

So what kind of space has room? Negative curvature. The defining fact about hyperbolic space is that volume
grows exponentially with radius instead of polynomially — which is why it embeds trees and hierarchies with
low distortion; there is exponentially more room near the boundary than a flat space provides. And graded
binding affinity *is* a hierarchy: within an assay ligands fan out from weak to strong, and across the library
there is a coarse-to-fine structure of "binds this family / binds this target / binds this pocket
specifically." If a curved space naturally encodes hierarchy, maybe it encodes the affinity hierarchy, and —
this is the bet — maybe the exponential growth gives me exactly the room to separate cliffs cheaply, *without*
uniformly stretching the metric.

"Exponential room" is a slogan until I can show a small structural difference becomes a large distance, and a
slogan is exactly what I refused to carry forward at the previous rungs, so let me make it quantitative before
committing. Work on the Lorentz model: points on the upper sheet of a hyperboloid with the Lorentzian inner
product, geodesic distance $d_L(x,y)=(1/\sqrt\kappa)\,\mathrm{arccosh}(-\kappa\langle x,y\rangle_L)$, and lift
Euclidean encoder outputs onto it with the exponential map at the origin. Take two ligands as tangent vectors
at the origin with nearly equal radial norm $r$ and a small angle $\theta\ll1$ between them. The hyperbolic
law of cosines gives, on a unit-curvature sheet, $\cosh d=\cosh^2 r-\sinh^2 r\cos\theta$. Expand for small
$\theta$ with $\cos\theta\approx1-\theta^2/2$ and use $\cosh^2 r-\sinh^2 r=1$: $\cosh d\approx1+\sinh^2 r\cdot
\theta^2/2$, so $\cosh d-1\approx(\sinh^2 r/2)\theta^2$. Now invert with $\mathrm{arccosh}(1+\epsilon)\approx
\sqrt{2\epsilon}$ — a step that carries the whole result, so I check it rather than trust it: numerically
$\mathrm{arccosh}(1.01)=0.14130$ against $\sqrt{0.02}=0.14142$, a ratio of $0.99915$, and the agreement
tightens at smaller $\epsilon$, so the leading inverse is genuinely $\sqrt{2\epsilon}$ with higher-order
error. With $\epsilon=(\sinh^2 r/2)\theta^2$, $\sqrt{2\epsilon}=\sinh r\cdot\theta$, and restoring curvature,
$d_H\approx(\sinh r/\sqrt\kappa)\,\theta$. Before I read meaning into that, confirm the whole chain — law of
cosines *and* inversion stacked — against an exact distance, because two approximations in series can drift.
At $\kappa=1,\,r=2,\,\theta=0.1$: the exact $\mathrm{arccosh}(\cosh^2 2-\sinh^2 2\cos0.1)=0.3610$, and the
approximation $\sinh(2)\cdot0.1=0.3627$, within 0.5%. Good — the formula is trustworthy in the small-$\theta$
regime that cliffs occupy.

So the separation is $\theta$ *amplified by $\sinh r$*, where Euclidean gives only $r\cdot\theta$, linear. Is
that amplification actually large at radii I would plausibly use, or is it a distinction without a difference?
The honest figure of merit is $\sinh(r)/r$ — the hyperbolic separation over the Euclidean one for the same
$\theta$ — and I should tabulate it rather than wave at "exponential." At $r=1$ it is only $1.18$; at $r=2$,
$1.81$; at $r=3$, $3.34$; at $r=4$, $6.82$. So the amplification is real but it *only switches on past
$r\approx2$*: near the origin hyperbolic space is locally flat and buys me essentially nothing. That is an
important caveat and it shapes the design — the geometry only helps if I can push the discriminating tier out
to a genuinely large radius, not merely nudge it off the origin. With that proviso the picture holds: a cliff
pair, tiny $\theta$ and very different function, can be separated *if* the geometry gives the stronger tier
access to larger radial scale and tighter angular control, because the angular sliver is amplified at the
radius where it matters, and I never distort the metric uniformly. The bet has teeth. The design now reduces
to three coupled demands: make the radial coordinate carry binding-strength tier, make angular position carry
identity, and push the discriminating tiers out past $r\approx2$ so the $\sinh r$ amplification is live where
the affinity hierarchy asks for it.

How do I *control* radial depth and angular spread per ligand as a function of affinity? I need a structured
prior on the manifold, not "embed and hope." Entailment cones give it: attach to each point a cone opening
away from the origin, and the cone's half-aperture has a closed form that *shrinks* as the point's norm grows
— a point farther out projects a narrower cone. On the Lorentz model the aperture is
$\omega(x)=\arcsin(2K/(\sqrt\kappa\,\lVert x_{\text{space}}\rVert))$ with $K$ a small constant. Let me plug
real norms in to check it gives a usable spread rather than saturating: at $\kappa=1,\,K=0.1$,
$\lVert x\rVert=0.3$ gives $\arcsin(0.667)=41.8^\circ$; $0.5\to23.6^\circ$; $1.0\to11.5^\circ$;
$2.0\to5.7^\circ$; $4.0\to2.9^\circ$. Monotone and well-spread from $\sim42^\circ$ down to $\sim3^\circ$ — a
genuinely tightening admissible cone, not something pinned at $90^\circ$ or collapsed to $0$. And a telling
boundary: the argument $2K/(\sqrt\kappa\lVert x\rVert)$ exceeds $1$ below $\lVert x\rVert=0.2$, where $\arcsin$
is undefined, so the cone is only meaningful once a point is pushed past a minimum radius — which lands on
exactly the same "the geometry does nothing near the origin" caveat the distance calculation produced. Read
in my setting: a pocket pushed deep toward the boundary is "more specific" and should admit only a tight set
of ligand directions, exactly the inductive bias I want for a specific binding pocket. But a single on/off
cone is binary, and I established I need *graded* tiers because affinity is graded. So I do the thing a plain
cone does not: turn it into a hierarchy of tiers indexed by affinity, using *both* the radial dimension
(geodesic pocket–ligand distance) and the angular dimension (the cone) as graded constraints.

Two measurements per ligand, and the argument order matters. The radial one is the geodesic distance $d_{ij}$
from pocket $i$ to ligand $j$. The angular one is the first-argument angle $\phi_{ij}=\mathrm{oxy\_angle}
(\text{ligand}_j,\text{pocket}_i)$ in the hyperbolic triangle O–ligand–pocket, and the aperture $\omega_i$ is
attached to the *pocket*. The constraint the code imposes is $\phi_{ij}\le\eta_{ij}\cdot\omega_i$ — the pocket
supplies the aperture, the ligand-first angle supplies the measured lean — and swapping the arguments would be
a different loss, so I take the order the harness's lorentz helpers compute. Now the tiers: bucket each
ligand's pIC50 by the standard activity thresholds $\{5,7,9\}$ (5 is the $\sim10$ µM "active" cutoff, each
step a decade), giving four buckets $b\in\{0,1,2,3\}$, and per ligand $r_k=r_0+b\,\Delta r$ and
$\eta_k=\eta_0-b\,\Delta\eta$ with $r_0,\eta_0$ the weakest-tier base and $\Delta r,\Delta\eta>0$. Stare at
the signs, because getting them backwards inverts the prior. A *stronger* binder (larger $b$) gets a *larger*
radial cap — a one-sided hinge permitting it to occupy the larger radial scales where the $\sinh r$
amplification can act — and a *smaller* angular tolerance, because a strong, specific binding event should
align more decisively with the pocket's admissible direction than a weak one. Strong = larger radial cap,
tighter cone; weak = smaller cap, wider cone. The two knobs move in opposite directions with affinity,
spreading the tiers along both distance and angular selectivity — the two-axis separation the cliff
calculation demanded, since $\sinh r\cdot\theta$ uses both.

Let me run an actual cliff pair through this with the constants I am about to choose ($r_0=0.5,\,\Delta r=0.5,
\,\eta_0=0.7,\,\Delta\eta=0.2$) and check the geometry separates them by an amount that survives noise. Take
the weak member at pIC50 5.5 and the strong member at 8.0. With thresholds $[5,7,9]$ and the bucketize
convention, I have to be careful at the boundary: 5.5 is *not* bucket 0, because the first threshold is 5 and
5.5 exceeds it, so 5.5 lands in bucket 1 and 8.0 in bucket 2. That gives the weak ligand a radial cap
$r=0.5+1\cdot0.5=1.0$ and the strong one $r=0.5+2\cdot0.5=1.5$; angular tolerances $\eta=0.7-0.2=0.5$ and
$0.7-0.4=0.3$. What separation does that produce? If both ligands lie near the same pocket-relative direction
but at their respective caps, they sit on roughly the same radial ray at radii 1.0 and 1.5, and the geodesic
distance between two points on a common ray is just $|1.5-1.0|=0.5$ (the law of cosines at $\theta=0$ returns
exactly $0.5$). So the radial tiering *alone* pries a structurally-identical pair apart by 0.5 in geodesic
distance, where the Euclidean dot-product gap was $\sim0.00125$ — a factor of four hundred. And the angular
axis compounds it: the strong ligand, held to the tighter $\eta$, is pushed toward the pocket axis at $r=1.5$,
where a residual $\theta=0.05$ maps to $\sinh(1.5)\cdot0.05=0.106$ against $\sinh(0.5)\cdot0.05=0.026$ if the
same sliver were left down at $r=0.5$ — a $4\times$ difference from radius alone. The two separations *add*
rather than fight, which is the whole point of moving the two knobs oppositely. And I only got the numbers
right by checking the bucket boundary: had 5.5 fallen into bucket 0, the two tiers would have been adjacent
and the radial gap only $\Delta r$ from a lower base, still positive but a weaker prior than I intended.

The cone losses are one-sided hinges — penalize only violations, never pull a satisfied ligand:
$L_{\text{rad}}=(1/\sqrt N)\sum\max(d_{ij}-r_{ij},0)$ and $L_{\text{ang}}=(1/\sqrt N)\sum\max(\phi_{ij}-
\eta_{ij}\omega_i,0)$, combined as $\lambda_{\text{rad}}L_{\text{rad}}+\lambda_{\text{ang}}L_{\text{ang}}$
with both weights $0.5$ — two halves of the same cone, no reason to prefer one. The $1/\sqrt N$ is the same
assay-size discipline I have used since rung one: sub-linear, so big assays count more but do not drown the
many small ones. Two regularizers fall out of what can go wrong with the cone. The angular hinge is zero the
instant $\phi\le\eta\omega$, so the optimizer has no pressure to do better than *touch* the boundary and could
even collapse angles trivially toward the axis with no discrimination; a margin $m=0.15$ rad —
$R_{\text{ang}}=(1/\sqrt N)\sum\max(\phi-\eta\omega+m,0)$ — keeps pushing until ligands are decisively
*inside* the cone rather than on its edge. And because the metric is dominated by the very top of the list, a
heterogeneity term weights threshold-selected entries by distance rank with $w_j=\exp(-\beta(\mathrm{rank}_j-1)
/L_i)$ at $\beta=80.5$ — the *same* focus parameter as BEDROC, so I am shaping the training weighting to match
the evaluation's early-enrichment emphasis rather than guessing a decay. The threshold convention stays
literal: this term follows the mask $v<5$, so it is a threshold-mask term, not a strong-binder term, and I
keep that explicit so the code and the intent agree. Both regularizers are auxiliary, weight $0.10$ each.

The cone hierarchy is the new part, but it rides on top of a retrieval objective that still has to find
binders at all — so I keep HCC's contrastive-plus-ranking core from rung two verbatim, applied now to the
hyperbolic embeddings, and I am careful about the similarity I feed the softmax. Geodesic distance is the
"correct" hyperbolic similarity, but at inference I need a plain dot product so retrieval stays a matmul over
a cached matrix — non-negotiable, it is the entire reason screening is feasible at billion scale, the
constraint that has survived from rung one untouched. So I score with the inner product of the spatial
components both inside the softmax logits and at inference, and let the cone losses do the geometric shaping;
the dot product is a cheap monotone proxy that the training geometry has already arranged to be faithful near
a pocket. And here the bookkeeping detail I deliberately *omitted* at rung two becomes load-bearing. At rung
two I noted there was no reserved coordinate to drop because the space was flat, and slicing one off would
have thrown away a real dimension. On the manifold that changes: the exp map returns only the space
components, the implementation treats the projector output as $[\text{lead},\,\text{space}\ldots]$ and drops
index 0 with `emb[:, 1:]` before the similarity, so the manifold coordinates stay aligned with how the lorentz
helpers bookkeep them. The `[:, 1:]` slice that was inert in the flat rung is exactly the convention that
keeps the contrastive and cone math consistent here. At *inference*, though, the score uses the *full*
embedding dot product — `pocket_reps @ mol_reps.T`, max over the target's pockets, plus the sequence
contribution — because that is the cached-retrieval matmul over the embeddings as produced; the geometry was
paid for at training time and the screen pays off with a plain matrix multiply.

The projection heads and the parameters the geometry needs come with initialization traps I have to defuse,
and this is where a careless lift onto the manifold simply diverges at step one. Each backbone feature goes
through the same NonLinearHead into 128-d, but then the exp map scales the tangent vector by
$\sinh(\sqrt\kappa\lVert v\rVert)/(\sqrt\kappa\lVert v\rVert)$, and CLIP-style init makes the Euclidean output
have norm $\approx\sqrt n$. Evaluate the blowup: $\sqrt{128}\approx11.31$, and $\sinh(11.31)\approx4.1\times
10^4$ — a per-coordinate magnitude of order $10^4$ feeding a softmax over inner products, with the time
coordinate $\sqrt{1/\kappa+\lVert x_{\text{space}}\rVert^2}$ squaring it before the root. That saturates the
logits and stalls the gradient immediately, and it gets dramatically worse where the curvature is sharpest:
at the clamp ceiling $\kappa=10$, $\sinh(\sqrt{10}\cdot11.31)=\sinh(35.8)\approx1.7\times10^{15}$, eleven more
orders of magnitude. So the danger is not hypothetical. The fix is a learnable per-tower scale $\alpha$
initialized to $1/\sqrt n=128^{-1/2}$ so the scaled embedding has expected unit norm at init, whereupon the
blowup factor is a tame $\sinh(1)\approx1.18$; I learn $\alpha$ in log space (so it cannot collapse all
embeddings to zero) and clamp $\exp(\alpha)\le1$ so the scale can shrink but never blow the exp map back up.
So each projection is $u=\text{head}(\text{feat})\cdot\exp(\alpha)$, then $h=\exp\_map0(u,\kappa)$. The
curvature $\kappa$ is itself learnable (init $\log1$), clamped to $[\log0.1,\log10]$ so it can neither collapse
to Euclidean nor detonate numerically. The contrastive softmax keeps the detached $\log13$ inverse
temperature from the earlier rungs.

One deliberate asymmetry in how far the geometry reaches. The sequence pathway runs the identical
contrastive-plus-ranking helper as a second query view, but I apply the cone supervision *only* to the pocket
branch. The pocket carries the structural, geometrically meaningful signal — the cone is about pocket–ligand
spatial alignment — and over-constraining the sequence branch geometrically would fight its role as an
auxiliary, structure-free view. So sequence participates in contrastive-plus-ranking but not in the cone. The
total is then $\alpha_{\text{poc}}\text{loss}_{\text{poc}}+\alpha_{\text{prot}}\text{loss}_{\text{seq}}+
\gamma_{\text{cone}}L_{\text{cone}}+\lambda_{\text{het}}R_{\text{het}}+\lambda_{\text{ang}}R_{\text{ang}}$ with
$\gamma_{\text{cone}}=0.1$, down-weighted because the cone is a structural *prior* shaping the space, not the
primary retrieval signal — if it dominated, the model would satisfy the geometry at the expense of actually
finding binders, and I would trade the very enrichment I am chasing for a well-shaped but empty cone.

So the delta from HCC is the geometry: keep the contrastive-plus-ranking core verbatim, lift every embedding
onto a learnable-curvature Lorentz hyperboloid through a clamped, scaled exp map, and add an affinity-tiered
radial-plus-angular cone hierarchy on the pocket branch, with a margin regularizer and a BEDROC-matched
heterogeneity regularizer. Reading HCC's numbers, the falsifiable claims are sharp and specifically about the
realistic benchmark, because that is where cliffs live. LIT-PCBA's AUROC and BEDROC should rise *again* past
HCC's 0.628 / 0.087 — if the $\sinh r$ room is real, the model should now separate the
near-identical-but-different-affinity pairs the Euclidean ranking term could not place, which is precisely the
residual that stalled at rung two. DUD-E, already near its ceiling at 0.920 AUROC, should hold or edge up on
the early-enrichment metrics where the cone's head-of-list shaping helps — and notably the cone might even
recover the small BEDROC/EF slip the ranking term cost on DUD-E, since it adds top-of-list structure rather
than diffuse ordering pressure. The honest risk is that the cone is a strong prior on top of already-strong
features: it could trade a little on a benchmark whose decoys are easy — DEKOIS, where the extra geometric
constraint is unnecessary — for a gain where cliffs dominate, so I would not be surprised to see DEKOIS hold
roughly flat rather than climb. The claim that decides whether the geometry was worth it is DUD-E's BEDROC/EF
and LIT-PCBA's AUROC moving up *together*: separation preserved where it was already good, plus the cliff room
the flat space never had. (The full scaffold module is in the answer.)
