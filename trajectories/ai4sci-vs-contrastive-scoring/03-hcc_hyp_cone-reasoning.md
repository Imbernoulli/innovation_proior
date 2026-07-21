The HCC numbers confirm the ordering diagnosis and, in the same breath, mark the limit the ranking term
cannot reach, so the residual is what dictates this step. LIT-PCBA moved exactly where I predicted: AUROC
0.576 → 0.628, BEDROC 0.065 → 0.087, EF@0.5% 7.27 → 11.15 ($\times1.53$). The relative gains *are*
concentrated on the realistic benchmark, just as the falsifiable claim said they would be if the
bottleneck were ordering. DUD-E even firmed its AUROC, 0.895 → 0.920, and DEKOIS held and firmed, 0.892 →
0.922. So activity-aware ranking plus the sequence view did the work I designed them for. But two things in
the same table say the ranking term has run out of room. First, LIT-PCBA in absolute terms is still soft —
0.628 AUROC is only $0.128$ above chance against DUD-E's $0.420$, and 0.087 BEDROC is still an order of
magnitude below DUD-E ($0.693/0.087=8.0\times$): the lever helped and *stopped* helping while the realistic
benchmark is mostly unsolved. Second, the ranking term cost something on the easy benchmark — DUD-E's
BEDROC slipped 0.703 → 0.693 and EF@0.5% 51.3 → 48.8 even as AUROC rose, the extra ranking pressure
reorganizing the space to satisfy within-assay order and giving back a hair of very-top enrichment where
DUD-E was already near ceiling. Both readings point the same way: the ordering lever delivered a bounded
gain, and the thing still failing on LIT-PCBA is not ordering. It is the alternative I flagged before — the
*geometry*, the Euclidean space itself collapsing the cases that matter most.

Where does Euclidean space fail, specifically, and can I make the failure quantitative? Activity cliffs.
Two ligands that are almost the same molecule — one substituent removed — that bind tens of fold
differently. A structure encoder, doing its job, maps near-identical molecules to near-identical vectors,
and in a flat, normalized space that is fatal, because the dot-product score is a smooth function of the
angle between vectors and dies *quadratically* as that angle shrinks. Put the pocket and two unit-norm
ligands separated by a small angle $\theta$; the score gap between the two is $\cos\theta-\cos0\approx
-\theta^2/2$. For a structurally tiny $\theta=0.05$ rad that is $1-\cos(0.05)=0.00125$ — three orders of
magnitude below anything I could distinguish from noise. So the geometry does not merely handle cliffs
badly; it *forces* the two scores together, and it worsens the closer the pair, since the gap shrinks as
$\theta^2$. The ranking term cannot fix this: it asks the score to order the pair, but the two embeddings
are so close that there is no room to *place* the order without forcing structurally similar molecules to
large distance — which would fight the encoder's smoothness everywhere and wreck the
genuinely-similar-and-similar-affinity pairs too. And this is exactly LIT-PCBA's regime, realistic actives
that include near-identical analogues, so the residual 0.628/0.087 is the model failing on precisely the
pairs where function diverges but structure does not.

So what kind of space has room? Negative curvature. The defining fact about hyperbolic space is that
volume grows exponentially with radius instead of polynomially — which is why it embeds trees and
hierarchies with low distortion; there is exponentially more room near the boundary than a flat space
provides. And graded binding affinity *is* a hierarchy: within an assay ligands fan out from weak to
strong, and across the library there is coarse-to-fine structure of "binds this family / this target /
this pocket specifically." The bet is that the exponential growth gives me exactly the room to separate
cliffs cheaply, without uniformly stretching the metric.

That is a slogan until I can show a small structural difference becomes a large distance, so make it
quantitative before committing. Work on the Lorentz model: points on the upper sheet of a hyperboloid with
the Lorentzian inner product, geodesic distance $d_L(x,y)=(1/\sqrt\kappa)\,\mathrm{arccosh}(-\kappa\langle
x,y\rangle_L)$, and lift Euclidean encoder outputs onto it with the exponential map at the origin. Take two
ligands as tangent vectors at the origin with nearly equal radial norm $r$ and a small angle $\theta\ll1$
between them. The hyperbolic law of cosines gives, on a unit-curvature sheet, $\cosh d=\cosh^2 r-\sinh^2 r
\cos\theta$; expand $\cos\theta\approx1-\theta^2/2$ and use $\cosh^2 r-\sinh^2 r=1$ to get $\cosh d-1
\approx(\sinh^2 r/2)\theta^2$, and invert with $\mathrm{arccosh}(1+\epsilon)\approx\sqrt{2\epsilon}$:
$\sqrt{2\epsilon}=\sinh r\cdot\theta$, so restoring curvature $d_H\approx(\sinh r/\sqrt\kappa)\,\theta$. The
separation is $\theta$ *amplified by $\sinh r$*, where Euclidean gives only $r\cdot\theta$, linear. Is that
amplification actually large at radii I would use, or a distinction without a difference? The honest figure
of merit is $\sinh(r)/r$: at $r=1$ it is only $1.18$; at $r=2$, $1.81$; at $r=3$, $3.34$; at $r=4$, $6.82$.
So the amplification is real but *only switches on past $r\approx2$* — near the origin hyperbolic space is
locally flat and buys me essentially nothing. That caveat shapes the design: the geometry helps only if I
can push the discriminating tier out to a genuinely large radius, not merely nudge it off the origin. With
that proviso the bet has teeth — a cliff pair, tiny $\theta$ and very different function, can be separated
if the geometry gives the stronger tier access to larger radial scale and tighter angular control, because
the angular sliver is amplified at the radius where it matters and I never distort the metric uniformly.
The design reduces to three coupled demands: make the radial coordinate carry binding-strength tier, make
angular position carry identity, and push the discriminating tiers out past $r\approx2$ where the $\sinh r$
amplification is live.

How do I *control* radial depth and angular spread per ligand as a function of affinity? I need a
structured prior on the manifold, not "embed and hope." Entailment cones give it: attach to each point a
cone opening away from the origin, whose half-aperture *shrinks* as the point's norm grows — a point
farther out projects a narrower cone. On the Lorentz model the aperture is $\omega(x)=\arcsin(2K/(\sqrt
\kappa\,\lVert x_{\text{space}}\rVert))$ with $K$ a small constant. Plugging real norms in ($\kappa=1,
K=0.1$): $\lVert x\rVert=0.3\to41.8^\circ$, $0.5\to23.6^\circ$, $1.0\to11.5^\circ$, $2.0\to5.7^\circ$,
$4.0\to2.9^\circ$ — monotone and well-spread, a genuinely tightening cone rather than something pinned at
$90^\circ$ or collapsed to $0$. And a telling boundary: the argument exceeds $1$ below $\lVert x\rVert=0.2$,
where $\arcsin$ is undefined, so the cone is only meaningful once a point is pushed past a minimum radius —
the same "geometry does nothing near the origin" caveat the distance calculation produced. In my setting a
pocket pushed deep toward the boundary is "more specific" and should admit only a tight set of ligand
directions — exactly the bias I want. But a single cone is binary, and affinity is graded, so I turn the
cone into a hierarchy of tiers indexed by affinity, using *both* the radial dimension (geodesic
pocket–ligand distance) and the angular dimension (the cone) as graded constraints.

Two measurements per ligand, and the argument order matters. The radial one is the geodesic distance
$d_{ij}$ from pocket $i$ to ligand $j$. The angular one is the first-argument angle $\phi_{ij}=
\mathrm{oxy\_angle}(\text{ligand}_j,\text{pocket}_i)$ in the hyperbolic triangle O–ligand–pocket, with the
aperture $\omega_i$ attached to the *pocket*; the constraint is $\phi_{ij}\le\eta_{ij}\cdot\omega_i$ — the
pocket supplies the aperture, the ligand-first angle supplies the measured lean — and swapping the arguments
would be a different loss, so I take the order the harness's lorentz helpers compute. Now the tiers: bucket
each ligand's pIC50 by the standard thresholds $\{5,7,9\}$ (5 is the $\sim10$ µM cutoff, each step a
decade), giving four buckets $b\in\{0,1,2,3\}$, and per ligand $r_k=r_0+b\,\Delta r$ and $\eta_k=\eta_0-b
\,\Delta\eta$ with $r_0,\eta_0$ the weakest-tier base. The signs are the whole point: a *stronger* binder
(larger $b$) gets a *larger* radial cap — permitting it to occupy the larger radial scales where $\sinh r$
amplification acts — and a *smaller* angular tolerance, because a strong, specific binding event should
align more decisively with the pocket's admissible direction than a weak one. Strong = larger radial cap,
tighter cone; weak = smaller cap, wider cone. The two knobs move oppositely with affinity, spreading the
tiers along both distance and angular selectivity — the two-axis separation the cliff calculation demanded,
since $\sinh r\cdot\theta$ uses both.

Run an actual cliff pair through this with the constants I am choosing ($r_0=0.5,\Delta r=0.5,\eta_0=0.7,
\Delta\eta=0.2$). Take the weak member at pIC50 5.5 and the strong at 8.0. With thresholds $[5,7,9]$ and
the bucketize convention, 5.5 exceeds the first threshold so it lands in bucket 1, not 0, and 8.0 in bucket
2 — giving radial caps $r=1.0$ and $r=1.5$ and angular tolerances $\eta=0.5$ and $0.3$. If both lie near the
same pocket-relative direction at their caps, they sit on roughly a common radial ray at radii 1.0 and 1.5,
geodesic distance $|1.5-1.0|=0.5$. So the radial tiering *alone* pries a structurally-identical pair apart
by 0.5 in geodesic distance, where the Euclidean dot-product gap was $\sim0.00125$ — a factor of four
hundred. And the angular axis compounds it: the strong ligand held to the tighter $\eta$ is pushed toward
the pocket axis at $r=1.5$, where a residual $\theta=0.05$ maps to $\sinh(1.5)\cdot0.05=0.106$ against
$\sinh(0.5)\cdot0.05=0.026$ down at $r=0.5$ — a $4\times$ difference from radius alone. The two separations
*add* rather than fight, which is the point of moving the two knobs oppositely.

The cone losses are one-sided hinges — penalize only violations, never pull a satisfied ligand: $L_{\text{
rad}}=(1/\sqrt N)\sum\max(d_{ij}-r_{ij},0)$ and $L_{\text{ang}}=(1/\sqrt N)\sum\max(\phi_{ij}-\eta_{ij}
\omega_i,0)$, combined as $\lambda_{\text{rad}}L_{\text{rad}}+\lambda_{\text{ang}}L_{\text{ang}}$ with both
weights $0.5$, two halves of the same cone. The $1/\sqrt N$ is the same assay-size discipline from the
start. Two regularizers fall out of what can go wrong with the cone. The angular hinge is zero the instant
$\phi\le\eta\omega$, so the optimizer has no pressure to do better than *touch* the boundary and could
collapse angles trivially toward the axis; a margin $m=0.15$ rad — $R_{\text{ang}}=(1/\sqrt N)\sum\max(\phi
-\eta\omega+m,0)$ — keeps pushing until ligands are decisively *inside* the cone. And because the metric is
dominated by the very top of the list, a heterogeneity term weights threshold-selected entries by distance
rank with $w_j=\exp(-\beta(\mathrm{rank}_j-1)/L_i)$ at $\beta=80.5$ — the *same* focus parameter as BEDROC,
shaping the training weighting to match the evaluation's early-enrichment emphasis rather than guessing a
decay. This term follows the mask $v<5$, so it is a threshold-mask term, not a strong-binder term, and I
keep that explicit so the code and the intent agree. Both regularizers are auxiliary, weight $0.10$ each.

The cone hierarchy rides on top of a retrieval objective that still has to find binders at all, so I keep
HCC's contrastive-plus-ranking core verbatim, applied now to the hyperbolic embeddings — and I am careful
about the similarity I feed the softmax. Geodesic distance is the "correct" hyperbolic similarity, but at
inference I need a plain dot product so retrieval stays a matmul over a cached matrix, the constraint that
has survived from the start untouched and is the entire reason screening is feasible at billion scale. So I
score with the inner product of the spatial components both inside the softmax logits and at inference, and
let the cone losses do the geometric shaping; the dot product is a cheap monotone proxy the training
geometry has already arranged to be faithful near a pocket. Here the coordinate bookkeeping I *omitted*
before becomes load-bearing. In the flat space there was no reserved coordinate to drop, and slicing one
off would have thrown away a real dimension. On the manifold that reverses: the exp map returns only the
space components, the implementation treats the projector output as $[\text{lead},\,\text{space}\ldots]$ and
drops index 0 with `emb[:, 1:]` before the similarity, so the manifold coordinates stay aligned with how the
lorentz helpers bookkeep them. At *inference*, the score uses the *full* embedding dot product — `pocket_
reps @ mol_reps.T`, max over the target's pockets, plus the sequence contribution — because that is the
cached-retrieval matmul over the embeddings as produced: the geometry is paid for at training time and the
screen pays off with a plain matrix multiply.

The projection heads and the geometry parameters come with initialization traps that a careless lift onto
the manifold hits at step one. Each backbone feature goes through the same NonLinearHead into 128-d, but the
exp map scales the tangent vector by $\sinh(\sqrt\kappa\lVert v\rVert)/(\sqrt\kappa\lVert v\rVert)$, and
CLIP-style init makes the Euclidean output norm $\approx\sqrt n=\sqrt{128}\approx11.31$, whereupon $\sinh(
11.31)\approx4.1\times10^4$ — a per-coordinate magnitude of order $10^4$ feeding a softmax over inner
products, with the time coordinate $\sqrt{1/\kappa+\lVert x_{\text{space}}\rVert^2}$ squaring it before the
root. That saturates the logits and stalls the gradient immediately, and it worsens where curvature is
sharpest: at the clamp ceiling $\kappa=10$, $\sinh(\sqrt{10}\cdot11.31)=\sinh(35.8)\approx1.7\times10^{15}$,
eleven more orders of magnitude. The fix is a learnable per-tower scale $\alpha$ initialized to $1/\sqrt n=
128^{-1/2}$ so the scaled embedding has expected unit norm at init, making the blowup factor a tame
$\sinh(1)\approx1.18$; I learn $\alpha$ in log space (so it cannot collapse all embeddings to zero) and
clamp $\exp(\alpha)\le1$ so the scale can shrink but never blow the exp map back up. Each projection is
$u=\text{head}(\text{feat})\cdot\exp(\alpha)$, then $h=\exp\_map0(u,\kappa)$. The curvature $\kappa$ is
itself learnable (init $\log1$), clamped to $[\log0.1,\log10]$ so it can neither collapse to Euclidean nor
detonate numerically. The contrastive softmax keeps the detached $\log13$ inverse temperature.

One deliberate asymmetry in how far the geometry reaches. The sequence pathway runs the identical
contrastive-plus-ranking helper as a second query view, but I apply the cone supervision *only* to the
pocket branch: the pocket carries the structural, geometrically meaningful signal — the cone is about
pocket–ligand spatial alignment — and over-constraining the sequence branch geometrically would fight its
role as an auxiliary, structure-free view. So the total is $\alpha_{\text{poc}}\text{loss}_{\text{poc}}+
\alpha_{\text{prot}}\text{loss}_{\text{seq}}+\gamma_{\text{cone}}L_{\text{cone}}+\lambda_{\text{het}}
R_{\text{het}}+\lambda_{\text{ang}}R_{\text{ang}}$ with $\gamma_{\text{cone}}=0.1$, down-weighted because
the cone is a structural *prior* shaping the space, not the primary retrieval signal — if it dominated, the
model would satisfy the geometry at the expense of finding binders, trading the very enrichment I am
chasing for a well-shaped but empty cone.

Reading HCC's numbers, the a-priori claims are sharp and specifically about the realistic benchmark, because
that is where cliffs live. LIT-PCBA's AUROC and BEDROC should rise *again* past 0.628 / 0.087 — if the
$\sinh r$ room is real, the model should now separate the near-identical-but-different-affinity pairs the
Euclidean ranking term could not place. DUD-E, already near ceiling at 0.920 AUROC, should hold or edge up
on the early-enrichment metrics where the cone's head-of-list shaping helps — and the cone might even
recover the small BEDROC/EF slip the ranking term cost there, since it adds top-of-list structure rather
than diffuse ordering pressure. The honest risk is that a strong prior on top of already-strong features
could trade a little on a benchmark whose decoys are easy — DEKOIS — for the gain where cliffs dominate, so
I would not be surprised to see DEKOIS hold roughly flat. What decides whether the geometry was worth it is
DUD-E's BEDROC/EF and LIT-PCBA's AUROC moving up *together*: separation preserved where it was already good,
plus the cliff room the flat space never had. The full module is in the answer.
