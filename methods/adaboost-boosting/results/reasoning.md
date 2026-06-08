Let me start from the thing that actually bothers me, which is a smell, not a theorem. I can almost always get a classifier that's a *little* better than guessing. A single threshold on one feature, a one-node decision rule, a shallow tree — something predictive, error a bit under one half. What I almost never get directly is a classifier that's *highly* accurate. So there are two regimes, "barely useful" and "very good," and they feel completely different in difficulty. The question I can't let go of: is that difference real, or is it an illusion? If I hand you a black box that, on any distribution over my data, returns a hypothesis with error at most 1/2 − γ for some fixed little γ, can I crank that into error ε for any ε I name, paying only polynomially in 1/γ and 1/ε? Kearns and Valiant drew the line precisely: a *strong* learner you can push to any ε, a *weak* learner only beats a coin by an inverse polynomial. Strong trivially gives weak. The open direction is the converse. Are they the same notion or not?

The naive thing first, because I should watch it fail before I respect the problem. Run the weak learner a hundred times on fresh independent samples from D, collect a hundred hypotheses, take the majority vote. Why doesn't that just work? Because they're all looking at the *same* D. They all find the same easy structure and they all stumble on the same hard region. Their errors are correlated — almost the same set of points keeps getting missed — and a majority vote over a hundred correlated wrong answers is still wrong on that region. Voting only helps when the mistakes are *independent*, and identical-distribution resampling gives me the opposite. So the naive vote is dead. But it tells me exactly what a real method has to do: it has to *change what the weak learner sees* on each call, deliberately pushing it onto the points the earlier hypotheses got wrong, so that the new hypothesis is forced to be right where the old ones were wrong. Decorrelate the errors by reshaping the distribution. That's the whole game.

Schapire already showed the equivalence holds — weak does imply strong — and his construction is the proof I should study, because it's the existence result and I want to know why it's not the end of the story. His idea: take the weak learner A, get h1 on the original D. Now build a second distribution D2 that's rigged so h1 is reduced to a *coin flip* — half its mass where h1 is right, half where it's wrong — and run A there to get h2. Then build D3 from exactly the examples where h1 and h2 *disagree*, and get h3. Output the majority of h1, h2, h3. The filtered-distribution bookkeeping bounds the majority's error by g(α) = 3α² − 2α³ when the three recursive calls are each controlled at error α; the same cubic is the familiar majority-of-three expression 3α²(1−α) + α³, but here it is earned by the construction of D2 and D3, not by assuming independent errors. Check that it actually shrinks. At α = 1/2, g = 3/4 − 2/8 = 3/4 − 1/4 = 1/2, fixed point, fine. The derivative g′ = 6α − 6α² = 6α(1 − α) is 3/2 at α = 1/2, so moving just below 1/2 sends g(α) below α; algebraically, g(α) − α = −α(1 − α)(1 − 2α) < 0 for 0 < α < 1/2. Recurse by asking each subcall for error g^{-1} of the current target, and after O(log(1/ε)) levels the error is below ε. So the equivalence is true.

But look at what it costs and what it can't do. The final hypothesis is a deep tree of three-input majority gates, a whole recursive circuit whose shape changes from run to run. Every level is dominated by its *worst* of three sub-hypotheses — the analysis only ever uses "error ≤ α," so if one of the three calls happened to return something far better than α, that good luck is wasted; the bound doesn't see it. And the filtering that reduces h1 to a coin presupposes I can control the bias. It's a proof, not an algorithm I'd want to run.

Freund's boost-by-majority cleans up the shape. Forget recursion: just call the weak learner T times in a row. Before round t, reweight the examples — put more mass on the ones the *running unweighted majority* of h1,…,h_{t-1} currently gets wrong — get h_t, and at the end output a single flat majority vote over all T hypotheses. The reweighting isn't ad hoc: he derives it from a binomial-tail calculation. Think of each example as needing a certain number of remaining correct votes to be safely on the right side of the final majority; the optimal weight to place on it is the probability that, with the votes still to come, it lands wrong — a binomial tail. That weighting is provably the best possible allocation, and the number of rounds T it needs to reach error ε matches an information-theoretic lower bound. So boost-by-majority is essentially optimal in rounds, and it's a single flat majority gate, not a recursive circuit. Beautiful.

So why am I not done? Two coupled flaws, and they're the same flaw. First: the whole schedule — the binomial weights, the very definition of how many votes each example still needs — is built from γ, the worst-case edge of the weak learner, and γ has to be **known before boosting starts**. Second: the final vote is *unweighted*. Every h_t counts the same. Now picture what really happens when I run this. The weak learner's edge is not some fixed γ. On the first, easy distribution it might come back with error 0.1; three rounds later, on a viciously reweighted distribution concentrated on the hard cases, it limps in at error 0.45. The edge *varies round to round*, and I never know it in advance. Boost-by-majority is forced to plan for the worst case γ throughout, and because the final vote is unweighted it can't even give extra credit to the round where the learner returned something excellent. It throws away precisely the information I most want to use: *how good was this particular hypothesis on this particular reweighting.*

That's the wall, and stating it sharply tells me what to change. I want a booster that (a) never needs γ told to it ahead of time, and (b) *measures* the edge of each h_t as it arrives and lets that measurement drive both how it reweights for the next round and how much that hypothesis counts in the final vote. The parameter that controls the reweighting can't be a constant fixed before the run — it has to be a function of the error I actually observe on round t.

Where do I get the reweighting machinery, though? The on-line prediction people — Littlestone and Warmuth — solve a problem that, squint, has the same skeleton. They maintain weights over N "experts," predict by weighted majority, and after each round multiply every erring expert's weight by a fixed β ∈ [0,1). Their potential argument bounds the learner's loss against the best expert. Multiplicative weight update, exponential potential, distribution-free, adversarial. Now flip the roles. In their setting the *experts* are fixed and the *rounds* bring new losses. In boosting I want the *examples* to play the role of the things carrying weight, and each *round* brings a new hypothesis that scores them. The weighted-majority update — multiply the weight of an example by some β when the current hypothesis classifies it well, leave it heavy when the hypothesis misses it — is exactly the "push mass onto the hard examples" move I need. And crucially, the multiplicative-weights analysis works for *any* sequence of losses, even adversarially chosen ones — which means it should let β float, react to whatever ε_t the weak learner hands me, instead of being nailed down in advance. So let me stop reasoning about boost-by-majority's binomial bookkeeping and instead steal the multiplicative-weights skeleton, but reverse it so examples are the weighted objects and let the multiplier be set *per round from the measured error*.

Let me build it concretely. Examples (x_1, y_1), …, (x_N, y_N), labels in {0,1} for now. Maintain a weight vector w over the examples, start it at the given distribution D (uniform if I have no prior), w^1_i = D(i). On round t, normalize to a distribution p^t = w^t / Σ_i w^t_i and hand p^t to the weak learner; it returns h_t. Measure its weighted error right there: ε_t = Σ_i p^t_i |h_t(x_i) − y_i|. (I'll allow h_t ∈ [0,1] so |h_t − y_i| reads as a soft error; for a {0,1} hypothesis it's just the indicator of a mistake.) Now the update. I want: examples h_t got *right* lose weight, examples it got *wrong* keep theirs, and the strength of that push should depend on ε_t. The cleanest multiplicative form: pick a factor β_t ∈ [0,1) and set

w^{t+1}_i = w^t_i · β_t^{1 − |h_t(x_i) − y_i|}.

Stare at the exponent. If h_t nails example i, |h_t(x_i) − y_i| ≈ 0, the exponent ≈ 1, so the weight gets multiplied by β_t < 1 — driven down. If h_t blows it, |h_t(x_i) − y_i| ≈ 1, exponent ≈ 0, weight multiplied by β_t^0 = 1 — untouched. So correctly-classified examples bleed weight and the survivors are the hard ones. Exactly the decorrelation move. And β_t is left open — that's the knob I'll set from ε_t.

For the final hypothesis I refuse to do an unweighted vote, because the whole point was to reward the good rounds. A hypothesis that came in with tiny ε_t should count more. The natural weighted vote in this {0,1} convention uses coefficient a_t = log(1/β_t) and thresholds at the midpoint,

h_f(x) = 1 if Σ_t a_t h_t(x) ≥ ½ Σ_t a_t, else 0, where a_t = log(1/β_t).

Why log(1/β_t)? Hold that — let the proof tell me what coefficient makes the bound tightest, and then I'll see if it is this. Don't guess the weights; derive them.

Now the part that has to actually work: bound the training error of h_f and see what choice of β_t makes the bound smallest. Let me track Σ_i w^t_i, the total weight, because the multiplicative update controls it and the threshold rule will let me lower-bound it by the errors. Start with the update's effect on the total. I need a handle on β^x for x ∈ [0,1]. The function x ↦ β^x is convex, and a convex function on [0,1] sits below the chord joining its endpoints: β^x ≤ β^0 + x(β^1 − β^0) = 1 − (1 − β)x for x ∈ [0,1], with equality at x = 0 and x = 1. Apply it with x = 1 − |h_t(x_i) − y_i| ∈ [0,1]:

Σ_i w^{t+1}_i = Σ_i w^t_i β_t^{1 − |h_t(x_i) − y_i|} ≤ Σ_i w^t_i [1 − (1 − β_t)(1 − |h_t(x_i) − y_i|)].

Pull out Σ_i w^t_i and recall p^t_i = w^t_i / Σ_i w^t_i, so Σ_i p^t_i (1 − |h_t(x_i) − y_i|) = 1 − ε_t. Then

Σ_i w^{t+1}_i ≤ (Σ_i w^t_i) · [1 − (1 − β_t)(1 − ε_t)].

Chain this across all T rounds; the weights start summing to 1 (w^1 = D, a distribution), so

Σ_i w^{T+1}_i ≤ Π_{t=1}^T [1 − (1 − ε_t)(1 − β_t)].

That's the upper bound on the surviving total weight. Now the lower bound, which is where the threshold rule earns its keep. The final weight of any single example, unrolling the update from D(i), is

w^{T+1}_i = D(i) · Π_{t=1}^T β_t^{1 − |h_t(x_i) − y_i|}.

When does h_f get example i *wrong*? With labels in {0,1}, h_f(x_i) ≠ y_i means the weighted vote landed on the wrong side of the midpoint threshold, i.e. the example accumulated at least half the total coefficient mass on the wrong side. Working that condition through the definition of h_f, a mistake on i forces

Π_t β_t^{−|h_t(x_i) − y_i|} ≥ (Π_t β_t)^{−1/2}.

Let me sanity-check the direction. β_t < 1 so log β_t < 0; the exponent −|h_t − y_i| accumulates the *errors* of the committee on i. A mistake means the committee's weighted error on i is at least half the total weight, which is precisely Σ_t (log 1/β_t)|h_t(x_i) − y_i| ≥ ½ Σ_t log(1/β_t), and exponentiating −(that) gives Π β_t^{−|h_t−y_i|} ≥ (Π β_t)^{−1/2}. Good, the inequality points the right way. Now substitute into the final weight of a *mistaken* example:

w^{T+1}_i = D(i) · Π_t β_t · Π_t β_t^{−|h_t(x_i) − y_i|} ≥ D(i) · (Π_t β_t) · (Π_t β_t)^{−1/2} = D(i) · (Π_t β_t)^{1/2}.

So every example h_f misclassifies carries final weight at least D(i)(Π_t β_t)^{1/2}. Sum over just the mistaken examples — and the total over *all* examples is at least the total over the mistaken ones:

Σ_i w^{T+1}_i ≥ Σ_{i: h_f(x_i) ≠ y_i} w^{T+1}_i ≥ (Σ_{i: mistake} D(i)) · (Π_t β_t)^{1/2} = ε · (Π_t β_t)^{1/2},

where ε = Σ_{i: mistake} D(i) is exactly the training error of h_f under D. Now squeeze the upper and lower bounds on Σ_i w^{T+1}_i together:

ε · (Π_t β_t)^{1/2} ≤ Π_t [1 − (1 − ε_t)(1 − β_t)]  ⟹  ε ≤ Π_t [1 − (1 − ε_t)(1 − β_t)] / √β_t.

There's the training-error bound, and it's a product over rounds of independent factors, one per t. So I can minimize the whole thing by minimizing each factor f(β) = [1 − (1 − ε_t)(1 − β)] / √β separately over β — and now the question of what β_t to choose answers itself instead of me guessing it. Differentiate. Write the numerator as 1 − (1 − ε_t) + (1 − ε_t)β = ε_t + (1 − ε_t)β. So f(β) = [ε_t + (1 − ε_t)β] · β^{−1/2} = ε_t β^{−1/2} + (1 − ε_t) β^{1/2}. Then f′(β) = −½ ε_t β^{−3/2} + ½(1 − ε_t) β^{−1/2}. Set to zero: (1 − ε_t)β^{−1/2} = ε_t β^{−3/2}, multiply by β^{3/2}: (1 − ε_t)β = ε_t, so

β_t = ε_t / (1 − ε_t).

There it is, and it's not a constant fixed in advance — it's read straight off the measured error of round t. Small ε_t gives small β_t gives a hard down-weighting and (since a_t = log 1/β_t) a large vote; ε_t near 1/2 gives β_t near 1, a gentle update and a near-zero vote. Exactly the adaptivity boost-by-majority couldn't have, because it didn't measure ε_t. Now plug β_t = ε_t/(1 − ε_t) back into the factor. The numerator ε_t + (1 − ε_t)β_t = ε_t + (1 − ε_t)·ε_t/(1 − ε_t) = ε_t + ε_t = 2ε_t. The denominator √β_t = √(ε_t/(1 − ε_t)). So the factor is

2ε_t / √(ε_t/(1 − ε_t)) = 2ε_t · √((1 − ε_t)/ε_t) = 2 √(ε_t (1 − ε_t)).

And the bound collapses to

ε ≤ Π_{t=1}^T 2 √(ε_t (1 − ε_t)).

Let me feel what this says. Each factor 2√(ε_t(1−ε_t)) equals 1 exactly when ε_t = 1/2 — a hypothesis at chance contributes nothing — and is strictly less than 1 when the weak learner beats chance, ε_t < 1/2. If a returned classifier is worse than chance, I can flip it before using the same formulas. Write ε_t = 1/2 − γ_t, where γ_t is the *measured* positive edge on round t. Then 4ε_t(1 − ε_t) = 4(1/2 − γ_t)(1/2 + γ_t) = 4(1/4 − γ_t²) = 1 − 4γ_t², so the factor is √(1 − 4γ_t²) and

ε ≤ Π_t √(1 − 4γ_t²).

Now 1 − u ≤ e^{−u}, so √(1 − 4γ_t²) ≤ √(e^{−4γ_t²}) = e^{−2γ_t²}, and the whole product is bounded by

ε ≤ exp(−2 Σ_t γ_t²).

The training error decays *exponentially* in the accumulated squared edges. If every round clears the weak-learning bar with γ_t ≥ γ > 0, then ε ≤ exp(−2Tγ²), and to reach a target ε it suffices to run

T = ⌈ (1 / 2γ²) ln(1/ε) ⌉

rounds. That's it — that's the equivalence, made constructive and efficient. A weak learner, error always at most 1/2 − γ, gets driven to *any* target ε in O((1/γ²) log(1/ε)) rounds, with no recursion, no circuit, and — the thing I was chasing — no need to know γ in advance: the algorithm measured each ε_t and set β_t from it. The bound even rewards rounds where the learner overperformed, because their γ_t² is bigger and pushes the product down harder. Weak learnability and strong learnability are the same notion, and this is the witness. I'll call it adaptive boosting for that reason — it adapts to the errors it sees.

I want to go back and understand *why* the multiplicative-exponential form was the right shape, because right now it feels like it fell out of a convexity inequality and a lucky optimization, and I'd trust it more if I saw it as the minimization of something meaningful. Let me re-derive the same machine in the symmetric {−1, +1} label convention, where the algebra is cleaner. Labels y_i ∈ {−1, +1}, hypotheses h_t : X → {−1, +1}, and the combined score F(x) = Σ_t α_t h_t(x), final prediction sign(F(x)). The quantity y_i h_t(x_i) is +1 when h_t is right on i and −1 when wrong — the *signed* agreement. The update I want, in this convention, is the multiplicative one written with an exponential:

D_{t+1}(i) = D_t(i) · exp(−α_t y_i h_t(x_i)) / Z_t,

where Z_t normalizes D_{t+1} to a distribution. Check the sign: when h_t is right, y_i h_t = +1, the factor is e^{−α_t} < 1 for α_t > 0 — weight down. When wrong, y_i h_t = −1, factor e^{+α_t} > 1 — weight up. Same push as before. Now unroll it all the way from the uniform start D_1(i) = 1/m:

D_{T+1}(i) = (1/m) · exp(−Σ_t α_t y_i h_t(x_i)) / Π_t Z_t = exp(−y_i F(x_i)) / (m Π_t Z_t).

The exponents *added up* across rounds into the single signed margin y_i F(x_i), and the normalizers multiplied into Π_t Z_t. Since D_{T+1} is a distribution, Σ_i D_{T+1}(i) = 1, which forces

m Π_t Z_t = Σ_i exp(−y_i F(x_i)).

Now the training error. h_f errs on i exactly when y_i F(x_i) ≤ 0, and the indicator of that event is bounded by the exponential: 1[y_i F(x_i) ≤ 0] ≤ exp(−y_i F(x_i)), because e^{−u} ≥ 1 whenever u ≤ 0. So

training error = (1/m) Σ_i 1[y_i F(x_i) ≤ 0] ≤ (1/m) Σ_i exp(−y_i F(x_i)) = Π_t Z_t.

The training error is bounded by the product of the per-round normalizers. So if on each round I just greedily make Z_t as small as I can, I'm directly driving down (an upper bound on) the training error. That reframes everything: the per-round job is "minimize Z_t," and the object the whole algorithm is descending is

(1/m) Σ_i exp(−y_i F(x_i)) — the exponential loss.

This is the meaning I was missing. The same procedure is greedy coordinate-wise minimization of the exponential loss: each round adds one more term α_t h_t to F, choosing h_t and α_t to cut the loss the most. Why the exponential and not something else? Because it's a smooth, convex *surrogate* that sits above the thing I actually care about — the 0/1 misclassification count, which is discontinuous and hopeless to minimize directly — and pushing it down forces sign(F) to agree with y. The exponential punishes a confidently-wrong example (large negative margin) viciously and rewards confidently-right ones, which is exactly the pressure that decorrelates errors round over round.

And it tells me what α_t should be, *deriving* the coefficient I earlier refused to guess. Compute Z_t for h_t ∈ {−1, +1}. Split the examples by whether h_t is right: those with y_i h_t = +1 carry total D_t-mass 1 − ε_t, those with y_i h_t = −1 carry mass ε_t. So

Z_t = Σ_i D_t(i) exp(−α_t y_i h_t(x_i)) = (1 − ε_t) e^{−α_t} + ε_t e^{+α_t}.

Minimize over α_t: dZ_t/dα_t = −(1 − ε_t)e^{−α_t} + ε_t e^{+α_t} = 0 ⟹ e^{2α_t} = (1 − ε_t)/ε_t ⟹

α_t = ½ ln((1 − ε_t)/ε_t).

So the vote coefficient *is* (half) the log-odds of the round's accuracy — large when ε_t is small, zero when ε_t = 1/2, and it would go negative if ε_t > 1/2 (in which case I'd just flip h_t). And it matches the earlier construction: there the {0,1} vote coefficient is a_t = log(1/β_t) = log((1 − ε_t)/ε_t), and the factor-of-½ difference is exactly because in the symmetric convention the margin y_i h_t swings the full ±1, doubling the exponent, so the optimal coefficient is halved. Same algorithm, two derivations. Plug this α_t back into Z_t: e^{−α_t} = √(ε_t/(1 − ε_t)), e^{+α_t} = √((1 − ε_t)/ε_t), so

Z_t = (1 − ε_t)√(ε_t/(1 − ε_t)) + ε_t √((1 − ε_t)/ε_t) = √(ε_t(1 − ε_t)) + √(ε_t(1 − ε_t)) = 2√(ε_t(1 − ε_t)).

The minimized normalizer is 2√(ε_t(1 − ε_t)) — exactly the per-round factor from the first proof. The two paths land on the identical bound ε_train ≤ Π_t 2√(ε_t(1 − ε_t)). Now I trust it: the convexity-chord proof and the exponential-loss proof are the same fact seen from two sides, and the α_t that minimizes the exponential loss is the same α_t that minimizes the training-error bound. The reweighting is exponential because the loss is exponential; the coefficient is the log-odds because that's where the round's contribution to the loss is stationary.

Driving training error to zero is only half of learning, though. I need to know the error on *new* examples — the generalization error ε_g = Pr_{(x,y)∼𝒫}[h_f(x) ≠ y] — and whether running for more rounds helps or hurts it. Standard route: bound ε_g by the empirical error plus a complexity penalty for the hypothesis class h_f lives in. What class is that? The final hypothesis is a thresholded linear combination of T base hypotheses from some class H. Let Θ_T(H) be the set of all functions x ↦ sign(Σ_{t=1}^T a_t h_t(x) − b) with h_t ∈ H — linear threshold over T members of H. I can read h_f as a two-layer network: first layer the T weak hypotheses, second layer a single linear threshold unit combining them. The VC-dimension of a linear threshold over T inputs is T + 1, and a counting argument (Baum–Haussler-style, on how many distinct labellings such a two-layer machine can realize on m points) gives the VC-dimension of Θ_T(H), when H has VC-dimension d ≥ 2, as at most

2(d + 1)(T + 1) log₂(e(T + 1)).

So the complexity of the final hypothesis grows roughly *linearly* in T. Feed that into a VC generalization bound — ε_g ≤ ε̂ + Õ(√(VCdim / m)) — and the penalty scales like √(d T / m). Combine with the training-error result: the empirical error ε̂ plunges to zero in O(log m) rounds, but the complexity term keeps climbing with T. So the predicted picture is a U: test error falls as training error falls, then rises again as the growing complexity overwhelms the vanishing training error. Run too long and this bound expects overfitting. Taken at face value, it says: stop boosting after a moderate number of rounds, or pick T by validation.

Except the observed behavior often refuses to look like that U. This is the part that nags. With tree learners, boosting can keep improving test error long after the training error is already zero and long after the raw size of the final vote has become enormous. That does not make the VC bound false — it is only an upper bound — but it makes the bound feel like the wrong measuring stick. It sees only *whether* each training point is correct, the 0/1 fit, and nothing past that.

What is changing after the training error has already reached zero? The *confidence* of the predictions. The committee already votes correctly on every training point, but by how much? Define, for the {−1,+1} combined classifier, the margin of an example as the signed, normalized vote:

margin(x, y) = y · (Σ_t α_t h_t(x)) / (Σ_t α_t) = y F(x) / Σ_t α_t ∈ [−1, +1].

It's the weighted fraction of the committee voting correctly minus the weighted fraction voting incorrectly — the "margin of victory" of the internal election. It's positive iff h_f is right on (x, y), and its magnitude is how lopsided the vote was, i.e. how confident. Training error only counts the *sign* of the margin. But the whole *distribution* of margins can keep changing after zero training error: examples that were correct-but-barely, with small positive margin, can be pushed toward more comfortable margins. Nothing is happening to the training *error*, but a great deal is happening to the low-margin tail. And it's exactly because the exponential loss keeps rewarding larger y F(x): even once sign(F) is right everywhere, e^{−yF} keeps shrinking as yF grows, so the algorithm keeps pressure on small margins.

That suggests the generalization bound I should believe depends on the margins, not on T. And one can prove such a bound: for any θ > 0, with high probability,

ε_g ≤ P̂[margin(x, y) ≤ θ] + O( √( d / (m θ²) ) ),

where P̂ is the empirical fraction of training examples whose margin is below θ, d is the VC-dimension of the base class, and m the sample size. The decisive feature: the right-hand side has **no T in it at all**. It depends on how the margins are distributed and on the complexity of the *base* hypotheses, not on how many of them I combined. So if the continued rounds reduce P̂[margin ≤ θ] for a fixed θ, the bound can improve without paying a raw T penalty. The U-shaped VC prediction was an artifact of measuring complexity by vote size while ignoring confidence; the margins bound measures the low-confidence tail directly. The equivalence and the training-error bound stand on their own, and the margins explain why continuing to optimize after zero training error can still matter.

Let me write down the landed algorithm and the proof it rests on. In the symmetric convention, with a decision stump as the weak learner:

```python
import numpy as np

def fit_stump(X, y, w):
    """Weak learner: the single feature/threshold/polarity stump of least *weighted* error.
       Returns (predict_fn, eps) with eps = sum_i w_i 1[h(x_i) != y_i]."""
    m, n = X.shape
    best = None
    for j in range(n):                                  # each feature
        values = np.sort(np.unique(X[:, j]))
        mids = (values[:-1] + values[1:]) / 2.0
        thresholds = np.concatenate(([-np.inf], mids, [np.inf]))
        for thr in thresholds:
            for polarity in (+1, -1):                   # polarity flips the stump
                pred = np.where(polarity * (X[:, j] - thr) >= 0, 1, -1)
                eps = float(np.sum(w * (pred != y)))    # weighted error under current w
                if best is None or eps < best[0]:
                    best = (eps, j, thr, polarity)
    eps, j, thr, polarity = best
    predict = lambda Z: np.where(polarity * (Z[:, j] - thr) >= 0, 1, -1)
    return predict, eps

def adaboost(X, y, T):                                  # y in {-1, +1}
    m = len(y)
    w = np.full(m, 1.0 / m)                             # D_1: uniform distribution
    hyps, alphas = [], []
    for t in range(T):
        h, eps = fit_stump(X, y, w)                     # measure the round's edge: no gamma needed
        eps = np.clip(eps, 1e-12, 1 - 1e-12)
        alpha = 0.5 * np.log((1 - eps) / eps)           # = the coefficient minimizing Z_t
        pred = h(X)
        w = w * np.exp(-alpha * y * pred)               # D_{t+1}: up-weight the misses
        w = w / w.sum()                                 # renormalize (this is the Z_t division)
        hyps.append(h); alphas.append(alpha)
    return hyps, alphas

def predict(hyps, alphas, X):
    F = np.zeros(X.shape[0])
    for a, h in zip(alphas, hyps):
        F += a * h(X)                                   # F(x) = sum_t alpha_t h_t(x)
    return np.where(F >= 0, 1, -1)                      # final weighted-majority vote
```

The causal chain, start to finish: a weak learner is cheap but its errors on a fixed distribution are correlated, so naive voting can't amplify it — I must reshape the distribution each round to force the next hypothesis onto the previous mistakes. Stealing the multiplicative-weights skeleton but reversing it so examples carry the weights, and letting the multiplier float with the *measured* round error instead of a pre-known γ, gives the update D_{t+1}(i) ∝ D_t(i)·β_t^{1−|h_t−y_i|} (equivalently exp(−α_t y_i h_t(x_i))). Tracking the total weight through a convexity inequality and lower-bounding it by the mistakes through the threshold rule squeezes the training error to ε ≤ Π_t [1−(1−ε_t)(1−β_t)]/√β_t; minimizing each factor sets β_t = ε_t/(1−ε_t), collapses the bound to Π_t 2√(ε_t(1−ε_t)) = Π_t √(1−4γ_t²) ≤ exp(−2Σγ_t²), and makes T = O((1/γ²)log(1/ε)) rounds drive any weak learner to any target accuracy — so weak and strong learnability are equivalent. Reading the same update as greedy coordinate descent on the exponential loss (1/m)Σ exp(−y_i F(x_i)) explains *why* the reweighting is exponential and derives α_t = ½ln((1−ε_t)/ε_t) as the coefficient that minimizes each per-round normalizer Z_t = 2√(ε_t(1−ε_t)). And because the exponential loss keeps pressure on low-margin examples even after the training error hits zero, the generalization story is better expressed by a margin distribution bound independent of the number of rounds than by the raw size of the vote.
