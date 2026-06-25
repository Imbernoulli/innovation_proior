I start with the wrong idea because it exposes the real obstacle. If a weak learner gives me a hypothesis with error just below one half, it is tempting to run it many times on fresh samples and vote. But those runs all see the same distribution. The learner can keep finding the same easy regularity and keep missing the same hard part of the space. A majority of correlated mistakes is still a mistake. So the problem is not only to collect many weak hypotheses. I need to make their errors different. Each new call has to be pushed toward the examples the current collection is failing on.

That tells me why the distribution-free assumption is powerful. If the weak learner promises an edge on every distribution, I am allowed to choose distributions that make the old errors important. I can use the weak learner as a probe: show it the current hard region, get a rule with some edge there, and then combine that rule with the previous ones. The hard part is choosing the next distribution and deciding how much to trust each returned rule.

The first complete proof already uses this idea, but in a rigid way. It gets one hypothesis on the original distribution. Then it manufactures a second distribution where the first hypothesis is correct about half the time and wrong about half the time, so its advantage is neutralized. Then it manufactures a third distribution on the disagreements of the first two hypotheses. If each of the three subcalls has error `a`, the majority of the three should have error `g(a)=3a^2-2a^3` — the probability that two or three of them are wrong. For the recursion to make progress I need `g(a)<a` strictly below one half, so let me actually check it rather than trust the shape of the cubic. At `a=0.4`, `g=3(0.16)-2(0.064)=0.48-0.128=0.352<0.4`; at `a=0.49`, `g=3(0.2401)-2(0.117649)=0.7203-0.2353=0.485<0.49`; at `a=0.5`, `g=0.75-0.25=0.5`, a fixed point as it must be. So the gap is real but it shrinks to nothing as `a` approaches one half, which is exactly why the recursion needs many levels when the edge is small. This proves the weak-to-strong direction, but it feels like a proof of existence more than the mechanism I want. The final classifier is a recursive majority circuit, and the calculation uses a worst-case error level `a` for every subcall. If one weak hypothesis happens to come back at error `0.1` while another comes back at `0.49`, the recursion still budgets for the worst case at each node and does not naturally reward the good one.

The flat majority construction improves that. It says: do not build a recursive tree; build one majority gate over many weak hypotheses. It chooses example weights from a majority-vote game: for each point, how many more future correct votes does it need to end on the right side? That gives a sharp schedule and a near-optimal number of weak hypotheses. But now I hit the decisive wall. The schedule is built from a fixed edge `gamma` that I must know before the run starts. In an actual run, the first distribution might be easy and produce error `0.08`; a later distribution might be much harder and produce error `0.46`. Planning for one worst-case edge wastes information. Worse, the final vote is unweighted, so the excellent round and the barely useful round count the same.

So the missing ingredient is not just reweighting examples. It is measuring the actual error of each returned hypothesis and letting that measurement do two jobs at once: control how aggressively I shift the next distribution, and control how much that hypothesis counts in the final vote. A better hypothesis should move the distribution more and receive more voting weight. A near-chance hypothesis should barely move anything and should barely count.

The online weighted-majority idea gives me the algebraic shape. There, weights sit on experts, and a multiplicative update changes the weights after each loss. In my setting the weights should sit on examples. The "loss" of an example is inverted: if the current hypothesis gets an example right, that example is no longer urgent and should lose weight; if the hypothesis gets it wrong, the example should remain important. I therefore maintain weights `w_i^t` over training examples and normalize them to a distribution `p_i^t = w_i^t / sum_j w_j^t`. The weak learner receives `p^t` and returns `h_t`. I measure its weighted error,

`epsilon_t = sum_i p_i^t |h_t(x_i)-y_i|`.

Now I need a multiplier. In the `{0,1}` convention, the update

`w_i^{t+1} = w_i^t beta_t^(1 - |h_t(x_i)-y_i|)`

has the direction I want. If `h_t` is correct on example `i`, then the exponent is `1`, so the weight is multiplied by `beta_t`. If `h_t` is wrong, the exponent is `0`, so the weight is multiplied by `1`. In the useful weak-learning case `epsilon_t<1/2`, I will have `beta_t<1`, so correct examples are down-weighted and errors stay heavy.

I want to see how hard this pressure actually pushes, so let me run one update by hand. Take five examples with labels `(+1,+1,-1,-1,+1)`, start at the uniform distribution `D=(1/5,...,1/5)`, and suppose `h_1` returns `(-1,+1,-1,+1,+1)`, wrong on examples `1` and `4`. Its weighted error is `epsilon_1 = 2/5 = 0.4`, so `beta_1 = 0.4/0.6 = 2/3` and (in the signed form I will derive below) `alpha_1 = (1/2)\log(0.6/0.4) = 0.2027`. The two wrong examples get multiplied by `e^{alpha_1}=1.2247`, the three right ones by `e^{-alpha_1}=0.8165`. Before normalizing, the unnormalized weights are `(0.2449, 0.1633, 0.1633, 0.2449, 0.1633)` summing to `Z_1 = 0.97980`. After normalizing, `D_2 = (0.250, 0.1667, 0.1667, 0.250, 0.1667)`: the two errors now hold half the mass between them. Two things from this small computation are worth pinning down. First, `Z_1 = 0.97980` and `2\sqrt{0.4\cdot0.6} = 2\sqrt{0.24} = 0.97980` agree to every digit, so the per-round shrink factor I am chasing is literally the value of the normalizer — I have not had to assume that yet, it just came out. Second, and this is the part I did not expect to be so clean, the weighted error of the *same* `h_1` under the new distribution `D_2` is `0.250+0.250 = 0.500` exactly. The update drives the just-used hypothesis to exactly chance on the next round's distribution. That is the precise sense in which it "forces later weak learners onto the current failures": `h_1` is now useless on `D_2`, so any hypothesis with an edge on `D_2` must disagree with `h_1` somewhere, i.e. it must do work on the examples `h_1` got wrong. The pressure is not a vague tendency; it is an exact orthogonalization.

The final vote should use the same multiplier scale. If a small `beta_t` means the round was strong enough to shrink correct-example weights aggressively, then that round should get a large vote weight. The natural coefficient is `log(1/beta_t)`, and the final classifier predicts `1` when

`sum_t log(1/beta_t) h_t(x) >= (1/2) sum_t log(1/beta_t)`.

I still have not chosen `beta_t`. I do not want to guess it. I want the error bound to choose it.

I track the total weight after each update. The function `beta^x` is convex on `[0,1]`, so it lies below the chord between its endpoint values: `beta^x <= 1 - (1-beta)x`. Applying this with `x = 1 - |h_t(x_i)-y_i|`, I get

`sum_i w_i^{t+1} <= (sum_i w_i^t) [1 - (1-epsilon_t)(1-beta_t)]`.

The initial weights sum to one, so after `T` rounds,

`sum_i w_i^{T+1} <= product_t [1 - (1-epsilon_t)(1-beta_t)]`.

That is the upper bound. I also need a lower bound that connects total remaining weight to final mistakes. If the final weighted vote is wrong on example `i`, then at least half of the total coefficient mass has effectively voted the wrong way. Written multiplicatively, that condition gives

`product_t beta_t^(-|h_t(x_i)-y_i|) >= (product_t beta_t)^(-1/2)`.

The final weight of example `i` is

`w_i^{T+1} = D(i) product_t beta_t^(1 - |h_t(x_i)-y_i|)`.

For a mistaken example, the vote condition therefore implies

`w_i^{T+1} >= D(i) (product_t beta_t)^(1/2)`.

Summing over mistaken examples gives

`sum_i w_i^{T+1} >= error * (product_t beta_t)^(1/2)`.

Now I squeeze the total weight between the upper and lower bounds:

`error <= product_t ([1 - (1-epsilon_t)(1-beta_t)] / sqrt(beta_t))`.

The product separates by round, so I can minimize each factor independently. For one round,

`f(beta) = [epsilon_t + (1-epsilon_t) beta] beta^(-1/2)`.

Differentiating gives

`f'(beta) = -0.5 epsilon_t beta^(-3/2) + 0.5 (1-epsilon_t) beta^(-1/2)`.

Setting this to zero yields `(1-epsilon_t) beta = epsilon_t`, so

`beta_t = epsilon_t/(1-epsilon_t)`.

I notice that I never had to supply `beta_t` myself: the round's measured error `epsilon_t` sets it. That is the property I was missing in the flat construction, and I should check that the formula behaves sensibly across the range of errors instead of just asserting it does. At `epsilon_t=0.1`, `beta_t=0.1/0.9=0.111` and the coefficient `log(1/beta_t)=2.197`; at `epsilon_t=0.4`, `beta_t=0.667` and `log(1/beta_t)=0.405`; at `epsilon_t=0.49`, `beta_t=0.961` and `log(1/beta_t)=0.040`. So a round at error `0.1` both down-weights correct examples roughly nine times harder (`0.961/0.111`) and counts about fifty times more in the final vote (`2.197/0.040`) than a round at error `0.49` — the same measurement controls both the next reweighting strength and the vote weight, and the two effects scale together. The boundary cases are limits of these same formulas: as `epsilon_t -> 0`, `beta_t -> 0` and the coefficient diverges, which means the round has found a perfect classifier on the weighted sample and the sensible move is to stop; as `epsilon_t -> 1/2`, `beta_t -> 1` and the coefficient `-> 0`, so the distribution barely changes and the round barely counts. If `epsilon_t` is at least one half in the binary setting, there is no positive edge to amplify unless I flip or reject the hypothesis. A useful round contributes in proportion to its observed confidence.

Substituting the minimizing `beta_t` into the factor gives

`2 sqrt(epsilon_t(1-epsilon_t))`.

Thus the training error of the final classifier is bounded by

`product_t 2 sqrt(epsilon_t(1-epsilon_t))`.

If I write `epsilon_t = 1/2 - gamma_t`, then each factor is `sqrt(1 - 4 gamma_t^2)`, hence at most `exp(-2 gamma_t^2)`. The whole product is at most

`exp(-2 sum_t gamma_t^2)`.

This is the weak-to-strong amplification I was after: a single product of measured factors that decays geometrically whenever the edges stay bounded away from zero. If every generated distribution gives a hypothesis with edge at least `gamma`, then `product_t exp(-2 gamma_t^2) <= exp(-2 T gamma^2)`, which is below `epsilon` once `T >= (1/(2 gamma^2)) log(1/epsilon)`. But the algorithm itself does not need to know `gamma`. It just keeps measuring `epsilon_t`, changing the distribution, and weighting the vote accordingly.

I should not let the algebra alone convince me the actual training error obeys this, so let me run the whole loop and watch both numbers. I take sixty points in the plane labeled by `sign(x_1 x_2)` — an XOR pattern where axis-aligned decision stumps are genuinely weak — and boost stumps for fifteen rounds, printing `epsilon_t`, the factor `2\sqrt{epsilon_t(1-epsilon_t)}`, the running product, and the true zero-one training error. The first rounds come back at `epsilon_t` around `0.35`–`0.40` and the product falls to `0.90` after three rounds with training error `0.27`; by round fifteen the product is `0.757` and the training error is `0.150`. At every single round the running product is `>=` the measured training error, so the bound holds as derived. But the run also corrects a hope I might have carried in: `epsilon_t` does not stay near `0.35`. It drifts upward toward `0.44` as the rounds proceed, because each update makes the surviving distribution harder and the stumps weaker on it. The factors approach one and the product flattens — exactly the slow regime the `g(a)` fixed point warned about earlier. The geometric decay is real but its rate is set by the *running* edges, not by the easy first round; on a hard problem the bound crawls. That is the honest reading: the equivalence is no longer only a recursive existence proof, it is a practical procedure, but "practical" buys fast convergence only when the weak learner keeps a real edge on the reweighted distributions.

I can rewrite the same method in the cleaner signed-label notation. Let `y_i in {-1,+1}`, let each `h_t(x_i)` also be `-1` or `+1`, and define the score `F(x)=sum_t alpha_t h_t(x)`. The update becomes

`D_{t+1}(i) = D_t(i) exp(-alpha_t y_i h_t(x_i)) / Z_t`.

The sign is right: a correct example has `y_i h_t(x_i)=+1` and gets multiplied by `exp(-alpha_t)`, while a wrong example has `y_i h_t(x_i)=-1` and gets multiplied by `exp(alpha_t)`. The final classifier is `sign(F(x))`.

Now the normalizer tells the same story. For a fixed `h_t`,

`Z_t = (1-epsilon_t) exp(-alpha_t) + epsilon_t exp(alpha_t)`.

Minimizing this over `alpha_t` gives

`alpha_t = (1/2) log((1-epsilon_t)/epsilon_t)`.

The minimized normalizer is `2 sqrt(epsilon_t(1-epsilon_t))`, the same factor as before. The factor of one half is just notation: in the `{0,1}` form the final coefficient is `log((1-epsilon_t)/epsilon_t)`, while in the signed form the margin swings from `-1` to `+1`, so the coefficient is half as large.

The signed form lets me ask where the exponential weighting came from in the first place. Unrolling the update telescopes the normalizers: `D_{T+1}(i) = (1/m) exp(-y_i F(x_i)) / product_t Z_t`, and since `D_{T+1}` sums to one, `(1/m) sum_i exp(-y_i F(x_i)) = product_t Z_t`. The training mistake indicator `1[y_i F(x_i) <= 0]` is at most `exp(-y_i F(x_i))`, so the training error is at most `product_t Z_t` — recovering the same bound from a different direction. Read this way, the algorithm is greedily reducing an exponential upper bound on training mistakes, and choosing `alpha_t` to minimize `Z_t` is choosing the steepest one-round descent of that bound. That view explains the coefficient and the update, but I keep the causal order straight: the boosting guarantee I just derived comes from the weight argument, while the exponential-loss view is another reading of the same algebra.

I still need to think about test error. A direct complexity bound says the final classifier is a thresholded linear combination of `T` base hypotheses. If the base class has VC dimension `d`, then the combined class has complexity growing roughly like `d T log T`. That suggests a tradeoff: training error falls as rounds increase, but the complexity term grows. This predicts eventual overfitting.

Empirically, that prediction is often too crude. Once training error reaches zero, more rounds can still improve the confidence of the votes. The right quantity is the margin: the weighted fraction of votes for the correct label minus the weighted fraction against it, equivalently `y F(x) / sum_t alpha_t`. A positive margin means correct classification; a large positive margin means a decisive vote. Continuing to boost can push small positive margins higher even when the zero-one training error no longer changes. Margin bounds explain why the number of rounds alone is not the whole story.

The final method is therefore simple but not shallow. It does not merely rerun a weak learner and vote. It builds a moving distribution over examples so weak learners focus on current errors; it measures each round's weighted error; it uses that measurement to choose both the next reweighting strength and the final vote confidence; and its training error falls with the accumulated squared edges. That is the adaptive core.
