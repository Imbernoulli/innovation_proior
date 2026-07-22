OK, let me think this through from scratch. The thing that bothers me is a result I can't shake: on Tetris, on hand-tuned locomotion, plain black-box search — cross-entropy method, CMA — is hard to beat, and it just rattles a search distribution over policy parameters and keeps the good samples. Meanwhile gradient methods, which on paper have far better sample-complexity (with a real gradient oracle the complexity need not even depend on dimension, whereas black-box scales like the parameter count), keep losing on the hard problems. That's backwards. And it's especially galling because supervised deep learning just showed that plain stochastic gradient descent scales to millions of parameters without drama. So why can't I take a policy that's a big neural net and just... ascend the return?

Let me try the obvious thing and watch it break. The return is `η(π_θ) = E[Σ_t γ^t r(s_t)]`, and the policy-gradient identity gives me `∇_θ η = E[ ∇_θ log π_θ(a|s) · A_π(s,a) ]` — score function times advantage, estimable from sampled trajectories. So I take `θ ← θ + α ∇η`. Now: how big is α? If I make it small, I crawl, and the variance of the estimate means I'm crawling through noise — that's exactly the regime where black-box random search keeps pace. If I make it large, sometimes the policy just falls off a cliff: one update and the return collapses, and because the next batch of data is collected *under the wrecked policy*, I might never recover. So the entire game is the step size, and I have no principled handle on it.

Why is there no handle? Because I'm measuring the step in the wrong space. `α ∇η` moves θ by some Euclidean amount `‖Δθ‖`. But θ are the weights of a network whose output is a *distribution* over actions. Euclidean distance in weight space has no fixed relationship to how much the distribution `π(·|s)` actually changed. Near a saturated softmax a tiny `Δθ` barely moves the probabilities; near a sensitive region the same `Δθ` flips them. So a single learning rate is being asked to mean "small move" in a geometry where "small" is wildly state- and location-dependent. No wonder it's brittle.

So the real question isn't "what step size in θ" — it's "how far should the *policy* move, per update, measured in distribution space, so that the true return reliably goes up?" Let me chase *that*.

To even ask whether the return goes up, I need to relate the new policy's return to the old one's. There's a clean identity for this. Write `η(π̃)` in terms of `η(π)` plus how much better `π̃` does, step by step. Claim:
`η(π̃) = η(π) + E_{τ∼π̃}[ Σ_t γ^t A_π(s_t,a_t) ]`,
where the advantage `A_π(s,a) = Q_π(s,a) − V_π(s)` is measured under the *old* policy but the trajectory `τ` is rolled out under the *new* one. Note `A_π(s,a) = E_{s'∼P(·|s,a)}[ r(s) + γ V_π(s') − V_π(s) ]`. Take the expectation over a `π̃`-trajectory:
`E_{τ∼π̃}[ Σ_t γ^t A_π(s_t,a_t) ] = E_{τ∼π̃}[ Σ_t γ^t ( r(s_t) + γ V_π(s_{t+1}) − V_π(s_t) ) ]`.
The `V` terms telescope: the partial sum through `T` is `Σ_{t=0}^T γ^t (γ V(s_{t+1}) − V(s_t)) = −V(s_0) + γ^{T+1}V(s_{T+1})`, and the tail vanishes since `γ<1` and values are bounded. So the right side is `E_{τ∼π̃}[ −V_π(s_0) + Σ_t γ^t r(s_t) ] = −η(π) + η(π̃)`. Rearrange and the identity holds — exact, no approximation.

Now collect by state instead of by timestep. Define the discounted visitation frequency `ρ_π(s) = Σ_t γ^t P(s_t = s)`. Then
`η(π̃) = η(π) + Σ_s ρ_{π̃}(s) Σ_a π̃(a|s) A_π(s,a)`.
Stare at this. It says: if at *every* state the new policy has nonnegative expected advantage `Σ_a π̃(a|s) A_π(s,a) ≥ 0`, the return can't decrease. That's the classic policy-iteration intuition — pick actions with positive advantage, you improve. But there's a poison pill: the weighting is `ρ_{π̃}`, the *new* policy's visitation. To know how good `π̃` is I'd have to already know where `π̃` spends its time, which depends on `π̃`, which is what I'm solving for. Circular. With function approximation and finite samples I can't enforce nonnegative advantage at literally every state anyway, so some states will go negative, and then the `ρ_{π̃}` weighting really matters and I can't evaluate it.

So I do the only tractable thing: pretend the visitation doesn't change. Define
`L_π(π̃) = η(π) + Σ_s ρ_π(s) Σ_a π̃(a|s) A_π(s,a)` —
same expression but with the *old* visitation `ρ_π`. Is that legitimate? It's exact in value at `π̃ = π`. For the gradient, write the exact objective as `η(π_θ) = η(π_{θ0}) + Σ_s ρ_{π_θ}(s) Σ_a π_θ(a|s)A_{π_{θ0}}(s,a)`. Differentiating at `θ0` gives two terms. The derivative of `π_θ(a|s)` is the policy-gradient term. The derivative of `ρ_{π_θ}(s)` is multiplied by `Σ_a π_{θ0}(a|s)A_{π_{θ0}}(s,a)`, which is zero at every state because advantage is centered under the old policy. So the visitation-derivative term drops exactly at the expansion point, and `∇_θ L_{π_{θ0}}(π_θ)|_{θ0} = ∇_θ η(π_θ)|_{θ0}` as well as `L_{π_{θ0}}(π_{θ0}) = η(π_{θ0})`. A sufficiently small step that improves `L` improves `η`. That's the same "sufficiently small" that stalled vanilla PG, one level up: I've reproduced the vanilla-PG problem at a higher level — I have a surrogate `L` I can actually optimize, and zero guidance on how far to trust it.

I need a *quantitative* statement: improve `L` by some amount, move the policy by at most some amount, and the true `η` is guaranteed to improve by at least something. A lower bound on improvement. Does anything like that exist? Yes — conservative policy iteration. Kakade and Langford proved exactly such a bound, but only for a peculiar update: the new policy is a *mixture* `π_new = (1−α)π_old + α π'`, where `π'` greedily maximizes `L`. For that mixture they show
`η(π_new) ≥ L_{π_old}(π_new) − (2εγ/(1−γ)^2) α^2`, with `ε = max_s |E_{a∼π'}[A(s,a)]|`.
This is the shape I want: a first-order-accurate surrogate `L` minus a penalty that's *quadratic* in how far you stepped (`α`). Improve the RHS and you provably improve η. The quadratic penalty is the leash. But the leash is tied to a policy class I'd never use — who parameterizes a deep net as an explicit mixture with a mixing coefficient α? Mixtures are unwieldy; I want the bound for *arbitrary* stochastic policies, with α replaced by some honest *distance* between `π_old` and `π_new`.

So can I rederive the bound with a distance in place of the mixing coefficient? The mixture had a clean meaning for α: with probability `1−α` the new policy acts like the old one. Let me generalize that notion. Say `(π, π̃)` is an α-coupled pair if I can define a joint distribution over action pairs `(a, ã)|s` whose marginals are `π` and `π̃` and for which `P(a ≠ ã | s) ≤ α` at every state. Operationally: fix a random seed, sample from each — they agree for at least a fraction `1−α` of seeds. The mixture was just one way to realize this; coupling is the general version.

Why does coupling buy me a quadratic error? Because the difference between `η(π̃)` and `L_π(π̃)` is *entirely* about visitation, i.e. about the trajectories drifting apart, and they only drift once `π` and `π̃` have actually disagreed. Let me make that precise. Write both quantities in the same form. Define the per-state mean advantage `Ā(s) = E_{a∼π̃(·|s)}[A_π(s,a)]`. Then from the exact identity,
`η(π̃) = η(π) + E_{τ∼π̃}[ Σ_t γ^t Ā(s_t) ]`, and by construction
`L_π(π̃) = η(π) + E_{τ∼π}[ Σ_t γ^t Ā(s_t) ]`.
The *only* difference is whether the states `s_t` are drawn under `π̃` or under `π`. So `|η(π̃) − L_π(π̃)| = | Σ_t γ^t ( E_{s_t∼π̃}[Ā(s_t)] − E_{s_t∼π}[Ā(s_t)] ) |`, and I just need to bound, per timestep, how different those two state-expectations of `Ā` can be.

First, how big is `Ā(s)` itself for a coupled pair? The old policy's expected advantage is zero, `E_{a∼π}[A_π(s,a)] = 0` (advantage is centered by definition). So I can write `Ā(s) = E_{ã∼π̃}[A(s,ã)] − E_{a∼π}[A(s,a)] = E_{(a,ã)}[ A(s,ã) − A(s,a) ]` under the coupling. When `a = ã` the integrand is zero, so only the disagreement events contribute: `Ā(s) = P(a≠ã|s) · E[ A(s,ã) − A(s,a) | a≠ã ]`. Each advantage is at most `ε := max_{s,a}|A_π(s,a)|` in magnitude, so the bracket is at most `2ε`, and `P(a≠ã)≤α`. Hence `|Ā(s)| ≤ 2 α ε`: disagreement is rare *and* each disagreement is bounded, so the mean advantage is `O(α)` and `O(ε)`.

Now the per-timestep visitation gap. Couple the *trajectories* with a shared seed: `τ` from `π`, `τ̃` from `π̃`, same random numbers. Let `n_t` = number of disagreements before time `t`. Split each state-expectation on whether `n_t = 0`:
`E_{s_t∼π̃}[Ā(s_t)] = P(n_t=0) E[Ā|n_t=0] + P(n_t>0) E_{π̃}[Ā|n_t>0]`, and the same for `π`. Crucially, conditioned on `n_t = 0` the two trajectories have been *identical* up to `t`, so `s_t` has the same distribution under both — those terms cancel exactly. What's left:
`E_{s_t∼π̃}[Ā] − E_{s_t∼π}[Ā] = P(n_t>0)( E_{π̃}[Ā|n_t>0] − E_{π}[Ā|n_t>0] )`.
Bound each piece. The policies agree each step with prob ≥ `1−α`, so `P(n_t=0) ≥ (1−α)^t`, giving `P(n_t>0) ≤ 1−(1−α)^t`. And each conditional expectation of `Ā` is at most `max_s|Ā(s)| ≤ 2αε`, so their difference is at most `4αε`. Therefore
`| E_{s_t∼π̃}[Ā(s_t)] − E_{s_t∼π}[Ā(s_t)] | ≤ 4 ε α (1−(1−α)^t)`.

Sum over time with the `γ^t` weights:
`|η(π̃) − L_π(π̃)| ≤ Σ_t γ^t · 4εα(1−(1−α)^t) = 4εα ( Σ_t γ^t − Σ_t (γ(1−α))^t ) = 4εα ( 1/(1−γ) − 1/(1−γ(1−α)) )`.
Combine the two fractions: numerator `(1−γ(1−α)) − (1−γ) = γα`, denominator `(1−γ)(1−γ(1−α))`, so the whole thing is `4εα · γα / [(1−γ)(1−γ(1−α))] = 4εγα^2 / [(1−γ)(1−γ(1−α))]`. Since `1−γ(1−α) ≥ 1−γ`, the denominator is at least `(1−γ)^2`, giving the clean bound
`|η(π̃) − L_π(π̃)| ≤ 4εγ α^2 / (1−γ)^2`.

Let me sanity-check this against limits I already know the answer to, because the constant rode through a telescoping sum and a fraction-combination where a stray `γ` or factor of 2 could hide. (i) `α = 0`: the policies are identical, `L_π(π) = η(π)` exactly, so the gap must be zero — and indeed the bound has `α²` out front, vanishing. (ii) Keep the *exact* pre-relaxation form `4εγα²/[(1−γ)(1−γ(1−α))]` and let `γ → 0` (myopic, single-step problem): the denominator `→ 1`, so the gap `→ 4εγα² → 0` linearly in `γ`. That's right too — with no discounting horizon, the visitation `ρ_π̃` and `ρ_π` differ only at the first step, so swapping them costs `O(γ)`, exactly what the bound shows. (iii) The factor that worried me most is the `1/(1−γ)²`: where does the *square* come from, intuitively? One `1/(1−γ)` is the horizon over which I sum the per-step gaps; the other is because the disagreement probability `1−(1−α)^t` *itself* grows over the horizon — late states are reached only after many chances to diverge. The product of "many timesteps" and "more divergence per late timestep" is the square. Each factor of `1/(1−γ)` has a mechanism, so the `(1−γ)²` makes sense.

These are limit checks, though, and a limit check can't catch an error that only shows up at a generic interior point — a wrong constant can still vanish correctly at `α=0` and scale correctly in `γ`. So before I build anything on this I want to watch the inequality hold on a fully concrete instance where I compute *both* sides from scratch. Take a 2-state, 2-action MDP: states `{0,1}`, reward `r = (0, 1)` (state 1 is the good one), `γ = 0.9`, transitions `P[0,·] = ([.8,.2],[.2,.8])`, `P[1,·] = ([.3,.7],[.7,.3])`, start always in state 0. Old policy `π_old = (.5, .5)` (prob of action 1 per state), new policy `π_new = (.9, .2)`. Everything here is exactly computable by solving the two linear systems `(I−γP_π)V = r` and `(I−γP_π^T)ρ = ρ0`. Doing that arithmetic: `η_old = 4.500`, the *true* new return `η_new = 6.011`, and the surrogate `L_π(π_new) = 6.174`. So the surrogate over-predicts, and the real gap is `|η_new − L| = 0.163`. Now the bound: `ε = max|A_old| = 0.270`, the per-state TV moves are `(|.5−.9|, |.5−.2|) = (.4, .3)` so `α = 0.4`, and `C = 4·0.270·0.9/(0.1)² = 97.2`. The predicted gap ceiling is `C·α² = 97.2·0.16 = 15.55`. And indeed `0.163 ≤ 15.55` — the inequality holds, with enormous room to spare. I re-ran the same computation for 2000 random `π_new` and the gap never once exceeded `C·α²` (smallest slack `≈ 7×10⁻⁴`, so the bound *is* attainable, just not at my hand-picked point). So the constant isn't just right in the limits; the actual inequality it claims is satisfied on a real instance and across a random sweep. (iv) Independently of all this, the perturbation proof below will produce `4γεα²/(1−γ)²` by a disjoint route — same square, same `4`, same `γ`. A symbolic re-derivation and a numeric stress-test agreeing is about as much assurance as I can get on a worst-case constant no experiment will ever pin down.

That same numeric instance flags something I should keep in mind, though. The true gap was `0.163` but the bound allowed `15.55` — the bound is loose by a factor of ~100 here. Worse, the surrogate itself was only `6.174` while the penalty `C·α²` was `15.55`, so `L − C·α² = −9.4`: the guaranteed lower bound on `η_new` came out *negative*, far below `η_old = 4.5`. If I were choosing the step by maximizing `L − C·D_KL`, this instance says the penalty would veto almost any real move. I'll come back to this — it's a concrete warning that the honest constant, used literally, will refuse to take useful steps — but first let me get the constant a second way.

The last link: I assumed an α-coupling, but I want to state this in terms of an actual distance between the policies. Total variation is exactly the right one. I claim: if `D_TV(p‖q) = α` then there exists a joint distribution `(X,Y)` with marginals `p,q` such that `P(X≠Y) = α` — and not just exists, I can construct it, which is what convinces me. Build the *maximal coupling*: with probability mass `Σ_i min(p_i,q_i)` set `X=Y` by drawing a common value `i` with probability `min(p_i,q_i)/Σ_j min(p_j,q_j)`; with the remaining mass draw `X` from the leftover `(p_i−min(p_i,q_i))_+` (normalized) and `Y` independently from `(q_i−min(p_i,q_i))_+`, which by construction never collide. Check the marginal: `P(X=i) = min(p_i,q_i) + (p_i−min(p_i,q_i))= p_i`, good. And `P(X=Y) = Σ_i min(p_i,q_i)`. Now `Σ_i min(p_i,q_i) = Σ_i (p_i − (p_i−q_i)_+) = 1 − Σ_i (p_i−q_i)_+ = 1 − ½Σ_i|p_i−q_i| = 1 − D_TV`, where the middle step uses that the positive and negative parts of `p−q` each sum to `D_TV` (they're equal because both `p` and `q` sum to 1). So `P(X≠Y) = D_TV = α` exactly. This is the construction behind "agrees with probability `1−α`" — I didn't have to take it on faith. So taking `α = D_TV^max(π,π̃) := max_s D_TV(π(·|s)‖π̃(·|s))` realizes a per-state coupling, and I land on the general bound:
`η(π̃) ≥ L_π(π̃) − (4εγ/(1−γ)^2) D_TV^max(π,π̃)^2`, `ε = max_{s,a}|A_π(s,a)|`.
So the mixture-only conservative-policy-iteration bound now holds for *any* pair of stochastic policies, with the mixing coefficient `α` replaced by their total-variation distance. The leash I was after — a quadratic penalty in how far the policy moved — survives the generalization, and it's stated in a quantity I can measure between two arbitrary policies rather than a mixing parameter I'd never use.

Let me get the same constant a completely different way, because a load-bearing constant deserves a second witness, and the coupling argument has a lot of moving parts (the trajectory split, the `(1−α)^t` bookkeeping) where I could have slipped a factor. Try perturbation theory on the discounted-occupancy operator. Write the resolvent `G = (1 + γP_π + (γP_π)^2 + …) = (1−γP_π)^{-1}`, where `P_π` is the state-to-state transition operator under `π`; likewise `G̃ = (1−γP_{π̃})^{-1}`. The bookkeeping convention I'll use: a density `ρ` over states is a column vector, the reward `r` is a row vector (a linear functional), so `rGρ₀` is a scalar — and indeed `η(π) = rGρ₀` (sum the geometric series: `rGρ₀ = r(Σ_t γ^t P_π^t)ρ₀ = Σ_t γ^t E[r(s_t)] = η(π)`), and `η(π̃) = rG̃ρ₀`. I want `η(π̃)−η(π) = r(G̃−G)ρ₀`. The standard resolvent trick: `G^{-1} − G̃^{-1} = (1−γP_π) − (1−γP_{π̃}) = γ(P_{π̃}−P_π) =: γΔ`. Left-multiply by `G`, right-multiply by `G̃`: `G̃ − G = γGΔG̃`, i.e. `G̃ = G + γGΔG̃`. That's still implicit in `G̃` on the right; substitute it into itself once to expose the order in `Δ`: `G̃ = G + γGΔ(G + γGΔG̃) = G + γGΔG + γ²GΔGΔG̃`. Therefore
`η(π̃) − η(π) = γ·rGΔGρ₀  +  γ²·rGΔGΔG̃ρ₀`.

Now the leading term. Note `rG = v` (it's the value function: `rG` is exactly the row vector whose `s`-component is `Σ_t γ^t E[r(s_t)|s_0=s] = V_π(s)`), and `Gρ₀ = ρ_π` (the resolvent applied to the start distribution is the discounted visitation). So `γ·rGΔGρ₀ = γ·v Δ ρ_π`. I claim this *equals* `L_π(π̃) − L_π(π)`, i.e. the perturbation expansion reproduces the surrogate's first-order change. Verify by expanding the surrogate directly:
`L_π(π̃) − L_π(π) = Σ_s ρ_π(s) Σ_a (π̃(a|s) − π(a|s)) A_π(s,a)`.
Substitute `A_π(s,a) = r(s) + γΣ_{s'} p(s'|s,a) v(s') − v(s)`. The `r(s) − v(s)` part is action-independent, so `Σ_a (π̃−π)(r(s)−v(s)) = (r(s)−v(s))·Σ_a(π̃−π) = 0` since both policies are normalized. Only the middle survives:
`= Σ_s ρ_π(s) Σ_{s'} γ v(s') Σ_a (π̃(a|s)−π(a|s)) p(s'|s,a) = Σ_s ρ_π(s) Σ_{s'} γ v(s') (p_{π̃}(s'|s) − p_π(s'|s)) = γ v Δ ρ_π`,
reading `Δ = P_{π̃}−P_π` as the state-to-state operator and contracting indices. So the leading perturbation term is identically the surrogate's increment — the surrogate `L` is *exactly* the first-order term of the true return in the policy perturbation, which is the cleanest possible statement of "`L` matches `η` to first order."

Now the second-order remainder `γ²·rGΔGΔG̃ρ₀`, which must be the `O(α²)` penalty. Split it as `γ·(rGΔ) · (γGΔG̃ρ₀)` and bound each factor with the matched norm so Hölder applies. The first factor `γrGΔ = γvΔ` is a row vector; bound it in `∞`-norm. Its `s`-component is `Σ_a (π̃(s,a)−π(s,a)) Q_π(s,a)`, and since `Σ_a(π̃−π) = 0` I may swap `Q` for `A` (subtracting the constant `V(s)` changes nothing): `= Σ_a (π̃−π)A_π(s,a)`, so `|(γvΔ)_s| ≤ (Σ_a|π̃(s,a)−π(s,a)|)·max_a|A_π(s,a)| ≤ 2α·ε`, using `Σ_a|π̃−π| = 2 D_TV ≤ 2α` and `ε = max_{s,a}|A_π|`. Hence `‖γvΔ‖_∞ ≤ 2αε`. The second factor I bound in `1`-norm via the `ℓ₁` operator norm `‖B‖₁ = sup_ρ ‖Bρ‖₁/‖ρ‖₁`. For a stochastic-matrix resolvent `‖G‖₁ = ‖1 + γP_π + γ²P_π² + …‖₁ ≤ Σ_t γ^t‖P_π‖₁ = Σ_t γ^t·1 = 1/(1−γ)` (each `P_π` is column-stochastic so `‖P_π‖₁=1`), and identically `‖G̃‖₁ = 1/(1−γ)`; and `‖Δ‖₁ = ‖P_{π̃}−P_π‖₁ = 2 max_s D_TV(p_{π̃}(·|s)‖p_π(·|s)) ≤ 2α` (the column-wise TV of the transition kernels is at most the action-TV that produced them). With `‖ρ₀‖₁ = 1`,
`‖γGΔG̃ρ₀‖₁ ≤ ‖G‖₁‖Δ‖₁‖G̃‖₁‖ρ₀‖₁ ≤ (1/(1−γ))·2α·(1/(1−γ)) = 2α/(1−γ)²`.
Hölder (`|u^Tw| ≤ ‖u‖_∞‖w‖₁`) on the two factors gives
`γ²|rGΔGΔG̃ρ₀| ≤ ‖γvΔ‖_∞ · ‖γGΔG̃ρ₀‖₁ ≤ 2αε · 2α/(1−γ)² = 4γεα²/(1−γ)²`,
where the stray `γ` rides along because `γvΔ = γ·(rGΔ)` already carried one factor of `γ` and the second factor carried the other. Same constant `4γε/(1−γ)²`, same `α²`, by an entirely disjoint route. So now three things line up on this constant: the coupling derivation, this resolvent perturbation, and the numeric instance where the inequality actually held — a symbolic re-derivation, an independent symbolic route, and a numerical stress-test, which is about as much triangulation as a worst-case constant admits. And the perturbation view threw in the bonus that `L` is *literally* the linearization of `η`.

TV is awkward to work with on parameterized distributions, but I have Pinsker's inequality lying around: `D_TV(p‖q)^2 ≤ D_KL(p‖q)`. So I can upper-bound the squared TV by KL and absorb it into the penalty:
`η(π̃) ≥ L_π(π̃) − C · D_KL^max(π,π̃)`, with `C = 4εγ/(1−γ)^2`,
where `D_KL^max = max_s D_KL(π(·|s)‖π̃(·|s))`. Now everything is in KL, which I can actually differentiate and estimate.

And this bound isn't just a one-shot guarantee — it gives me an *algorithm* with monotonic improvement. Let `M_i(π) = L_{π_i}(π) − C·D_KL^max(π_i,π)`. The bound says `η(π) ≥ M_i(π)` for all π, and at `π = π_i` it's tight: `L` and the KL penalty are both exact there, so `η(π_i) = M_i(π_i)`. So if I set `π_{i+1} = argmax M_i`, then
`η(π_{i+1}) − η(π_i) ≥ M_i(π_{i+1}) − M_i(π_i) ≥ 0`,
the first inequality because `η ≥ M_i` everywhere and `η = M_i` at `π_i`, the second because `π_{i+1}` maximizes `M_i`. So `η` never decreases. This is a minorize-maximize scheme: `M_i` is a surrogate that touches `η` from below at the current policy and I keep maximizing it. (It also smells like proximal gradient / mirror descent — I'm maximizing an affine approximation of the objective minus a divergence penalty; the KL is the Bregman divergence of the entropy regularizer. Nice that it fits a known mold, but I don't need that to proceed.)

So in principle I'm done: repeatedly maximize `L_{θ_old}(θ) − C·D_KL^max(θ_old,θ)`. Let me try to actually run this and watch where it hurts.

First wall — and it's the one the numeric instance already warned me about: that constant `C = 4εγ/(1−γ)^2`. Even at the mild `γ = 0.9` of my toy MDP, `C` was `97.2` and the penalty `C·α²` swamped the surrogate, sending the guaranteed bound to `−9.4`. At the `γ = 0.99` I actually care about, `(1−γ)^2 = 10^{-4}`, so `C` is on the order of `10^4 · ε` — two orders of magnitude worse again. The penalty is enormous, which means the maximizer of `M_i` barely moves from `θ_old`: microscopic steps. The bound is honest but worst-case, and the toy instance made the looseness concrete — the *true* gap there was `0.16` while the penalty charged `15.55`, off by ~100×. Worst-case is far too pessimistic for the typical step. If I literally used the theoretical `C` I'd train forever. I want big steps when they're safe.

The fix: stop treating the KL as a *penalty* with a fixed coefficient and treat it as a *constraint* with a budget I choose. Instead of `max L − C·D_KL`, solve
`max_θ L_{θ_old}(θ)  s.t.  D_KL^max(θ_old, θ) ≤ δ`.
This is a trust region: optimize the surrogate but only within a region where the policy is allowed to move by `δ` in KL. The penalty form has a coefficient that's hard to choose robustly (and the theory's value is uselessly conservative); the constraint form has `δ`, which is *interpretable* — it directly says "the policy may move this far per update" — and gives consistent step sizes across iterations and across problems. This is the trade I'll make: keep the theory's *shape* (improve a surrogate, bounded policy movement) but swap the unusable constant for a tunable trust-region radius.

Second wall: `D_KL^max`, the max over *all* states, is one constraint per state — infinitely many in a continuous space, and a max is nasty for sample-based estimation and for the optimizer. I can't enforce a separate constraint at every state. Soften it to the *average* KL over the normalized discounted state distribution generated by the old policy:
`D̄_KL^ρ(θ_old,θ) = E_{s∼d_{θ_old}}[ D_KL(π_{θ_old}(·|s)‖π_θ(·|s)) ] ≤ δ`.
This is a heuristic — averaging is weaker than bounding the max — but it's one scalar constraint I can estimate from samples, and the line search can enforce this sampled constraint on the nonlinear policy after the local step is proposed.

Now make the objective something I can compute from a batch of `π_old` rollouts. Expand `L`:
`L_{θ_old}(θ) = Σ_s ρ_{θ_old}(s) Σ_a π_θ(a|s) A_{θ_old}(s,a)` (dropping the additive constant `η(θ_old)`, which doesn't affect the argmax). Three substitutions make it an expectation over samples. Let `d_{θ_old}(s) = (1−γ)ρ_{θ_old}(s)` be the normalized discounted visitation distribution, so `Σ_s ρ_{θ_old}(s)[…] = (1/(1−γ))E_{s∼d_{θ_old}}[…]`; the positive factor `1/(1−γ)` scales the objective but leaves the constrained step unchanged after the KL-normalized step length is computed. Replace the advantage `A_{θ_old}` by the Q-value `Q_{θ_old}`; they differ by `V_{θ_old}(s)`, which is constant in the action, so `Σ_a π_θ(a|s)V(s)=V(s)` is independent of θ. Replace the inner `Σ_a π_θ(a|s)(…)` by an importance-sampling expectation under a sampling distribution `q`: `Σ_a π_θ(a|s) A(s,a) = E_{a∼q}[ (π_θ(a|s)/q(a|s)) A(s,a) ]`. Putting it together, the problem becomes exactly
`max_θ E_{s∼d_{old}, a∼q}[ (π_θ(a|s)/q(a|s)) Q_{old}(s,a) ]  s.t.  E_{s∼d_{old}}[ D_KL(π_{old}(·|s)‖π_θ(·|s)) ] ≤ δ`.
For the simplest estimator — single-path — I roll out `π_old` itself, so `q = π_old`, the ratio at the sampled action is `π_θ/π_old`, and `Q_old` is just the discounted return along the trajectory from that state-action.

There's a lower-variance alternative, *vine*: pick a set of states (the "rollout set"), branch off `K` actions each, estimate each `Q` by a short rollout, and crucially reuse one random-number seed across the `K` rollouts so the *differences* in Q between actions — which is all that drives the gradient — have their shared noise cancel. Let me actually establish that the gradient depends only on Q-*differences*, because the whole common-random-numbers (CRN) argument hangs on it. Take a finite two-action space `{1,2}` and the non-importance-sampled per-state loss `L_n(θ) = Σ_a π_θ(a|s_n) Q̂(s_n,a) = π_θ(1|s_n)Q̂(s_n,1) + π_θ(2|s_n)Q̂(s_n,2)`. Use `π_θ(2|s_n) = 1 − π_θ(1|s_n)` to eliminate the second probability:
`∇_θ L_n = ∇_θ[π_θ(1|s_n)Q̂(s_n,1) + (1−π_θ(1|s_n))Q̂(s_n,2)] = ∇_θ π_θ(1|s_n)·(Q̂(s_n,1) − Q̂(s_n,2))`.
So the per-state gradient is `∇π · (Q̂₁ − Q̂₂)` — the absolute Q-levels drop out entirely, only the difference survives. Its variance is therefore proportional to `Var[Q̂₁ − Q̂₂]`. Now CRN: model each rollout's `Q̂(s,a)` as a deterministic function `Q(s,a,z)` of a shared noise vector `z` (the `U(0,1)` draws feeding both the stochastic dynamics and the stochastic policy over the rollout). With CRN both actions are rolled out under the *same* `z`, so `Var_{+CRN} = Var_z[Q(s,1,z) − Q(s,2,z)] = σ₁² + σ₂² − 2ρσ₁σ₂`; without it, independent `z₁,z₂` give `Var_{−CRN} = σ₁² + σ₂²`. So `Var_{+CRN} < Var_{−CRN}` exactly when the correlation `ρ > 0`, which it is whenever the two action-branches share most of their downstream randomness — they do, since after the first differing action the trajectories experience the same dynamics noise. The reason this isn't a minor tweak: define the signal-to-noise ratio `|Q(s,a₁)−Q(s,a₂)| / sqrt(Var[Q̂(s,a₁)])`. In a continuous-time control problem discretized at `Δt`, a single action held for one step changes the long-run return by `O(Δt)` (the signal), while the rollout-estimate variance stays `O(1)` (the noise accumulates over the whole trajectory regardless of `Δt`), so without CRN the SNR `→ 0` as `Δt → 0` — the gradient drowns in noise exactly in the fine-discretization limit I care about. With CRN the shared noise cancels in the difference and the SNR limit is finite. So CRN is what keeps the vine estimator from degenerating as the control becomes fine-grained. The cost is many simulator resets, so vine needs a resettable simulator; single-path runs on anything.

For large or continuous action spaces I can't enumerate the `K` actions, so I importance-sample. The basic IS estimator `(1/K)Σ_k (π_θ(a_k)/q(a_k)) Q̂_k` is unbiased, but it misbehaves in a way worth catching before I lean on it. The weights `w_k = π_θ(a_k)/q(a_k)` are sampled under `q = π_old`, so at `θ = θ_old` every weight is exactly 1 and the estimator equals `(1/K)Σ_k Q̂_k`, a raw average of Q-values. Two problems. First, with only `K` (say 4) actions per state, that average has whatever scale and offset the `Q̂_k` happen to have — a large state-dependent constant rides on top of the signal, and since I'm differentiating, that constant's *sampling noise* is pure variance with no useful gradient. Second, as `θ` moves off `θ_old` the weights spread out, and a single action that became much more likely under `π_θ` gets a weight `≫ 1`; with a handful of samples the estimate is then dominated by one or two terms — heavy-tailed, the classic IS blow-up. The unbiasedness is cold comfort when the variance is this large at small `K`. So the basic estimator is the wrong tool; I need one that cancels the state-dependent offset and is robust to weight spread. Use instead the *self-normalized* estimator, dividing by the sum of the weights: `L_n(θ) = [Σ_k w_k Q̂_k] / [Σ_k w_k]` with `w_k = π_θ(a_{n,k})/q(a_{n,k})`. It's biased (the ratio of two estimators), but the bias is `O(1/K)` and the variance is far smaller, and — the payoff — its gradient subtracts a baseline for free. Watch what happens to its gradient — quotient rule, and the magic is in what the numerator becomes. Differentiating `f̂_sn = Σ_k w_k Q̂_k / Σ_k w_k`:
`∇_θ f̂_sn = [ (Σ_k ∇w_k Q̂_k)(Σ_k w_k) − (Σ_k w_k Q̂_k)(Σ_k ∇w_k) ] / (Σ_k w_k)²`,
and pulling `Σ_k w_k` out of the denominator and recognizing `Σ_k w_k Q̂_k / Σ_k w_k = f̂_sn` itself,
`∇_θ f̂_sn = [ Σ_k ∇w_k·(Q̂_k − f̂_sn) ] / Σ_k w_k`.
The `Q̂_k` got *replaced* by `Q̂_k − f̂_sn` — the estimator subtracts its own weighted mean, automatically, as a baseline. And recall from the likelihood-ratio identity that `∇E[f] = E[∇log p·(f − β)]` for any constant baseline `β`, with variance minimized when `β ≈ E[f]`: the self-normalized estimator picks `β = f̂_sn ≈ E[Q]` for free, out of the quotient rule, without my having to fit a separate baseline. That's why I can drop a hand-built baseline from the vine objective and still get the variance reduction — the normalization *is* the baseline. (And this is consistent with my earlier claim that swapping `A` for `Q` in the surrogate is harmless: adding any state-constant to `Q` leaves the gradient unchanged precisely because the estimator centers `Q` itself.)

Now the real work: how do I actually *solve*
`max_θ L(θ)  s.t.  D̄_KL(θ_old,θ) ≤ δ`
for a network with thousands of parameters, every iteration, cheaply? The objective and constraint are both nonlinear in θ, but δ is small, so the step stays in a neighborhood of `θ_old` — I can use local approximations. Expand both around `θ_old`. To first order the surrogate is linear: `L(θ) ≈ L(θ_old) + g^T(θ−θ_old)` with `g = ∇_θ L|_{θ_old}` (the policy gradient). The constraint: `D̄_KL` is zero at `θ_old`, and its *gradient* is also zero there (KL is minimized at the matching distribution), so the leading term is quadratic: `D̄_KL(θ_old,θ) ≈ ½(θ−θ_old)^T A (θ−θ_old)`, where `A = ∇^2_θ D̄_KL|_{θ_old}` is the Hessian of the average KL — which is precisely the (average) Fisher information matrix. So the subproblem is
`max_x g^T x  s.t.  ½ x^T A x ≤ δ`, with `x = θ−θ_old`.

This is a quadratic-constrained linear program; solve it with a Lagrangian. The optimum lies on the constraint boundary and aligns the gradient of objective and constraint: `g = λ A x`, so `x ∝ A^{-1} g`. The direction is `A^{-1} g` — and wait. `A` is the Fisher information matrix; the step direction `A^{-1} g` is exactly the *natural gradient*. I didn't set out to build the natural gradient; it fell out of "linearize the objective, quadraticize the KL constraint." So the natural policy gradient is just the special case of this trust-region update where I take a quadratic model of the KL and a fixed step. And vanilla PG is the *other* special case: if instead of the KL metric I'd used a Euclidean trust region `½‖x‖^2 ≤ δ`, the same Lagrangian gives `x ∝ I^{-1} g = g`, the raw gradient. Same machine, different metric — and the whole earlier complaint about Euclidean geometry is now a one-line consequence of which `A` I plug in. (Drop the constraint entirely and fully maximize `L` and you recover policy iteration. One update, three classical methods as limits.)

For the step length, don't leave it as a free learning rate — saturate the constraint. With direction `x = A^{-1}g`, scale it: `θ = θ_old + β x`. Plug into the quadratic constraint at equality: `½(βx)^T A (βx) = ½ β^2 (x^T A x) = δ`, so
`β = sqrt( 2δ / (x^T A x) )`.
This is the move that distinguishes me from plain natural gradient: natural gradient picks `β` (or the penalty `1/λ`) as a fixed hyperparameter and lives with whatever KL that produces; I *solve* for the `β` that hits the KL budget `δ` exactly, every step. That's the thing I think actually matters — it's why a fixed-step natural-gradient method can take steps that are fine on an easy task and fatal on a hard one, while a fixed-δ method moves the policy a controlled distance regardless.

But forming `A` is a non-starter. For `n` parameters `A` is `n×n`; with tens of thousands of parameters that's hundreds of millions of entries to build, store, and *invert*. The whole point was to scale. So I must never materialize `A`. What do I actually need? I need `x = A^{-1}g`, i.e. to solve `A x = g`. Conjugate gradient solves a linear system using *only* matrix-vector products `v ↦ A v` — no explicit matrix. A handful of CG iterations (≈10) gives a good-enough `A^{-1}g`. So the whole problem reduces to: compute Fisher-vector products `A v` cheaply.

Two ways to get `A v` without forming `A`. The clean general trick: `A` is the Hessian of `D̄_KL`, so `A v = ∇_θ( (∇_θ D̄_KL)^T v )` — take the gradient of the average KL, dot it with `v` (a scalar), differentiate again. Two backward passes, no matrix. That's a Hessian-vector product and any autodiff package does it. Even better, exploit the structure, and I want to actually do the second derivative rather than wave at it, because the cancellation is the whole reason this is cheap. The policy maps state `x` to a distribution-parameter vector `μ_θ(x)` (the Gaussian mean, or the softmax logits), and the per-state KL depends on θ *only through* `μ`: write `D_KL(π_old(·|x)‖π_θ(·|x)) = kl(μ_old(x), μ_θ(x))`, a fixed scalar function `kl(·,·)` of two parameter vectors. Differentiate once with the chain rule: `∂/∂θ_i kl = (∂μ_a/∂θ_i)·kl'_a`, where `kl'_a` is the derivative of `kl` w.r.t. the `a`-th component of its *second* argument (sum over `a` implied). Differentiate again, product rule on the two θ-dependent factors:
`∂²/∂θ_i∂θ_j kl = (∂μ_a/∂θ_i)(∂μ_b/∂θ_j)·kl''_{ab}  +  (∂²μ_a/∂θ_i∂θ_j)·kl'_a`.
Two terms: a `J^T(kl'')J` piece and a curvature-of-`μ` piece weighted by `kl'_a`. Now the cancellation: I evaluate the Hessian at `θ = θ_old`, where the two distributions coincide, and `kl(μ, ·)` as a function of its second argument is minimized (value 0) exactly at the matching point — so its first derivative there is zero, `kl'_a(μ_old, μ_old) = 0` for every `a`. That kills the entire second term, the one carrying the expensive second derivatives `∂²μ/∂θ²` of the network. What survives is `A = J^T M J`, with `J = ∂μ/∂θ` the Jacobian of the distribution parameters and `M = kl''_{ab}` the Hessian of the KL *in mean-parameter space* — which is the Fisher information of the distribution family itself, a small object with a closed form (for a diagonal Gaussian, `M` is diagonal: `1/σ²` on the mean coordinates, `2` on each log-σ coordinate, in the natural coordinates). Sanity check the cancellation against the generic Hessian-vector route: if I'd just autodiff'd `∇(∇D̄_KL·v)` blindly I'd have implicitly computed *both* terms — but at `θ_old` the second is numerically zero, so the two routes agree, the structured one just never forms the `∂²μ/∂θ²` tensor in the first place. Then `A v = J^T (M (J v))`: multiply by `J` (a Jacobian-vector / forward op), by the tiny `M` (closed form), by `J^T` (which is exactly the backprop operation). Either way the Fisher-vector product costs about one gradient evaluation; the structured form is the one that never touches a dense Hessian or a per-network-weight second derivative.

One more economy: `A` is only a *metric* — it shapes the direction, it doesn't carry the reward signal. So I can estimate it on a *subsample* of the batch (say 10%) without much harm, while the gradient `g` uses all the data. With CG at ~10 iterations and FVPs on 10% of the data, computing the natural step `A^{-1}g` costs about the same as computing `g` alone. So the trust-region step is essentially free relative to a vanilla gradient step. (Practically I also add a tiny multiple of the identity to `A` inside the FVP — `A v → (A + ηI)v` — so CG stays numerically well-behaved; it's a small damping, nothing conceptual.)

Also, when I build `A`, I build it as the analytic Hessian of the KL — `A_{ij} = (1/N)Σ_n ∂²/∂θ_i∂θ_j D_KL(π_old(·|s_n)‖π_θ(·|s_n))`, integrating over the action at each state — rather than as the empirical covariance of the per-sample score gradients `A_{ij} = (1/N)Σ_n ∂_i log π_θ(a_n|s_n)·∂_j log π_θ(a_n|s_n)`. The two are the same matrix in expectation (the Fisher information has both forms), but they differ operationally in a way that matters at scale. The empirical-covariance form depends on the actually-sampled action `a_n`: it's a rank-one outer product per sample, so to apply it as a metric I'd either materialize the dense `n×n` average of outer products, or keep every per-sample score gradient `∇log π(a_n|s_n)` around to re-form the product on demand — `O(N · #params)` storage just for the metric. The analytic form has *integrated the action out* — `D_KL(π_old‖π_θ)` at state `s_n` is an expectation over actions with no sampled `a_n` left in it — so its Hessian-vector product is the `J^T M J` operation above, needing only the current network and the tiny closed-form `M`, no stored per-sample gradients and no dense Hessian. It's also lower-variance for the same reason (no Monte-Carlo over the action). Same metric in expectation, but the analytic one is the only one that's actually cheap to *apply* for a big net — and empirically the policy improves at the same rate either way, so I'm giving up nothing by integrating the action out. Same metric, cheaper and cleaner.

Now the last hole. I solved a *quadratic* model of the constraint and a *linear* model of the objective, then stepped `β` to saturate the quadratic model. But the true KL and true surrogate are nonlinear; the model is only good locally, and `β` was computed to land exactly on the model's boundary — so the true KL after the step might exceed `δ`, or the true surrogate might not actually improve (these are the catastrophic-collapse steps that wreck training). I shouldn't trust the full computed step blindly. So: backtracking line search. The honest object I want to maximize along the ray is the *constrained* surrogate — write it as `L_old(θ) − X[D̄_KL(θ_old,θ) ≤ δ]`, where the indicator penalty `X[·]` is `0` when the true average KL is within budget and `+∞` when it isn't. That single expression encodes both requirements: a step that violates the budget has value `−∞` (instantly rejected), and among feasible steps I want the one that raises the true `L`. Operationally, start at the full step and shrink geometrically: try `θ_old + α^j β x` for `j = 0,1,2,…` with `0<α<1`, and accept the first `j` for which the *true* (nonlinear) surrogate `L` improves over `θ_old` *and* the *true* average KL is `≤ δ` (the two conditions that make `L − X[·]` finite and improved). The shrink factor `α^j` is geometric rather than, say, halving the model-predicted KL, because the true KL grows roughly quadratically in step length near `θ_old` (its gradient is zero there), so shrinking `β` by `α` shrinks the realized KL by about `α²` — a few backtracks cover orders of magnitude of KL and quickly find the feasible regime. If none of the `j` qualifies (rare), take no step at all. Without this check the algorithm occasionally takes a huge step that collapses performance; with it, the step is both an improvement and within budget by the real measures, not just the approximate ones. This is exactly the robustness that fixed-step natural gradient lacks: it commits to `β` from the quadratic model; I verify against the truth and back off if the model lied.

Let me also nail down the policy parameterization, since `M`/the KL depend on it. For continuous control: a Gaussian with mean `μ_θ(s)` from the network and a diagonal covariance whose log-standard-deviations are state-independent learned parameters — concretely the network is a couple of fully-connected hidden layers with a smooth nonlinearity (a soft rectifier `σ(x) = log(1+e^x)` works well — smooth so that the Jacobian `J = ∂μ/∂θ` and hence the Fisher-vector product are well-behaved everywhere, unlike a hard ReLU whose kinks make the second-order geometry ragged), producing `μ_θ(s)`, plus a separate parameter vector `r` of per-dimension log-standard-deviations, so `π_θ(a|s) = N(a; μ_θ(s), diag(exp(2r)))`. The KL between two diagonal Gaussians has a closed form. Let me write it out so I can read off `M`: for one coordinate with means `μ₀,μ` and standard deviations `σ₀,σ = e^{r₀}, e^r`,
`D_KL(N(μ₀,σ₀²)‖N(μ,σ²)) = log(σ/σ₀) + (σ₀² + (μ₀−μ)²)/(2σ²) − ½`.
At the matching point `μ=μ₀, r=r₀`: `∂/∂μ = −(μ₀−μ)/σ² = 0` and `∂/∂r = 1 − (σ₀² + (μ₀−μ)²)/σ² = 1−1 = 0` — both first derivatives vanish, which is the concrete instance of the `kl'_a = 0` cancellation I used to drop the `∂²μ/∂θ²` term. The second derivatives there: `∂²/∂μ² = 1/σ² = e^{−2r}`, and `∂²/∂r² = 2(σ₀² + (μ₀−μ)²)/σ²|_{match} = 2`, with the cross term `∂²/∂μ∂r = 0`. Let me confirm these aren't an algebra slip, since they *are* the matrix the whole inner loop multiplies by — finite-difference the closed-form KL at a generic matching point `μ₀ = 0.3, r₀ = −0.5` (so `σ₀² = e^{−1} ≈ 0.368`, predicted `∂²/∂μ² = e^{1} ≈ 2.718`). Numerically I get `∂/∂μ ≈ 0` and `∂/∂r ≈ −6×10⁻¹¹` (both first derivatives vanish, as the algebra said), and `∂²/∂μ² ≈ 2.71828`, `∂²/∂r² ≈ 2.00000`, cross term `≈ 0` — matching `e^{−2r₀}` and `2` to five figures. So `M = diag(e^{−2r}, …, e^{−2r}, 2, …, 2)` — diagonal, the mean-block scaled by inverse variance and the log-σ-block a constant 2. That's the tiny closed-form `M` the Fisher-vector product needs. For discrete actions (Atari): a *factored* categorical/softmax, where the action is a tuple `(a_1,…,a_K)` and each factor `μ_k = softmax(logits_k)` is its own categorical — so `μ` is the concatenation of the factors' probability vectors and the KL is the sum of per-factor categorical KLs, each again with first derivative zero and a closed-form second derivative (`M` block `= diag(1/p) − 11^T` in probability coordinates). State-independent log-σ is a deliberate simplification — it makes the policy's exploration a smooth global quantity rather than something the net can collapse pointwise (a state-dependent σ-head can drive `σ(s) → 0` in a high-advantage region, becoming nearly deterministic there and making the KL geometry blow up like `1/σ²`), and it keeps the Fisher block for the σ-parameters clean (the constant `2` above, independent of the data).

So the loop assembles itself. Collect trajectories under `π_old`; estimate advantages (subtract a learned value-function baseline `V(s)` fit by regression to returns to cut variance — and note the importance-sampling surrogate is baseline-invariant, so a self-normalized estimator already behaves like it has a baseline). Form `g = ∇_θ L` (the surrogate's gradient, which is the policy gradient weighted by advantages). Get the search direction `x = A^{-1}g` by conjugate gradient using Fisher-vector products. Compute the step `β = sqrt(2δ/(x^T A x))`. Backtrack along `α^j β x` until true KL ≤ δ and true surrogate improves. Refit the value function. Repeat. Every piece traces back to one demand: move the policy a controlled distance in KL each step so the true return reliably goes up.

```python
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# --- policy: state -> Gaussian over actions; diagonal, state-independent log-std.
#     KL between two diagonal Gaussians is closed form, so its Hessian (the Fisher
#     metric A) has a cheap matrix-vector product. (categorical for discrete actions)
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64)):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]; last = h
        layers += [nn.Linear(last, act_dim)]
        self.mu_net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))  # state-independent

    def dist(self, obs):
        return Normal(self.mu_net(obs), torch.exp(self.log_std))

    def logp(self, obs, act):
        return self.dist(obs).log_prob(act).sum(-1)

class ValueNet(nn.Module):                      # learned baseline V(s)
    def __init__(self, obs_dim, hidden=(64, 64)):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]; last = h
        layers += [nn.Linear(last, 1)]
        self.v = nn.Sequential(*layers)
    def forward(self, obs):
        return self.v(obs).squeeze(-1)

def flat(grads):                                 # flatten a list of tensors
    return torch.cat([g.reshape(-1) for g in grads])

def set_flat_params(model, flat_params):          # write a flat vector into the net
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(flat_params[i:i + n].view_as(p)); i += n

def discount_cumsum(x, discount):
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + discount * running
        out[t] = running
    return out

def gae(rewards, values, last_val, gamma=0.99, lam=0.97):
    rews = np.append(rewards, last_val)
    vals = np.append(values, last_val)
    deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    adv = discount_cumsum(deltas, gamma * lam)
    ret = discount_cumsum(rews, gamma)[:-1]
    return adv, ret

# --- surrogate L: importance-weighted advantage under old-policy samples.
#     This is the affine model whose gradient g is the policy gradient.
def surrogate(policy, obs, act, adv, logp_old):
    ratio = torch.exp(policy.logp(obs, act) - logp_old)
    return (ratio * adv).mean()

# --- average KL: zero and zero-gradient at theta_old; its Hessian is the Fisher A.
def mean_kl(policy, obs, mu_old, std_old):
    d = policy.dist(obs)
    d_old = Normal(mu_old, std_old)
    return torch.distributions.kl_divergence(d_old, d).sum(-1).mean()

# --- Fisher-vector product A v = grad( (grad mean_kl) . v ): no matrix formed.
def fisher_vector_product(policy, obs, mu_old, std_old, v, damping=0.1):
    kl = mean_kl(policy, obs, mu_old, std_old)
    g = flat(torch.autograd.grad(kl, policy.parameters(), create_graph=True))
    gv = (g * v).sum()
    hv = flat(torch.autograd.grad(gv, policy.parameters(), retain_graph=True))
    return hv + damping * v                       # damping keeps CG well-conditioned

# --- conjugate gradient solves A x = g using only matrix-vector products.
def conjugate_gradient(Avp, b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone(); p = b.clone(); r_dot = torch.dot(r, r)
    for _ in range(iters):
        Ap = Avp(p)
        alpha = r_dot / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p; r -= alpha * Ap
        r_dot_new = torch.dot(r, r)
        if r_dot_new < tol: break
        p = r + (r_dot_new / r_dot) * p; r_dot = r_dot_new
    return x

def trpo_step(policy, obs, act, adv, logp_old, mu_old, std_old,
              delta=0.01, backtrack_coeff=0.8, backtrack_iters=10):
    # g = gradient of the surrogate = policy gradient
    L_old = surrogate(policy, obs, act, adv, logp_old)
    L_old_value = L_old.detach()
    g = flat(torch.autograd.grad(L_old, policy.parameters(), retain_graph=True))

    # search direction x = A^{-1} g  (natural gradient), via CG + Fisher-vector products
    Avp = lambda v: fisher_vector_product(policy, obs, mu_old, std_old, v)
    x = conjugate_gradient(Avp, g)

    # step length that saturates the KL budget:  1/2 beta^2 x^T A x = delta
    xAx = torch.dot(x, Avp(x))
    beta = torch.sqrt(2 * delta / (xAx + 1e-8))
    full_step = beta * x

    # backtracking line search on the TRUE objective + TRUE KL constraint:
    # the quadratic/linear models are only local, so verify before committing.
    old_params = flat([p.data for p in policy.parameters()])
    for j in range(backtrack_iters):
        step = (backtrack_coeff ** j) * full_step
        set_flat_params(policy, old_params + step)
        with torch.no_grad():
            kl = mean_kl(policy, obs, mu_old, std_old)
            L_new = surrogate(policy, obs, act, adv, logp_old)
        if kl <= delta and L_new > L_old_value:
            return                                # accept first step that truly improves & obeys delta
    set_flat_params(policy, old_params)           # nothing worked: don't move

def fit_value(value_net, obs, returns, opt, iters=80):
    for _ in range(iters):
        opt.zero_grad()
        loss = ((value_net(obs) - returns) ** 2).mean()
        loss.backward(); opt.step()

def reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, terminated or truncated, info
    return out

def train(env, policy, value_net, epochs=50, steps=4000, gamma=0.99, lam=0.97):
    v_opt = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    for epoch in range(epochs):
        O, Ac, Rew, LogpOld, MuOld = [], [], [], [], []
        o, ep_rew, ep_val = reset_env(env), [], []
        for t in range(steps):
            ot = torch.as_tensor(o, dtype=torch.float32)
            with torch.no_grad():
                d = policy.dist(ot); a = d.sample()
                logp = d.log_prob(a).sum(-1); val = value_net(ot)
            o2, r, done, _ = step_env(env, a.numpy())
            O.append(o); Ac.append(a.numpy()); ep_rew.append(r); ep_val.append(val.item())
            LogpOld.append(logp.item()); MuOld.append(d.mean.numpy()); o = o2
            if done or t == steps - 1:
                with torch.no_grad():
                    last_val = 0.0 if done else value_net(torch.as_tensor(o, dtype=torch.float32)).item()
                adv, ret = gae(np.array(ep_rew), np.array(ep_val), last_val, gamma, lam)
                Rew.append((adv, ret)); o, ep_rew, ep_val = reset_env(env), [], []
        adv = np.concatenate([a for a, _ in Rew]); ret = np.concatenate([r for _, r in Rew])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.as_tensor(np.array(O), dtype=torch.float32)
        act = torch.as_tensor(np.array(Ac), dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        logp_old = torch.as_tensor(np.array(LogpOld), dtype=torch.float32)
        mu_old = torch.as_tensor(np.array(MuOld), dtype=torch.float32)
        std_old = torch.exp(policy.log_std).detach().expand_as(mu_old)

        trpo_step(policy, obs, act, adv_t, logp_old, mu_old, std_old, delta=0.01)
        fit_value(value_net, obs, ret_t, v_opt, iters=80)
```
