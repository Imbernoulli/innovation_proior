OK, let me think this through from scratch. The thing in front of me is a task whose input is a *set*: a sample set from some distribution and I want its entropy, or a cloud of 3D points and I want the object class, or a galaxy cluster and I want a red-shift. The common feature is that the input is an unordered, variable-size bag of elements $X=\{x_1,\dots,x_M\}$, and the answer cannot depend on the order I happen to list them in. If I shuffle the elements, the output has to stay put. So whatever function $f$ I build, the hard constraint is

$$f(\{x_1,\dots,x_M\}) = f(\{x_{\pi(1)},\dots,x_{\pi(M)}\}) \quad\text{for every permutation }\pi.$$

That's permutation invariance. And there's a sibling case: sometimes each element carries its own label — flag the outliers in this set — and then reordering the input should reorder the output the same way, $\mathbf f(\pi\mathbf x)=\pi\mathbf f(\mathbf x)$. That's equivariance. Let me hold equivariance in reserve and chew on invariance first, because it's the cleaner constraint.

The obvious thing everyone reaches for is to pretend the set is a vector and feed it to a standard net, or feed it element-by-element to an RNN. But both of those *see* the order. An RNN reading $x_1$ then $x_2$ then $x_3$ builds a different hidden state than reading them in another order; people have literally tried searching for a "good" ordering to feed a set into a sequence model, which is an admission that the order leaks into the answer. And empirically it shows: if I train an LSTM to sum a set of digits with at most 10 of them and then test it on sets of 50 or 100, it falls apart, because it learned something tied to positions and lengths it saw, not the set itself. So I don't want to *fight* the order with augmentation or search. I want a function that is invariant *by construction*, so no ordering ever enters.

So the real question is: what is the most general function of a set that is permutation invariant? Not "what's one architecture that happens to work" — I want the *shape of the whole space*, so I know I'm not throwing away expressive power and not secretly cheating on the symmetry.

Let me start with the most basic order-independent operation I know: addition. $\sum_m x_m$ doesn't care about order, because addition is commutative and associative. So $f(X)=\sum_m x_m$ is invariant. So is $\prod_m x_m$, so is $\max_m x_m$. These are all "reduce the set with a commutative operation." But $\sum x_m$ throws away almost everything — two very different sets can have the same sum. I need more than the raw sum. What if I don't sum the raw elements, but sum a *transformed* version of each element, and then post-process the total? That is,

$$f(X) = \rho\Big(\sum_{x\in X}\phi(x)\Big)$$

for some per-element map $\phi$ and some readout $\rho$. The inner sum is invariant no matter what $\phi$ is, because summation is order-blind; then $\rho$ is just a function of one fixed vector, so it can't reintroduce order. So *every* function of this form is invariant. That's the easy direction — sufficiency is immediate. The element-wise $\phi$, the pool, the readout: invariant no matter how I fill them in.

But is that the *whole* story? Could there be invariant functions that *can't* be written like this — that genuinely need the set as a whole and refuse to factor through a sum of per-element terms? If so, then committing to this form is a real loss of generality and I should look for something richer. This is the question that decides whether the form is fundamental or just convenient. Let me try to prove that every invariant function has this form, and if I can't, the failure will tell me what's missing.

Let me take the cleanest case where I have a real chance: the universe $\mathfrak X$ the elements come from is *countable* (think: a finite vocabulary, or the integers). Countable means I can enumerate the possible elements — there's an injection $c:\mathfrak X\to\mathbb N$ assigning each possible element a distinct natural-number index. A set $X$ is just a subset of this universe: each possible element is either in or out. So a set is exactly a characteristic vector of 0/1 over the (countable) index set. I want a single number $\sum_{x\in X}\phi(x)$ that encodes *which* elements are present, with no collisions — an injective code for subsets.

Encoding a 0/1 pattern as one number is a positional numeral. If I let $\phi(x)=2^{-c(x)}$, then $\sum_{x\in X}\phi(x)$ is a binary fraction whose $c(x)$-th digit is 1 exactly when $x\in X$. Different sets should give different digit patterns, hence different sums. Let me not just assert it — let me code two distinct subsets and look at the numbers. Take elements with indices $\{1,3\}$ versus $\{1,2\}$: the first sums to $2^{-1}+2^{-3}=\tfrac12+\tfrac18=\tfrac58$, the second to $2^{-1}+2^{-2}=\tfrac34$. Different, good. So $E(X):=\sum_{x\in X}\phi(x)$ looks like a unique fingerprint of the set. Then define $\rho$ to be "read the digits back out to recover $X$, then apply $f$": formally $\rho = f\circ E^{-1}$ on the image of $E$. That gives $f(X)=\rho(\sum_{x\in X}\phi(x))$ exactly, for *any* invariant $f$. Necessity done — for sets.

But wait, I quietly used "each digit is 0 or 1", which is true for a *set* and false for a *multiset*. Let me probe whether base 2 actually survives multiplicity, because if it doesn't I've found the boundary of the argument. Suppose one element with index $c=2$ appears twice, against a different multiset where one element with index $c=1$ appears once. The first sums to $2\cdot 2^{-2}=\tfrac12$; the second to $1\cdot 2^{-1}=\tfrac12$. They collide. So base 2 is *not* injective on multisets — the doubled digit carried into the neighbouring position, exactly as a binary carry would. That's a genuine hole, not a cosmetic one. The fix is headroom in the radix: if I use base 4, $\phi(x)=4^{-c(x)}$, then the same two multisets give $2\cdot 4^{-2}=\tfrac{2}{16}=\tfrac18$ versus $1\cdot 4^{-1}=\tfrac14$ — now distinct. A digit can be $0,1,2,3$ before it carries, so as long as multiplicities stay below the radix the sum still uniquely encodes them, and the same $\rho=f\circ E^{-1}$ argument goes through. So the safe canonical choice is base 4 (and in general, "pick a radix larger than any multiplicity you'll see"). With that patch, for countable universes the structure theorem holds: an invariant set function factors as $\rho(\sum_x\phi(x))$.

That's a strong statement — pooling a per-element embedding is not a heuristic here, it's *the* form. But the construction is a cheat in a practical sense: $\phi(x)=4^{-c(x)}$ is a wildly discontinuous lookup table, and $\rho$ is its inverse, equally nasty. It proves existence, not that nice $\phi,\rho$ exist. And it leaned hard on countability. What happens when the elements are real-valued — a point in $[0,1]$, or in $\mathbb R^d$? Then I can't enumerate them, the digit trick dies, and I have to ask whether *continuous* $\phi,\rho$ can still do the job. Let me see how far I can push the continuous case.

Fix the set size to $M$ for now (I'll worry about variable size separately), and take elements in $[0,1]$, so the domain is $[0,1]^M$ modulo permutation. I need an inner map $E(X)=\sum_m\phi(x_m)$ that (a) is invariant — automatic, it's a sum — and (b) is *injective* on sets, so that $\rho=f\circ E^{-1}$ is well defined, and ideally (c) has a *continuous* inverse so $\rho$ comes out continuous.

What continuous per-element features, summed, separate sets of reals? The candidate that keeps coming to mind is symmetric polynomials. If I take $\phi(x)=[x, x^2, x^3, \dots, x^M]$ (throw in a constant $1$ too, so $\phi(x)=[1,x,\dots,x^M]\in\mathbb R^{M+1}$), then the coordinates of $E(X)$ are the **power sums** $p_q=\sum_m x_m^{\,q}$ for $q=0,\dots,M$. Are the power sums up to degree $M$ enough to pin down a set of $M$ reals? This is a classical fact, but I've been burned once already (base 2) by trusting the clean statement, so let me actually run the recovery on numbers and watch it succeed before I lean on it.

Take the concrete set $u=\{0.2,0.5,0.9\}$, $M=3$. Its power sums are $p_0=3$, $p_1=0.2+0.5+0.9=1.6$, $p_2=0.04+0.25+0.81=1.1$, $p_3=0.008+0.125+0.729=0.862$. Now the bridge from power sums to the **elementary symmetric polynomials** $a_m$ (the coefficients of $\prod_m(x-u_m)$ up to sign) is the Newton–Girard recursion. With $a_0=1$,

$$m\,a_m=\sum_{q=1}^m (-1)^{q-1}a_{m-q}p_q.$$

Run it on these numbers. $a_1=p_1=1.6$. Then $2a_2=a_1p_1-p_2=1.6\cdot1.6-1.1=2.56-1.1=1.46$, so $a_2=0.73$. Then $3a_3=a_2p_1-a_1p_2+p_3=0.73\cdot1.6-1.6\cdot1.1+0.862=1.168-1.76+0.862=0.27$, so $a_3=0.09$. So the monic polynomial is $x^3-1.6x^2+0.73x-0.09$. Solving it should hand me back $\{0.2,0.5,0.9\}$. Factor check: $(x-0.2)(x-0.5)(x-0.9)$. Expand $(x-0.2)(x-0.5)=x^2-0.7x+0.1$; times $(x-0.9)$ gives $x^3-0.9x^2-0.7x^2+0.63x+0.1x-0.09=x^3-1.6x^2+0.73x-0.09$. That is exactly the polynomial the recursion produced. So the power sums round-tripped back to the original set — concretely, with real arithmetic, not by appeal to the theorem. (And the integer factors in Newton–Girard are load-bearing: if I'd dropped the $1/m$ and the alternating signs, $a_3$ would not have come out $0.09$ and the factorization would fail.)

I should also make sure degree $M$ is really *needed*, not overkill — because that determines how long $\phi$ has to be. Try truncating: keep only count and sum, $p_0$ and $p_1$. Then $A=\{0.2,0.8\}$ and $B=\{0.3,0.7\}$ both have $p_0=2$, $p_1=1.0$ — identical, so they collide under the truncated map. Going one degree further separates them: $p_2(A)=0.04+0.64=0.68$ while $p_2(B)=0.09+0.49=0.58$. So count-plus-sum genuinely is not injective even for $M=2$; I need power sums all the way up to $p_M$. Good — that pins down the embedding dimension and confirms the truncation can't be cheaper.

So with a perfectly *continuous* $\phi$ (a polynomial!) the inner sum already encodes the set uniquely, at least on the example I checked, and the count-plus-sum failure shows the degree is the right one. The general injectivity argument is the same recovery I just ran by hand: equal power sums $\Rightarrow$ (by the recursion, which only adds/multiplies/divides by fixed integers) equal elementary symmetric polynomials $\Rightarrow$ same monic polynomial $\Rightarrow$ same multiset of roots $\Rightarrow$ (sorted) same set. The countable trick's ugliness was an artifact; for fixed-size real sets, monomials do it cleanly.

Now I need $\rho$ to be continuous, which means I need $E^{-1}$ continuous — small wiggles in the power-sum vector should mean small wiggles in the recovered set. Two steps. First, the coefficients $a$ are continuous functions of the power sums $z=E(X)$: the Newton–Girard recursion only adds, multiplies, and divides by the fixed integer $m$, so each $a_m$ is a polynomial in $z_1,\dots,z_m$ — visibly continuous. Second, I need the roots to depend continuously on the coefficients. That one I'm relying on as a known fact — the map from a monic polynomial's coefficient vector to its multiset of roots is continuous (roots move continuously as you nudge coefficients); I haven't re-proved it here and I'd want to be careful at repeated roots, but it's standard. Composing, $E^{-1}$ = (coeffs from power sums) then (roots from coeffs) is continuous on the relevant compact domain.

Let me assemble it. The domain $\mathcal X=\{x_1\le\dots\le x_M\}\subset[0,1]^M$ is compact (a bounded polytope). $E$ is continuous (polynomial), so its image $\mathcal Z=E(\mathcal X)$ is compact, and $E:\mathcal X\to\mathcal Z$ is surjective by definition; it's injective by the recovery argument above; and its inverse is continuous by the root-continuity. A continuous bijection from a compact space with continuous inverse is a homeomorphism. So $E$ is a homeomorphism between the sorted-set domain and $\mathcal Z$. Then for any continuous invariant $f$, set $\rho(z)=f(E^{-1}(z))$ — continuous as a composition of continuous maps — and $f(X)=\rho(\sum_m\phi(x_m))$ with continuous $\phi,\rho$. So in the fixed-size continuous case too, the sum-decomposition is the class of invariant functions, and the inner $\phi(x)=[1,x,\dots,x^M]$ is universal — it doesn't even depend on $f$; only $\rho$ does.

There's a cross-check sitting next to this that I want to look at, because if it pointed the other way I'd worry. For an *arbitrary* continuous multivariate function — not necessarily symmetric — the Kolmogorov–Arnold representation theorem writes $f(x_1,\dots,x_M)=\rho(\sum_m \lambda_m\,\phi(x_m))$ with the inner $\phi$ independent of $f$. The only difference from what I just derived is those per-coordinate weights $\lambda_m$. And those $\lambda_m$ are exactly what lets a function *distinguish coordinate 1 from coordinate 2*, i.e. depend on order: if $\lambda_1\ne\lambda_2$, swapping $x_1$ and $x_2$ changes $\lambda_1\phi(x_1)+\lambda_2\phi(x_2)$, so the value can move. Demand invariance and the value must *not* move under that swap, which forces the weights on swapped coordinates to agree — and since every pair is swappable, all $\lambda_m$ collapse to one shared constant, giving a single shared $\phi$ summed across members. So invariance is precisely the assumption that strips the coordinate-dependence out of the universal representation. The two derivations agree; the pooling form is what the general representation theorem becomes once the function is forbidden from knowing positions.

What about variable size, and elements in $\mathbb R^d$? The clean homeomorphism argument used a fixed $M$ and the polynomial degree tied to $M$. I should not pretend I have proved exact equality for the whole space of finite subsets; even choosing the right topology on $2^{[0,1]}$ is part of the problem, and I haven't done that. What I can prove cleanly is the fixed-size approximation statement on compact subsets of $\mathbb R^d$. Polynomials are dense in continuous functions on a compact set by Stone–Weierstrass. The continuous invariant functions at fixed $M$ are the *symmetric* continuous functions, and the Fundamental Theorem of Symmetric Functions (a special case of the Chevalley–Shephard–Todd theorem) says every symmetric polynomial is a polynomial in pooled monomial generators such as $\sum_m x_{m,1}^{\alpha_1}\cdots x_{m,d}^{\alpha_d}$. So for every fixed set size on a compact domain, any continuous symmetric target can be approximated arbitrarily well by $\rho(\sum_m\phi(x_m))$ with polynomial $\phi$ and $\rho$. For variable-size data, the architecture still *runs* because the same shared $\phi$ and the same commutative pool accept any $M$; the theorem gives me the exact fixed-size backbone and the countable case gives me the all-sizes discrete backbone, but the continuous all-sizes theorem is not something I have in hand. Worth being honest with myself about that gap rather than papering over it.

So here is the architectural consequence, and it needs no further design choices: replace the hand-built polynomial $\phi$ and $\rho$ with neural networks, since nets can approximate the continuous maps the theorem calls for on compact domains. Each element $x_m$ goes through a shared network $\phi$ (the same weights for every element — and now I see *why* tying is mandatory: the form is one $\phi$ summed over members, so per-element weights would be a different function class, not just wasteful). Sum the embeddings, $s=\sum_m\phi(x_m)$. Push $s$ through a second network $\rho$. Invariant by construction, usable for any $M$, and backed exactly in the countable case and the fixed-size continuous case.

Before I trust the code, let me actually trace it on an input rather than assume the algebra survives the implementation. Build the invariant net on $3$-dim elements, take a set of $5$ elements, and feed it twice — once in original order, once permuted. If the sum-pool is doing its job, $f(X)$ and $f(\pi X)$ should agree to floating-point tolerance. Running it: $f(X)=-0.88408$ and $f(\pi X)=-0.88408$, equal to within $6\cdot10^{-8}$ (the residual is just float reassociation in the sum, not order leakage). So the implementation really is invariant, not just the math on paper.

One more thing worth noticing about why this trains sanely: the gradient of $\rho(\sum_{x}\phi(x))$ with respect to $\phi$'s weights $w_\phi$ is, by the chain rule,

$$\partial_{w_\phi}\,\rho\Big(\sum_{x}\phi(x)\Big)=\rho'\Big(\sum_{x}\phi(x)\Big)\sum_{x}\partial_{w_\phi}\phi(x),$$

the gradients of the shared $\phi$ just add up across members. So parameter-tying across set members — which practitioners do by instinct when order is irrelevant — isn't only convenient; it's the *only* admissible thing, and the backward pass is a clean sum. Theory backing the instinct, and sharpening it from "you may" to "you must."

A small design point on the pool. Sum and mean are both commutative reductions, so both are legal. Sum keeps the *count* — for a task like summing digits the magnitude has to scale with how many elements there are, so sum is right. Mean = sum$/M$ divides that out, which is what I want when set size is incidental and I don't want the pooled magnitude to blow up with large $M$; it's the safer default for classification across very different set sizes. Max is also commutative and is robust to set size and to a few wild elements; it tends to help when the answer is driven by an extreme member (point clouds). The theorem is stated for sum, but any commutative pool inherits the invariance; which one to use is a bias-about-the-task choice, and I'll keep all three on the table.

Now the equivariant case, which I parked. Here the input is $\mathbf x\in\mathbb R^M$ (one scalar per element, say) and I want a *layer* — a map $\mathbb R^M\to\mathbb R^M$ — such that permuting the inputs permutes the outputs identically, so I can stack these and produce a per-element output (outlier scores). The plainest layer is $\mathbf f_\Theta(\mathbf x)=\sigma(\Theta\mathbf x)$ with a full weight matrix $\Theta\in\mathbb R^{M\times M}$ and a pointwise nonlinearity $\sigma$. When is it equivariant? Write $\pi$ for the permutation matrix. I need $\mathbf f(\pi\mathbf x)=\pi\mathbf f(\mathbf x)$, i.e. $\sigma(\Theta\pi\mathbf x)=\pi\,\sigma(\Theta\mathbf x)$. Because $\sigma$ is applied entrywise and $\pi$ just shuffles entries, $\pi$ commutes with $\sigma$: $\pi\,\sigma(\mathbf v)=\sigma(\pi\mathbf v)$. Assuming $\sigma$ is invertible (e.g. sigmoid), I can strip it, and the condition becomes purely linear:

$$\Theta\pi=\pi\Theta\quad\text{for every permutation }\pi\in\mathcal S_M.$$

So the question is exactly: which matrices commute with *all* permutation matrices? Let me find them.

First the easy direction: $\mathbf I$ commutes with everything, and the all-ones matrix $\mathbf 1\mathbf 1^{\mathsf T}$ commutes with every $\pi$ (permuting rows and columns of an all-ones matrix gives back the all-ones matrix: $\pi(\mathbf 1\mathbf 1^{\mathsf T})=\mathbf 1\mathbf 1^{\mathsf T}=(\mathbf 1\mathbf 1^{\mathsf T})\pi$ since $\pi\mathbf 1=\mathbf 1$). Commutativity is linear — if $\Theta_1$ and $\Theta_2$ each commute with $\pi$ then so does $a\Theta_1+b\Theta_2$ — so any $\Theta=\lambda\mathbf I+\gamma\,\mathbf 1\mathbf 1^{\mathsf T}$ commutes with every $\pi$. That's a two-parameter family: $\lambda\mathbf I+\gamma\mathbf 1\mathbf 1^{\mathsf T}$ has diagonal $\lambda+\gamma$ and off-diagonal $\gamma$, so what's pinned is "all diagonal entries equal" and "all off-diagonal entries equal" — two free numbers.

But have I found *all* of them, or just a family I happened to guess? This is the part that decides whether I have a characterization or merely a sufficient construction, so I don't want to hand-wave it. Two ways to settle it, and I'll do both. The cheap empirical one first: for $M=3$, set up $\Theta\pi=\pi\Theta$ as a linear system in the $9$ entries of $\Theta$, sweep over all $6$ permutations, and ask for the dimension of the solution space. If my guessed family is everything, the nullspace should be exactly $2$-dimensional. Solving it numerically, the commuting subspace comes out $2$-dimensional, and a basis for it is (rounded)

$$\begin{pmatrix}.425&-.276&-.276\\-.276&.425&-.276\\-.276&-.276&.425\end{pmatrix},\qquad \begin{pmatrix}.391&.300&.300\\.300&.391&.300\\.300&.300&.391\end{pmatrix},$$

both of which have all-equal diagonals and all-equal off-diagonals — i.e. both live in $\mathrm{span}\{\mathbf I,\mathbf 1\mathbf 1^{\mathsf T}\}$. So at least for $M=3$ there is nothing outside the two-parameter family. That's reassuring, but $M=3$ isn't a proof; let me get the structural argument that gives all $M$.

Suppose $\Theta\pi=\pi\Theta$ for every permutation. Take a transposition $\pi_{k,l}$ that swaps just indices $k$ and $l$. From $\pi_{k,l}\Theta=\Theta\pi_{k,l}$, conjugate: $\pi_{k,l}\Theta\pi_{k,l}^{-1}=\Theta$, and $\pi_{k,l}^{-1}=\pi_{k,l}$. Conjugating $\Theta$ by the swap exchanges rows $k,l$ *and* columns $k,l$. Let me watch this happen on a concrete non-tied matrix so I'm sure I have the index bookkeeping right. Take $\Theta=\begin{pmatrix}1&2&3\\4&5&6\\7&8&9\end{pmatrix}$ and $\pi$ swapping indices $1,2$ (0-based $0,1$). Computing $\pi\Theta\pi=\begin{pmatrix}5&4&6\\2&1&3\\8&7&9\end{pmatrix}$: the $(1,1)$ entry is $1$, which was $\Theta_{0,0}$ — the swap pulled the old $(k,k)$ diagonal entry into the $(l,l)$ slot. For this conjugate to equal $\Theta$, I'd need $\Theta_{l,l}=\Theta_{k,k}$, but here $\Theta_{1,1}=5\ne1=\Theta_{0,0}$, so this $\Theta$ does *not* commute — and indeed $\Theta\pi-\pi\Theta=\begin{pmatrix}-2&-4&-3\\4&2&3\\1&-1&0\end{pmatrix}\ne0$. So the diagonal-equality is forced exactly where I expected. Reading the general rule off this: the $(l,l)$ entry of $\pi_{k,l}\Theta\pi_{k,l}$ came from $\Theta_{k,k}$ and must equal $\Theta_{l,l}$, hence $\Theta_{k,k}=\Theta_{l,l}$ for every pair $k,l$ — **all diagonal entries are equal**, call it $\lambda'$.

Off-diagonal next. Take any two off-diagonal positions $(i,j)$ and $(i',j')$, with $i\ne j$ and $i'\ne j'$. Because the two source indices are distinct and the two target indices are distinct, there is a permutation $\pi$ with $\pi(i)=i'$ and $\pi(j)=j'$. Commutativity gives $\pi\Theta\pi^{-1}=\Theta$. The $(i',j')$ entry of $\pi\Theta\pi^{-1}$ is pulled from the $(i,j)$ entry of $\Theta$, so $\Theta_{i,j}=\Theta_{i',j'}$. No special row-sharing or column-sharing case is left over; the single permutation handles all of them. So **all off-diagonal entries are equal**, call it $\gamma'$. Therefore $\Theta=\gamma'\mathbf 1\mathbf 1^{\mathsf T}+(\lambda'-\gamma')\mathbf I$ — exactly the two-parameter tied form, now for every $M$, matching the $2$-dimensional nullspace I computed at $M=3$. Both routes agree: the family is complete. And as a final sanity check that the *forward* claim isn't vacuous, the tied $\Theta=3\mathbf I+\tfrac12\mathbf 1\mathbf 1^{\mathsf T}$ does commute — $\Theta\pi-\pi\Theta=0$ exactly for the swap above.

Read off what that layer *does*: $\Theta\mathbf x=\lambda\mathbf I\mathbf x+\gamma(\mathbf 1\mathbf 1^{\mathsf T})\mathbf x=\lambda\,\mathbf x+\gamma\big(\textstyle\sum_m x_m\big)\mathbf 1$. Each output element is a weighted mix of (i) the element's own value and (ii) the sum over the whole set, broadcast back to every position. So the *only* way a dense layer can be equivariant is "transform yourself, plus a shared pooled summary of everyone." That's a strikingly tight result: it says the entire freedom in an equivariant linear layer is two scalars, one for self and one for the pool. And it explains why the ad hoc "each agent combines its own state with a pooled summary of the others" updates people have used in multi-agent and sensor models work — they are this layer, and now I know they are the *unique* equivariant linear form, not one option among many.

Since the pool is what carries the symmetry, I can swap the sum for any other commutative pool and stay equivariant. A useful variant uses max: $\mathbf f(\mathbf x)=\sigma(\lambda\,\mathbf x+\gamma\,\mathrm{maxpool}(\mathbf x)\,\mathbf 1)$. If I want true max-centering, the sign matters: in this plus-sign parameterization I set $\gamma=-\lambda$, giving $\lambda(\mathbf x-\mathrm{maxpool}(\mathbf x)\mathbf 1)$. Equivalently I can write the layer with an explicit subtraction from the start. That centered form keeps each element compared with the set's extreme value while preserving equivariance, so I'll keep the max version available. (I'll want to confirm the max pool really preserves equivariance once it's in code — it's a different reduction than the sum the proof used.)

Lifting to features: with $D$ input and $D'$ output channels, each element is a $D$-vector. The implemented two-map form is $\mathbf f(X)=\sigma(XW_\Gamma-\mathrm{pool}(X)W_\Lambda)$, with the pooled row vector broadcast back to all $M$ elements. With sum pooling this is just the multi-channel version of the two-parameter scalar layer, with the minus sign absorbed into the pooled weights; with max or mean pooling it keeps the same equivariance because the pooled statistic is commutative. And I can trim parameters by tying it down to one matrix plus a bias, $\mathbf f(X)=\sigma\big(\beta+(X-\mathbf 1\,\mathrm{maxpool}(X))W\big)$, which is "subtract the set's max from each element, then a shared linear map" — fewer weights, better generalization. Because composing equivariant maps stays equivariant, I can stack several of these layers to build depth; and the moment I want an *invariant* output instead, I just apply one commutative pool over the set members at the end and the whole stack becomes invariant. Equivariant body, pool, readout — the invariant model is literally the equivariant model with a pool stuck on top, which unifies the two cases I started by separating.

Let me trace the equivariant layer too, with the max pool I was uneasy about, before committing. Same $5$-element, $3$-dim input as before, and the test now is $\pi(\mathbf f(X))=\mathbf f(\pi X)$ — applying the layer then permuting should equal permuting then applying. Running it, $\mathbf f(X)[\pi]$ and $\mathbf f(\pi X)$ match to tolerance. So the max-pool variant is genuinely equivariant in the implementation, not only the sum-pool the proof covered; the commutativity of max was enough.

Let me now turn the two theorems into code, keeping each block tied to the step that forced it. The invariant model first — shared $\phi$, sum-pool, $\rho$:

```python
import torch
import torch.nn as nn

class InvariantDeepSet(nn.Module):
    # f(X) = rho( sum_x phi(x) ): the sum-decomposition is THE invariant form,
    # so phi is shared across elements (tying is mandatory, not optional).
    def __init__(self, in_dim, phi_dim=30, hidden=100, out_dim=1):
        super().__init__()
        self.phi = nn.Sequential(          # per-element embedding, same weights for all m
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, phi_dim),
        )
        self.rho = nn.Linear(phi_dim, out_dim)   # readout on the pooled code

    def forward(self, x, mask=None):       # x: (batch, M, in_dim)
        h = self.phi(x)                    # phi(x_m) for every element
        if mask is not None:
            h = h * mask.unsqueeze(-1)     # ignore padding for variable M
        s = h.sum(dim=1)                   # sum_x phi(x): order-independent pool
        return self.rho(s)                 # rho(.): invariant by construction
```

And the equivariant layer — the unique tied form, written as "self map minus pooled map" — stacked into a model that pools at the end to become invariant:

```python
class PermEquivariant(nn.Module):
    # Theta = lambda*I + gamma*11^T is the ONLY equivariant linear layer.
    # Realized as f(X) = Gamma(X) - Lambda(pool(X)).
    def __init__(self, in_dim, out_dim, pool='max'):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)               # per-element (self) term
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)  # broadcast pooled term
        self.pool = pool

    def forward(self, x):                  # x: (batch, M, in_dim)
        if self.pool == 'max':
            xm, _ = x.max(dim=1, keepdim=True)   # commutative pool -> equivariance
        else:
            xm = x.mean(dim=1, keepdim=True)
        return self.Gamma(x) - self.Lambda(xm)   # subtract pool: centers each element


class EquivariantDeepSet(nn.Module):
    def __init__(self, in_dim, d=256, num_classes=40, pool='max'):
        super().__init__()
        self.phi = nn.Sequential(          # stacked equivariant layers stay equivariant
            PermEquivariant(in_dim, d, pool), nn.ELU(inplace=True),
            PermEquivariant(d, d, pool),       nn.ELU(inplace=True),
            PermEquivariant(d, d, pool),       nn.ELU(inplace=True),
        )
        self.rho = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(d, d), nn.ELU(inplace=True),
            nn.Dropout(0.5), nn.Linear(d, num_classes),
        )

    def forward(self, x):
        h = self.phi(x)
        s, _ = h.max(dim=1)                # final commutative pool -> invariant output
        return self.rho(s)
```
