The problem is to certify a fundamental limit for statistical estimation. For a model family, a target parameter, and a loss, we care about the minimax risk, which takes an infimum over all estimators and a supremum over all parameters. An upper bound only says that one estimator is good enough; it cannot rule out a better estimator that has not yet been imagined. To prove that no procedure can beat a certain error scale, the argument must be independent of any particular algorithm and instead show that the data themselves do not contain enough information to resolve the parameter at the desired resolution.

The central difficulty is that the risk quantifies over every measurable estimator. We cannot attack one rule at a time. We need a single reduction that forces every estimator to fail whenever the data channel is too noisy. The classical tool is the Fano minimax method, which converts an estimation problem into a testing problem on a carefully constructed finite set of alternatives. This method is canonical because it cleanly connects geometric separation in the parameter space with the information that the sample carries about the hidden truth.

The Fano minimax method proceeds in three conceptual steps. First, we choose a finite hard set of parameters that are well separated under the loss metric. Specifically, we pick parameters θ_1, ..., θ_M such that any two distinct members are at least 2ε apart. Second, we draw an index V uniformly from this set and generate the data from the distribution indexed by θ_V. Third, we observe that any estimator θ̂ naturally induces a test for V by rounding θ̂ to the nearest hard-set parameter. If θ̂ lands within ε of the true θ_V, the rounding recovers V exactly, because every other hard-set parameter is at least 2ε away. Therefore, failure to identify V implies estimation error of at least ε.

Once this reduction is in place, Fano's inequality gives a lower bound on the probability that any test fails to recover V. For a uniform index over M alternatives, the probability of error is at least 1 minus the ratio of the mutual information I(V; X) plus a small binary-entropy constant to log M. Combining the reduction and Fano's inequality yields the basic lower bound: the minimax risk is at least Φ(ε) times the Fano error probability, where Φ is a nondecreasing loss transform. The remaining analytical work in any application is to construct a large packing with small mutual information. A common way to bound the mutual information is by averaging KL divergences, either against a reference distribution or pairwise over the hard set.

A refined form, sometimes called the Duchi-Wainwright corollary, avoids exact index recovery. Instead of asking whether the estimator identifies the exact index, it asks whether the estimator lands inside a neighborhood of radius t in an index metric. The effective number of distinguishable alternatives then becomes the total number of alternatives divided by the largest neighborhood size, and the separation scale is the parameter distance guaranteed for indices farther than t apart. This generalization recovers the classical packing proof as the special case t = 0 and can simplify proofs where the parameter space has nontrivial local geometry.

The method has limitations. It often gives constant-probability or weak-converse lower bounds, and the quality of the bound depends on a careful choice of packing and on a sharp information bound. In adaptive or sequential settings, controlling the mutual information can become more involved. Nevertheless, when the parameter space contains many separated alternatives whose induced data laws remain statistically close, the Fano minimax method is the standard way to prove that the obstacle is not a missing algorithm but the information content of the sample.

Collecting the packing construction, the rounding reduction, and Fano's inequality into one statement gives the object I actually certify. For a model family $\mathcal P$, parameter map $\theta(P)$, semimetric $\rho$, and nondecreasing loss transform $\Phi$, the minimax risk

$$
M_n(\theta(\mathcal P),\Phi\circ\rho)
=
\inf_{\hat\theta}\sup_{P\in\mathcal P}
\mathbb E_P[\Phi(\rho(\hat\theta(X_1^n),\theta(P)))]
$$

satisfies, for any hard set $\theta_1,\ldots,\theta_M$ that is a $2\epsilon$-packing of the parameter space under $\rho$, with $V$ uniform on $\{1,\ldots,M\}$ and $X_1^n\sim P_{\theta_V}$,

$$
M_n(\theta(\mathcal P),\Phi\circ\rho)
\ge
\Phi(\epsilon)
\left(
1-\frac{I(V;X_1^n)+\log 2}{\log M}
\right).
$$

The distance-based, Duchi-Wainwright form of the same certificate drops exact index recovery in favor of recovery to within radius $t$ under an index metric $\rho_{\mathcal V}$ on a finite index set $\mathcal V$. Writing $N_t^{\max}=\max_{v\in\mathcal V}\mathrm{card}\{w\in\mathcal V:\rho_{\mathcal V}(v,w)\le t\}$ for the largest such neighborhood and $\delta(t)$ for the parameter separation guaranteed whenever $\rho_{\mathcal V}(v,w)>t$,

$$
M_n(\theta(\mathcal P),\Phi\circ\rho)
\ge
\Phi(\delta(t)/2)
\left(
1-\frac{I(X_1^n;V)+\log 2}{\log(|\mathcal V|/N_t^{\max})}
\right),
$$

which reduces to the classical packing bound at $t=0$. Either form is closed by one further step: bounding the mutual information itself, most commonly through the average KL divergence to a reference law $Q$,

$$
I(V;X)\le \frac{1}{M}\sum_{v=1}^M D(P_v\|Q),
$$

or through the pairwise average

$$
I(V;X)\le \frac{1}{M^2}\sum_{v,w}D(P_v\|P_w).
$$

This is the complete certificate: a packing, a randomized index, a rounding test, Fano's inequality, and an information bound, combined into a single inequality that no estimator — however it is built — can beat.
