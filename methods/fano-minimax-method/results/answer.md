# Fano Minimax Lower Bound

Given a model family $\mathcal P$, parameter map $\theta(P)$, semimetric $\rho$, and nondecreasing loss transform $\Phi$, define

$$
M_n(\theta(\mathcal P),\Phi\circ\rho)
=
\inf_{\hat\theta}\sup_{P\in\mathcal P}
\mathbb E_P[\Phi(\rho(\hat\theta(X_1^n),\theta(P)))].
$$

Choose a finite hard set $\theta_1,\ldots,\theta_M$ with pairwise separation

$$
\rho(\theta_i,\theta_j)\ge 2\epsilon \qquad (i\ne j).
$$

Let $V\sim\mathrm{Unif}\{1,\ldots,M\}$ and draw data from $P_{\theta_V}$. Any estimator induces a test

$$
\hat V(X)=\arg\min_j \rho(\hat\theta(X),\theta_j).
$$

If $\rho(\hat\theta,\theta_V)<\epsilon$, nearest-neighbor rounding recovers $V$. Therefore testing failure implies estimation error, and Fano's inequality gives

$$
M_n(\theta(\mathcal P),\Phi\circ\rho)
\ge
\Phi(\epsilon)
\left(
1-\frac{I(V;X_1^n)+\log 2}{\log M}
\right).
$$

The distance-based Duchi-Wainwright form replaces exact index recovery by approximate recovery. For a finite index set $\mathcal V$, define

$$
N_t^{\max}=\max_{v\in\mathcal V}
\mathrm{card}\{w\in\mathcal V:\rho_{\mathcal V}(v,w)\le t\},
$$

and let $\delta(t)$ be the parameter separation guaranteed whenever $\rho_{\mathcal V}(v,w)>t$. Then

$$
M_n(\theta(\mathcal P),\Phi\circ\rho)
\ge
\Phi(\delta(t)/2)
\left(
1-\frac{I(X_1^n;V)+\log 2}{\log(|\mathcal V|/N_t^{\max})}
\right).
$$

Applications finish the proof by bounding the mutual information, commonly via

$$
I(V;X)\le \frac{1}{M}\sum_v D(P_v\|Q)
$$

or

$$
I(V;X)\le \frac{1}{M^2}\sum_{v,w}D(P_v\|P_w).
$$

The method proves that if many parameter values are well separated while their data laws carry too little information to identify the hidden index, then every estimator has risk at the corresponding separation scale.
