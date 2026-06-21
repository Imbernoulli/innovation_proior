# Expectation Propagation

Approximate a product-form target, keeping any tractable base term such as a prior exact:

`p(x | D) = (1/Z) p0(x) prod_i f_i(x)`.

Choose a tractable exponential family `Q` and maintain one site approximation `t_i(x)` per non-base exact factor:

`q(x) = (1/Z_q) p0(x) prod_i t_i(x)`.

For site `i`, repeat:

1. Cavity deletion:
   `q_{-i}(x) = (1/Z_{-i}) q(x) / t_i(x)`.
2. Exact local insertion:
   `p_hat_i(x) = (1/Z_i) f_i(x) q_{-i}(x)`,
   with `Z_i = integral f_i(x) q_{-i}(x) dx`.
3. Moment projection:
   `q_i_new = argmin_{q in Q} KL(p_hat_i || q)`.
   For exponential-family `Q`, this means
   `E_{q_i_new}[phi(x)] = E_{p_hat_i}[phi(x)]`.
4. Site replacement:
   `t_i_new(x) proportional to q_i_new(x) / q_{-i}(x)`.
   In natural parameters, `eta_i_new = eta(q_i_new) - eta(q_{-i})`; damping uses `eta_i <- (1-rho) eta_i + rho eta_i_new`.

At convergence, every local tilted distribution agrees with the global approximation on the retained sufficient-statistic moments:

`E_q[phi(x)] = E_{p_hat_i}[phi(x)]` for all `i`.

The approximate evidence is the normalizing constant of `p0(x) prod_i t_i(x)` after including the site log-scales. In Minka's BPM MATLAB code this is why `obj.s` combines `sum(a)`, posterior/prior log determinants, and quadratic natural-parameter terms rather than using tilted normalizers alone.

The distinctive move is replacing one intractable global posterior projection with repeated local repair: delete one old approximate factor, insert the true factor in the current context, compute the exact tilted moments, project them back, and rewrite the site. It extends one-pass assumed-density filtering by revisiting old approximations, and it recovers loopy belief propagation when the approximation is fully disconnected.

EP is useful when each tilted update is much cheaper than full inference, such as a non-Gaussian likelihood times a Gaussian cavity. It is not a guaranteed convex algorithm: fixed points can be multiple, iterations can oscillate or diverge, and poor approximating families can hide multimodality. The canonical BPM implementation reflects this: the first pass is ADF, cavity variances are checked, optional `restrict` skips negative site-variance updates, and convergence is judged by posterior-mean change.
