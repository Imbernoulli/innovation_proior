I start with a discomfort that is too basic to ignore. Fitting one model is not the problem. If I hand myself a basis, a noise model, and a regularizer, I can find the best weights and put error bars on them. The trouble starts when I have several models. A polynomial of degree ten can imitate a polynomial of degree three and then use the remaining coefficients to chase whatever noise is left. A dense radial-basis expansion can reproduce a sparse one and then wiggle in between the data. If I rank models by the height of the best likelihood, I have built a ratchet: every extra adjustable direction can only help or leave the fit unchanged. Best fit is not a model-comparison principle; it is an overfitting machine.

The usual repair is to bolt on a cost for complexity. Count parameters, add `2k`, add `k log N`, count bits, reserve a test set, or measure a worst-case capacity. These are all attempts to say the same intuitive thing: do not buy flexibility unless the data pay for it. But I would rather the cost come from the inference itself than be attached afterward as a fine because the result looks too flexible. If the model genuinely made probabilistic predictions before seeing the data, then somewhere in those predictions it must already have committed probability to every other data set it was willing to explain. I want to find out whether that commitment, taken seriously, is already a penalty — or whether I will be forced to add one by hand after all.

I go back to Bayes' theorem for parameters inside one model. For model `H` with parameters `w`, I write

`P(w|D,H) = P(D|w,H) P(w|H) / P(D|H)`.

When I am only fitting `w`, the denominator is a nuisance. It is constant in `w`, so it cannot change the location of the maximum or the Hessian around that maximum. I normally throw it away. But now I write Bayes' theorem one level higher:

`P(H|D) proportional to P(D|H) P(H)`.

The thing I discard during fitting is exactly the data-dependent quantity I need for comparing models. This is not a recombination of prior and likelihood at the same level. The numerator `P(D|w,H)P(w|H)` ranks parameter values after `H` has already been assumed. The denominator asks a different question: how much probability did the whole model assign to this data set before the parameter value was selected? And it has only one possible value, because it must normalize the parameter posterior:

`P(D|H) = int P(D|w,H) P(w|H) dw`.

So a candidate model-comparison score would be an average likelihood over the parameter values the model itself considered plausible. Whether this average actually behaves like a razor is not obvious from the formula. Averaging could just smear the best-fit likelihood downward by a roughly constant factor for every model, in which case it would not discriminate at all. I need to compute it on something small enough to see all the pieces.

Take the smallest non-trivial case: one parameter `w`, one observation. Let the likelihood be Gaussian in `w` and let the prior be Gaussian, mean zero. Concretely I set a design coefficient `phi=2`, observation `t=3`, noise precision `beta=4`, prior precision `alpha=0.5` (so prior width `sigma_w = 1/sqrt(alpha) ~= 1.41`). The integral `int P(t|w)P(w)dw` is a Gaussian convolution, and because `t = phi w + noise` with `w ~ N(0,1/alpha)` and noise `~ N(0,1/beta)`, the marginal is `t ~ N(0, phi^2/alpha + 1/beta)`. That variance is `4/0.5 + 0.25 = 8.25`, so `P(t|H) = exp(-t^2/(2*8.25))/sqrt(2*pi*8.25)`, which comes out to `0.08050`. I also did the integral numerically as a sanity check and got `0.08050`, agreeing to seven digits. Good — the closed form is the evidence.

Now I want to see what the evidence is made of. The posterior precision is `A = alpha + beta phi^2 = 0.5 + 16 = 16.5`, so the posterior width is `1/sqrt(16.5) ~= 0.246`. The best-fit weight is `w_MP = beta phi t / A = 24/16.5 ~= 1.455`, and the likelihood there, `like_MP`, is `0.7849`. Dividing, `evidence / like_MP = 0.08050 / 0.7849 = 0.1026`. So the evidence is the best-fit likelihood multiplied by a factor of about `0.10`, strictly less than one. That factor is what I am after. To see it as a width ratio cleanly, redo the same example with a flat prior of width `W` instead of Gaussian: I take `W=8`, integrate, and find `evidence/like_MP = 0.0783`, while `sqrt(2*pi) * sigma_post / W` with `sigma_post = 1/sqrt(beta phi^2) = 0.25` is also `0.0783`. They match exactly. So the multiplier really is the posterior width as a fraction of the prior width — the fraction of the parameter space that survives the data — and it is below one whenever the data localize the parameter. The average did not smear uniformly; it produced a genuine, geometry-dependent discount. So in one dimension,

`P(D|H) ~= P(D|w_MP,H) P(w_MP|H) sigma_w|D`,

and with a prior roughly uniform over width `sigma_w` so that `P(w_MP|H) ~= 1/sigma_w`,

`P(D|H) ~= P(D|w_MP,H) * sigma_w|D / sigma_w`.

The first factor is the best-fit likelihood. The second is the fraction of the prior width that remains plausible after seeing the data — the inverse of the number of distinguishable parameter settings the model offered beforehand, of which only one small neighborhood survives. This is a penalty that was not attached from outside. It is the width of the integral.

I picture the model's predictions as a probability distribution over possible data sets, integrating to one. A simple model puts its mass on a narrow region of data space. A flexible model spreads its mass over many more possible data sets. If the observed data lie in the narrow region predicted by the simple model, the flexible model may still be able to fit them after tuning, but it assigned them less probability beforehand because it also had to reserve probability for all the other data sets it could fit. Predictive probability is conserved, so flexibility cannot be free.

There is a caveat I have to handle before trusting this. Someone can invent the hypothesis "the data are exactly the data that occurred," which assigns huge probability to the observation. But it is one member of an enormous family of equally specific hypotheses, and a sane prior over models assigns each such after-the-fact hypothesis tiny probability. So model priors do not disappear. The point is narrower and stronger: for ordinary candidate models proposed on comparable footing, the data-dependent term alone already contains a razor, as the one-dimensional number above showed. I do not need to dislike complex models in advance to make an over-flexible model lose.

The decomposition also clarifies what "complexity" means here. It is not the number of lines of code needed to evaluate the model, nor only the number of parameters. A model is complex to the extent that it spreads predictive probability over alternatives that the data do not select. A parameter with a wide prior costs more than one with a narrow prior. A parameter that has to be tuned to a tiny posterior width costs more than one that remains coarse. A direction the data do not measure is mostly controlled by the prior and contributes little. The penalty is geometric and data-dependent.

In `k` dimensions, the same Laplace step uses the Hessian. Let

`A = -nabla nabla log P(w|D,H)` at `w_MP`.

The Gaussian volume around the peak is `(2*pi)^(k/2)|A|^(-1/2)`, so

`P(D|H) ~= P(D|w_MP,H) P(w_MP|H) (2*pi)^(k/2)|A|^(-1/2)`.

The matrix I need is not new: it is the same curvature matrix I use for error bars. The determinant is the posterior volume, corrected for correlations among parameters. Best-fit model selection keeps only the height of the peak; the comparison integral keeps height times volume. And in the one-dimensional example everything was Gaussian, so the Laplace formula was not an approximation there — I checked that `like_MP * P(w_MP) * sqrt(2*pi)/sqrt(A)` reproduces the closed-form evidence `0.08050` to seven digits. That one missing volume is what bends the best-fit ratchet back down.

The interpolation problem has a second version of the same issue hidden inside it. Maximum-likelihood interpolation through noisy data is ill-posed, so I introduce a regularizer. With Gaussian noise and a quadratic weight penalty I write

`M(w)=alpha E_w + beta E_n`.

For fixed `alpha` and `beta`, fitting is easy. The posterior over `w` is proportional to `exp(-M(w))`, and its Hessian is `A=alpha C+beta B`. But `alpha` is itself a complexity knob. If `alpha` is too large, the curve is too stiff and underfits. If `alpha` is too small, the weights can roam and the curve chases noise. I want the same comparison principle to set `alpha`, not a separate discrepancy rule.

The normalizers give it. The likelihood has normalizer `Z_n(beta)`, the prior has normalizer `Z_w(alpha)`, and the posterior numerator over weights has normalizer `Z_M(alpha,beta)=int exp(-M(w))dw`. Therefore the probability of the data at fixed `alpha,beta` is

`P(D|alpha,beta,H)=Z_M/(Z_w Z_n)`.

Now the regularizer's trade-off is visible. Small `alpha` makes the prior over weights broad, so `Z_w` grows and divides the score down. Large `alpha` prevents a good fit, so `Z_M` is small. The preferred `alpha` should sit where the data are fit without buying unnecessary parameter volume.

At this point I hit the objection that matters. Am I choosing my prior after seeing the data? It sounds like I define a family of priors indexed by `alpha`, look at the data, choose the best `alpha`, and then call that prior legitimate. That cannot be the fundamental story. The honest Bayesian object is an ensemble of priors before the data arrive, and exact inference should integrate over that ensemble:

`P(w|D,H)=int P(w|D,alpha,beta,H) P(alpha,beta|D,H) d alpha d beta`.

So using the best `alpha,beta` is not the rule; it is an approximation. If the posterior over `alpha,beta` has a sharp peak, the integral is dominated by that peak, just as the parameter integral was dominated by `w_MP`. The apparent post-hoc prior choice is really another Laplace approximation to a larger marginalization.

I want to make sure this distinction — maximize versus marginalize — actually changes an answer, because if it did not I would be free to keep maximizing everything. The cleanest test is the textbook Gaussian: data `x_1..x_N` drawn from `N(mu, v)`, and I want the variance `v`. Joint maximum likelihood over `(mu, v)` gives `v_ML = (1/N) sum (x_i - xbar)^2`. The marginalize-the-mean route is different: with a flat prior on `mu`, integrate `mu` out of the likelihood first, then look at the `v` that maximizes the resulting marginal. I did this on a small simulated sample, `N=6`. With `S = sum (x_i - xbar)^2`, the maximizer of the marginal likelihood came out at `v_hat = 5.7069`, and `S/(N-1) = 5.7069`, while `S/N = 4.7557`. So marginalizing the mean turns the `1/N` into `1/(N-1)` exactly — the fitted mean has absorbed one degree of freedom of noise, and integration, not a moral preference for simplicity, is what books it. Maximizing and marginalizing give genuinely different numbers. I have to marginalize.

Now I derive the cleaner hyperparameter equations. I move to the basis where the quadratic regularizer is whitened, so `E_w=0.5 sum w_a^2` and `A=alpha I + beta B`. The log evidence contains a `-0.5 log det A` term from the posterior volume and a `+(k/2) log alpha` term from the prior normalizer. When I differentiate with respect to `alpha`, the motion of `w_MP` drops out because `w_MP` is already the stationary point of `M`. The only nontrivial derivative is of the log-determinant, and the matrix identity I want is `d log det A / d alpha = Tr(A^-1 dA/dalpha) = Tr(A^-1)`, since `dA/dalpha = I`. I do not want to take that on faith, so I check it numerically on a random `4x4` example: the analytic `Tr(A^-1)` is `0.71119`, and a central finite difference of `log det A` in `alpha` gives `0.71119` as well. Identity confirmed. Setting the `alpha`-derivative of the log evidence to zero then gives

`2 alpha E_w^MP = k - alpha Tr(A^-1)`.

The right side deserves a name. I claim that if the eigenvalues of `beta B` are `lambda_a`, then

`k - alpha Tr(A^-1) = sum_a lambda_a/(lambda_a+alpha)`.

This is the same `4x4` matrix, so I check it: `k - alpha Tr(A^-1) = 3.07546` and `sum lambda_a/(lambda_a+alpha) = 3.07546`. They agree. Each term `lambda_a/(lambda_a+alpha)` lies between zero and one. A direction with `lambda_a >> alpha` is measured by the data and contributes almost one. A direction with `lambda_a << alpha` is held by the prior and contributes almost zero. So the raw number of parameters `k` is the wrong measure. The relevant number is `gamma = sum_a lambda_a/(lambda_a+alpha)`, the effective number of well-measured parameters, and the `alpha` equation reads `2 alpha E_w^MP = gamma`.

Differentiating with respect to `beta` should give a matching noise equation, and the discrepancy-correction structure suggests

`2 beta E_n^MP = N - gamma`.

I would rather see these two equations hold together at an actual evidence maximum than trust each derivative in isolation, because it is easy to drop a term and not notice. So I set up a small regression — `N=7` data points, `k=4` weights, data generated as `Phi w_true + noise` — and numerically maximize the log evidence jointly over `alpha` and `beta`. At the optimum the solver returns `alpha ~= 3.258`, `beta ~= 3.068`. Evaluating the two sides: `2 alpha E_w^MP = 2.8461` and `gamma = 2.8461`; `2 beta E_n^MP = 4.1539` and `N - gamma = 4.1539`. Both stationarity conditions hold simultaneously at the jointly-optimized hyperparameters, to four digits. That is the generalized degrees-of-freedom correction I was after. The discrepancy principle says `N`; ordinary least squares says `N-k`; the regularized Bayesian calculation says `N-gamma`. Well-measured parameters each fit about one unit of noise; poorly measured parameters do not. The same volume calculation that penalizes model flexibility also tells me how much noise the fitted parameters have inevitably absorbed.

Finally I return to whole models. Different bases, regularizers, and noise assumptions are ranked by integrating the `alpha,beta` evidence over their prior scales. When that surface is peaked, I approximate the integral by the peak evidence times its width in log `alpha` and log `beta`. As model complexity increases, the best-fit likelihood alone keeps rising, but the accumulated volume ratios — each one a fraction below one, as the very first one-dimensional number showed — eventually pull the total score down. On a plot, the score should climb steeply while the model is too simple to fit, then descend gently once the extra flexibility is no longer measured by the data. So the method is not "prior plus likelihood" dressed up with a penalty. It is model selection by predictive mass: integrate over the parameter volume, and the complexity penalty appears, computed and confirmed above, from the volume that the data fail to use.
