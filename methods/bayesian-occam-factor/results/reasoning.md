I start with a discomfort that is too basic to ignore. Fitting one model is not the problem. If I hand myself a basis, a noise model, and a regularizer, I can find the best weights and put error bars on them. The trouble starts when I have several models. A polynomial of degree ten can imitate a polynomial of degree three and then use the remaining coefficients to chase whatever noise is left. A dense radial-basis expansion can reproduce a sparse one and then wiggle in between the data. If I rank models by the height of the best likelihood, I have built a ratchet: every extra adjustable direction can only help or leave the fit unchanged. Best fit is not a model-comparison principle; it is an overfitting machine.

The usual repair is to bolt on a cost for complexity. Count parameters, add `2k`, add `k log N`, count bits, reserve a test set, or measure a worst-case capacity. These are all attempts to say the same intuitive thing: do not buy flexibility unless the data pay for it. But I want the cost to come from the inference itself. I do not want to first maximize the likelihood and then attach a separate fine because the result looks too flexible. If the model genuinely made probabilistic predictions before seeing the data, then somewhere in those predictions it must already have paid for all the other data sets it was willing to explain.

I go back to Bayes' theorem for parameters inside one model. For model `H` with parameters `w`, I write

`P(w|D,H) = P(D|w,H) P(w|H) / P(D|H)`.

When I am only fitting `w`, the denominator is a nuisance. It is constant in `w`, so it cannot change the location of the maximum or the Hessian around that maximum. I normally throw it away. But now I write Bayes' theorem one level higher:

`P(H|D) proportional to P(D|H) P(H)`.

The thing I discard during fitting is exactly the data-dependent quantity I need for comparing models. This is not a recombination of prior and likelihood at the same level. The numerator `P(D|w,H)P(w|H)` ranks parameter values after `H` has already been assumed. The denominator asks a different question: how much probability did the whole model assign to this data set before the parameter value was selected?

That denominator has only one possible value, because it must normalize the parameter posterior:

`P(D|H) = int P(D|w,H) P(w|H) dw`.

So the model-comparison score is an average likelihood over the parameter values the model itself considered plausible. This changes the game. A model is no longer rewarded for having one excellent parameter setting if that setting was only one tiny target inside a huge prior volume. The model is rewarded for predicting the data as a model, not for becoming clairvoyant after the data arrive.

I picture the model's predictions as a probability distribution over possible data sets. That distribution integrates to one. A simple model puts its mass on a narrow region of data space. A flexible model spreads its mass over many more possible data sets. If the observed data lie in the narrow region predicted by the simple model, the flexible model may still be able to fit them after tuning, but it assigned them less probability beforehand because it also had to reserve probability for all the other data sets it could fit. This is the geometric form of the insight: flexibility is not free because predictive probability is conserved.

There is a caveat I have to handle. Someone can invent the hypothesis "the data are exactly the data that occurred." That hypothesis assigns huge probability to the observation. But it is one member of an enormous family of equally specific hypotheses, and a sane prior over models assigns each such after-the-fact hypothesis tiny probability. So I should not pretend model priors disappear. The point is narrower and stronger: for ordinary candidate models proposed on comparable footing, the data-dependent term alone already contains a razor. I do not need to dislike complex models in advance to make an over-flexible model lose.

Now I want the penalty to appear in a formula. The integral `int P(D|w,H)P(w|H)dw` is usually dominated by a peak near the fitted parameter vector `w_MP`. A peaked integral invites the same approximation I already use for error bars: expand the log integrand to second order. In one dimension, the integral is peak height times accessible width:

`P(D|H) ~= P(D|w_MP,H) P(w_MP|H) sigma_w|D`.

If the prior is roughly uniform over a width `sigma_w`, then `P(w_MP|H) ~= 1/sigma_w`, so

`P(D|H) ~= P(D|w_MP,H) * sigma_w|D / sigma_w`.

The first factor is the best-fit likelihood. The second factor is the fraction of the prior width that remains plausible after seeing the data. That fraction is less than one when the data have localized the parameter. It is the inverse number of distinguishable parameter settings the model offered before the data, of which only one small neighborhood survives. This is the penalty I was looking for. It is not attached from outside. It is the width of the integral.

This also clarifies what "complexity" means here. It is not the number of lines of code needed to evaluate the model. It is not only the number of parameters. A model is complex to the extent that it spreads predictive probability over alternatives that the data do not select. A parameter with a wide prior costs more than one with a narrow prior. A parameter that has to be tuned to a tiny posterior width costs more than one that remains coarse. A direction the data do not measure is mostly controlled by the prior and contributes little. The penalty is geometric and data-dependent.

In `k` dimensions, the same calculation uses the Hessian. Let

`A = -nabla nabla log P(w|D,H)` at `w_MP`.

The Gaussian volume around the peak is `(2*pi)^(k/2)|A|^(-1/2)`, so

`P(D|H) ~= P(D|w_MP,H) P(w_MP|H) (2*pi)^(k/2)|A|^(-1/2)`.

This is a pleasing proof because the matrix I need is not new. It is the same curvature matrix I use for error bars. The determinant is the posterior volume, corrected for correlations among parameters. Best-fit model selection keeps only the height of the peak; the comparison integral keeps height times volume. That one missing volume is what bends the best-fit ratchet back down.

The interpolation problem has a second version of the same issue hidden inside it. Maximum-likelihood interpolation through noisy data is ill-posed, so I introduce a regularizer. With Gaussian noise and a quadratic weight penalty I write

`M(w)=alpha E_w + beta E_n`.

For fixed `alpha` and `beta`, fitting is easy. The posterior over `w` is proportional to `exp(-M(w))`, and its Hessian is `A=alpha C+beta B`. But `alpha` is itself a complexity knob. If `alpha` is too large, the curve is too stiff and underfits. If `alpha` is too small, the weights can roam and the curve chases noise. I need the same comparison principle to set `alpha`, not a separate discrepancy rule.

The normalizers give it. The likelihood has normalizer `Z_n(beta)`, the prior has normalizer `Z_w(alpha)`, and the posterior numerator over weights has normalizer `Z_M(alpha,beta)=int exp(-M(w))dw`. Therefore the probability of the data at fixed `alpha,beta` is

`P(D|alpha,beta,H)=Z_M/(Z_w Z_n)`.

Now the regularizer's trade-off is visible. Small `alpha` makes the prior over weights broad, so `Z_w` grows and divides the score down. Large `alpha` prevents a good fit, so `Z_M` is small. The preferred `alpha` is where the data are fit without buying unnecessary parameter volume.

At this point I hit the objection that matters. Am I choosing my prior after seeing the data? It sounds like I define a family of priors indexed by `alpha`, look at the data, choose the best `alpha`, and then call that prior legitimate. That cannot be the fundamental story. The honest Bayesian object is an ensemble of priors before the data arrive. Exact inference should integrate over that ensemble:

`P(w|D,H)=int P(w|D,alpha,beta,H) P(alpha,beta|D,H) d alpha d beta`.

So using the best `alpha,beta` is not the rule; it is an approximation. If the posterior over `alpha,beta` has a sharp peak, the integral is dominated by that peak, just as the parameter integral was dominated by `w_MP`. The apparent post-hoc prior choice is really another Laplace approximation to a larger marginalization.

This also tells me why I must marginalize rather than jointly maximize. In a familiar Gaussian problem, if I estimate both the mean and the variance by maximum likelihood, the variance divides by `N`; if I account for the fitted mean, the correction divides by `N-1`. The fitted mean has absorbed one degree of freedom of noise. That correction is not a moral preference for simplicity. It is the consequence of integrating over the nuisance parameter instead of pretending the peak height is the whole mass.

Now I derive the cleaner hyperparameter equations. I move to the basis where the quadratic regularizer is whitened, so `E_w=0.5 sum w_a^2` and `A=alpha I + beta B`. The log evidence contains a `-0.5 log det A` term from the posterior volume and a `+(k/2) log alpha` term from the prior normalizer. When I differentiate with respect to `alpha`, the motion of `w_MP` drops out because `w_MP` is already the stationary point of `M`. The only nontrivial derivative is

`d log det A / d alpha = Tr(A^-1)`.

Setting the derivative to zero gives

`2 alpha E_w^MP = k - alpha Tr(A^-1)`.

The right side deserves a name. If the eigenvalues of `beta B` are `lambda_a`, then

`k - alpha Tr(A^-1) = sum_a lambda_a/(lambda_a+alpha)`.

Each term is between zero and one. A direction with `lambda_a >> alpha` is measured by the data and contributes almost one. A direction with `lambda_a << alpha` is held by the prior and contributes almost zero. So the raw number of parameters `k` is the wrong measure. The relevant number is `gamma`, the effective number of well-measured parameters.

Differentiating with respect to `beta` gives the matching noise equation:

`2 beta E_n^MP = N - gamma`.

This is exactly the generalized degrees-of-freedom correction I wanted. The discrepancy principle says `N`; ordinary least squares says `N-k`; the regularized Bayesian calculation says `N-gamma`. Well-measured parameters each fit about one unit of noise. Poorly measured parameters do not. The same volume calculation that penalizes model flexibility also tells me how much noise the fitted parameters have inevitably absorbed.

Finally I return to whole models. Different bases, regularizers, and noise assumptions are ranked by integrating the `alpha,beta` evidence over their prior scales. When that surface is peaked, I can approximate the integral by the peak evidence times its width in log `alpha` and log `beta`. As model complexity increases, the best-fit likelihood alone keeps rising, but the accumulated volume ratios eventually lower the total score. On a plot, the score climbs steeply while the model is too simple to fit, then descends gently when extra flexibility is not measured by the data. The final method is therefore not "prior plus likelihood" dressed up with a penalty. It is model selection by predictive mass: integrate over the parameter volume, and the complexity penalty appears from the volume that the data fail to use.
