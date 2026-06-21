I start with a signal that is too rich. If I compress it without asking what the receiver needs, I can only preserve the signal itself or minimize a distortion someone has already chosen for me. That is exactly where the usual rate-distortion setup becomes unsatisfying for learning. The hard part is not just to trade bits against error; the hard part is to say what error should mean when the useful structure is not known in the coordinates of `X`.

Prediction error alone does not settle it. A predictor can succeed while its internal summary still carries many details of `X` that have no stable relation to `Y`. Dimension alone does not settle it either. A small representation can keep the wrong detail, and a larger stochastic one can forget a great deal. I need the criterion to talk about information: how much about the input survives, and how much about the relevant variable survives.

The variable `Y` is the missing source of pressure. If `Y` tells me what is relevant, then each input value `x` has an operational meaning through the conditional law `p(y|x)`. I do not have to guess a Euclidean distance between raw inputs. What matters for relevance is how much information about `Y` remains after I replace `x` by some compressed code.

So I make the representation a channel from `X` to a code variable `T`, with the Markov chain `Y -> X -> T`. That direction is important: the code is generated from `X`; it is not a hidden cause that generates the data. The price of a channel is `I(X;T)`, the amount of input information allowed through. The value is `I(T;Y)`, the amount of relevant information preserved. The constrained problem is to keep as much information about `Y` as possible while allowing only a limited amount of information about `X`; in Lagrangian form that is `I(X;T) - beta I(T;Y)`, with `beta >= 0`.

Now I need to see what the stationary encoder must look like. The variables I can vary are the rows of `p(t|x)`, with `sum_t p(t|x)=1` for every `x`. The other distributions are tied to that encoder: `p(t)=sum_x p(x)p(t|x)`, and by the Markov chain `p(t|y)=sum_x p(t|x)p(x|y)`. Taking the derivative of `I(X;T) - beta I(T;Y)` with respect to one entry `p(t|x)` gives, up to the row-normalization multiplier,

```text
log[p(t|x)/p(t)] - beta sum_y p(y|x) log[p(y|t)/p(y)].
```

That still is not in a useful assignment form. But I can add and subtract the term `beta sum_y p(y|x) log[p(y|x)/p(y)]`. It depends on `x` but not on `t`, so it can be absorbed into the same row multiplier. What remains beside the log assignment ratio is

```text
beta sum_y p(y|x) log[p(y|x)/p(y|t)]
  = beta D_KL[p(y|x) || p(y|t)].
```

Setting the derivative to zero then forces

```text
p(t|x) proportional to p(t) exp(-beta D_KL[p(y|x) || p(y|t)]).
```

After normalization over `t`,

```text
p(t|x) = p(t) exp(-beta D_KL[p(y|x) || p(y|t)]) / Z(x,beta).
```

This is the rate-distortion shape I wanted, but the distortion has not been supplied by hand. It appears from the relevance constraint. The penalty for sending `x` to `t` is the mismatch between what `x` predicts about `Y` and what the codeword `t` predicts about `Y`.

The codeword prediction itself has to be consistent with the encoder. Bayes' rule gives

```text
p(y|t) = sum_x p(y|x) p(t|x) p(x) / p(t),
```

where `p(t)=sum_x p(x)p(t|x)`. This also fixes a possible index trap: the average is over input values `x`, not over `y`; `y` is the coordinate of the output distribution being computed.

The earlier distributional-clustering machinery now fits naturally. There, soft memberships, averaged context distributions, KL mismatch, and annealing were already doing useful work for one application. Here they become a general principle: the objects being averaged are the relevance distributions induced by the joint law of `(X,Y)`, and the annealing parameter is the multiplier on preserved relevance.

I also want the algorithmic form to minimize a single quantity. The expected KL distortion is

```text
E_{p(x,t)} D_KL[p(y|x) || p(y|t)].
```

Expanding it gives `H(Y|T) - H(Y|X)`, because the first term averages `-log p(y|t)` under the joint law and the second averages `-log p(y|x)`. Under `Y -> X -> T`, this is also `I(X;Y) - I(T;Y)`. Therefore

```text
I(X;T) + beta E D_KL[p(y|x) || p(y|t)]
  = I(X;T) - beta I(T;Y) + beta I(X;Y).
```

The last term is constant for the data distribution, so minimizing this free energy is the same tradeoff. The sign is important: the encoder Lagrangian has `- beta I(T;Y)`, while the free-energy form has `+ beta` times the relevance distortion.

Now the fixed-point loop is forced. Given `p(t|x)`, compute `p(t)`. Given `p(t|x)` and `p(t)`, compute `p(y|t)` by the posterior average. Given `p(t)` and `p(y|t)`, update `p(t|x)` with the exponential KL rule. Each subproblem is convex in the distribution being updated while the others are fixed, so the alternating procedure decreases the same free energy. The joint problem is still not convex in all distributions at once, so initialization and the chosen number of codewords can matter.

The boundary cases are a useful check on the formula. If `beta=0`, the exponential term is one, so `p(t|x)=p(t)` and the code carries no information about `X`; with a minimal active alphabet, everything collapses to one codeword. As `beta` grows, the KL term dominates and inputs prefer codewords with matching predictive distributions. In the KL itself, a `p(y|x)=0` term contributes zero, while `p(y|t)=0` with positive `p(y|x)` gives infinite distortion, so that assignment must receive zero probability.

The finite implementation is just this loop with careful normalization and log-space assignment:

```python
def ib_step(p_x, p_y_given_x, p_t_given_x, beta):
    p_t = p_x @ p_t_given_x
    p_y_given_t = (p_x[:, None] * p_t_given_x).T @ p_y_given_x / p_t[:, None]
    d = kl_rows_to_centers(p_y_given_x, p_y_given_t)
    logits = broadcast(log(p_t)) if beta == 0 else log(p_t)[None, :] - beta * d
    return softmax_over_t(logits)
```

Putting it together, the representation I get is an approximate minimal sufficient statistic: it keeps the information in `X` that matters for `Y` and discards as much other information as the tradeoff allows. The code is not a reconstruction of `X`; it is a compressed carrier of relevance. The distinctive move is that the relationship between `X` and `Y` supplies the distortion through `D_KL[p(y|x) || p(y|t)]`, turning feature selection into a variational information problem.
