I start with the symmetric-loss theorem because it is the only clean route from a loss formula to a label-noise guarantee. If `sum_j L(f(x), j) = C` is constant, then under uniform noise the noisy risk is `R_eta(f) = (1 - eta*K/(K - 1)) R(f) + C*eta/(K - 1)`. For `eta < (K - 1)/K`, the noisy risk preserves the ordering of the clean risk, so the same global minimizer survives the corruption. Cross entropy does not have a constant class-sum. MAE and RCE do, but they are weak training losses in deep nets. The first question is whether the constant-sum condition has to be discovered loss by loss, or whether I can impose it mechanically.

The condition itself gives the construction. For any nonnegative loss whose class-sum is positive, I can define

`L_norm(f(x), y) = L(f(x), y) / sum_{j=1}^K L(f(x), j)`.

Then `sum_y L_norm(f(x), y) = 1` for every prediction. The uniform-noise derivation becomes the same affine transform with `C = 1`, namely `R_eta(f) = (1 - eta*K/(K - 1)) R(f) + eta/(K - 1)`. That is the normalization move: it does not merely find another robust loss; it turns any suitable base loss into a symmetric one. For class-conditional noise the guarantee is narrower — it needs extra assumptions: zero clean risk, bounded normalized losses on off-target classes, and a diagonally dominant transition condition. I should keep that caveat attached to the stronger-looking phrase "any loss can be made robust."

Normalized cross entropy is the first concrete test. With `p_k = p(k|x)`, one-hot `q`, and label `y`,

`NCE = (-sum_k q_k log p_k) / (-sum_j log p_j) = (-log p_y) / (-sum_j log p_j)`.

Equivalently, since both logs are nonpositive, this is `log p_y / sum_j log p_j`. The signs matter: the implemented numerator is positive `-log p_y`, the denominator is the positive class-sum `-sum_j log p_j`, and the ratio is in `[0, 1]`. This is robust under the symmetric theorem, but it is not automatically a good training loss. The denominator contains `Q = -sum_{k != y} log p_k`, so the loss can change by changing the non-target distribution even when `p_y` is fixed. That coupling can reduce the useful pressure on the labeled class, and empirically NCE and normalized focal loss underfit on hard noisy datasets even though CE and focal loss overfit the noise.

So robustness alone is not the final criterion. I need a way to explain why a robust active loss underfits and what companion term can repair it without losing symmetry. I write a loss as `L(f(x), y) = sum_k ell(f(x), k)`. An active loss has `ell(f(x), k) = 0` for every `k != y`; it explicitly acts only at the labeled coordinate. CE, focal loss, NCE, and normalized focal loss have this form. A passive loss has at least one nonzero off-label component; it explicitly penalizes probability assigned to some wrong class. MAE, normalized MAE, RCE, and normalized RCE have this form.

This active/passive split clarifies the underfitting fix. A robust active term gives a direct "raise the labeled probability" signal, but the normalization denominator leaves a route for the loss to improve through the off-label distribution. A passive term directly controls that off-label part. If both terms are themselves noise-tolerant, then their positive linear combination is also symmetric: `sum_y (alpha L_A + beta L_P)(f, y) = alpha C_A + beta C_P`, still constant. The result is an active-passive objective that keeps the theorem-level robustness while using two complementary optimization directions.

For the concrete active term I choose NCE, because it is the robust version of CE and keeps a CE-like dependence on the labeled probability. For the passive term I choose RCE, because it is already symmetric and therefore does not need to be normalized for this pairing. With `q_y = 1` and off-label entries clamped so `log q_k = A < 0`, reverse cross entropy is

`RCE = -sum_k p_k log q_k = -A sum_{k != y} p_k = -A(1 - p_y)`.

The class-sum over possible labels is `sum_y RCE(f, y) = -A(K - 1)`, a positive constant because `A < 0`. If I did normalize RCE, the correct scaling would be division by `-A(K - 1)`, giving `(1 - p_y)/(K - 1)`. For the NCE+RCE objective, the unnormalized RCE is fine because scalar multiples of symmetric losses remain symmetric.

I also check the logit-gradient signs for RCE, since a sign error would reverse the passive term's meaning. From `RCE = -A(1 - p_y)`, the derivative with respect to the labeled logit is `A p_y(1 - p_y)`, which is nonpositive because `A < 0`; gradient descent therefore raises the labeled logit. For a wrong logit `z_j`, the derivative is `-A p_y p_j`, which is nonnegative; gradient descent lowers wrong logits. The term is also gated by `p_y`, so it is strongest once the model already gives the provided label some probability and weaker on examples the model strongly rejects. That matches the intended passive behavior.

The resulting loss is

`L = alpha * NCE + beta * RCE`.

For uniform noise, its class-sum is `alpha * 1 + beta * (-A)(K - 1)`, so it is still a constant and the same affine-risk proof applies. The weights are balancing parameters, not a universal constant. Equal weights `alpha = beta = 1` are the natural starting point and work on an easier setting like CIFAR-10, but harder, many-class settings such as CIFAR-100 and web-label data need a larger active weight and a smaller passive weight to lean on the fitting term. So the code should expose `alpha` and `beta` rather than bury a claim that one pair is canonical for every dataset.

Now I translate the formula into the reference implementation faithfully. NCE must use `F.log_softmax(logits, dim=1)`, a one-hot label vector, numerator `-(one_hot * log_probs).sum(dim=1)`, and denominator `-log_probs.sum(dim=1)`. RCE must use `F.softmax(logits, dim=1)`, clamp probabilities at `1e-7`, clamp one-hot labels at `1e-4`, and compute `-(probs * log(one_hot)).sum(dim=1)`. The final minibatch scalar is the mean of each term after scaling by `alpha` and `beta`. This is exactly the `NCEandRCE` composition in the reference code: `NormalizedCrossEntropy(scale=alpha) + ReverseCrossEntropy(scale=beta)`.
