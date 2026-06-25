I start with the ordinary Siamese idea: one image gives me two augmented views, and I want the two representations to agree. The problem is that pure agreement has the wrong trivial optimum. If every image maps to one constant vector, the agreement objective is perfectly happy and the representation is useless. So I need to identify the smallest change that keeps the agreement signal but changes the training dynamics enough that the optimizer does not fall into that constant solution.

The existing recipes each pay for non-collapse in a different currency. Contrastive learning pays with negatives: pull the two views of the same image together and push views of different images apart. That makes collapse impossible for the contrastive objective, but it also makes the number of negatives matter, which is why large batches and queues show up. Clustering pays with assignments and a balance constraint: make views agree through cluster codes, but force the codes to spread across prototypes. Momentum-target methods pay with a second network: the online branch predicts a target branch that moves slowly.

The last recipe is the one I want to take apart. A momentum target bundles two things that are usually shipped together: the target parameters are a moving average of the online parameters, and the target branch is excluded from the current loss gradient. I cannot tell from the recipe alone which of those two properties is doing the work against collapse. The cheapest way to separate them is to delete the moving average entirely — make the target branch literally the same encoder, same parameters, same step — and keep only the rule that the target side gets no gradient. If that alone trains, the moving average was never the anti-collapse mechanism. If it collapses, the moving average was load-bearing and I have learned something different.

So I build the stripped model. I take two augmentations \(x_1, x_2\). A shared encoder \(f\), meaning backbone plus projection MLP, gives \(z_1=f(x_1)\) and \(z_2=f(x_2)\). A predictor \(h\) maps each projection to \(p_1=h(z_1)\) and \(p_2=h(z_2)\). I compare a prediction from one view to the projection of the other view with negative cosine similarity:

\[
D(p,z) = - {p \over \|p\|_2} \cdot {z \over \|z\|_2}.
\]

Minimizing \(D\) maximizes cosine similarity, and the minimum is \(-1\). Before I commit to negative cosine, I want to know whether the sign and the normalization are an arbitrary choice or whether they tie back to something I trust. The natural reference is squared error between the normalized vectors. Expanding,

\[
\|\hat p-\hat z\|_2^2 = \hat p\cdot\hat p - 2\,\hat p\cdot\hat z + \hat z\cdot\hat z = 1 - 2\,\hat p\cdot\hat z + 1 = 2 - 2\,\hat p\cdot\hat z = 2 + 2D(p,z),
\]

using \(\|\hat p\|=\|\hat z\|=1\). Let me check that on actual numbers rather than trust the algebra. Taking two random 8-vectors, normalizing them, I get \(\|\hat p-\hat z\|_2^2 = 2.49773\) and \(2+2D = 2.49773\), agreeing to the last digit. So negative cosine and normalized squared error differ only by a positive scale and an additive constant; they have the same minimizers, and I can move between the cosine view and a regression view freely later. That equivalence is going to matter when I try to explain the predictor.

The two-view symmetric loss is

\[
L = {1\over2}D(p_1,z_2) + {1\over2}D(p_2,z_1).
\]

Now I put the only special operation on the target side. In the first term \(z_2\) is held constant, and in the second term \(z_1\) is held constant:

\[
L = {1\over2}D(p_1,\operatorname{sg}(z_2)) + {1\over2}D(p_2,\operatorname{sg}(z_1)).
\]

I want to be careful about what this actually freezes, because it is easy to misread \(\operatorname{sg}\) as "view 2 is detached." It is not. The encoder on \(x_2\) receives no gradient through \(z_2\) in the first term, but it still receives gradient through \(p_2\) in the second term, and symmetrically for \(x_1\). Each view is a target once and a predicted branch once. To make sure I implement this and not something stronger, I trace the gradient through the actual code path: the forward returns the predictions together with detached projections, and the loss reads those detached projections as the targets. Running a tiny version, the loss value is identical with and without the detach — \(0.086822\) either way, because the detach changes nothing about the forward computation — but the encoder weight gradient differs (their normed difference is non-zero, ratio of norms about \(1.96\)). So the detach is genuinely altering the update and not just the bookkeeping; the loss curve alone would never reveal whether stop-gradient is present.

Now the controlled experiment is clean, because the only thing I am toggling is this target-side gradient rule with architecture and hyperparameters fixed. With it in place, the loss decreases without slamming into \(-1\), the kNN monitor improves, and the per-channel std of the l2-normalized output sits near \(1/\sqrt{d}\). I should sanity-check that reference value before I lean on it as a non-collapse diagnostic. A constant output has per-channel std zero. The opposite extreme is outputs scattered isotropically on the unit sphere: drawing many random unit vectors in dimension \(d\) and averaging the per-channel std, I get \(0.50000\) for \(d=4\), \(0.12500\) for \(d=64\), \(0.02210\) for \(d=2048\), against \(1/\sqrt{d}=0.5,\,0.125,\,0.02210\). So \(1/\sqrt d\) really is the scattered-sphere value, and a std reading near it means the outputs are spread, not piled on a point. When I drop the target-side rule, with everything else unchanged, the loss reaches \(-1\) fast, the output std goes to zero, and linear evaluation is chance. The constant solution exists for this structure, and this one switch is what decides whether the optimizer falls into it. It does not prove collapse is impossible in every setting; it shows which switch flips collapse in this design.

I should not stop there, because a hidden architectural component might be the real cause and the stop-gradient might be incidental. The most suspicious co-conspirator is the predictor. So I remove it and keep the target-side constant. With \(h\) the identity, the symmetric loss becomes

\[
{1\over2}D(z_1,\operatorname{sg}(z_2)) + {1\over2}D(z_2,\operatorname{sg}(z_1)).
\]

I expect this to behave like the plain agreement loss \(D(z_1,z_2)\) with no stop-gradient at all, but I do not want to assert that — the whole point of \(\operatorname{sg}\) was supposed to be that it changes the gradient. Let me actually compute both gradients. For the stop-gradient loss, the gradient with respect to \(z_1\) comes only from the first term (in the second term \(z_1\) is the frozen target), so \(\nabla_{z_1}L = \tfrac12\,\nabla_a D(a,z_2)|_{a=z_1}\). For the plain loss \(D(z_1,z_2)\), the gradient with respect to \(z_1\) is the full \(\nabla_a D(a,z_2)|_{a=z_1}\). Numerically, on a random pair in dimension 6, \(\nabla_{z_1}L_{\mathrm{sg}}\) equals exactly half of \(\nabla_{z_1}D(z_1,z_2)\), component by component, and the same holds for \(z_2\) using the second-argument derivative. So in the symmetric no-predictor case the stop-gradient has become algebraically vacuous: the two terms reassemble the two sides of the ordinary agreement gradient, scaled by \(1/2\). Collapse is therefore expected here for exactly the same reason plain agreement collapses, and that is what I observe. I have to keep this identity scoped to the symmetric loss; the asymmetric no-predictor variant also fails empirically, but that is an observation, not the same algebra.

Freezing the predictor at random initialization fails in a different way. The loss stays high and training does not converge — that is not the collapse signature, where the loss races to \(-1\). The two failure modes are distinct, which tells me the predictor is not just a passive piece of architecture: it has to be trained, but a trained-and-frozen predictor is not enough either. It has to keep adapting to the moving representation, which is consistent with the recipe choice to keep the predictor learning rate high rather than decaying it.

Batch normalization is the next candidate to rule out. Removing BN from the heads hurts accuracy badly, but the std diagnostic does not show collapse — the outputs stay spread even as accuracy drops, so this is an optimization problem, not the constant solution. Putting BN in the hidden layers recovers most of the accuracy; adding BN to the projection output gives the default result; BN on the predictor output is unstable. Crucially, the with/without target-side-gradient comparison used the same BN configuration in both arms, so BN cannot be the causal collapse switch in that experiment.

The similarity function is not the explanation either: replacing negative cosine with a symmetrized cross-entropy similarity still trains without collapse, just at lower accuracy. Symmetrization is not it: a one-direction loss still avoids collapse, and using two sampled directions mainly improves the empirical estimate of the augmentation expectation. Batch size behaves the ordinary way: normal batches work, a very large batch hurts under plain SGD in the familiar large-batch optimization regime.

So the stop-gradient survives every attempt to attribute its effect to something else. But a code trick that survives ablation is not yet an explanation, and the fact that the loss value is unchanged by the detach (I checked this above) is nagging at me. A method whose update cannot be written as the gradient of the loss it reports is usually doing something other than plain gradient descent on a single objective. The phrase "hold this fixed while optimizing that" is the signature of alternating optimization — k-means holds centers fixed to reassign points, EM holds parameters fixed to estimate latents. Let me try to write the SimSiam update as one block of such an alternation and see whether the stop-gradient falls out for free or has to be bolted on by hand.

I introduce a second block of variables, one target vector per image:

\[
\mathcal L(\theta,\eta)=\mathbb E_{x,T}\|F_\theta(T(x))-\eta_x\|_2^2.
\]

Here \(F_\theta\) is the encoder-like network, \(T\) is the augmentation distribution, and \(\eta_x\) is not a network output — it is an optimization variable indexed by the image, like a per-sample latent code or cluster center. I solve by alternating:

\[
\theta^t \leftarrow \arg\min_\theta \mathcal L(\theta,\eta^{t-1}),
\qquad
\eta^t \leftarrow \arg\min_\eta \mathcal L(\theta^t,\eta).
\]

In the \(\theta\) subproblem \(\eta^{t-1}\) is fixed, so by construction no gradient flows into it. This is the part I was hoping for: the stop-gradient is not an extra heuristic in this view, it is forced by the act of optimizing one block while the other is held constant. That is a much better reason to believe it than "it happens to prevent collapse."

The \(\eta\) subproblem separates by image. For fixed \(x\), I minimize \(\mathbb E_T\|\,F_{\theta^t}(T(x))-\eta_x\|^2\) over the single vector \(\eta_x\). I claim the minimizer is the augmentation mean. Let me verify the general fact that \(\arg\min_c \mathbb E\|v-c\|^2 = \mathbb E[v]\): sampling \(1000\) random vectors and setting \(c\) to their mean, the finite-difference gradient of the objective at that point has norm \(0\) to eight digits, and perturbing \(c\) by \(0.3\) in every coordinate raises the objective from \(4.9603\) to \(5.4103\). So the mean is the minimizer, and

\[
\eta_x^t = \mathbb E_T[F_{\theta^t}(T(x))].
\]

Computing that expectation exactly is not practical, so I approximate it with one augmentation sample \(T'\):

\[
\eta_x^t \approx F_{\theta^t}(T'(x)).
\]

Substituting into the \(\theta\) subproblem,

\[
\theta^{t+1}\leftarrow \arg\min_\theta
\mathbb E_{x,T}\|F_\theta(T(x))-F_{\theta^t}(T'(x))\|_2^2,
\]

where the second term is evaluated at the previous parameters and is constant for this subproblem. The two augmentations \(T\) and \(T'\) are exactly the two views. If I do not solve the subproblem fully but take one SGD step, I recover the simple shared-weight Siamese update with the target projection treated as constant — which is the update I built at the start. So the stripped recipe and the alternation land on the same place, with the stop-gradient explained rather than assumed.

This also gives the predictor a job. The one-sample approximation \(\eta_x^t\approx F_{\theta^t}(T'(x))\) threw away the augmentation expectation; the predictor is the obvious place to put it back. As a regression map trained to send one view's projection toward the other's, \(h\) minimizing \(\mathbb E\|h(z_1)-z_2\|^2\) has optimum the conditional mean \(h^*(z_1)=\mathbb E[z_2\mid z_1]\). I want to confirm the conditional mean really is the regression optimum and not just repeat the textbook line. On a toy joint where \(z_2=\tanh(z_1)+0.5\,\varepsilon\), the best per-bin constant predictor (the empirical conditional mean) gives MSE \(0.2503\), matching the noise variance \(0.5^2=0.25\), while the best global constant gives \(0.6408\); and the per-bin predictor tracks the true \(\mathbb E[z_2\mid z_1]=\tanh(z_1)\) to within \(0.068\) across bins. So the conditional mean is both optimal and informative. In the per-image picture that conditional target is the augmentation average \(\mathbb E_T[f(T(x))]\), so the predictor is a learned stand-in for the expectation the one-sample \(\eta\) update ignores. The moving-average target experiment is consistent with this reading: if I approximate the per-image expectation explicitly with a memory-style moving average, I can drop the predictor and still get a nontrivial representation, though much worse than the default.

I want to be honest about what this derivation does and does not give me. It does not prove non-collapse: the objective \(\mathcal L(\theta,\eta)\) still has a constant solution where every \(\eta_x\) and every output is the same vector. What the alternation buys me is a different trajectory, not a different set of optima. The per-image targets start from a randomly initialized network, so initially they are scattered, not constant; the \(\eta\) updates are per image, so nothing pulls all targets toward a shared value the way a joint gradient would; and empirically the optimizer follows the scattered-output path with std near \(1/\sqrt d\) rather than the constant path. That is the most I can claim from the evidence in front of me.

Putting it together, the method is a minimal Siamese recipe: one shared encoder, a projection MLP, a trained predictor MLP, negative cosine between normalized prediction and target projection, and the target projection held constant in each loss term via stop-gradient. Stripped this far, it sits where the three baseline families meet — it is what is left of a momentum-target method after deleting the moving average, of a contrastive method after deleting the negatives, of a clustering method after deleting the online clustering. The reliable claim is not that a theorem forbids collapse, but that in these controlled experiments this small asymmetric update is enough to learn useful representations without negatives, online clustering, a momentum encoder, or large batches.
