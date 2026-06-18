I start with the ordinary Siamese idea: one image gives me two augmented views, and I want the two representations to agree. The problem is that pure agreement has the wrong trivial optimum. If every image maps to one constant vector, the agreement objective is perfectly happy and the representation is useless. So I need to identify the smallest change that keeps the agreement signal but changes the training dynamics enough that the optimizer does not fall into that constant solution.

The existing recipes each pay for non-collapse in a different currency. Contrastive learning pays with negatives: pull the two views of the same image together and push views of different images apart. That makes collapse impossible for the contrastive objective, but it also makes the number of negatives matter, which is why large batches and queues show up. Clustering pays with assignments and a balance constraint: make views agree through cluster codes, but force the codes to spread across prototypes. Momentum-target methods pay with a second network: the online branch predicts a target branch that moves slowly.

The last recipe is the suspicious one. A momentum target is not only a moving average. It also means the target branch is not trained by the current loss gradient. Those two properties are tied together in the usual implementation. If I want to know which one matters, I should remove the moving average and keep only a single shared encoder, then ask whether the target-side gradient rule is enough.

So I build the stripped model. I take two augmentations \(x_1, x_2\). A shared encoder \(f\), meaning backbone plus projection MLP, gives \(z_1=f(x_1)\) and \(z_2=f(x_2)\). A predictor \(h\) maps each projection to \(p_1=h(z_1)\) and \(p_2=h(z_2)\). I compare a prediction from one view to the projection of the other view with negative cosine similarity:

\[
D(p,z) = - {p \over \|p\|_2} \cdot {z \over \|z\|_2}.
\]

This sign is important. Minimizing \(D\) maximizes cosine similarity, and the minimum is \(-1\). The squared distance between normalized vectors is

\[
\|\hat p-\hat z\|_2^2 = 2 - 2\hat p\cdot\hat z = 2 + 2D(p,z),
\]

so negative cosine and normalized squared error differ only by a positive scale and an additive constant. They define the same minimizers.

The two-view symmetric loss is

\[
L = {1\over2}D(p_1,z_2) + {1\over2}D(p_2,z_1).
\]

Now I put the only special operation on the target side. In the first term \(z_2\) is a constant target, and in the second term \(z_1\) is a constant target:

\[
L = {1\over2}D(p_1,\operatorname{sg}(z_2)) + {1\over2}D(p_2,\operatorname{sg}(z_1)).
\]

This does not freeze either view globally. The encoder on \(x_2\) receives no gradient through \(z_2\) in the first term, but it receives gradient through \(p_2\) in the second term, and symmetrically for \(x_1\). Each view is a target once and a predicted branch once.

The controlled experiment is then clean. With this operation in place, the loss decreases without hitting the degenerate minimum immediately, the kNN monitor improves, and the per-channel std of the l2-normalized output stays near \(1/\sqrt{d}\). Without it, with architecture and hyperparameters otherwise unchanged, the loss quickly reaches \(-1\), the output std goes to zero, and linear evaluation is chance. That tells me the constant solution exists for this structure, and this target-side gradient rule is the load-bearing difference in the reported baseline. It does not prove that collapse is impossible in every setting; it shows exactly which switch flips collapse in this design.

I should not stop there, because a hidden architectural component might still be responsible. I remove the predictor and keep the target-side constant. With \(h\) as the identity, the symmetric loss becomes

\[
{1\over2}D(z_1,\operatorname{sg}(z_2)) + {1\over2}D(z_2,\operatorname{sg}(z_1)).
\]

Its gradient has the same direction as the gradient of \(D(z_1,z_2)\), scaled by \(1/2\). In the symmetric no-predictor case, the stop operation has become algebraically vacuous: one term gives the \(z_1\) side of the ordinary gradient and the other gives the \(z_2\) side. Collapse is expected and observed. I have to keep the algebra scoped to the symmetric loss; the asymmetric no-predictor variant also fails empirically, but that second fact is an observation rather than the same identity.

Freezing the predictor at random initialization fails in a different way. The loss stays high and training does not converge; this is not the collapse signature. That distinction matters. A trained predictor is required for the method to work, but the predictor alone is not the anti-collapse proof. It has to keep adapting to the moving representation, which explains why keeping the predictor learning rate constant can improve the result.

Batch normalization is another candidate. Removing BN from the heads hurts accuracy badly, but the std diagnostic does not show collapse. Putting BN in the hidden layers recovers most of the accuracy, and adding BN to the projection output gives the default result. BN on the predictor output is unstable. Since the with/without target-side-gradient comparison uses the same BN configuration in both arms, BN is an optimization and accuracy component, not the causal collapse switch in this experiment.

The similarity function is not the explanation either. Replacing negative cosine with a symmetrized cross-entropy similarity still trains without collapse, although accuracy is lower. Symmetrization is also not the explanation. A one-direction loss still avoids collapse, and using two sampled directions mainly improves the empirical estimate of the augmentation expectation. Batch size behaves similarly: ordinary batches work, while a very large batch hurts with plain SGD in the familiar large-batch optimization way.

At this point the target-side constant is not just a code trick. It suggests that I am not doing ordinary gradient descent on a single loss over one set of network parameters. A natural place where "hold this thing fixed while optimizing that thing" appears is alternating optimization. I can make that precise by introducing a second block of variables, one target vector per image:

\[
\mathcal L(\theta,\eta)=\mathbb E_{x,T}\|F_\theta(T(x))-\eta_x\|_2^2.
\]

Here \(F_\theta\) is the encoder-like network, \(T\) is the augmentation distribution, and \(\eta_x\) is not a network output. It is an optimization variable indexed by the image. The analogy to k-means is useful: \(\theta\) is like the learned centers or model parameters, and \(\eta_x\) is like a per-sample latent code or assignment.

I can solve this problem by alternating:

\[
\theta^t \leftarrow \arg\min_\theta \mathcal L(\theta,\eta^{t-1}),
\]

\[
\eta^t \leftarrow \arg\min_\eta \mathcal L(\theta^t,\eta).
\]

In the \(\theta\) subproblem, \(\eta^{t-1}\) is fixed. Therefore no gradient flows into it. The target-side constant is not an extra heuristic in this view; it is the direct consequence of optimizing one block while holding the other block fixed.

The \(\eta\) subproblem separates by image. For a fixed image \(x\), I minimize the expected squared distance from \(\eta_x\) to \(F_{\theta^t}(T(x))\). The minimizer is the augmentation mean:

\[
\eta_x^t = \mathbb E_T[F_{\theta^t}(T(x))].
\]

Computing that expectation exactly is not practical, so I approximate it with one augmentation sample \(T'\):

\[
\eta_x^t \approx F_{\theta^t}(T'(x)).
\]

Substituting this into the \(\theta\) subproblem gives

\[
\theta^{t+1}\leftarrow \arg\min_\theta
\mathbb E_{x,T}\|F_\theta(T(x))-F_{\theta^t}(T'(x))\|_2^2.
\]

Now the second term is evaluated with the previous parameters and is constant for the current subproblem. The two augmentations \(T\) and \(T'\) are exactly the two views. If I reduce this subproblem by one SGD step instead of solving it fully, I get the simple shared-weight Siamese update with the target projection treated as constant.

This also gives the predictor a role. The one-sample approximation throws away the augmentation expectation. A regression predictor trained to map one view's projection toward another view's projection has an optimal form as a conditional mean: \(h^*(z_1)=\mathbb E[z_2\mid z_1]\). In the per-image picture I am using, that conditional target is the augmentation average \(\mathbb E_T[f(T(x))]\). So the predictor is a learned stand-in for the expectation that the one-sample \(\eta\) update ignores. The moving-average target experiment supports this interpretation: if I approximate the per-image expectation explicitly with a memory-style moving average, I can remove the predictor and still get a nontrivial representation, though much worse than the default.

This derivation does not prove non-collapse. The objective with \(\eta\) still has a constant solution. The safer conclusion is the one the evidence supports: alternating optimization changes the trajectory. The per-image targets start from a randomly initialized network, so they are not constant; the target updates are per image rather than a joint gradient pulling all targets together; and empirically the optimizer follows the scattered-output path instead of the constant path.

So the method closes as a minimal Siamese recipe: one shared encoder, a projection MLP, a trained predictor MLP, negative cosine between normalized prediction and target projection, and the target projection held constant in each loss term. It is BYOL with the moving-average encoder removed, SimCLR with negatives removed, and SwAV with clustering removed. The reliable claim is not that a theorem forbids collapse, but that in the controlled experiments this small asymmetric update is enough to train useful representations without negatives, online clustering, a momentum encoder, or large batches.
