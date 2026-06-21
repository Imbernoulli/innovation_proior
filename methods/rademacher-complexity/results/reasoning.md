I start with the part of learning that is easy to underestimate. If I fix one function before seeing any data, the empirical average and the true expectation have ordinary concentration between them. That is not the learning problem. The learner looks at the same sample that creates all the fluctuations and then chooses a function from the class. So I need to control the largest upward accident the class can exploit, not the accident of any one preselected function. I write this as a supremum of true minus empirical averages, and I ask what property of the class makes that supremum large on the sample in front of me.

The usual answer is a fixed combinatorial size. That answer is valid but feels misaligned with model selection. A class can have a large worst-case dimension because there exists some malicious arrangement of points on which it can do many things. But my sample is one particular arrangement from one distribution. If the penalty refuses to look at that arrangement, it cannot know whether the class is flexible in the directions that matter here. I want the capacity of the class on this sample, not the capacity of the class over every possible sample.

What does "capacity on this sample" mean operationally? It cannot mean success on the true labels, because that mixes capacity with signal. A rich class and a correct class can both fit labels. I need a target that contains no signal at all. I put independent fair signs on the sample points and ask how much the class can align with them. If some function in the class can achieve a large signed correlation with pure coin flips, then the class has enough freedom to explain noise on these points. If every function has small correlation with the signs, the class cannot chase arbitrary sample accidents very well.

So the object I want is an average over random signs of a supremum over the class. I use the absolute-value convention with a `2/n` normalization:

$$
\widehat R_n(\mathcal F)=E_\sigma\left[\sup_{f\in\mathcal F}\left|\frac{2}{n}\sum_{i=1}^n\sigma_i f(X_i)\right|\mid X_1,\ldots,X_n\right].
$$

This is already the conceptual answer: the capacity of the class is its ability to fit pure random signs on the realized sample. The signs have mean zero, so any correlation is overfitting ability, not information extraction.

Now I need to show that this noise-fitting score is not just a metaphor. The uniform gap contains an unknown expectation \(Pf\), so I introduce an independent ghost sample. For each \(f\), the expectation of the ghost empirical mean is \(Pf\). Pulling the ghost expectation outside the supremum only enlarges the expression. I am left comparing the empirical average on a ghost sample with the empirical average on the real sample. The paired real and ghost observations are exchangeable, so flipping which member of a pair contributes with a positive sign and which contributes with a negative sign does not change the distribution. That exchangeability is exactly where the random signs enter. The proof has not imported arbitrary noise from nowhere; it uncovers signs that were already hidden in the symmetry between the sample and its ghost copy.

Once I insert the signs, the difference between ghost and real empirical processes splits into two signed sums. This is why a factor of two appears in the usual symmetrization argument. The expected uniform gap is controlled by a signed empirical process, and the signed empirical process is precisely the noise-correlation quantity I am trying to use as capacity.

I still have an expectation statement, and a learning guarantee needs a high-probability statement. Here bounded differences supply the next step. If the loss or cost values stay in a bounded interval, changing one data point changes the supremum by only a controlled amount. McDiarmid's inequality converts the expected supremum into a bound that holds on the realized sample with a confidence term. The same bounded-differences reasoning also says the empirical signed-average estimate is concentrated around its expectation, so I can estimate the penalty from the sample by drawing signs and solving the corresponding optimization problem.

For binary classification, I need to be careful about the loss class. The loss is \(1(Y\ne f(X))\), but for labels and predictions in \(\{\pm1\}\) this is \((1-Yf(X))/2\). The constant part does not affect the supremum over \(f\), and multiplying a fair sign by the fixed label \(Y_i\) leaves a fair sign. This cancels the factor from the loss transformation and gives a clean classifier-class penalty:

$$
P(Y\ne f(X))\le \widehat P_n(Y\ne f(X))+\frac{R_n(\mathcal F)}2+\sqrt{\frac{\ln(1/\delta)}{2n}}
$$

in this same convention. The label reduction matters because it says the class is penalized for its ability to align with random labels on the observed inputs, which is exactly the overfitting diagnostic I want.

For real-valued predictors, I hit a second wall. A hard zero-one loss ignores margins, and margin behavior is the thing that explains boosting, neural networks with controlled weights, and support vector machines. I replace the step by a Lipschitz cost that dominates it. Now the class is composed with a nonlinear function, so I need a comparison principle. The contraction inequality says that a Lipschitz transformation that passes through zero cannot increase the signed-average complexity by more than a Lipschitz factor, with the constants set by the absolute `2/n` convention. This lets me bound the margin-cost class by the underlying score class and obtain

$$
P(Yf(X)\le0)\le \widehat E_n\phi(Yf(X))+2L R_n(\mathcal F)+\sqrt{\frac{\ln(2/\delta)}{2n}}.
$$

This is the point where the method becomes usable. I am no longer stuck with the exact loss class; I can analyze the score class and pay a controlled Lipschitz price.

I also need the penalty to behave under constructions. If a class is contained in another, the supremum can only shrink. Scaling the class scales the signed correlations. Taking a convex hull does not change the value because a linear functional reaches its supremum on the hull at the original extreme points. Sums subadd by the triangle inequality. A fixed translation costs only a small bounded term, and a bounded power loss inherits that translation cost plus the Lipschitz factor from the power map. These rules are simple, but they are why the method applies to voting, neural networks, trees, and kernels. A boosted classifier lies in a convex hull, so the hull itself adds no extra signed-correlation complexity beyond the base class. A neural network can be peeled layer by layer through Lipschitz composition and sums. A kernel norm ball can be bounded by Cauchy-Schwarz, producing the trace-style sample quantity

$$
\widehat R_n(\mathcal F)\le \frac{2B}{n}\left(\sum_i k(X_i,X_i)\right)^{1/2}.
$$

I have to check that this new penalty does not lose the old theory. If I restrict a binary class to the sample, I get a finite set of sign vectors. The signed supremum over the class is just the maximum of inner products between a random sign vector and that finite set. A finite-class maximal inequality bounds this by a square root of the logarithm of the number of realized patterns. If the class has VC dimension \(d\), Sauer-style growth bounds recover the usual VC rate up to logarithmic factors. So the noise-fitting penalty can fall back to the classical bound in the worst case, but it is allowed to be smaller on an easier sample.

I now see the whole method in one chain. I start from the uniform gap caused by choosing after looking at data. I reject a purely worst-case penalty because model selection needs a sample-sensitive diagnostic. I test the class against random signs because pure noise separates flexibility from signal. Ghost-sample symmetrization proves that this diagnostic controls the uniform gap. Concentration turns it into a high-probability bound and lets the sample version estimate the population version. Contraction and structural rules make the quantity calculable for real learning architectures. The distinctive insight is therefore not merely "there is a generalization bound"; it is that learnability is measured by how well the class can fit random labels on the observed points.
