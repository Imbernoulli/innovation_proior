I start from the scaling pressure rather than from a routing trick. A dense Transformer gives me one reliable feed-forward computation per token, but it also forces every token to use every feed-forward parameter. If I widen the layer, both parameter count and FLOPs rise. If I shard a dense layer because it no longer fits on one accelerator, I add communication to every forward and backward pass. The useful target is therefore a layer with many possible feed-forward parameters but only one small slice active for any particular token.

Mixture-of-experts is the natural first object to inspect. I can put \(N\) feed-forward networks behind a router, compute \(p_i(x)=\exp h_i(x)/\sum_j\exp h_j(x)\), select the top \(k\) experts, and return \(\sum_{i\in\mathcal T}p_i(x)E_i(x)\). That already separates total capacity from per-token expert compute. The problem is the cost of the route. With top-2 routing, every token claims two expert slots, runs two FFNs, and may need to be sent to two devices and gathered back twice. If communication and capacity buffers are the practical bottlenecks, I want to know whether the second expert is truly necessary.

The old objection to \(k=1\) is that the router would not learn because the argmax is hard. I write the top-1 output as \(y=p_e(x)E_e(x)\), where \(e=\arg\max_i p_i(x)\). The identity of \(e\) is not differentiable, but the selected probability \(p_e(x)\) is still a smooth softmax value multiplying the expert output. If a downstream loss has gradient \(g=\partial \ell/\partial y\), then, holding the chosen expert fixed for the local derivative,
\[
  \frac{\partial \ell}{\partial h_j}
  = \langle g,E_e(x)\rangle\,p_e(x)\,(\mathbf 1\{j=e\}-p_j(x)).
\]
So the router has a real gradient through the selected gate. What top-1 lacks is counterfactual feedback about the experts not chosen, not differentiability itself. I can treat that as an exploration and load-balancing problem instead of paying for a second expert on every token.

Now the hardware constraint bites. The router's choices are dynamic, but the expert input tensors need static sizes. For \(T\) routed tokens and \(N\) experts, the even share is \(T/N\), so I reserve \(C=(T/N)\) times a capacity factor for each expert, with integer rounding and a minimum capacity in implementation. If an expert receives fewer than \(C\) tokens, the empty slots are padding. If it receives more than \(C\), I cannot grow the buffer, so the overflowed tokens must skip this sparse FFN contribution. That is acceptable only because the Transformer block adds the FFN output to the residual stream; a dropped token contributes zero from the sparse FFN and remains unchanged through the residual.

I need to be exact about the slot cases because an off-by-one error changes which token is dropped. With an exclusive cumulative count for each expert, the first token routed to that expert gets position 0, the next gets 1, and a token is kept exactly when position \(< C\). With an inclusive cumulative count, the first token has count 1 and the equivalent keep test is count \(\le C\), with the stored slot equal to count minus 1. These are the same rule in different indexing conventions. The Mesh implementation uses the exclusive-position form for the switch gate; a simple PyTorch translation can use `cumsum(one_hot) - 1` and `position < capacity`.

Dropping is only rare if the router spreads tokens out. The hard load I care about is
\[
  f_i=\frac{1}{T}\sum_x \mathbf 1\{\arg\max_j p_j(x)=i\}.
\]
That is the actual fraction of tokens sent to expert \(i\), but it is an argmax count and has no useful gradient. The smooth quantity I can move is
\[
  P_i=\frac{1}{T}\sum_x p_i(x),
\]
the average probability mass assigned to expert \(i\). A separate penalty on \(P\) alone would equalize the router's soft probabilities without necessarily reacting to the experts currently overloaded by hard assignments. I therefore couple the two quantities with \(N\sum_i f_iP_i\) and stop gradients through \(f\).

The sign works out. For one token, the load term contributes \(\alpha N\sum_i f_i p_i(x)/T\). Differentiating through the softmax gives
\[
  \frac{\partial L_{aux}}{\partial h_j(x)}
  = \frac{\alpha N}{T}p_j(x)\left(f_j-\sum_i f_i p_i(x)\right).
\]
Under gradient descent, logits for experts whose hard load \(f_j\) is above the current probability-weighted average are pushed down, and logits for underloaded experts are pushed up. At the uniform target \(f_i=P_i=1/N\), the unweighted dot product is \(1/N\); multiplying by \(N\) makes the auxiliary loss value \(\alpha\), independent of expert count. The Mesh code's `mean(f * P) * N**2` is the same expression because the mean over experts divides the sum by \(N\).

Precision is the next failure mode. The fragile operation is not the expert matrix multiplication; it is the router softmax and the hard decisions downstream of it. If I train the whole sparse model in float32, I make the all-to-all routing traffic more expensive. If I keep the router softmax in bfloat16, the exponentials and small mantissa can perturb the probabilities enough to destabilize routing. The clean compromise is local selective precision: cast router inputs to float32, compute logits and softmax in float32, choose the expert and form the combine weight, then cast the combine and dispatch tensors back to the model dtype before they cross devices.

I also want the early routing logits not to be unnecessarily large. The weight initialization is a truncated normal with standard deviation \(\sqrt{s/n}\), where \(n\) is fan-in. Reducing the default scale \(s\) by a factor of ten gives smaller early activations and router logits, which is exactly where hard switching is most sensitive. This is not a new route; it is a stability condition that makes the route trainable at scale.

Top-1 routing leaves a real exploration issue: once an expert wins a token, the model does not observe what the runner-up expert would have done. Sampling from the softmax would explore but would also add high-variance routing noise. A lighter perturbation is to jitter the router input during training by multiplying hidden states by uniform noise near 1. That can flip borderline argmax decisions while leaving confident assignments mostly intact. This matches the canonical Mesh code path, where input jitter is applied before the router projection.

I consider two tempting extensions and keep them out of the core design. First, if a token overflows its first-choice expert, I could reroute it to the next-choice expert. That would reduce drops, but it sends a token to an expert the router ranked lower and complicates the dispatch path; the residual skip is simpler and preserves the token state rather than forcing a mismatched expert computation. Second, I could make the attention projections sparse too, but the feed-forward sublayer is the high-parameter, token-independent part of the Transformer block and is the safer insertion point for the first stable design.

The final layer is therefore deliberately narrow in its moving parts. A float32 router computes softmax probabilities, picks the top-1 expert, keeps the selected gate as the differentiable scalar, assigns a 0-based slot within the expert capacity, and zeros the gate for overflow. Experts are ordinary feed-forward networks. The sparse layer returns only the expert update, so dropped tokens remain zero in that update and survive through the block residual. The training loss is the language-modeling loss plus \(\alpha N\sum_i f_iP_i\) from each sparse layer. The result is not a general top-k MoE; it is the simplest useful expert route: one token, one expert, one balancing loss, and float32 only where the router needs it.
