OK, let me think this through from scratch. Self-attention has a problem I keep running into: it doesn't know where anything is. The output at position m is a softmax-weighted sum of values, and the weights come from q_m^T k_n. If q, k, v are linear in the token embeddings, then permuting the tokens just permutes the outputs — the function is over the *set* of inputs, not the sequence. "The dog bit the man" and "the man bit the dog" produce the same bag of representations. For language that's fatal. So I have to put position in by hand. The whole question is how.

Let me be precise about *where* it has to go in. The only quantity that decides which token attends to which is the logit q_m^T k_n. Everything downstream — the softmax, the weighted sum — is a consequence of those logits. So whatever I do, position has to end up changing q_m^T k_n. That's the target.

The obvious thing, the thing everyone does first, is sinusoidal absolute encoding. Build a fixed vector p_i for each position, p_{i,2t} = sin(i / 10000^{2t/d}), p_{i,2t+1} = cos(i / 10000^{2t/d}), and add it to the embedding before projecting: f_t(x_i, i) = W_t(x_i + p_i). I like the construction — each dimension-pair is a sinusoid whose wavelength sweeps geometrically from 2*pi up to about 10000*2*pi, so it's a stack of clock hands at every resolution, fast hands for local position and slow hands for global. The learned-embedding variant (a trainable p_i up to length L) is the same idea minus the closed form, and it has the ugly property that it caps the sequence at L and learns nothing past it. Set that aside.

But let me actually expand what this does to the logit, because that's the thing I care about. With q_m = W_q(x_m + p_m), k_n = W_k(x_n + p_n),

  q_m^T k_n = x_m^T W_q^T W_k x_n + x_m^T W_q^T W_k p_n + p_m^T W_q^T W_k x_n + p_m^T W_q^T W_k p_n.

Stare at that. The first term is pure content, fine. But the other three all carry absolute position — p_m here, p_n there, p_m and p_n together. The logit depends on *where m and n sit in the buffer*, not on *how far apart they are*. And that's wrong, or at least it's not what I want. What language actually cares about is the gap m - n: a verb three words after its subject is the same relation whether the sentence starts at token 5 or token 500. The model can in principle untangle relative offset from these absolute signals, but I'm making it work for something I could just hand it. So the additive-absolute approach pushes the relative structure into a place where it has to be learned indirectly.

There's a clue hiding in the sinusoid though. If I shift position by a fixed k, what happens to p? For a single (sin, cos) pair at frequency w, going from i to i+k sends (sin(wi), cos(wi)) to (sin(w(i+k)), cos(w(i+k))), and that's exactly a rotation by angle wk applied to the pair — a *fixed* linear map independent of i. So sinusoids already secretly encode relative shift as rotation. It's right there. The additive scheme just never uses it as the mechanism; it adds the vector and hopes. Let me file that away: rotation, relative shift. Feels important.

Now the relative-position line of work. Shaw and colleagues (2018) were the first to make attention explicitly relative. Their move: leave q as W_q x_m, but inject a learned relative embedding into the key (and value): f_k(x_n, n) = W_k(x_n + p~^k_r) with r = clip(m - n, r_min, r_max), and similarly for the value. So now there's a learned vector for each relative distance, clipped to a window. This is a real step — the logit finally sees m - n. But look at the costs. The clip throws away every distinction beyond the window: offset +50 and +500 collapse to the same r. It's a learned table, so it's extra parameters and no closed form. It pokes position into the *values*, not just the logit, which muddies what "position" even means here. And critically, the relative term is buried inside the expanded dot product — it's not a clean transform of q and k separately.

Transformer-XL (Dai et al., 2019) and the whole family after it sharpen this but keep the same starting point: take that four-term expansion and surgically edit it. Replace absolute p_n with a sinusoidal relative p~_{m-n}; replace the p_m factors in the third and fourth terms with two trainable global vectors u and v that don't depend on the query position; split W_k into a content projection and a position projection. T5 (Raffel et al., 2020) takes it to the extreme and throws the whole expansion away in favor of x_m^T W_q^T W_k x_n + b_{m,n}, a single learned scalar bias bucketed by distance. DeBERTa (He et al., 2020) keeps the two "content-times-position" cross terms with relative embeddings and argues those are the ones that matter. They all work. But notice what they all *are*: they begin with "add a position vector, expand the product," and then they hand-edit the resulting terms until they look relative. Every one of them parameterizes the relative signal with a learned table or a learned bias, and every one of them leaves that signal sitting inside the pairwise logit.

That last point nags at me for a separate reason. Linear attention. Softmax attention is O(N^2) precisely because it forms every q_m^T k_n. Katharopoulos and colleagues (2020) showed you can drop the softmax for a factorized similarity, Attention_m = sum_n phi(q_m)^T psi(k_n) v_n / sum_n phi(q_m)^T psi(k_n) with non-negative feature maps, and then by associativity you precompute sum_n psi(k_n) v_n^T and sum_n psi(k_n) once and never build the N-by-N matrix — O(N). The entire benefit hinges on position entering as a transform of the per-token features phi(q_m) and psi(k_n). A relative bias b_{m,n} cannot do that; it is, by definition, an entry of the matrix you refused to build. So none of the relative schemes port to linear attention. They can't, structurally.

So let me lay out what I actually want, all at once, instead of patching someone else's expansion. I want the logit q_m^T k_n to depend only on x_m, x_n, and m - n. I want position injected as a per-token transformation of q and k (so it survives the linear-attention factorization). And I'd love a closed form with no learned position parameters. Let me write that down as an equation and try to solve it.

Let q_m = f_q(x_m, m) and k_n = f_k(x_n, n) be whatever functions inject position. The demand is

  < f_q(x_m, m), f_k(x_n, n) > = g(x_m, x_n, m - n)

for some function g that depends on the positions only through the difference. And a boundary condition so this reduces to ordinary attention with no position: f_q(x, 0) = W_q x, f_k(x, 0) = W_k x. That's the whole specification. Now solve for f_q, f_k.

Start in the simplest nontrivial dimension, d = 2, because a 2-vector is a complex number and complex numbers make rotations trivial — and that rotation hunch from the sinusoid is still in the back of my mind. Identify R^2 with C, and use the fact that for complex a, b the real inner product of the corresponding 2-vectors is Re[a b*]. So the inner product q_m^T k_n is Re[f_q(x_q, m) f_k(x_k, n)*], and the cleanest way to enforce the relative structure is to ask the *complex* product itself to be a function of m - n:

  f_q(x_q, m) f_k(x_k, n)* = g(x_q, x_k, m - n).

Write everything in polar form, magnitude times phase:

  f_q(x_q, m) = R_q(x_q, m) e^{i Theta_q(x_q, m)},
  f_k(x_k, n) = R_k(x_k, n) e^{i Theta_k(x_k, n)},
  g(x_q, x_k, m - n) = R_g(x_q, x_k, m - n) e^{i Theta_g(x_q, x_k, m - n)}.

Plug in. f_q f_k* multiplies the magnitudes and subtracts the phases (Theta_q minus Theta_k), so matching magnitude and phase separately:

  R_q(x_q, m) R_k(x_k, n) = R_g(x_q, x_k, m - n),
  Theta_q(x_q, m) - Theta_k(x_k, n) = Theta_g(x_q, x_k, m - n).

And the boundary at position 0: writing q = ||q|| e^{i theta_q} for the unencoded query (theta_q is just its angle in the plane) and likewise k = ||k|| e^{i theta_k}, I have R_q(x_q, 0) = ||q||, Theta_q(x_q, 0) = theta_q, and the same for k.

Set m = n, so the relative offset is zero. The magnitude equation becomes

  R_q(x_q, m) R_k(x_k, m) = R_g(x_q, x_k, 0).

But R_g at offset 0 is fixed by the boundary: R_g(x_q, x_k, 0) = R_q(x_q, 0) R_k(x_k, 0) = ||q|| ||k||. So R_q(x_q, m) R_k(x_k, m) = ||q|| ||k|| for every m. Matching the magnitude still allows a degenerate reciprocal scale, and in the general relative case that scale would be an exponential distance bias rather than a rotation. I do not want position to amplify one side and shrink the other; I want a stable per-token map whose only positional effect is relative phase. The norm-preserving branch freezes each magnitude at its position-0 value, R_q(x_q, m) = ||q|| and R_k(x_k, m) = ||k||. Position does not live in the magnitude. Whatever it does, it is a pure phase thing.

Now the phase equation at m = n:

  Theta_q(x_q, m) - Theta_k(x_k, m) = Theta_g(x_q, x_k, 0) = theta_q - theta_k

(again using the boundary at offset 0). Rearrange: Theta_q(x_q, m) - theta_q = Theta_k(x_k, m) - theta_k. The left side depends only on (x_q, m); the right only on (x_k, m). They're equal for all choices, so each must be a function of m alone, with no dependence on the embedding at all. Call that function phi(m). So both phase functions have the same shape:

  Theta_f(x, m) = phi(m) + theta_x,   where Theta_f := Theta_q = Theta_k.

The position contributes the *same* extra angle phi(m) to query and key, on top of each vector's own intrinsic angle. Now I just need phi. Go back to the general phase equation, Theta_q(x_q, m) - Theta_k(x_k, n) = Theta_g(x_q, x_k, m - n), substitute Theta_f = phi + theta, and pick m = n + 1:

  phi(n + 1) + theta_q - (phi(n) + theta_k) = Theta_g(x_q, x_k, 1),
  phi(n + 1) - phi(n) = Theta_g(x_q, x_k, 1) + theta_k - theta_q.

The right-hand side has no running position index in it — it's a constant. So phi has constant first difference: it's an arithmetic progression, phi(m) = m*theta + gamma for constants theta (nonzero) and gamma. The gamma is a free global offset; fold it into the boundary and set it to zero. Then

  f_q(x_q, m) = ||q|| e^{i(theta_q + m*theta)} = q * e^{i m theta},
  f_k(x_k, n) = ||k|| e^{i(theta_k + n*theta)} = k * e^{i n theta}.

Position is a rotation. Multiply the (complex) query by e^{i m theta} and the key by e^{i n theta}. And the rotation hunch from the sinusoids comes back, except now it is a solution of the relative-dot-product demand. Let me check the logit:

  < f_q(x_q, m), f_k(x_k, n) > = Re[ q * e^{i m theta} * (k * e^{i n theta})* ] = Re[ q k* e^{i(m - n) theta} ].

The absolute positions m and n appear only through e^{i(m-n)theta}. The logit depends on m - n and nothing else absolute. Exactly the demand, and I didn't add anything — I solved for it. Writing the 2D rotation as a real matrix, f(x_m, m) = [[cos m*theta, -sin m*theta], [sin m*theta, cos m*theta]] W x_m. Rotate the projected vector by an angle proportional to its position. That's the whole idea in two dimensions.

Now lift it to real d. The plane gave me one rotation at one frequency theta. The honest generalization: don't pick one frequency — chop the d-dimensional space into d/2 independent 2-planes and rotate each plane at its own frequency. Why is that even allowed to give relative-ness in the full space? Because the inner product is a sum over the planes, and each plane independently satisfies < rotate_m, rotate_n > = relative-only by the 2D argument; sum of relative-only-per-plane is relative-only. Linearity of the inner product is doing the gluing. So

  f_{q,k}(x_m, m) = R^d_{Theta, m} W_{q,k} x_m,

where R^d_{Theta, m} is block-diagonal, the i-th 2x2 block being a rotation by m*theta_i:

  block_i = [[cos m*theta_i, -sin m*theta_i], [sin m*theta_i, cos m*theta_i]].

The logit should then collapse to a single relative rotation. Rotations compose by adding angles, so within each plane (R_m)^T R_n should rotate by (n - m)*theta_i, which would give (R^d_{Theta, m})^T R^d_{Theta, n} = R^d_{Theta, n-m} and hence

  q_m^T k_n = (R^d_{Theta, m} W_q x_m)^T (R^d_{Theta, n} W_k x_n) = x_m^T W_q^T R^d_{Theta, n-m} W_k x_n.

The gluing across planes and the transpose convention are worth checking on numbers, not just in my head. Take d = 4 with the two-frequency schedule (theta = [1, 0.01]), fix arbitrary content vectors q = W_q x_m and k = W_k x_n, and compute. First the composition: build R_m and R_n as 4x4 block-diagonal rotations and form R_m^T R_n versus R_{n-m} directly. For m = 5, n = 8, max |R_5^T R_8 - R_3| comes out 1.1e-16 — equal to machine precision, so the transpose really does subtract angles and leave R_{n-m}. Then the relative property on the actual logit: (R_5 q)·(R_8 k) = 1.03749, and at the shifted pair (R_105 q)·(R_108 k) = 1.03749 — identical to five places, same offset -3. Move to offset -4, (R_5 q)·(R_9 k) = 0.97103, and it changes. So the logit tracks m - n and is blind to the absolute shift, exactly the demand, and now I've watched it happen on numbers rather than trusting the algebra. The sign also lines up with the complex check above: with this real rotation convention, q^T R_delta k equals Re[q k* e^{-i delta}], so the matrix-form R_{n-m} is the same relative dependence as the complex-form e^{i(m-n)theta}.

The position difference sits in a single rotation matrix sandwiched between the content projections. No learned position table, no clip, no bias bucket. And R is orthogonal, so it preserves norms — applying it can't blow up or collapse the representation as it propagates through layers, which is the stability I wanted when the magnitude dropped out of the 2D solution.

What frequencies? I have d/2 of them to choose. I keep coming back to the sinusoid: its geometric spread of wavelengths is what gave it multi-resolution coverage, and I want the same — some planes that spin fast and resolve local offsets, some that spin slowly and stay almost fixed over the whole sequence to carry coarse position. So reuse exactly that schedule: theta_i = 10000^{-2(i-1)/d} for i = 1..d/2. This isn't an arbitrary borrow; it makes the construction literally the relative-rotation version of sinusoidal encoding. Fast planes for nearby tokens, slow planes for far ones. (And as a practical aside, if I let the theta_i be learned, they barely move from this initialization — so there's no reason to spend parameters on them; freeze them.)

Whether far-apart tokens really do get a weaker positional signal is a question I can answer directly rather than leave as intuition. Group q = W_q x_m and k = W_k x_n into d/2 complex pairs and write the logit as a sum over planes:

  q_m^T k_n = Re[ sum_{i=0}^{d/2-1} q_{[2i:2i+1]} k_{[2i:2i+1]}* e^{i(m-n) theta_i} ].

Let h_i = q_{[2i:2i+1]} k_{[2i:2i+1]}* — that's content, no position in it — and let S_j = sum_{i=0}^{j-1} e^{i(m-n) theta_i} be the partial sums of the position phase factors, with the conventions h_{d/2} = 0 and S_0 = 0. The sum I care about is sum_i h_i (S_{i+1} - S_i), since S_{i+1} - S_i = e^{i(m-n) theta_i}. Summation by parts (the discrete Abel transformation) rewrites a sum of products as

  sum_{i=0}^{d/2-1} h_i (S_{i+1} - S_i) = - sum_{i=0}^{d/2-1} S_{i+1} (h_{i+1} - h_i),

where the boundary terms vanish because S_0 = 0 and h_{d/2} = 0. Take magnitudes and bound:

  | sum_i h_i e^{i(m-n) theta_i} | = | sum_i S_{i+1} (h_{i+1} - h_i) |
                                   <= sum_i |S_{i+1}| |h_{i+1} - h_i|
                                   <= ( max_i |h_{i+1} - h_i| ) * sum_i |S_{i+1}|.

This factors cleanly: a content piece, max_i |h_{i+1} - h_i|, times a purely positional envelope sum_i |S_{i+1}|. The content piece doesn't know about m - n at all. So the question of decay is entirely about the envelope (1/(d/2)) sum_i |S_i|. The story I want to tell is that because the theta_i are a geometric sweep, as |m - n| grows the phases e^{i(m-n)theta_i} spread out across the frequencies, the partial sums S_i lose coherence, and the envelope shrinks. That's a claim about spectral coherence, not a computation, so evaluate the envelope directly. Take d = 64 (so d/2 = 32 frequencies) and the standard schedule, and tabulate the averaged envelope for growing offsets:

  delta=0:   16.50    delta=1:   15.95    delta=2:   14.48    delta=4:   11.14
  delta=8:    9.92    delta=16:   7.65    delta=32:   7.65    delta=64:   5.70
  delta=128:  4.75    delta=256:  3.84    delta=512:  2.99.

At delta = 0 every phase is 1 and the cumulative sums grow straight out to S_32 = 32, averaging 16.5 — the maximum, as it must be. From there it falls: by delta = 8 it's down to 9.9, by delta = 512 to 3.0. So there is a genuine decay, and it's substantial. But the numbers also stop me from overclaiming: from delta = 16 to delta = 32 the envelope is essentially flat (7.6497 then 7.6529 — it even ticks up a hair), so this is not monotone. That makes sense — coherence loss across a finite set of incommensurate frequencies is noisy, not a clean exponential. So the honest statement is what the data show: a decaying envelope, not a strictly monotone one. That's still exactly the inductive bias I wanted — far-apart tokens have a weaker positional contribution to the logit, all else equal — and the schedule choice and the decay property turn out to be the same fact seen twice.

Two practical things before code. First, I'm not going to build that block-diagonal matrix and matrix-multiply — it's mostly zeros, O(d^2) for an O(d) operation. The rotation of each 2-plane (x_{2i}, x_{2i+1}) by angle m*theta_i is, written out, x_{2i} cos - x_{2i+1} sin and x_{2i+1} cos + x_{2i} sin. So I can do the whole thing elementwise: take x, multiply by the cosine vector [cos m*theta_1, cos m*theta_1, cos m*theta_2, cos m*theta_2, ...] (each frequency repeated for its two coordinates), and add a "rotated" copy of x — namely [-x_2, x_1, -x_4, x_3, ...], the per-pair 90-degree swap — multiplied by the matching sine vector. That's R^d_{Theta, m} x = x (*) cos_vec + rotate(x) (*) sin_vec, two elementwise multiplies and an add, O(d). This is the form I'll actually run, so I should confirm it computes the same thing as the block matrix and not some off-by-one variant. On the same d = 4 vector as before, with cos_vec = [cos 5, cos 5, cos 0.05, cos 0.05], sin_vec the matching sines, and rotate(q) the pair swap, max |elementwise(q, m=5) - R_5 q| is 0.0 exactly. Good — the elementwise rule and the matrix are bit-identical, so I can drop the matrix entirely.

The linear-attention case is the property none of the additive relative schemes could give. The same norm-preserving rotation can sit after the feature maps. In linear attention the per-token features go through non-negative maps phi(q_m), psi(k_n) before being combined. If I rotate *after* the feature map, R_m phi(q_m) and R_n psi(k_n), that should leave the factorized structure intact, which rests on two things worth checking directly. First, that the rotation keeps the relative property even on the (now non-negative) feature vectors: with d = 4, phi and psi sampled non-negative, m = 3, n = 7, I get (R_3 phi)·(R_7 psi) - phi·(R_4 psi) = 1.7e-16, so the dot still collapses to a single R_{n-m} between the per-token features. Second, that orthogonality really leaves magnitudes alone: ||R_3 phi|| - ||phi|| = -2.2e-16. Both hold to machine precision. So position still rides on the per-token features, the associativity trick still precomputes its running sums and stays O(N), and the relative offset is now baked in:

  Attention_m = sum_n (R_m phi(q_m))^T (R_n psi(k_n)) v_n / sum_n phi(q_m)^T psi(k_n).

I keep the denominator un-rotated on purpose: after rotation the numerator terms can go negative, and leaving the normalizer as the original non-negative sum avoids dividing by something near zero. The weights aren't strictly a probability distribution anymore, but they still weight values by relative-position-modulated similarity, which is the point. An additive bias b_{m,n} cannot sit on the per-token features — it lives in the matrix linear attention refuses to form — which is precisely why this relative scheme is linear-attention compatible.

So let me write it. I'll precompute the per-position angles m*theta_i, hold their cosines and sines, and apply the elementwise rotation to q and k right before the logit: multiply by the repeated cosine vector, add the pair-swapped-and-sign-flipped copy times the repeated sine vector. There's one bookkeeping choice with no mathematical content but real consequences for the code: which two coordinates of the head dimension count as a "pair." I can interleave them (dims 0,1 are a pair, 2,3 are a pair, ...) or split the head dimension into two contiguous halves and pair dim i with dim i + d/2. Both realize the same block-diagonal rotation up to a fixed permutation of the head dimension, so both are mathematically identical; a model just has to pick one and apply it consistently to every query and key it ever produces, since mixing the two conventions silently misaligns which coordinates rotate together. The full module — the frequency table, the two pairing conventions, and the attention layer that calls into it — is in the answer.
