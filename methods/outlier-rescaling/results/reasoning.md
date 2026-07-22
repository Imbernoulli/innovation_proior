OK, let me start from the thing I keep seeing and cannot explain. I take trained transformer language models with different attention designs, feed in ordinary sequences, and record attention maps and residual-stream activations layer by layer. Two patterns keep appearing. A few tokens, usually the first token, receive attention logits much larger than the rest, so many queries send a large part of their softmax mass there. At the same time, if I sort feature dimensions by mean absolute hidden-state magnitude, a few dimensions form a vertical stripe: they are large for most tokens, across many inputs, in the same coordinates. The token-axis sink and the feature-axis stripe look like the same shape rotated into different axes.

The engineering problem is immediate. These values dominate quantization scales. W8A8 already has to waste range on them, and W4A4 is even more brittle. But the obvious fixes are suspicious. Clipping large activations can make the model unstable or poor; replacing the normalizer that seems to create the outlier can also hurt. So I do not want to start by declaring the outliers bad. I want to know what computation becomes unavailable when I remove them.

The attention sink gives me a place to start. The sink token gets a huge attention weight, yet its value vector has unusually small norm. If attention weight meant "fetch this token's content," that pairing would be strange — why pay almost all your probability mass for a token that carries almost nothing? So let me write out what the output actually is and see what the sink is buying. Let the real-token logits be `z_i`, their values be `v_i`, the sink logit be `z_s`, and the sink value be `v_s`. Write `S = Σ_i exp(z_i)` and `w_s = exp(z_s)/(exp(z_s)+S)`. Every real-token weight shares the same denominator, so

`w_i = exp(z_i)/(exp(z_s)+S) = (1-w_s) exp(z_i)/S`,

where the last step just multiplies and divides by `S`. The output is then

`Σ_i w_i v_i + w_s v_s = (1-w_s)Σ_i softmax(z)_i v_i + w_s v_s`.

I want to be sure I have not dropped a term, so let me put numbers to it. Take five real tokens with random logits, a sink logit `z_s = 6`, and random value vectors, then compare the direct weighted sum `w·v + w_s v_s` against the right-hand side `(1-w_s)·softmax(z)·v + w_s v_s`. The two agree to machine precision (max coordinate difference `~3e-17`), so the rewrite is an identity, not an approximation. With `z_s = 6` here, `w_s ≈ 0.986`, so `1-w_s ≈ 0.0137`.

The identity by itself does not say the sink is "just a scale knob" — that depends on `v_s`. So let me push on the value vector. I shrink `v_s` toward zero and watch the two pieces of the output. When `v_s` is comparable to the real values, the sink's content term `w_s v_s` dominates everything (it is heavily weighted). But that is not the regime I measured: the sink's value norm is small. When I scale `v_s` down to zero, the output becomes exactly `(1-w_s)·(real-token mixture)` — the same direction the head would attend to without any sink, attenuated by a constant `1-w_s`. So in the small-`v_s` regime the sink is not adding a direction to the output; it is multiplying the real-token mixture by a scalar the head controls through `z_s`. The sink is a scale knob created through softmax's denominator. One thing the numbers make honest: at `w_s ≈ 0.99` that scalar is severe attenuation, so when a head wants this, it wants to nearly mute its own output — which is a real function, not a pathology.

Now I can ask whether the fixed residual stripe is doing the analogous thing through RMSNorm. RMSNorm divides every coordinate by `sqrt(mean(x^2)+eps)` and then applies a learned weight `λ`. One huge coordinate raises the shared denominator and shrinks all coordinates after the division. If the residual stripe were playing the sink's role, then the coordinate it lives in should not contribute directly — and the fingerprint of that would be a tiny post-normalization weight on that same dimension, the residual analogue of the sink's small value vector. So I go look at the measured `λ` on that dimension. Most RMSNorm weights are near one, while the residual-sink dimension can carry a weight around a few thousandths. The fingerprint is there. So the candidate hypothesis is: the network makes the coordinate large before the denominator, then suppresses its direct output immediately after the denominator. But "raises the denominator and shrinks the rest" is a verbal story; I do not yet know whether a large outlier actually shrinks the surviving features or whether the small `λ_d` is doing something I have not accounted for. I need the magnitude relation in closed form.

Let me work it out without hiding a factor of `D`. I will ignore `eps` for the algebra; with `eps` included the denominator is only larger, so whatever upper bound I get only loosens. Take `h ∈ R^D`, one outlier dimension `d`, and a post-norm weight vector `λ`. Let `r = |h_d|/||h||_2`, and write the suppression as `|λ_d| ≤ ε||λ||_∞` with `ε < 1`. Define `u = h/||h||_2`, so `|u_d| = r` and `Σ_{i≠d}u_i^2 = 1-r^2`. The RMS denominator is `||h||_rms = ||h||_2/√D`, so dividing `h` by it gives `√D·u`; multiplying by `λ` and taking the RMS norm of the result (`||·||_2/√D`) cancels the `√D`, leaving `||RMSNorm(h)||_rms = ||λ⊙u||_2` exactly. Now

`||λ⊙u||_2^2 = λ_d^2 r^2 + Σ_{i≠d} λ_i^2 u_i^2`.

The non-outlier part is at most

`Σ_{i≠d} λ_i^2 u_i^2 ≤ ||λ_{-d}||_∞^2 Σ_{i≠d} u_i^2 = ||λ_{-d}||_∞^2 (1-r^2)`,

so

`||RMSNorm(h)||_rms ≤ sqrt(||λ_{-d}||_∞^2 (1-r^2) + λ_d^2 r^2)`.

Relaxing `||λ_{-d}||_∞ ≤ ||λ||_∞` and `λ_d^2 ≤ ε^2||λ||_∞^2`,

`||RMSNorm(h)||_rms ≤ ||λ||_∞ sqrt((1-r^2)+ε^2 r^2) = ||λ||_∞ sqrt(1-(1-ε^2)r^2)`.

Algebraically, `ε<1` makes `1-(1-ε^2)r^2` shrink as `r` grows, so the bound should fall with the outlier fraction. But I have been wrong about monotonicity in finite-`D` derivations before — the bound and the actual norm can diverge — so I check it directly. I build `h` in `D=64` with `λ` drawn near one except `λ_d = 0.003` (giving `ε ≈ 0.0023`), sweep `r` from `0.1` to `0.99` by placing fraction `r` of the unit norm on dimension `d` and the rest at random, and compute both the true `||λ⊙u||_2` and the bound:

```text
 r       actual ||out||_rms     bound ||λ||∞·sqrt(1-(1-ε²)r²)
 0.10     1.0557                  1.2906
 0.30     1.0003                  1.2373
 0.50     0.8658                  1.1233
 0.70     0.6768                  0.9263
 0.90     0.4574                  0.5654
 0.99     0.1472                  0.1830
```

The bound holds at every `r`, and both the actual norm and the bound fall as `r` grows; sampling the bound on a fine grid of `r` confirms it is monotonically decreasing. So a larger residual outlier really does shrink the post-normalization feature magnitude, while its own dimension is muted by the small `λ_d`. The attention sink and the residual sink now sit in the same algebraic role: a large pre-normalization component sets the scale of the surviving components through a shared denominator, and a tiny accompanying weight keeps the outlier itself from leaking into the output.

This picture makes several diagnostics fall into place. If I remove softmax from token mixing with sigmoid or linear attention, attention-side massive activations shrink because the token-mixing denominator no longer needs a sink. But a residual sink can remain, because RMSNorm is still present. If I replace RMSNorm with Dynamic Tanh, `tanh(alpha*x)*weight+bias`, the operation is pointwise. Each coordinate only sees itself. That removes the cross-dimensional denominator, and the residual outlier mostly vanishes, but the model has also lost the ability for one coordinate to rescale the rest. The instability at ordinary learning rates and the need for a much smaller learning rate are exactly what I would expect if the outlier was carrying a useful scaling function.

Clipping separates the two pieces more cleanly. If RMSNorm is still there and I only cap the residual activation, then the denominator exists but the model cannot drive it with a large coordinate. When clipping at low thresholds causes divergence, and clipping at a high threshold still creates loss spikes, the message is not merely "normalization matters." The large value and the normalizer are acting as a unit. In a model whose attention side already has gated attention, aggressive residual clipping is less catastrophic, which tells me the attention sink was the primary stability hazard; the remaining degradation still says the residual sink has its own job.

The FFN source of the outlier also makes more sense now. A SwiGLU block computes `down(silu(gate(x)) * up(x))`. Swish is unbounded on the positive side, so it can generate large intermediate values that the down projection can amplify into residual outliers. A sigmoid GLU caps the gate in `(0,1)` and therefore starves that route. If the outlier is useful for rescaling, sigmoid GLU should suppress outliers and pay a quality cost even if the rest of the architecture is unchanged. That is the pattern I see. Swish is not only a better nonlinearity in the abstract; in this training regime it gives the model more room to manufacture the scale knob it wants.

Now the goal changes. I do not want to remove the rescaling. I want to remove the need to store the rescaling signal as a huge activation. The first route is to put the fixed residual sink into a parameter. The residual stripe uses the same dimensions across tokens and inputs, so a static per-dimension vector is a plausible carrier. If I multiply by a learned `λ1` before computing RMSNorm,

`PreAffineRMSNorm(x) = RMSNorm(λ1 ⊙ x)`,

then a large `λ1_d` can make the normalizer input large in dimension `d` even when the residual activation itself is ordinary. This is not the same as the usual RMSNorm weight. The usual `λ` sits after the division; it can scale a coordinate's direct contribution but cannot affect the denominator. `λ1` sits before the division, so it can change the RMS and rescale other dimensions. This relocates the outlier-driven denominator control from activation space into parameter space.

That still leaves a large internal value inside `λ1⊙x`. If I want the computation to stop depending on any large value, the model needs a direct scale path after normalization. Attention already has the template: gated attention lets a head reduce its output without donating mass to a sink. The residual-side version should take `y = RMSNorm(x)` and multiply it by a learned gate `g(y)`. The gate has to be input-dependent, because a fixed scale would only replace the average effect. It has to be cheap, because it appears at every normalization site. It has to address dimensions separately, because a single scalar per token cannot decide which coordinates should be damped.

A low-rank self-gate fits those constraints: down-project `y` from `d` to a small rank `r`, apply a nonlinearity, project back to `d`, and pass through a sigmoid:

`g = sigmoid(W_up(silu(W_down(y))))`, `y' = g ⊙ y`.

The low rank keeps the cost linear in hidden size with a small constant. The element-wise output gives each coordinate its own scale. The choice of squashing function is the part I least want to take on faith, because the whole point is to *avoid* manufacturing large values, and a gate that can exceed one would reintroduce them. So I check the candidate output function `g = sigmoid(·)` against the alternatives. With sigmoid, `g ∈ (0,1)`, so `|y'| = |g⊙y| ≤ |y|` coordinate-wise — running the module on random inputs with random small projection weights, the gate lands in roughly `(0.45, 0.54)` and `|g⊙y| ≤ |y|` holds on every coordinate. That is the property I want: the path can only attenuate, never amplify. Tanh would let `g` go negative and flip signs; SiLU and identity are unbounded above, so the gate could grow a coordinate and recreate exactly the large activations I am trying to eliminate. Only the bounded, nonnegative sigmoid gives a pure down-scaling path. With it, the model can shrink a component directly instead of inventing a residual sink to shrink it indirectly through the RMS denominator.

If the explicit gate supplies the residual rescaling function, then choices that were useful mainly because they helped manufacture outliers should become less important. Dynamic Tanh with a residual gate should recover part of the missing cross-coordinate scale path. Sigmoid GLU should no longer be starved of rescaling capacity just because it makes smaller FFN activations. Low-bit quantization should also be easier because the residual stream no longer needs a few coordinates to carry enormous scale signals.

Let me write the modules in the same shape as the Qwen3-style code I am modifying: RMSNorm before attention, gated attention that splits gate logits out of `q_proj` and applies `sigmoid(gate_score)` before `o_proj`, RMSNorm before the MLP, and the residual normalization slot filled either by a pre-affine RMSNorm or by the explicit gated RMSNorm.

Tracing the chain back, the sink denominator algebra tells me an attention sink with a small value vector scales real-token content instead of adding content. The RMSNorm bound gives the residual version: a large coordinate with a tiny post-norm weight can shrink the rest through the shared RMS denominator. Removing normalizers removes both outliers and that scale path; clipping keeps the normalizer but blocks the value it was using. So I keep the scale path and change where it lives: PreAffine moves the denominator control into a pre-RMS parameter, while GatedRMSNorm replaces the indirect denominator trick with a direct bounded element-wise gate.
