Lowering the softcap to 15 let the step count drop to 1390 and pulled wallclock to ~3.4 minutes, with the new record validated below the bar across 80 runs (mean 3.2791, $p=0.0001$). At this point the step count is nearly as low as it goes without a deeper change, so I shift where I am looking: wallclock is steps times time-per-step, and after rung upon rung knocking down the count it is the *per-step time* that has not moved. A step is a forward and backward pass over a packed token stream, dominated by matmuls — attention and the MLPs across twelve blocks, the embedding lookup, and the language-model head at the end. Most of those touch the 768-dim residual stream and already run on the tensor cores in bf16. But one matmul is structurally different and by far the biggest in the model: the head. It takes the final hidden state, `n_embd=768` wide, out to 50304 logits, for *every token* in a ~64K-token packed stream — roughly $64\text{K} \times 768 \times 50304$ — and it does this twice, once forward and once in the backward pass. That 50304 vocabulary dimension dwarfs the 768 the body works in; the head is the fattest target in the model by a wide margin, and it is the obvious place to cut milliseconds.

I propose to **run the head matmul in FP8**. Hopper's tensor cores do FP8 — eight-bit floating point — at roughly twice the bf16 throughput, so casting the head's inputs and weights to float8 and running the matmul there roughly halves the time of the single largest matmul, on both passes. That is not a step-count win, it is a pure per-step-time win landing on the most expensive op — exactly the trade a speedrun wants: same number of steps, each step cheaper. The obvious worry is precision, and the catch is not only mantissa but *dynamic range*. The forward format I want is **e4m3** — four exponent bits, three mantissa — because for activations and weights I care more about a couple of mantissa bits than a huge exponent range, but e4m3's largest representable value is only about 448, a tiny window: if the head's inputs or weights wander outside roughly $\pm448$ they saturate or flush and the matmul is garbage. FP8 only works if the magnitudes are *bounded* and placed inside that narrow window — and this is where the earlier rungs pay off in a way I did not plan. I have a tanh softcap on the logits, just tightened to $15\tanh(\text{logits}/15)$, so the head's outputs are bounded by construction. The thing that feeds e4m3 most dangerously — the head's output side — is exactly what the softcap keeps in a controlled range. The regularizer I introduced and then tuned for step count turns out to be the safety rail that makes an FP8 head feasible at all; I am not betting FP8 is safe in general, I am betting *this* head with *this* softcap has outputs tame enough for e4m3.

Bounded magnitudes still are not conveniently near 448 — the hidden states might sit around order 1 or 100, the weights elsewhere, the gradients elsewhere again — and casting a value near 1 straight to e4m3 wastes almost the whole format. So I use **per-tensor scaling**: before casting, divide each tensor by a scale chosen so its typical values land near the top of the e4m3 range, use the full eight bits, then multiply the scale back out after the matmul. The three tensors — input activations $x$, head weights $w$, and the gradient — have *different* natural magnitudes, so they need *asymmetric* scales, one `x_s`, one `w_s`, one `grad_s`, each tuned to push its tensor into the representable band. I divide by the scale before the cast and hand the scale to `torch._scaled_mm`, which takes `scale_a`/`scale_b`, runs the FP8 matmul on the tensor cores, and folds the scales back into a bf16 output. The backward pass needs a *different* format, the standard mixed-FP8 convention: forward activations and weights are bounded and want mantissa, so e4m3, but gradients have a much wider dynamic range — tiny in some directions, large in others — and jamming them into e4m3's narrow $\pm448$ exponent range loses the small ones to underflow. So for the gradient I switch to **e5m2** — five exponent bits, two mantissa — trading a mantissa bit for a much larger exponent range so the spread of gradient magnitudes survives the cast. Precision where the values are bounded, range where they are wild.

Concretely I write a custom autograd op: a forward `mm_op` that casts $x$ and $w$ to e4m3 by their scales, runs the scaled matmul to bf16, and returns the cast tensors so the backward does not recompute them; and a backward `mm_backward_op` that casts the incoming gradient to e5m2 by `grad_s` and runs the scaled matmuls for the input and weight gradients in FP8. It wires into the head as `CastedLinear(model_dim, vocab_size, use_fp8=True)` with the three scales tuned against the e4m3 max of 448 — `x_s` near 100/448, `w_s` near 1.6/448, `grad_s` a fraction of the global grad scale over 448 — each constant just placing its tensor's values inside the window. While I am in the head I fold in two small companions that ride along — offsetting the logits, and decaying the learning rate to 0.1 of peak at the end instead of all the way to 0.0, a hair more learning late rather than a dead tail — but those are sidecars; the headline is running the biggest matmul in the model on FP8 tensor cores. I expect the step count to stay roughly flat, since FP8 does not change *what* the model learns, only how fast each step runs, while the per-step time drops because the fattest matmul now runs at twice the throughput.

```python
# Custom FP8 matmul for the LM head: e4m3 forward, e5m2 backward gradients,
# asymmetric per-tensor scales (x_s, w_s, grad_s). E4M3_MAX ~ 448.
@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)        # forward inputs -> e4m3
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)        # weights        -> e4m3
        out = torch._scaled_mm(
            x_f8, w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8
    return impl(x, w)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)   # gradients -> e5m2 (wider range)
        grad_x = torch._scaled_mm(
            grad_f8, w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad.new_tensor(grad_s, dtype=torch.float32),
            scale_b=grad.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=False,
        )
        # ... (weight gradient computed similarly in fp8)
        return grad_x, grad_w
    return impl(g, x_f8, w_f8)

# the LM head uses fp8 with asymmetric scales tuned to the e4m3 range (448):
# self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=True, x_s=100/448, w_s=1.6/448, grad_s=grad_scale*0.75/448)
```
