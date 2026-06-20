**Problem (from step 9).** With the lower softcap landing 3.2785 in 1390 steps (≈3.4 min), the step count
is nearly bottomed out, so wallclock is now governed by per-step time. The single fattest matmul in the
model is the language-model head: the final hidden state (n_embd=768) times the head matrix out to 50304
logits, for every token in a ~64K-token packed stream, forward and backward, in bf16. It dwarfs every body
matmul and is the obvious place to cut milliseconds.

**Key idea (run the head matmul in FP8).** Hopper's tensor cores do FP8 at ~2× the bf16 throughput, so cast
the head's inputs and weights to float8 and run the matmul via `torch._scaled_mm`. Use the standard mixed-
FP8 convention: **e4m3** on the forward pass (more mantissa bits, for the bounded activations and weights),
**e5m2** on the backward pass (more exponent range, for the wide dynamic range of gradients). Because FP8's
window is tiny (e4m3 max ≈ 448), use **asymmetric per-tensor scales** — separate `x_s`, `w_s`, `grad_s`,
each chosen to land its tensor inside the representable range; divide by the scale before the cast and hand
the scale to `_scaled_mm`. The tanh softcap from the earlier rungs is what makes the head's outputs bounded
enough for e4m3 to be safe. (Bundled in the same record: offset logits and decaying the LR to 0.1 instead of
0.0 — minor companions; FP8 is the headline.)

**Why it works.** FP8 doesn't change *what* the head learns, only how fast it computes, so the step count
stays roughly flat while the most expensive matmul runs at twice the throughput, on both forward and
backward — a pure per-step-time cut on the model's biggest op. The precision risk that normally sinks an FP8
head is already neutralized: the softcap bounds the logits so the head's outputs stay inside e4m3's ±448
window, and the asymmetric scales place each of input/weight/gradient where it uses the full eight bits.
Forward e4m3 keeps mantissa where values are bounded; backward e5m2 keeps range where gradients are wild.

**Change / code.** Custom FP8 autograd op for the head: cast x and w to e4m3 by x_s/w_s and run the scaled
matmul forward; cast the gradient to e5m2 by grad_s and run the scaled matmuls backward; wire it into the
head as `CastedLinear(..., use_fp8=True)` with the three scales tuned to 448.

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
