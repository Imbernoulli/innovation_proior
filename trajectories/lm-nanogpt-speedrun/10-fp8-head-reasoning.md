Three and a half minutes now, 1390 steps, val_loss 3.2785. Lowering the softcap to 15 bought me another
hundred steps off the count, and at this point the step count is almost as low as it's going to go without
a deeper change. So I shift where I'm looking: the metric I really care about is wallclock, and wallclock is
steps times time-per-step. I've spent rung after rung knocking down the step count; now the per-step time
is the thing that hasn't moved, and it's worth asking where the milliseconds in a single step actually go.

A step is a forward pass and a backward pass over a packed stream of tokens, and the time is dominated by
matmuls — attention and the MLPs across twelve blocks, plus the embedding lookup and the language-model
head at the very end. Most of those matmuls touch a 768-dim residual stream and operate on the body, and
they're already running on the tensor cores in bf16, which is fast. But there's one matmul that's
structurally different from all the others, and it's the biggest single one in the whole model: the head.
The head takes the final hidden state — n_embd=768 wide — and multiplies it by the head matrix out to 50304
logits, and it does this for *every token* in the packed stream, which is on the order of 64K tokens per
step. So the head matmul is roughly 64K × 768 × 50304. That 50304 vocabulary dimension is enormous compared
to the 768 the body works in; the head is doing far more arithmetic per token than any individual block
matmul, and it's doing it twice — once forward, and again in the backward pass to get the gradient into the
hidden state and the gradient of the head weights. If I'm hunting for per-step time, this is the fattest
target in the model by a wide margin.

So the question is whether I can run *that* matmul faster than bf16. And here the hardware gives me an
opening I haven't used yet. These are H100s — Hopper — and Hopper's tensor cores support FP8, eight-bit
floating point, at roughly twice the throughput of bf16. Two bytes down to one byte, double the matmul
rate. If I could do the head matmul in FP8 instead of bf16 without hurting the loss, I'd roughly halve the
time of the single largest matmul in the model, on both the forward and backward passes. That's not a step-
count win, it's a pure per-step-time win, and it lands squarely on the most expensive op. That's exactly
the kind of trade a speedrun wants: same number of steps, each step cheaper.

The obvious worry is precision. FP8 is brutally low-precision — there just aren't many bits. And the catch
isn't only mantissa, it's *dynamic range*. The format I'd want for the forward pass is e4m3 — four exponent
bits, three mantissa bits — because for activations and weights I care more about having a couple of
mantissa bits of precision than about a huge exponent range, and e4m3's largest representable value is only
about 448. That's a tiny window. If the head's inputs or weights have values that wander outside roughly
±448, they saturate to infinity or flush, and the matmul is garbage. So FP8 only works if the magnitudes
going into it are *bounded* and I can place them inside that narrow representable window.

And this is where the work I did in the earlier rungs pays off in a way I didn't plan. I have a tanh
softcap on the logits, and I just *tightened* it — `15*tanh(logits/15)` — which means the head's outputs are
bounded by construction. The thing that feeds e4m3 most dangerously, the output side of the head, is exactly
the thing the softcap keeps in a controlled range. The softcap, which I introduced as a regularizer and
then tuned for step count, turns out to be the safety rail that makes an FP8 head feasible at all. The
bounded logits keep the head's activations from blowing through the e4m3 window. So the precision risk that
would normally kill an FP8 head is already mitigated by an earlier decision. That's the insight that makes
me willing to try this: I'm not betting on FP8 being safe in general, I'm betting that *this* head, with
*this* softcap, has outputs tame enough for e4m3.

Now, how do I actually place values inside the ±448 window? Even though the magnitudes are bounded, they're
not going to be conveniently near 448 — the inputs to the head, the hidden states, might sit around order 1
or order 100, and the weights somewhere else, and the gradients somewhere else again, each with its own
natural scale. If I cast a value that's around 1 straight to e4m3, I'm wasting almost the entire format —
all the resolution near 448 is unused and I'm quantizing a small number coarsely. The fix is per-tensor
scaling: before casting to FP8, divide each tensor by a scale factor chosen so that its typical values land
near the top of the e4m3 range, use the full eight bits, then multiply the scale back out after the matmul.
And crucially the three tensors involved — the input activations x, the head weights w, and the gradient —
have *different* natural magnitudes, so they need *different*, asymmetric scales: one x_s for the input, one
w_s for the weight, one grad_s for the gradient. Each scale is tuned to push its tensor's values into the
representable band of the format. I divide by the scale before the cast and hand the scale to the matmul so
it can fold it back into the bf16 output — `torch._scaled_mm` takes exactly these scale_a and scale_b
arguments and runs the FP8 matmul on the tensor cores for me.

The backward pass needs a different format, though, and this is the standard mixed-FP8 convention. Forward
activations and weights are bounded and want mantissa, so e4m3. But gradients are different animals — their
dynamic range is much wider, they can be tiny in some directions and large in others, and if I jam them into
e4m3's narrow ±448 exponent range I'll lose the small ones to underflow. So for the gradient I switch to
e5m2 — five exponent bits, two mantissa bits — trading a mantissa bit for a much larger exponent range, so
the wide spread of gradient magnitudes survives the cast. Forward in e4m3, backward in e5m2: precision where
the values are bounded, range where the values are wild. The custom op casts the incoming gradient by its
own grad_s scale into e5m2 and runs the backward `_scaled_mm` to get the gradient into the hidden state, and
computes the weight gradient the same way in FP8.

So I write a custom autograd op — a forward `mm_op` that casts x and w to e4m3 by their scales and runs the
scaled matmul to bf16, returning the cast tensors so the backward doesn't recompute them, and a backward
`mm_backward_op` that casts the gradient to e5m2 by grad_s and runs the two scaled matmuls for the input and
weight gradients. I wire it into the head as a `CastedLinear(model_dim, vocab_size, use_fp8=True)` with the
three scales passed in, tuned against the e4m3 max of 448: x_s near 100/448, w_s near 1.6/448, grad_s a
fraction of the global grad scale over 448. Those constants are just placing each tensor's values inside the
window. While I'm in here touching the head I'll also fold in two small companions that ride along
naturally: offsetting the logits, and decaying the learning rate down to 0.1 of peak at the end instead of
all the way to 0.0 — a hair more learning late in the run rather than a dead tail. But those are
sidecars; the headline, the thing that moves the wallclock, is running the biggest matmul in the model on
FP8 tensor cores.

The risk is that even with the softcap and the asymmetric scales, eight bits in the head is too coarse and
the loss creeps up above the bar, or that the e5m2 gradients are too noisy and the step count climbs to pay
for it. But the head's outputs are bounded by the softcap, the scales put every tensor in range, and the
mixed e4m3/e5m2 convention is exactly designed for this forward-bounded/backward-wide split. I expect the
step count to stay roughly flat — FP8 shouldn't change *what* the model learns much, only how fast each step
runs — while the per-step time drops because I've halved the cost of the fattest matmul. That's the whole
bet: same steps, cheaper steps.

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

The chain: the head is the single fattest matmul in the model — 768 into 50304 logits for 64K tokens, twice
per step; Hopper has FP8 tensor cores at ~2× the bf16 rate; FP8 only works if values are bounded and placed
inside e4m3's tiny ±448 window, which the softcap from earlier rungs already guarantees for the head's
outputs; asymmetric per-tensor scales x_s/w_s/grad_s push each tensor into range, e4m3 forward for bounded
activations and e5m2 backward for wide-range gradients, run through `torch._scaled_mm`. Same step count,
but the biggest matmul runs at twice the throughput, so the per-step time falls.
