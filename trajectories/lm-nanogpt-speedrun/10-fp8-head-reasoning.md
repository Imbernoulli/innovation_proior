Three and a half minutes now, 1390 steps, val_loss 3.2785, about 148 ms a step. Lowering the softcap to 15
bought me the last hundred steps off the count (and the step_avg is down to 148 from the value-embed record's
174, via intervening systems tuning I won't relitigate), and at this point the step count is almost as low as
it's going to go without a deeper change — I've spent rung after rung knocking it down from 6200 to 1390, a
factor of 4.5, and each further cut is getting harder. On the product, 1390 × 148.08 ≈ 204k ms, and of those two factors the
step count is the one I've been draining — so I shift where I'm looking. The metric I really care
about is wallclock, and wallclock = steps × time-per-step. I've been draining the first factor; now the
per-step time is the thing that hasn't structurally moved, and it's worth asking where the milliseconds in a
single step actually go.

A step is a forward pass and a backward pass over a packed stream of tokens, and the time is dominated by
matmuls — attention and the MLPs across twelve blocks, plus the embedding lookup and the language-model
head at the very end. Most of those matmuls touch a 768-dim residual stream and operate on the body, and
they're already running on the tensor cores in bf16, which is fast. But there's one matmul that's
structurally different from all the others, and it's the biggest single one in the whole model: the head.
Let me actually size it against a body matmul so I'm sure it's the target. The head takes the final hidden
state — n_embd = 768 wide — and multiplies it by the head matrix out to 50304 logits, for every token in the
packed stream, on the order of 64K tokens per step. So the head forward is ~2 × 65536 × 768 × 50304 ≈ 5.1·10¹²
FLOPs. Compare a fat body matmul, the MLP's c_fc (768 → 3072): ~2 × 65536 × 768 × 3072 ≈ 3.1·10¹¹, so the head
forward alone is ~16× a single c_fc. Summing the whole body — four attention projections and two MLP matrices
per block, twelve blocks — comes to ~1.1·10¹³ FLOPs forward, so the head's 5.1·10¹² is nearly half the body's
entire matmul load in one operation. And it's doing it twice — once forward, and again in the backward pass to
get the gradient into the hidden state and the gradient of the head weights, so ~1.5·10¹³ total for the head
per step. That 50304 vocabulary dimension is what makes it fat: it's ~65× wider than the 768 the body works
in. If I'm hunting for per-step time, this is the single fattest target in the model by a wide margin, and
cutting it is a pure step_avg win that doesn't touch what the model learns.

So the question is whether I can run *that* matmul faster than bf16. And here the hardware gives me an
opening I haven't used yet. These are H100s — Hopper — and Hopper's tensor cores support FP8, eight-bit
floating point, at roughly twice the throughput of bf16. Two bytes down to one byte, double the matmul rate.
If I could do the head matmul in FP8 instead of bf16 without hurting the loss, I'd roughly halve the ~1.5·10¹³
FLOPs' worth of time on the single largest matmul in the model, on both the forward and backward passes. And
there's a second axis of saving: FP8 halves the *bytes* too. The head weight matrix is 768 × 50304 ≈ 38.6M
elements, which is ~77 MB in bf16 but ~38.6 MB in FP8, and the activations and gradients through the head
likewise halve — so a matmul that's partly memory-bandwidth-bound at this shape (the 50304-wide output is a
lot of bytes to write) gets relief on both compute and traffic. Two bytes to one byte is a 2× on the tensor
cores and a 2× on the bytes moved, both landing on the fattest op.
That's not a step-count win, it's a pure per-step-time win, and it lands squarely on the most expensive op —
exactly the kind of trade a speedrun wants once the step count has bottomed: same number of steps, each step
cheaper.

Before I chase the head specifically, the obvious greedier idea is: why not FP8 the *whole* model, every
matmul? Because the head is uniquely suited and the body is uniquely unsuited, for the same reason. FP8 works
only if the values fed to it are *bounded* and can be placed inside the format's narrow representable window,
and the head's *output* is bounded — I have a tanh softcap pinning the logits — whereas the body's internal
activations have no such bound and would routinely blow through e4m3's tiny range or flush to zero, corrupting
the matmul. On top of that, the body matmuls are individually ~16× smaller than the head, so each carries less
payoff for the same precision risk. So FP8-the-head is the high-payoff, low-risk target and FP8-the-body is
low-payoff, high-risk; I take the head and leave the body in bf16. The accounting sharpens it: the body is ~24
matmuls each ~16× smaller than the head, so FP8-ing the body would mean 24 separate scale-tuning problems and
24 separate overflow/underflow risks — since each body matmul's activations are unbounded, each needs its own
carefully-placed scale that could drift as training changes the activation statistics — for a per-matmul
payoff a sixteenth of the head's. One bounded, fat, softcapped matmul is worth far more than two dozen small,
unbounded, fragile ones, so the head is where the single FP8 investment goes.

The obvious worry even for the head is precision. FP8 is brutally low-precision — there just aren't many
bits. And the catch isn't only mantissa, it's *dynamic range*. The format I'd want for the forward pass is
e4m3 — four exponent bits, three mantissa bits — because for activations and weights I care more about having
a couple of mantissa bits of precision than about a huge exponent range, and e4m3's largest representable
value is only about 448. That's a tiny window. If the head's inputs or weights have values that wander outside
roughly ±448, they saturate or flush and the matmul is garbage. Let me confront the mantissa coarseness head
on, because three mantissa bits (plus the implicit leading 1) is only ~4 significant bits, a relative
quantization error of ~2⁻⁴ ≈ 6% per element — which sounds fatal. The reason it isn't is that the head matmul
is a dot product over the 768-dim hidden state, a *sum* of 768 products, and if the per-element quantization
errors are roughly independent they partially cancel in the sum: the relative error on the accumulated logit
scales like 6% / √768 ≈ 0.2%, not 6%. So the averaging over the contraction dimension buys back most of the
precision the format lost — the logit that comes out is good to a couple of parts in a thousand, which against
a val_loss measured to a few parts in ten-thousand is tolerable, especially since the accumulation itself is
done in higher precision (bf16 output) and the logits are then softcapped anyway. That's the quantitative
reason an 8-bit head can be safe where naïve intuition says it can't.

And this is where the work I did in the earlier rungs pays off in a way I didn't plan. I have a tanh softcap
on the logits, and I just *tightened* it to `15*tanh(logits/15)`, which means the head's outputs are bounded
by construction to ±15 — comfortably inside e4m3's ±448 window. The thing that feeds e4m3 most dangerously,
the output side of the head, is exactly the thing the softcap keeps in a controlled range. The softcap, which
I introduced as a regularizer and then tuned for step count, turns out to be the safety rail that makes an FP8
head feasible at all. So the precision risk that would normally kill an FP8 head is already mitigated by an
earlier decision, and I'm not betting on FP8 being safe in general — I'm betting that *this* head, with *this*
softcap, has outputs tame enough for e4m3.

Now, how do I actually place values inside the ±448 window? Even though the magnitudes are bounded, they're
not going to be conveniently near 448 — the inputs to the head, the hidden states, might sit around order 1 or
order 100, and the weights somewhere else, and the gradients somewhere else again, each with its own natural
scale. If I cast a value that's around 1 straight to e4m3, I'm wasting almost the entire format — all the
resolution near 448 is unused and I'm quantizing a small number coarsely (the 6%-per-element error becomes
much worse when the value sits near the bottom of the format's range where the exponent is small). The fix is
per-tensor scaling: before casting to FP8, divide each tensor by a scale factor chosen so that its typical
values land near the top of the e4m3 range, use the full eight bits, then multiply the scale back out after
the matmul. And crucially the three tensors involved — the input activations x, the head weights w, and the
gradient — have *different* natural magnitudes, so they need *different*, asymmetric scales: one x_s for the
input, one w_s for the weight, one grad_s for the gradient. Each scale is tuned to push its tensor's values
into the representable band. Concretely x_s ≈ 100/448, w_s ≈ 1.6/448, grad_s ≈ grad_scale·0.75/448 — read each
as "divide by (typical magnitude)/448" so the typical value maps to ~448, the top of the range. Let me verify
the placement literally: a hidden state of magnitude ~100 divided by x_s = 100/448 gives 100 × 448/100 = 448,
landing exactly at the top of e4m3; a weight of magnitude ~1.6 divided by w_s = 1.6/448 gives 1.6 × 448/1.6 =
448 likewise. So the scale constants are nothing mysterious — each is (that tensor's characteristic magnitude)
÷ 448, which is precisely the divisor that maps the characteristic value onto the format's ceiling and uses
the full eight-bit range instead of a coarse sliver near zero. That the three constants are so different
(100 vs 1.6 vs a fraction of the grad scale) is the whole justification for making them *asymmetric*: a single
shared scale would map at most one of the three tensors to the top of the range and leave the other two either
saturating or wasting bits. I divide by the scale
before the cast and hand the scale to the matmul so it can fold it back into the bf16 output — `torch._scaled_mm`
takes exactly these scale_a and scale_b arguments and runs the FP8 matmul on the tensor cores for me.

The backward pass needs a different format, though, and this is the standard mixed-FP8 convention. Forward
activations and weights are bounded and want mantissa, so e4m3. But gradients are different animals — their
dynamic range is much wider, they can be tiny in some directions and large in others, and if I jam them into
e4m3's narrow ±448 exponent range I'll lose the small ones to underflow. So for the gradient I switch to e5m2
— five exponent bits, two mantissa bits — trading a mantissa bit for a much larger exponent range (e5m2's max
is ~57344, ~128× e4m3's 448), so the wide spread of gradient magnitudes survives the cast. The grad_s scale
is itself built from a global `grad_scale` (grad_s = grad_scale·0.75/448), which is the standard loss-scaling
trick for low-precision gradients: gradients are the tensor most at risk of *underflow* — small components
flushing to zero in a narrow format and vanishing from the update — so I pre-scale the gradient by a large
factor before casting, lifting the small components up out of the underflow region, and divide it back out
after. The 0.75/448 places the scaled gradient near the top of e5m2's range, and folding grad_scale in ties
that placement to the run's actual gradient magnitude rather than a fixed guess. So e5m2 gives the range and
the loss-scaling makes sure the small gradients actually land inside it. The logic is a
clean split: precision where the values are bounded (forward, e4m3), range where the values are wild
(backward, e5m2). The custom op casts the incoming gradient by its own grad_s scale into e5m2 and runs the
backward `_scaled_mm` to get the gradient into the hidden state, and computes the weight gradient the same way
in FP8.

So I write a custom autograd op — a forward `mm_op` that casts x and w to e4m3 by their scales and runs the
scaled matmul to bf16, returning the cast tensors so the backward doesn't recompute them, and a backward
`mm_backward_op` that casts the gradient to e5m2 by grad_s and runs the two scaled matmuls for the input and
weight gradients. Two implementation details fall out of the precision split. The forward returns the cast
tensors x_f8, w_f8 so the backward doesn't have to re-quantize them — the cast is cheap, but re-doing it would
re-introduce a slightly different rounding, so reusing the exact forward-quantized tensors keeps forward and
backward consistent. And the forward uses `use_fast_accum=True` while the backward uses `use_fast_accum=False`:
fast accumulation is fine on the forward where the bounded, softcapped outputs tolerate the lower-precision
accumulate, but the backward wants the more careful accumulation because gradient accuracy compounds across the
run and is exactly where a precision slip would cost steps. I wire it into the head as a
`CastedLinear(model_dim, vocab_size, use_fp8=True)` with the three scales passed in. While I'm in here touching the head I'll also fold in two small companions that ride
along naturally: offsetting the logits (a constant shift so the softcapped, FP8-fed logits sit centered in a
range the head handles cleanly), and decaying the learning rate down to 0.1 of peak at the end instead of all
the way to 0.0 — the intuition being that a schedule that dies to exactly zero wastes the final steps taking
vanishing updates, whereas holding a floor of 0.1× keeps a hair of real learning in the tail rather than a
dead zero-LR stretch. Both are cheap tweaks I bundle because I'm already editing the head and the schedule;
neither is the point. Those are sidecars;
the headline, the thing that moves the wallclock, is running the biggest matmul in the model on FP8 tensor
cores.

The risk is that even with the softcap and the asymmetric scales, eight bits in the head is too coarse and
the loss creeps up above the bar, or that the e5m2 gradients are too noisy and the step count climbs to pay
for it. But the head's outputs are bounded by the softcap to ±15, the √768 averaging over the contraction
dimension buys back most of the mantissa loss, the scales put every tensor near the top of its range, and the
mixed e4m3/e5m2 convention is exactly designed for this forward-bounded/backward-wide split. If the mechanism
is right the falsifiable signature is precise: the step count should stay roughly *flat* — FP8 shouldn't
change *what* the model learns much, only how fast each step runs — while the step_avg drops because I've
halved the cost of the fattest matmul. So I expect step_avg below 148 ms at ~1390 steps and val_loss holding
under 3.28. If instead the step count *climbs* (the model needing more steps to overcome FP8 noise) or
val_loss pokes over the bar, the eight bits were too coarse and I'd widen the head back toward bf16. That's
the whole bet: same steps, cheaper steps.

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

The chain: with the step count bottomed at 1390, wallclock is now governed by per-step time, and the head is
the single fattest matmul — 768 into 50304 logits for 64K tokens, ~5·10¹² FLOPs forward and ~1.5·10¹³ with
the backward, nearly half the body's entire matmul load in one op; Hopper has FP8 tensor cores at ~2× the
bf16 rate; FP8 only works if values are bounded and placed inside e4m3's tiny ±448 window, which the softcap
(tightened to ±15 last rung) already guarantees for the head's outputs, while the √768 averaging over the
contraction dimension buys back the mantissa coarseness (6%/√768 ≈ 0.2% logit error); asymmetric per-tensor
scales x_s/w_s/grad_s push each tensor near the top of its range, e4m3 forward for the bounded activations and
e5m2 backward for the wide-range gradients, run through `torch._scaled_mm`. Same step count (~1390), but the
biggest matmul runs at twice the throughput, so the per-step time falls below 148 ms while val_loss holds
under 3.28.
