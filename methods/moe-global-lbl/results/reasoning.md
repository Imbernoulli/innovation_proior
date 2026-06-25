The Switch aux loss cut the imbalance sharply against the control, so the `f·P` surrogate is clearly
doing its job of steering the router toward uniform. But the previous rung left me with a specific
suspicion about *what* it was making uniform. The counts `f_i` were measured on the micro-batch — the
small noisy slice one forward pass sees — and the loss demands that every such slice be balanced. The
cross-entropy did not come down as far as I hoped, and my guess is that some of the balancing pressure
was being spent flattening structure I wanted to keep. If that guess is right, the diagnosis points at
exactly one knob, and I want to turn only that knob: the set of tokens over which `f_i` is computed.

Before I commit to that, I should make sure I actually believe the mechanism, because "the scope is
the whole story" is the kind of thing that sounds good and turns out to be a quarter of the story. Let
me strip the problem down to something I can compute by hand. Take `N=4` experts and, just for the
arithmetic, top-1 routing so the counts are unambiguous. Suppose the corpus has two kinds of content,
and the router has learned to specialize: a "code" micro-batch sends all its tokens to experts 0 and 1,
a "prose" micro-batch sends all of its to experts 2 and 3. This is the *good* router — the one I want.
Over the corpus, every expert pulls a quarter of the work; the global usage is perfectly uniform. Now I
ask what each version of the penalty reports on this exact routing.

I need the penalty's value, so let me make it concrete. The form is `N · Σ_i f_i P_i`. To get a number
I take the gate mass to track the selection, `P_i ≈ f_i`, which turns the penalty into `N · Σ_i f_i²`.
First, what does a perfectly balanced router score? With every `f_i = 1/N`, that is `N · N · (1/N)² = 1`.
So `1.0` is the floor — and incidentally that is *why* the loss is written with the leading `N`: it pins
the balanced optimum at a fixed value independent of how many experts there are, which is what makes the
coefficient `α` mean the same thing across configurations. Good, that explains a choice I had been
carrying without justifying.

Now the two scopes on my specialized router. The micro-batch penalty evaluates each slice and averages.
Each slice has `f = (½, ½, 0, 0)` on its two active experts, scoring `4 · (¼+¼) = 2.0`; averaging the two
slices gives `2.0`. The global penalty evaluates the summed counts: `f = (¼,¼,¼,¼)`, scoring `1.0`. I ran
this through to be sure I had not fooled myself, and the numbers come out micro `= 2.0`, global `= 1.0`,
against a floor of `1.0`. That is the whole argument in two numbers. The global penalty sees my good
router and reports it as *optimal* — it has nothing to complain about. The micro-batch penalty sees the
same good router and reports `2.0`, double the floor, a penalty as large as it would assign to genuine
collapse on a slice. It literally cannot tell the code slice's legitimate skew apart from the pathological
skew of a dying expert, because both look like a slice that isn't uniform. So the micro-batch loss is
enforcing the right constraint — corpus-level balance — at the wrong scope, and the cost it charges for
specialization is not small: it is the entire gap between `1.0` and `2.0`.

That settles it, and it tells me the change is surgical. Measure `f_i` over the global batch — the union
of all the micro-batches, synchronized — instead of each micro-batch alone. Everything else in the
penalty stays: the form `α · N · Σ_i f_i P_i`, the same `α`, the same `N` factor I just re-derived the
purpose of, the detached counts and differentiable probabilities. The only thing that moves is the scope
of the frequency. Now the penalty asks one question — across all this data, is the usage uniform? — and
is silent about whether any individual slice is. A micro-batch is free to be as specialized as its content
demands, as long as the specializations of different slices cover all the experts when summed. The worked
example is exactly that freedom made arithmetic: the router that scored `2.0` under the old scope scores
the floor under the new one.

I have to be honest that in this small single-process reproduction there is only one device and the
training batch already *is* the global batch — so the way I make the distinction real is to compute the
previous rung's penalty on a handful of micro-splits of the batch (emulating the micro-batch locality),
and this rung's penalty on the full batch. The contrast is therefore between "uniformity demanded of each
quarter of the batch" and "uniformity demanded of the whole batch," the same contrast my `N=4` example
makes, just at the scale of the real task. I should temper my expectation accordingly: in the toy the two
slices were maximally different (all-code vs all-prose), which is why the gap was a clean factor of two;
in the reproduction the four micro-splits are drawn from the same distribution and are already fairly
representative of each other, so the absolute gap between the two scopes will be muted. It is a faithful
stand-in for the per-device-vs-global `f` distinction, but I would want to see it at real scale, with
genuinely heterogeneous device batches, before I'd claim the effect is large rather than just correctly
signed.

There is a complementary lever worth pairing with this, even though it is not a loss at all. DeepSeek's
auxiliary-loss-free scheme keeps a per-expert bias used *only* to break ties in the top-K selection —
added to the routing scores for ranking, but excluded from the gate weights that actually combine the
experts — and nudges it once per step by a small constant `u` in the direction that cools overloaded
experts and warms underloaded ones. It carries no gradient; it is a slow control loop on the counts that
runs beside the differentiable penalty. The reason pairing it is attractive is that the two act on
different surfaces, and I can see exactly where that matters by looking at where the loss's gradient goes.
In the real penalty `f_i` is detached, so the only gradient is through `P_i`, and it is
`∂(penalty)/∂P_i = α · N · f_i` — the downward push on an expert's gate is proportional to its *current*
load. Put numbers on it at `N=8`, `α=1e-2`: an overloaded expert at twice its fair share feels a force of
`0.02` on its gate, while an expert sitting at a tenth of its fair share feels `0.001`. So the loss does
its work mostly by pressing down on the hogs, and is nearly silent on the starved. The selection bias is
the opposite kind of actuator: it adjusts the hard top-K decision directly, by count, and can warm a cold
expert regardless of how weak the smooth gradient on it is. I will run the global-batch loss both alone
and with this bias added, to see whether the count-level controller buys anything on top.

So the rung is: the same `f·P` penalty, but with `f` measured over the global batch rather than the
micro-batch, optionally augmented by the gradient-free selection bias. On the imbalance metric I expect
to land in the same good range as the Switch loss — both are the same penalty form and both break
collapse — and the worked example gives me reason to expect the cross-entropy to come out at least as
low, since the global scope stops charging the slices for specialization. But the same gradient I just
computed names the honest caveat I will carry into the next rung. The loss equalizes the *average* usage,
yet because its force on each expert scales with that expert's load, it has almost nothing to say to the
experts that have fallen well below their fair share: the dying expert at `0.1×` fair share feels a `0.001`
push, twenty times weaker than the `0.02` the loss spends on an overloaded one — weakest exactly in the
tail where rescue matters most. Balancing the mean is not the same as resurrecting the dying, and that
gap — the under-utilized tail the global-batch loss leaves on the table — is the opening the next rung
has to attack.
