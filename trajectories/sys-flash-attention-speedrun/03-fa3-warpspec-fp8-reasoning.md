The partitioned kernel hits 72% MFU on A100 — near peak for Ampere. Now I move it to Hopper (H100), and the
number that should make me uncomfortable: the same well-tuned FlashAttention-2 kernel reaches only about 35%
utilization of the H100's tensor-core peak. Two-thirds of the new chip is idle. The kernel didn't get worse; the
hardware changed underneath it, and a kernel written for Ampere's synchronous model leaves the Hopper-specific
machinery untouched. So the question is: what did Hopper add that my kernel ignores, and how do I restructure
attention to use it? Let me catalog the new hardware and then redesign around it.

Hopper added three things that matter here. A **Tensor Memory Accelerator (TMA)**: a dedicated copy engine that
bulk-moves a whole tile between HBM and shared memory as a *single asynchronous instruction*, with the addressing
and bounds handled in hardware, so the threads that issue it don't sit there computing addresses and waiting on
loads. **WGMMA** (warpgroup matrix-multiply-async): a tensor-core GEMM issued by an entire warpgroup (4 warps,
128 threads) that reads its operands straight from shared memory and runs *asynchronously* — the issuing threads
fire it and move on, then synchronize when they need the result. And a faster, more asynchronous tensor core
generally, plus an **FP8** path at roughly double the FP16 rate. The common thread is *asynchrony*: Hopper wants
the data movement, the matmuls, and the other math all in flight at once, on different engines, overlapping. My
FA2 kernel is synchronous — a warp loads, then computes `S`, then does the softmax, then computes `P·V`, each
waiting for the last. On Ampere that was fine because there was no async copy engine and the tensor core was less
asynchronous. On Hopper it means the TMA engine is idle while the warps compute, the tensor cores are idle while
the warps do softmax, and the warps are idle while they wait on loads. Everything takes turns. The 35% is the
cost of taking turns.

Let me get precise about why the *same* kernel that hit 72% MFU on A100 collapses to ~35% on H100, because the
number is the whole motivation and I want the mechanism, not a lament. The A100's BF16 tensor peak is 312 TFLOPs/s;
the H100's is about 989 TFLOPs/s — the matmul got roughly `989/312 ≈ 3.2×` faster. But the things the synchronous
kernel *hides behind* the matmul did not speed up in step: the softmax still runs on the SFU, and the K/V loads
still cross HBM. On Ampere the matmul was slow enough that the softmax and the load latency tucked into its shadow
and the kernel read as compute-bound at 72%. Shrink the matmul time by 3.2× and those fixed-cost stalls no longer
fit in the shadow — they stick out. Put it on the roofline: at 35% the kernel delivers `0.35 · 989 ≈ 346` TFLOPs/s
on H100, which is barely above the `0.72 · 312 ≈ 225` TFLOPs/s it already did on A100. I moved to a chip with 3.2×
the tensor throughput and gained almost nothing, because I am now paying, in the open, for the serialization the old
hardware masked. The redesign has to *un-serialize*: overlap the load, the matmul, and the softmax so the tensor
cores never wait.

So the redesign principle is: stop making one warp do everything in sequence; *specialize* the warps so that
loading and computing happen concurrently on different warps, and pipeline so that while one stage runs on the
tensor cores another runs on the softmax units. Two ideas, and they compose.

First, **warp specialization** — a producer/consumer split. Instead of every warpgroup doing the full load-then-
compute cycle, dedicate one warpgroup to be the *producer*: its only job is to drive the TMA engine, issuing the
asynchronous loads of the next K and V tiles into shared memory, staying ahead of the consumers. The other
warpgroups are *consumers*: they do no loading at all; they wait for a tile to be ready in shared memory, then run
the WGMMA matmuls and the softmax on it. The producer and consumers run *at the same time* — while the consumers
crunch tile `j` on the tensor cores, the producer is already TMA-loading tile `j+1` from HBM. The data movement is
hidden completely behind the compute, which is exactly what the synchronous kernel failed to do. A circular
buffer of shared-memory stages, guarded by a pipeline of barriers, connects them: the producer `acquire`s an empty
stage, fires the TMA copy with a barrier token, and `commit`s; a consumer `wait`s on that stage's barrier, uses
the data, and `release`s the stage back to the producer. It's a classic producer/consumer queue, mapped onto
warpgroups and Hopper's async-barrier hardware.

There's a register subtlety that makes this *pay*, and it is worth doing the arithmetic because it is the reason
specialization beats a naive async loop. The SM has a fixed register file of 65536 32-bit registers. Say I run one
producer warpgroup (128 threads) and two consumer warpgroups (256 threads), 384 threads total. Split the file
evenly and every thread gets `65536 / 384 ≈ 170` registers — and 170 is not enough for a consumer that must hold a
large FP32 output accumulator plus the score tile plus the WGMMA operands, so it would spill to local memory, which
is HBM-backed and murderous in the hot loop. But the producer needs almost nothing: it computes a few TMA
descriptors and loops. Hopper lets a warpgroup *deallocate* registers it won't use and *donate* them to the others
(`warpgroup_reg_dealloc` on the producer, `warpgroup_reg_alloc` on the consumers). Shrink the producer to ~32
registers/thread — it uses `128 · 32 = 4096` — and the remaining `65536 − 4096 = 61440` registers go to the 256
consumer threads: `61440 / 256 = 240` registers each, right up against the 255-per-thread ceiling. That is the
difference between a consumer that spills at 170 and one that keeps its whole `128 × d` accumulator and score tile in
registers at 240. The specialization is not just about overlap; it is about putting the register budget where the
compute is — and a monolithic kernel where every warp both loads and computes could never do this, because every
warp would carry both the load bookkeeping *and* the fat accumulators at once, forced back to the ~170-register even
split and its spills. That is the concrete reason I split roles rather than just sprinkling async instructions into
one loop.

There is a matching shared-memory budget that only closes on Hopper. The pipeline needs several K/V stages resident
so the producer can run ahead of the consumers. For `d = 128`, `BLOCK_N = 128`, FP16, one K tile is `128 · 128 · 2 =
32` KB and one V tile another 32 KB — 64 KB per stage. Triple-buffering (3 stages, enough to keep a load always in
flight ahead of compute) is `3 · 64 = 192` KB, plus a 32 KB resident Q tile ≈ 224 KB. That does not fit in A100's
164 KB of SRAM at all — it is only possible because Hopper widened shared memory to ~228 KB/SM. And it only *works*
because the consumer accumulators live in the donated registers rather than competing for that SRAM. So the two
resource moves are coupled: the register donation clears SRAM for a deep TMA pipeline, and the deep pipeline is what
keeps the tensor cores fed. As for depth: a 32 KB tile over H100's ~3.35 TB/s HBM3 takes `32768 / 3.35e12 ≈ 10` ns
to land, and the two WGMMAs that consume a block take longer than that, so two-to-three in-flight stages are enough
to bury the load latency completely behind compute — which is exactly the "loads hidden" I could not get on the
synchronous kernel.

The kernel-level dispatch is then: read the warpgroup index; warpgroup 0 takes the producer role (dealloc
registers, loop issuing TMA loads of Q/K/V into the pipeline), the rest take the consumer role (alloc registers,
loop running the matmuls + softmax over the loaded tiles). The two roles run concurrently for the whole kernel.

Second, **overlapping the softmax with the matmuls within the consumer** — and this is the subtler win, because
even with the loads hidden, a consumer still alternates between two kinds of work that use different units. The
attention inner loop, per key block, is: WGMMA for `S = QKᵀ` (tensor cores) → softmax of `S`: max, `exp`, sum,
the online-softmax rescale (CUDA cores / SFU) → WGMMA for `P·V` (tensor cores). If I run these strictly in order,
the tensor cores idle during the softmax and the softmax units idle during the two GEMMs. But the WGMMAs are
*asynchronous* — I can *issue* the `QKᵀ` GEMM for the *next* key block and let it run on the tensor cores while
the *current* block's softmax runs on the SFU. The dependency structure allows it: the next block's `QKᵀ` depends
only on Q (resident) and the next K tile (the producer already loaded it), not on the current block's softmax. So
I pipeline across iterations: each step issues `gemm(QKᵀ)` for block `n+1` (async, doesn't block), does the
softmax for block `n` (on the SFU, overlapping the just-issued matmul), then issues `gemm(P·V)` for block `n` and
the rescale. The two matmuls and the softmax of adjacent iterations interleave so that *some* tensor-core work is
always in flight while the softmax runs. That's the intra-warpgroup overlap; with two consumer warpgroups it can
be taken further into a ping-pong where one warpgroup's softmax overlaps the other's GEMMs, scheduled by named
barriers. The whole point is to never let the tensor cores stall on the softmax — keep a matmul always running.

None of this changes the result. The producer just loads the same tiles earlier; the consumers run the same two
GEMMs and the same online softmax, only reordered so independent work overlaps; warp specialization is a
scheduling of which warpgroup does what. The output is bit-for-bit the same exact attention as FA1 and FA2. What
changes is that the TMA engine, the tensor cores, and the softmax units are now all busy at once instead of
taking turns — and the utilization climbs from ~35% toward ~75% of the H100's FP16 peak.

Now the last lever, and it *does* touch the numbers, so I have to be careful: **FP8**. Hopper's tensor cores run
FP8 (e4m3) at roughly double the FP16 rate. If I can do the attention matmuls in FP8 I nearly double throughput
again. The hazard is precision, and it is worth spelling out in the format's own bits. FP8 e4m3 is 1 sign, 4 exponent, 3
mantissa bits: its max normal is 448, and 3 mantissa bits means a relative quantization step of `2^{-3} = 12.5%`.
Attention activations are known to carry occasional large-magnitude *outlier* entries — a handful of coordinates
tens of times bigger than the median. Quantize a vector like that with a single per-tensor scale and the outlier
dictates the scale: to keep the outlier under 448 I divide everything by a big factor, which pushes the ordinary
entries down toward the bottom of the range where only a bit or two of mantissa survives — or into subnormals — and
their relative error explodes far past the nominal 12.5%. Naive FP8 attention does not lose a little accuracy; the
outliers make it lose a lot. So I can't just cast and matmul. I need to shrink the dynamic range *within each
vector* before I quantize.

Two pieces. First, per-tensor *descaling*: keep the FP8 values scaled into range and carry per-tensor scale
factors `q_descale, k_descale, v_descale` that are applied at the right points — the `QKᵀ` product is in units of
`q_descale · k_descale`, the `P·V` product in units of `v_descale`, and folding these scalars into the softmax
scaling and the final normalization recovers the correctly-scaled output. The FP8 GEMMs run fast; the descales
are a handful of scalar multiplies. (And since FP8 inputs can't faithfully represent the output, the result is
accumulated and emitted in higher precision — BF16 — not FP8.)

Second, and this is the real idea for the outliers: **incoherent processing**. The trouble with outliers is that
they're concentrated — a few entries are huge, most are small, and FP8 can't serve both at one scale. But suppose
I first multiply Q and K each by a random orthogonal matrix `R` (`Q → QR`, `K → KR`). This leaves the scores
untouched: `(QR)(KR)ᵀ = Q R Rᵀ Kᵀ = QKᵀ` for orthogonal `R`. Concretely with a Hadamard — the `d × d` matrix `H`
has entries `±1` and `H Hᵀ = d·I`, so `R = H/√d` gives `R Rᵀ = (d·I)/d = I`, exactly invertible inside the
product. What the rotation buys, at no cost to the output, is dynamic range. Take an outlier of magnitude `M` sitting in a single
coordinate of a vector; after multiplying by a dense `±1/√d` transform that concentrated energy `M²` is spread
across all `d` coordinates, so the per-coordinate magnitude falls to roughly `M/√d`. For `d = 128`, `√d ≈ 11.3`, so
the spike that forced the FP8 scale is cut by about `11×`. With the worst outlier `11×` smaller, the per-vector
max-to-median ratio collapses, the shared per-tensor scale no longer has to be blown up to cover a lone spike, and
the ordinary entries keep their mantissa bits instead of falling into subnormals — the values become "incoherent,"
no single coordinate dominating. I use the Hadamard because it is a fast structured transform, `O(d log d)` by the
butterfly, so the rotation is nearly free next to the matmul and the score math is preserved to the identity above.
The result: FP8 attention with the matmul quantization error cut by a large factor — enough that the near-2× FP8
speedup arrives without the accuracy collapse naive FP8 would suffer, the error ending up several-fold smaller than
the un-rotated FP8 baseline.

These give falsifiable directions, not a table I can read off in advance. In FP16 the starting point is ~35% util
on H100; if the load-hiding and the softmax/matmul overlap succeed, util heads toward the ~75% band the async
engines can sustain — a ceiling of `0.75/0.35 ≈ 2.1×`, which WGMMA latency, pipeline fill/drain, and imperfect
softmax burial should discount to a realized `speedup_over_prev` in the `1.5–2.0×` range with `mfu` in the
low-to-mid 70s of FP16 peak. FP8 is a separate axis: the e4m3 path roughly doubles the tensor peak, so absolute
`tflops_per_sec` should climb toward the petaflop range while `mfu` sits *lower* than FP16 — the Hadamard rotation
and the descales are overhead the FP16 path does not pay. The sharpest falsifier is the accuracy metric: if the
`√d ≈ 11×` outlier argument holds, FP8-with-rotation should show materially smaller error than plain FP8 at the
same speed; unchanged error would mean the rotation is dead weight and the FP8 case collapses to naive
quantization.

The structure that lands all this: at kernel entry the warpgroup index picks a role — warpgroup 0 deallocates its
registers and loops issuing TMA loads (`producer_acquire` a pipeline stage, fire the `tma_load` with the stage's
barrier, commit), the rest allocate registers and loop running the mainloop `mma`. Inside a consumer, each step of
`fwd_step` issues the two WGMMAs for the pipelined blocks async, releases the consumed stage, then runs the
`max`/`online_softmax`/`rescale_o` for the current block so the softmax overlaps the just-issued matmuls; on the FP8
path a register permute precedes the descale-finalize (`softmax.finalize(v_descale)`), and the interface routes
`float8_e4m3fn` inputs to a BF16 output. The full Hopper kernel is in the answer.

The arc across the three rungs stayed on one fixed output — `softmax(QKᵀ/√d)·V`, bit-for-bit — while everything
under it moved: first *what touches HBM* (never materialize the score matrix), then *how the on-chip work is
partitioned* across grid, warps, and the softmax normalization, and now *how it maps onto asynchronous silicon*
(warp-specialized TMA producers feeding WGMMA consumers, softmax overlapped with the matmuls, FP8 made accurate by
Hadamard incoherent processing). Each rung left the math alone and only rescheduled the computation onto the
hardware, and the utilization tracked that: from a bandwidth-bound floor, through 72% of Ampere's peak, to
three-quarters of an H100 in FP16 and into the petaflop range in FP8.
