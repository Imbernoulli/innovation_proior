The matmuls are fast and the storage is BF16, so the profile has shifted: a real slice of every step is now the elementwise and reduction kernels — GELU, the residual adds, the LayerNorm, the AdamW update — and these are pure memory traffic. Their job is to read some bytes, do a trivial amount of arithmetic, and write some bytes; their speed is bounded entirely by how close they get to the GPU's HBM bandwidth. So the question is sharp: are my elementwise kernels actually saturating memory bandwidth? Let me look at what my BF16 GELU kernel does per thread.

It reads one BF16 element, computes the GELU, writes one BF16 element. A BF16 is 2 bytes. So each thread issues a 2-byte load and a 2-byte store. The problem is that the GPU's memory system doesn't transact in 2-byte units. The hardware coalesces the threads of a warp into memory *transactions*, and the widest, most efficient load/store instructions move 128 bits — 16 bytes — at once (the `LDG.128` / `STG.128` instructions, the same width as a `float4`). When each thread asks for only 2 bytes, I am either wasting most of the transaction width or relying on the compiler to coalesce 8 adjacent threads' 2-byte requests into one 16-byte transaction — which it can do for a perfectly contiguous access pattern, but it is fragile and it still costs me one in-flight memory instruction per *element* rather than per *16 bytes*. The number of memory instructions in flight, not just the bytes, is part of what bounds a bandwidth-limited kernel: too few wide instructions and the memory pipeline isn't kept busy; too many narrow ones and I am instruction-issue limited. I want each thread to move a *wide* chunk per instruction.

The clean way to force this is to make each thread load and store a 128-bit packet explicitly. With BF16 elements at 2 bytes each, 128 bits is 8 BF16 values. So I want a type that is exactly 16 bytes wide, holds 8 BF16s, is 16-byte aligned (so the compiler is *allowed* to emit the 128-bit instruction), and that I can index element-by-element for the math. The trick is to reinterpret the 16-byte packet as an `int4` (CUDA's native 128-bit vector type) for the load/store — because `*reinterpret_cast<const int4*>(address)` compiles to a single `LDG.128` — while keeping a typed `payload[]` array for the arithmetic. A small templated struct does exactly this:

    template<class ElementType>
    struct alignas(16) Packed128 {
        __device__ ElementType& operator[](int index) { return payload[index]; }
        __device__ int4 get_bits() const { int4 bits; memcpy(&bits, &payload, sizeof(bits)); return bits; }
        static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
        ElementType payload[size];
    };
    template<class ElementType>
    __device__ Packed128<ElementType> load128(const ElementType* address) {
        return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
    }
    template<class ElementType>
    __device__ void store128(ElementType* target, Packed128<ElementType> value) {
        *reinterpret_cast<int4*>(target) = value.get_bits();
    }

`alignas(16)` is load-bearing: without it the compiler cannot assume the address is 16-byte aligned and will refuse to emit the 128-bit instruction, falling back to narrower loads. `size` comes out to 8 for BF16 (16/2) and 4 for FP32 (16/4) automatically, so the same code vectorizes correctly whatever `floatX` is. I typedef `x128 = Packed128<floatX>` for the activation precision and `f128 = Packed128<float>` for the FP32 reductions.

Now the GELU kernel becomes: each thread handles a *block* of `x128::size` consecutive elements. It does one `load128` (one 128-bit transaction), loops over the 8 packed values doing the GELU math in FP32 internally (promote each BF16 to float, compute, demote back — the math should always be done in FP32 even when storage is BF16, to keep the nonlinearity accurate), packs the 8 results, and does one `store128`. The grid is sized so that `thread_index · x128::size` strides over the array:

    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N) {
        x128 packed_out;
        x128 packed_inp = load128cs(inp + i); // load and do not keep in cache
        for(int k = 0; k < packed_inp.size; ++k) {
            float xi = (float)packed_inp[k];
            float cube = 0.044715f * xi * xi * xi;
            packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
        }
        store128(out + i, packed_out);
    }

Two refinements fall out of thinking about the cache, not just the bandwidth. The GELU input is read once and, for this kernel, never needed again as *input* — so I use `load128cs`, the streaming variant (`__ldcs`), which hints the hardware not to keep it resident in L2, leaving cache room for data that *will* be reused. But the GELU *output* feeds straight into the next matmul, so for the store I use the plain `store128` (cache it) rather than the streaming `store128cs`, betting the next kernel wants it hot in cache. These cache hints (`cs` = streaming/evict-first, `cg` = cache-global-bypass-L1) are the second knob the Packed128 wrappers expose, and they matter precisely because these kernels are bandwidth-bound: keeping the right bytes in cache and streaming the throwaway ones is free bandwidth.

The same Packed128 treatment applies everywhere the old code touched memory one element at a time: the residual add reads two x128 packets and writes one; LayerNorm reads the row as x128 packets to compute mean and variance (still accumulating the reductions in FP32 via f128, because summing 768 BF16 values in BF16 would lose precision); the AdamW update reads gradients and writes weights in packed form. Each of these was issuing narrow 2-byte transactions and is now issuing 16-byte ones, cutting the number of memory instructions by up to 8× and letting each kernel actually approach the HBM roofline.

Does this change the result? No — it is purely how the bytes are moved. The arithmetic is unchanged (and the internal math is still FP32), so the kernels produce the same outputs to the bit, and the correctness check against the reference is unaffected. What changes is that the memory-bound layers, which became a visible fraction of the step once the matmuls got fast, now run at close to peak bandwidth instead of leaving most of the memory pipeline idle.

What's left after this. The elementwise layers are vectorized, but they are still *separate* kernel launches — the residual add writes its result to HBM, then the LayerNorm kernel reads that same result right back. Two kernels that touch the same tensor back-to-back are paying for a round-trip through HBM that fusing them into one kernel would eliminate; the Packed128 machinery makes each kernel fast but doesn't remove the round-trips *between* kernels. And the two big structural memory costs are still there: attention materializes the O(T²) score matrix, and the classifier materializes the (B,T,V) logits. The next obvious move is to stop the round-trips — fuse adjacent memory-bound ops so a tensor produced by one is consumed by the next without ever leaving the chip.
