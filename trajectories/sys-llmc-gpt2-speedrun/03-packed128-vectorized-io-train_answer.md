With the matmuls now BF16 on the tensor cores (the A100's 312-TFLOPS half-precision lane in the perf table) and the storage halved, the profile has shifted: a real slice of every step is now the elementwise and reduction kernels — GELU, the residual adds, LayerNorm, the AdamW update — which are pure memory traffic, their speed bounded entirely by how close they get to HBM bandwidth. So the sharp question is whether those kernels actually saturate memory bandwidth, and the BF16 GELU kernel shows they do not: each thread reads one BF16 element (2 bytes), computes, and writes one BF16 element. The GPU's memory system does not transact in 2-byte units. Hardware coalesces a warp's threads into transactions, and the widest, most efficient load/store instructions move 128 bits — 16 bytes — at once (`LDG.128`/`STG.128`, the width of a `float4`). Asking for 2 bytes per thread either wastes most of the transaction width or leans on the compiler to coalesce eight adjacent threads' requests into one 16-byte transaction — fragile, and still one in-flight memory instruction per *element* rather than per *16 bytes*. A bandwidth-bound kernel is bounded by the number of in-flight memory instructions as well as the bytes: too few wide ones and the memory pipeline starves; too many narrow ones and it is instruction-issue limited. I want each thread to move a *wide* chunk per instruction.

I propose the **`Packed128` 128-bit vectorized I/O** data structure: force every memory-bound kernel to load and store a 128-bit packet explicitly. With BF16 at 2 bytes, 128 bits is 8 BF16 values, so I need a type that is exactly 16 bytes wide, holds 8 BF16s, is 16-byte aligned (so the compiler is *allowed* to emit the 128-bit instruction), and that I can still index element-by-element for the math. The trick is to reinterpret the 16-byte packet as an `int4` — CUDA's native 128-bit vector type — for the load/store, because `*reinterpret_cast<const int4*>(address)` compiles to a single `LDG.128`, while keeping a typed `payload[]` array for the arithmetic. A small templated `struct alignas(16) Packed128<ElementType>` holds `payload[size]` with `size = sizeof(int4) / sizeof(ElementType)`, exposes `operator[]` for indexing and `get_bits()` to recover the `int4`, and `load128`/`store128` do the reinterpret-cast load and store.

`alignas(16)` is load-bearing: without it the compiler cannot assume the address is 16-byte aligned and refuses to emit the 128-bit instruction, falling back to narrower loads. `size` comes out to 8 for BF16 ($16/2$) and 4 for FP32 ($16/4$) automatically, so the same code vectorizes correctly whatever `floatX` is; I typedef `x128 = Packed128<floatX>` for activations and `f128 = Packed128<float>` for the FP32 reductions.

The GELU kernel then has each thread handle a *block* of `x128::size` consecutive elements: one `load128` (one 128-bit transaction), a loop over the 8 packed values doing the GELU in FP32 internally — promote each BF16 to float, compute $0.5\,x\,(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715\,x^3)))$, demote back, because the math is always done in FP32 even when storage is BF16 to keep the nonlinearity accurate — then one `store128`. The grid is sized so `thread_index \cdot x128::size` strides over the array.

Two refinements come from thinking about the cache, not just the bandwidth. The GELU input is read once and never needed again *as input*, so the load uses `load128cs`, the streaming variant (`__ldcs`), which hints the hardware not to keep it resident in L2, leaving cache room for data that *will* be reused. But the GELU *output* feeds straight into the next matmul, so the store is the plain cached `store128`, betting the next kernel wants it hot. These cache hints (`cs` = streaming/evict-first, `cg` = cache-global-bypass-L1) are the second knob the wrappers expose, and they matter precisely because these kernels are bandwidth-bound: keeping the right bytes in cache and streaming the throwaway ones is free bandwidth.

The same treatment applies everywhere the old code touched memory one element at a time: the residual add reads two `x128` packets and writes one; LayerNorm reads the row as `x128` packets to compute mean and variance, still accumulating the reductions in FP32 via `f128` because summing 768 BF16 values in BF16 would lose precision; the AdamW update reads gradients and writes weights in packed form. Each was issuing narrow 2-byte transactions and now issues 16-byte ones, cutting memory instructions by up to 8× and letting each kernel actually approach the HBM roofline. The arithmetic is unchanged — purely how the bytes are moved, math still FP32 internally — so the kernels produce the same outputs to the bit and the 3.29 target is untouched; what changes is that these now-visible memory-bound layers run near peak bandwidth instead of leaving most of the memory pipeline idle.

What remains: the elementwise layers are vectorized but still *separate* kernel launches — the residual add writes its result to HBM, then the LayerNorm kernel reads that same result right back. Two kernels touching the same tensor back-to-back pay for an HBM round-trip that fusing them would delete; Packed128 makes each kernel fast but does not remove the round-trips *between* kernels. And the two big structural costs remain: attention materializes the $O(T^2)$ score matrix, the classifier the $(B,T,V)$ logits. The next move is to stop the round-trips — fuse adjacent memory-bound ops so a tensor produced by one is consumed by the next without leaving the chip. The `Packed128` structure with its 128-bit load/store, and the BF16 GELU kernel that uses it:

```c
// forces 128-bit (LDG.128 / STG.128) loads/stores: 8 BF16 or 4 FP32 per instruction
template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;
    __device__ explicit Packed128(int4 bits) { memcpy(&payload, &bits, sizeof(bits)); }
    __device__ ElementType& operator[](int index) { return payload[index]; }
    __device__ const ElementType& operator[](int index) const { return payload[index]; }
    __device__ int4 get_bits() const { int4 bits; memcpy(&bits, &payload, sizeof(bits)); return bits; }
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}
template<class ElementType>                    // streaming hint: load and do not keep in cache
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
typedef Packed128<float> f128;     // FP32 reductions
typedef Packed128<floatX> x128;    // BF16 activations: x128::size == 8

// GELU forward, one 128-bit packet per thread, math in FP32 internally
__global__ void gelu_forward_kernel2(floatX* out, const floatX* inp, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N) {
        x128 packed_out;
        x128 packed_inp = load128cs(inp + i); // read-once: stream, don't pollute cache
        for(int k = 0; k < packed_inp.size; ++k) {
            float xi = (float)packed_inp[k];
            float cube = 0.044715f * xi * xi * xi;
            packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
        }
        // plain store (cache it): the next matmul will consume this immediately
        store128(out + i, packed_out);
    }
}
```
