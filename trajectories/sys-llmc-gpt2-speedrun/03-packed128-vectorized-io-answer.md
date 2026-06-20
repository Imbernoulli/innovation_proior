**Problem (from step 2).** With fast BF16 matmuls, the memory-bound elementwise/reduction kernels (GELU,
residual, LayerNorm, AdamW) are now a visible slice of the step — and they move one 2-byte BF16 element per
memory instruction. The GPU's widest, most efficient transaction is 128 bits (16 bytes); a 2-byte load wastes
most of that width and issues one memory instruction per element. These kernels are bandwidth-bound and not
reaching the HBM roofline.

**Key idea.** Force every memory-bound kernel to move **128 bits per load/store**. A 16-byte-aligned
`Packed128<T>` holds `16/sizeof(T)` elements (8 BF16s, 4 floats) and reinterprets the packet as an `int4` so
the load/store compiles to a single `LDG.128`/`STG.128`. Each thread processes one packet: one wide load, the
math on the packed elements (done in FP32 internally), one wide store — cutting memory instructions by up to 8×.
Use streaming cache hints (`load128cs`) for read-once data and plain cached stores for data the next kernel
will reuse.

**Why it works.** A bandwidth-bound kernel is limited by how well it saturates HBM and keeps the memory pipeline
busy; wide 128-bit transactions move the same bytes in far fewer instructions and let each kernel approach peak
bandwidth. `alignas(16)` is what permits the compiler to emit the 128-bit instruction. The arithmetic is
unchanged (FP32 internally), so the outputs match bit-for-bit and the target loss is untouched. Still open: the
vectorized kernels are still *separate launches* that round-trip tensors through HBM between them (fusion comes
next); attention still materializes T×T; the classifier still materializes (B,T,V).

**Change / code.** The `Packed128` data structure and its 128-bit load/store, plus the BF16 GELU kernel that
uses it (`llmc/cuda_utils.cuh`, `dev/cuda/gelu_forward.cu`):

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
