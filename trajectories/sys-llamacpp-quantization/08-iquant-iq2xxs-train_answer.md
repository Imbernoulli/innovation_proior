The imatrix changed the picture: weighting each weight's error by its activation importance gave a free quality gain at every width — q4_K_S's 8B gap dropping from 0.269 to 0.178 at byte-identical size — and, crucially, it made the *very* low widths stop being garbage. With importance telling the quantizer which coordinates to keep and which to throw away, sub-3-bit models hold together where they used to collapse. So the question is how low the bits-per-weight can go while the model stays coherent, and I want to push toward 2 bits. But here is the wall: every format so far, Q4_0 through the imatrix Q4_K, is a *scalar* quantizer — it treats each weight independently, picking a per-block scale and rounding each weight on its own. Two bits is four levels per weight, and scalar rounding there falls apart for a plain reason. When I quantize eight weights independently to 2 bits each I am placing a point in 8-dimensional space, constrained to the corners of a regular grid: every combination of four levels in each of eight coordinates is allowed, $4^8 = 65536$ points tiling the cube uniformly. But the eight weights are not uniform in that cube — after scaling they cluster, most 8-tuples near the origin, a few with one or two large components, almost none with all eight large. A uniform grid spends as many codepoints on the empty far corners as on the dense center. Independence forces a uniform tiling onto data that is not uniform, and that is the waste scalar quantization cannot escape at 2 bits.

I propose **IQ2_XXS**, a codebook (vector) quantizer: stop quantizing weights one at a time and quantize them in *groups*, to a dictionary of points chosen to match where the data actually lives. Take a group of eight weights as a single 8-D vector and, instead of rounding each coordinate, find the nearest entry in a fixed set of 256 carefully chosen 8-D points and store its 8-bit index — eight weights, one byte for the group's *shape*. The 256 points need not tile the cube; they concentrate where 8-tuples of weights actually fall, so no codepoints are wasted on empty corners. To build the codebook, each coordinate carries a two-bit code unpacked by the odd-positive-magnitude rule $2l+1$, so the generic map permits $\{1, 3, 5, 7\}$; the signs are factored out separately. Positive magnitudes let me store signs in a separate bitfield (a group of 8 has 8 signs, but parity lets me store only 7 and recover the eighth), and odd values place the representable magnitudes at bin centers. For the 256-point IQ2_XXS grid I do not take all $4^8$ two-bit tuples but the curated low-magnitude subset in $\texttt{kgrid\_2bit\_256}$ — exactly where scaled weight 8-tuples are dense.

The block layout makes this nearly true two-bit storage. A super-block is 256 weights = 32 groups of 8, organized as 8 sub-blocks of 32 (each sub-block = 4 groups). Per super-block I store one fp16 $d$, and for each sub-block a 32-bit word packing the four groups' 8-bit grid indices and a second 32-bit word packing the four groups' 7-bit sign masks plus a 4-bit per-sub-block scale in its top bits. That is $\texttt{QK\_K}/8 \cdot \text{sizeof}(\texttt{uint16\_t}) = 64$ bytes of indices/signs/scales plus one fp16, i.e. 66 bytes over 256 weights → **2.0625 bits/weight** raw; the published whole-model 8B artifact lands at 2.3824 bpw after amortization. The scale hierarchy is three deep: each weight 8-tuple → codebook index; each sub-block's magnitude → a 4-bit scale via $d = \text{max\_scale}/31$ and index $\text{round}(0.5\,(\text{scale}_{ib}/d - 1))$; the super-block → fp16 $d$. Reconstruction is just a gather: read the 8-bit index, fetch the eight codebook magnitudes, apply the seven stored signs (and the parity-recovered eighth), and multiply by the sub-block scale times $d$ — cheap enough to stay usable on commodity hardware.

The encoder is the hard part, and it is where the imatrix is not optional but *required* — the format asserts $\texttt{quant\_weights}$ must be present, because at 2 bits there is no quality without knowing which weights matter. For each group of 8 it takes absolute values and records the signs with the parity trick: if an odd number are negative, it flips the sign of the *least-important* entry (the one minimizing $\texttt{weight}\cdot x^2$) so the stored 7-bit mask has even parity, recoverable at decode. Then it searches for the best codebook point and scale jointly: get an initial scale from $\texttt{make\_qp\_quants}$ on the magnitudes, sweep the inverse-scale over a small grid, and for each candidate scale the eight magnitudes, round each to a level in $\{0,1,2\}$ (the codebook's magnitude indices before $2l+1$), and look up the resulting tuple in $\texttt{kmap\_q2xs}$. If that exact tuple is a codebook point it is used; if not — and most rounded tuples are *not* among the curated 256 — it falls back to $\texttt{iq2\_find\_best\_neighbour}$, which searches the precomputed neighbor list for the codepoint minimizing the *importance-weighted* distance

$$\sum_i w_i\,(\text{scale}\cdot g_i - x_i)^2$$

to the actual weights. The importance weights $w_i = \texttt{qw}_i\cdot\sqrt{\sigma^2 + x_i^2}$ are the same activation-aware weights from the imatrix, so the nearest-codebook search is nearest in the metric that *counts* — protecting the high-activation coordinates and letting the low-activation ones snap to whatever codepoint is cheap. The candidate (codepoint, scale) with the best weighted fit is kept, then refined once more before packing.

Step back and this is the synthesis of the whole ladder: per-block adaptation from the legacy formats, super-block metadata sharing from the k-quants, and activation importance from the imatrix, all of it — but now applied to *groups* of weights against a *data-fit codebook* instead of single weights against a uniform grid, which is the only way the round-off survives 2 bits. This is the bottom of the bits axis I set out to reach. The bar is qualitative-into-quantitative: a 2-bit scalar code diverges, so the test is whether the codebook scheme produces a *coherent* model near the two-bit floor — a perplexity high relative to the 4-bit k-quants, as it must be, but finite. The raw format sits at 2.0625 bpw, the published 8B artifact at 2.3824 bpw, with the gap to the 4-bit formats largest where the activation distribution is benign (the importance signal cannot save much) and smallest where it is skewed (where dropping unimportant coordinates costs nothing). And the mixed-allocation policy still rides on top: the most sensitive tensors can be assigned a safer type than the base IQ2_XXS format. The clean success is a model that fits on hardware that could not hold any earlier rung and still writes coherent text — the lowest-bits-per-weight rung the perplexity-vs-bpw frontier reaches.

```c
// (Almost) "true" 2-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 2.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == sizeof(ggml_half) + QK_K/8*sizeof(uint16_t), "wrong iq2_xxs block size/padding");
```

```c
    static const uint16_t kgrid_2bit_256[256] = {
            0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
          100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
         1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
         1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
         2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
         4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
         5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
         8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
        10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
        16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
        17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
        20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
        22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
        25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
        33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
        37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
    };
```

```c
    const uint16_t * kgrid = type == GGML_TYPE_IQ2_XXS ? kgrid_2bit_256 :
                             type == GGML_TYPE_IQ2_XS  ? kgrid_2bit_512 :
                             type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? kgrid_1bit_2048 : kgrid_2bit_1024;
    uint64_t * kgrid_q2xs;
    int      * kmap_q2xs;
    uint16_t * kneighbors_q2xs;

    //printf("================================================================= %s(grid_size = %d)\n", __func__, grid_size);
    uint64_t * the_grid = (uint64_t *)malloc(grid_size*sizeof(uint64_t));
    for (int k = 0; k < grid_size; ++k) {
        int8_t * pos = (int8_t *)(the_grid + k);
        for (int i = 0; i < 8; ++i) {
            int l = (kgrid[k] >> 2*i) & 0x3;
            pos[i] = 2*l + 1;
        }
    }
    kgrid_q2xs = the_grid;
    iq2_data[gindex].grid = the_grid;
    kmap_q2xs = (int *)malloc(kmap_size*sizeof(int));
    iq2_data[gindex].map = kmap_q2xs;
```

```c
static int iq2_find_best_neighbour(const uint16_t * GGML_RESTRICT neighbours, const uint64_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, const float * GGML_RESTRICT weight, float scale, int8_t * GGML_RESTRICT L) {
    int num_neighbors = neighbours[0];
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}
```

```c
static void quantize_row_iq2_xxs_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights) {

    const int gindex = iq2_data_index(GGML_TYPE_IQ2_XXS);

    const uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    const int      * kmap_q2xs       = iq2_data[gindex].map;
    const uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n/QK_K;

    block_iq2_xxs * y = vy;

    float scales[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float  waux[32];
    uint8_t block_signs[4];
    uint32_t q2[2*(QK_K/32)];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        memset(q2, 0, QK_K/4);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = sumx2/QK_K;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float * xb = xbl + 32*ib;
            const float * qw = quant_weights + QK_K*ibl + 32*ib;
            for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float scale = make_qp_quants(32, kMaxQ+1, xval, (uint8_t*)L, weight);
            float eff_max = scale*kMaxQ;
            if (eff_max <= 0) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float best = 0;
            for (int is = -6; is <= 6; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/eff_max;
                float this_scale = 1/id;
                for (int k = 0; k < 4; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    memcpy(L, Laux, 32);
                }
            }
            if (scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 4; ++k) {
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 2*i);
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q2xs + grid_index);
                    for (int i = 0; i < 8; ++i) L[8*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                q2[2*ib+0] |= ((uint32_t) grid_index << 8*k);
                q2[2*ib+1] |= (block_signs[k] << 7*k);
            }
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            memset(y[ibl].qs, 0, QK_K/4);
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            q2[2*ib+1] |= ((uint32_t)l << 28);
        }
        memcpy(y[ibl].qs, q2, QK_K/4);
    }
}
```

```c
void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}
```
