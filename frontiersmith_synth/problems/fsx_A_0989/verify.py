import os, sys, math
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

# ================= fixed apparatus constants (must match statement.md) =================
RA = 2                                   # activator disk radius (pixels), global, fixed
FEED = 1.5                               # activation gain, global, fixed
KILL_GAIN = 0.7                          # inhibition gain, global, fixed
KILL_REACH = [5, 7, 9, 11, 13, 16]       # per-formulation inhibitor reach radius (pixels)
P = len(KILL_REACH)                      # number of formulations (0 .. P-1)
L_CALIB = [10.8, 13.0, 14.6, 16.7, 18.9, 20.1]   # published expected wavelength per formulation
B_BLOCK = 48                             # region block size (pixels)
MARGIN = 8                               # crop margin removed from each block before spectral read
STEPS = 220                              # CA relaxation steps
DT = 1.0
LN_TOL = math.log(1.55)                  # log-ratio tolerance -> match hits 0 past this
BASELINE_FORMULATION = 0                 # the checker's own "do nothing" formulation index
MIN_STD = 0.05                           # below this, a crop is judged pattern-less


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


# ---------------------------- kernels & convolution ----------------------------
def disk_kernel(r):
    sz = int(math.ceil(r))
    ys, xs = np.mgrid[-sz:sz + 1, -sz:sz + 1]
    d = np.sqrt(xs ** 2 + ys ** 2)
    k = (d <= r + 0.5).astype(float)
    k /= k.sum()
    return k


def ring_kernel(r_in, r_out):
    sz = int(math.ceil(r_out))
    ys, xs = np.mgrid[-sz:sz + 1, -sz:sz + 1]
    d = np.sqrt(xs ** 2 + ys ** 2)
    k = ((d > r_in + 0.5) & (d <= r_out + 0.5)).astype(float)
    k /= max(k.sum(), 1.0)
    return k


def kernel_fft(k, N):
    kh, kw = k.shape
    pad = np.zeros((N, N))
    ch, cw = kh // 2, kw // 2
    pad[:kh, :kw] = k
    pad = np.roll(pad, (-ch, -cw), axis=(0, 1))
    return np.fft.fft2(pad)


def simulate(N, formulation_field, seed):
    """Deterministic discrete reaction-diffusion relaxation. `formulation_field` is an
    (N,N) int array giving, per pixel, which of the P published formulations governs the
    LOCAL inhibition reach at that pixel. Neighbouring pixels' actual state values still
    bleed across formulation boundaries (real spatial coupling), which is exactly the
    boundary-interaction effect an isolated per-formulation calibration cannot see."""
    rng = np.random.default_rng(seed)
    x = 0.1 * rng.standard_normal((N, N))
    Ka_fft = kernel_fft(disk_kernel(RA), N)
    Ki_ffts = [kernel_fft(ring_kernel(RA, r), N) for r in KILL_REACH]
    masks = [formulation_field == p for p in range(P)]
    for _ in range(STEPS):
        Xf = np.fft.fft2(x)
        A = np.real(np.fft.ifft2(Xf * Ka_fft))
        I = np.zeros((N, N))
        for p in range(P):
            m = masks[p]
            if m.any():
                Ip = np.real(np.fft.ifft2(Xf * Ki_ffts[p]))
                I[m] = Ip[m]
        net = FEED * A - KILL_GAIN * I
        x = x + DT * (np.tanh(net) - x)
    return x


# ---------------------------- spectral read-out ----------------------------
def block_wavelength(block):
    f = block - block.mean()
    if f.std() < MIN_STD:
        return None
    n0, n1 = block.shape
    win = np.outer(np.hanning(n0), np.hanning(n1))
    F = np.fft.fft2(f * win)
    Pw = np.abs(F) ** 2
    Pw[0, 0] = 0.0
    fx = np.fft.fftfreq(n0)
    fy = np.fft.fftfreq(n1)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    rad = np.sqrt(FX ** 2 + FY ** 2)
    mask = rad > (1.4 / max(n0, n1))
    idx = np.unravel_index(np.argmax(Pw * mask), Pw.shape)
    r = rad[idx]
    if r < 1e-9:
        return None
    return 1.0 / r


def region_measure(field, R):
    out = [[None] * R for _ in range(R)]
    for i in range(R):
        for j in range(R):
            blk = field[i * B_BLOCK:(i + 1) * B_BLOCK, j * B_BLOCK:(j + 1) * B_BLOCK]
            crop = blk[MARGIN:B_BLOCK - MARGIN, MARGIN:B_BLOCK - MARGIN]
            out[i][j] = block_wavelength(crop)
    return out


def match_score(measured, target):
    if measured is None or measured <= 0:
        return 0.0
    rel = abs(math.log(measured / target))
    return max(0.0, 1.0 - rel / LN_TOL)


def objective(formulation_field, targets, R, seed):
    N = R * B_BLOCK
    x = simulate(N, formulation_field, seed)
    meas = region_measure(x, R)
    tot = 0.0
    for i in range(R):
        for j in range(R):
            tot += match_score(meas[i][j], targets[i][j])
    return tot / (R * R)


# ---------------------------- main ----------------------------
def main():
    try:
        toks = open(sys.argv[1]).read().split()
        it = iter(toks)
        R = int(next(it))
        seed = int(next(it))
        targets = [[int(next(it)) for _ in range(R)] for _ in range(R)]
    except Exception:
        fail("bad instance")
    if R < 3 or R > 6:
        fail("bad R in instance")

    out_toks = open(sys.argv[2]).read().split()
    if len(out_toks) != R * R:
        fail("expected %d integers, got %d" % (R * R, len(out_toks)))

    field = np.zeros((R, R), dtype=np.int64)
    for k, tok in enumerate(out_toks):
        try:
            v = int(tok)
        except Exception:
            fail("non-integer token %r" % tok)
        if v < 0 or v >= P:
            fail("formulation index %d out of range [0,%d]" % (v, P - 1))
        field[k // R, k % R] = v

    # expand the per-region choice into a per-pixel formulation field
    formulation_field = np.repeat(np.repeat(field, B_BLOCK, axis=0), B_BLOCK, axis=1)

    F = objective(formulation_field, targets, R, seed)
    if not math.isfinite(F) or F < 0:
        fail("non-finite objective")

    baseline_field = np.full((R, R), BASELINE_FORMULATION, dtype=np.int64)
    baseline_pixels = np.repeat(np.repeat(baseline_field, B_BLOCK, axis=0), B_BLOCK, axis=1)
    Bv = objective(baseline_pixels, targets, R, seed)
    Bv = max(1e-6, Bv)

    F = round(F, 9)
    Bv = round(Bv, 9)
    sc = min(1000.0, 100.0 * F / Bv)
    print("F=%.9f B=%.9f Ratio: %.6f" % (F, Bv, sc / 1000.0))


if __name__ == "__main__":
    main()
