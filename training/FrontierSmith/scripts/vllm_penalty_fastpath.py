"""Incremental replacement for vLLM's per-step penalty tensor construction.

WHY
---
`vllm/v1/sample/ops/penalties.py::_convert_to_tensors` rebuilds a padded
[batch, max_output_len] int64 tensor from a list-of-lists on EVERY decode step,
then copies it to the GPU. The lists grow by ~one token per step, so the work is
O(batch x length) per step to communicate O(batch) new information. vLLM's own
comment says: "The penalties implementation is currently quite inefficient and
will be reworked anyhow."

Measured here (vocab 248320, int64):
    B=128 L=32768 -> 162 ms PER DECODE STEP (32 MiB rebuilt + pinned + copied)
~70% of our 240 ms decode step; py-spy caught the sampler in this path in 12 of
14 samples while the GPUs idled at 0-37%. It is also why throughput scaled as
1/context_length instead of with batch size.

DESIGN (v2, after an adversarial review found two ways v1 could change sampling)
-------------------------------------------------------------------------------
* Persistent pinned CPU buffer; only the new tail of each row is written.
* The H2D copy is BLOCKING. v1 used non_blocking=True from a buffer it mutated on
  the next call: CUDA stream ordering protects the GPU destination but NOT the
  host source, so a still-in-flight transfer could read tokens from the next
  step. The copy is ~32 MiB (~2 ms) -- irrelevant next to the 162 ms it replaces.
* vLLM edits token lists IN PLACE at the tail (async-scheduling placeholder
  repair rewrites a -1; streaming clears the list). Identity+length alone misses
  a same-length tail edit, so every step also re-copies the last TAIL_RECHECK
  tokens of each row. Cost stays O(batch); any tail-local mutation is absorbed.
  A shrink or a different list object still triggers a full row rebuild.
* install() keeps the stock function as `_fs_original_convert` so selftest()
  always compares against the REAL original (v1's installer overwrote it, so the
  deployed selftest silently compared the fast path against itself).

The result is bit-identical to stock: same dtype, shape, padding and device.
This is a pure speed patch; sampling is unchanged.

USAGE
-----
    scripts/apply_vllm_penalty_fastpath.sh          # install (env-gated)
    FS_VLLM_PENALTY_FASTPATH=1 <run>                # activate
    python -c "import sys;sys.path.insert(0,'scripts');import vllm_penalty_fastpath as f;f.install();f.selftest()"
    scripts/apply_vllm_penalty_fastpath.sh --revert # back to stock
"""

from __future__ import annotations

import numpy as np
import torch

# Re-copy this many trailing tokens per row every step, so an in-place tail edit
# (vLLM's -1 placeholder repair) can never be missed. It must exceed the longest
# suffix vLLM can rewrite in one step: with async scheduling that is 1 ordinary
# placeholder + up to max_num_speculative_tokens (vLLM caps at 128, see
# v1/sample/rejection_sampler.py) = 129. 192 leaves margin and costs
# ~192 x batch int64 writes per step (0.2 MB at batch 128) -- nothing.
TAIL_RECHECK = 192

import threading as _threading

# One state per (process, device): colocated engines in the SAME interpreter
# would otherwise invalidate each other's cache every step -- and race while
# filling the numpy view. Our four engines live in separate processes, but the
# lock and the per-device keying make the patch safe if that ever changes.
_lock = _threading.Lock()
_states: dict = {}


def _state_for(device) -> dict:
    key = str(device)
    st = _states.get(key)
    if st is None:
        st = {"buf": None, "ids": None, "lens": None, "cap": 0, "rows": 0}
        _states[key] = st
    return st


_state: dict = {"buf": None, "ids": None, "lens": None, "cap": 0, "rows": 0}  # legacy alias for tests


def _reset() -> None:
    with _lock:
        _states.clear()


def _fast_convert(output_token_ids, vocab_size: int, device) -> torch.Tensor:
    with _lock:
        return _fast_convert_locked(output_token_ids, vocab_size, device)


def _fast_convert_locked(output_token_ids, vocab_size: int, device) -> torch.Tensor:
    st = _state_for(device)
    rows = len(output_token_ids)
    if rows == 0:
        # Still invalidate: a cached row whose list object is later mutated in
        # place (same length) would otherwise be served stale on its return.
        if st["ids"] is not None:
            st["ids"] = [None] * st["rows"]
            st["lens"] = [0] * st["rows"]
        return torch.empty((0, 0), dtype=torch.int64, device=device)
    # NOTE: stock returns width 0 when every row is empty -- do NOT floor this at 1.
    max_len = max((len(x) for x in output_token_ids), default=0)

    if st["buf"] is None or st["rows"] < rows or st["cap"] < max_len:
        cap = max(max_len * 2, 1024)
        nrows = max(rows, st["rows"], 1)
        buf = torch.full((nrows, cap), vocab_size, dtype=torch.int64)
        if torch.cuda.is_available():
            buf = buf.pin_memory()
        st.update(buf=buf, cap=cap, rows=nrows, ids=[None] * nrows, lens=[0] * nrows)

    buf, ids, lens, cap = st["buf"], st["ids"], st["lens"], st["cap"]
    np_buf = buf.numpy()

    for i in range(rows):
        toks = output_token_ids[i]
        n = len(toks)
        prev = lens[i]
        # Full rebuild when the slot holds a different request, or the row shrank
        # (streaming clears the list), or nothing was cached yet.
        if ids[i] is not toks or n < prev:
            start = 0
        else:
            start = max(0, prev - TAIL_RECHECK)  # absorb in-place tail edits
        if n > start:
            np_buf[i, start:n] = np.fromiter(toks[start:], dtype=np.int64, count=n - start)
        if n < cap:
            np_buf[i, n:] = vocab_size  # exact padding, including after a shrink
        ids[i] = toks
        lens[i] = n
    for j in range(rows, st["rows"]):  # rows that vanished this step
        ids[j] = None
        lens[j] = 0

    # BLOCKING copy: the source is reused next step (see DESIGN). On CPU, .to()
    # would return a VIEW of the cache -- the caller does masked_fill_ on the
    # result, which would corrupt it -- so clone there.
    out = buf[:rows, :max_len]
    return out.clone() if out.device == torch.device(device) else out.to(device)


def install() -> bool:
    """Monkey-patch vLLM's _convert_to_tensors, preserving the original."""
    from vllm.v1.sample.ops import penalties as P

    if getattr(P, "_fs_fastpath_installed", False):
        return True
    if not hasattr(P, "_fs_original_convert"):
        P._fs_original_convert = P._convert_to_tensors
    P._convert_to_tensors = _fast_convert
    P._fs_fastpath_installed = True
    return True


def _stock():
    """The REAL stock implementation, whatever has been patched over it."""
    import importlib

    P = importlib.import_module("vllm.v1.sample.ops.penalties")
    orig = getattr(P, "_fs_original_convert", None)
    if orig is None or orig is _fast_convert:
        raise RuntimeError(
            "cannot find the stock _convert_to_tensors -- refusing to 'verify' the fast path "
            "against itself (install() must run before selftest, and the installer must keep "
            "_fs_original_convert)"
        )
    return orig


def selftest(trials: int = 400, seed: int = 0, device: str = "cpu") -> None:
    """Bit-exactness vs the stock implementation across every state vLLM can
    produce: growth, slot reuse, shrink, ragged, empty rows, zero-row batch,
    batch-size changes, in-place TAIL edits (-1 placeholder repair) and
    speculative multi-token appends."""
    orig = _stock()
    dev = torch.device(device)
    rng = np.random.default_rng(seed)
    V = 1000

    def check(rows, label):
        got = _fast_convert(rows, V, dev)
        want = orig(rows, V, dev)
        assert got.shape == want.shape, f"{label}: shape {tuple(got.shape)} != {tuple(want.shape)}"
        assert torch.equal(got.cpu(), want.cpu()), f"{label}: values differ"

    # --- deterministic transitions that the randomized loop is unlikely to hit
    _reset()
    a = [1, 2, 3]
    rows = [a]
    check(rows, "init")
    a[-1] = -1                      # in-place SAME-LENGTH tail edit (placeholder)
    check(rows, "in-place tail -1")
    a[-1] = 7                       # placeholder repaired in place
    check(rows, "in-place tail repair")
    a.extend([8, 9, 10])            # speculative multi-append
    check(rows, "multi-append")
    del a[:]                        # streaming clear (same object, shrink to 0)
    check(rows, "in-place clear")
    a.extend([4, 5])
    check(rows, "regrow after clear")
    rows.append([9] * 40)           # batch grows
    check(rows, "batch grow")
    rows.pop()                      # batch shrinks
    check(rows, "batch shrink")
    check([], "zero-row batch")
    check([[], []], "all-empty rows")

    # --- Codex review probes (v2 findings 1 and 4) --------------------------
    _reset()
    deep = list(range(300))
    rows = [deep]
    check(rows, "deep row cached")
    for k in (33, 64, 129):                       # same-length repair k from the tail
        for j in range(1, k + 1):
            deep[-j] = 900 + j
        check(rows, f"same-length repair of the last {k} tokens")
    _reset()
    keep = list(range(100))
    rows = [keep]
    check(rows, "cache a row")
    check([], "zero-row batch invalidates")
    keep[0] = 777                                  # in-place edit while absent
    check([keep], "row returns after an in-place edit")

    # --- randomized soak
    rows = [[] for _ in range(8)]
    for t in range(trials):
        if rng.random() < 0.08:                       # batch size changes
            if len(rows) > 2 and rng.random() < 0.5:
                rows.pop()
            else:
                rows.append(list(rng.integers(0, V, size=int(rng.integers(0, 6)))))
        for i in range(len(rows)):
            r = rng.random()
            if r < 0.05:
                rows[i] = list(rng.integers(0, V, size=int(rng.integers(0, 5))))   # new request
            elif r < 0.10 and rows[i]:
                rows[i][-1] = -1                                                    # tail placeholder
            elif r < 0.13 and rows[i]:
                del rows[i][:]                                                      # in-place clear
            elif r < 0.20:
                rows[i].extend(int(x) for x in rng.integers(0, V, size=2))          # speculative
            elif r < 0.95:
                rows[i].append(int(rng.integers(0, V)))
        check(rows, f"random step {t}")
    print(f"selftest OK on {device}: {trials} randomized steps + 10 deterministic transitions, bit-identical")
