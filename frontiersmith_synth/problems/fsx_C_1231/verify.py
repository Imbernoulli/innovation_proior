#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans is an unused placeholder, per format-C contract)

Deterministic scorer for "Undocumented Logs" (log-schema-inference).

Instance (stdin, from gen.py):
  N W
  line_1        (W whitespace tokens)
  ...
  line_N        (W whitespace tokens)

Artifact (stdout): a set of templates plus an assignment of every log line
to exactly one of them.
  T
  tmpl_1        (W tokens: either a literal string, or "*" for a wildcard slot)
  ...
  tmpl_T
  a_1 a_2 ... a_N      (1-indexed template id assigned to line i, in input order)

All tokens are read positionally (whitespace-split), independent of newline
placement: 1 (T) + T*W (templates) + N (assignment) tokens total, no more,
no less.

Feasibility (strict -> Ratio 0.0 on any violation):
  * T parses as int, 1 <= T <= N.
  * exactly T*W template tokens, exactly N assignment tokens.
  * every a_i parses as int with 1 <= a_i <= T.
  * for every line i assigned to template t: at every position p where
    tmpl_t[p] != "*", line_i[p] must equal tmpl_t[p] EXACTLY (a template's
    constant claims must actually hold for every line it explains).

Objective (minimize a description-length cost; "*" is the free-form slot a
decoder must additionally be told a value for on every covered line):
  for each template t used by n_t >= 1 lines, with v_t = #wildcard positions:
      cost(t) = W + n_t * v_t          (W: pay once to state the template
                                         itself; n_t*v_t: pay to state every
                                         wildcard slot's actual value, once
                                         per covered line)
  F = sum over used templates of cost(t)

Baseline B = N * W  (cost of describing every line with its own private,
all-constant, single-use template -- i.e. explaining nothing at all).

  Ratio = min(1.0, B / (10 * F))     (F > 0 always since W >= 1)

A template that is *too general* (many wildcards covering many lines) makes
F grow past B and scores BELOW the do-nothing baseline; a partition that is
*too specific* (one template per line) exactly reproduces B (Ratio = 0.1).
Genuine structure recovery -- few templates, each with only the positions
that truly vary marked wildcard -- is what pushes F well below B.

O(N*W); pure integer/string arithmetic; deterministic.
"""
import sys

WILDCARD = "*"


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    it = iter(toks)
    try:
        N = int(next(it))
        W = int(next(it))
    except StopIteration:
        raise ValueError("truncated header")
    if N <= 0 or W <= 0:
        raise ValueError("bad header")
    lines = []
    for _ in range(N):
        row = []
        for _ in range(W):
            try:
                row.append(next(it))
            except StopIteration:
                raise ValueError("truncated instance body")
        lines.append(row)
    return N, W, lines


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        N, W, lines = read_instance(in_path)
    except Exception as e:
        print(f"BAD_INPUT: {e}")
        print("Ratio: 0.0")
        sys.exit(0)

    try:
        with open(out_path, "r") as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    if not raw:
        fail("empty output")

    try:
        T = int(raw[0])
    except ValueError:
        fail(f"non-integer T token: {raw[0]!r}")
    except OverflowError:
        fail(f"T token out of range: {raw[0]!r}")

    if T < 1 or T > N:
        fail(f"T={T} out of range [1, {N}]")

    expected = 1 + T * W + N
    if len(raw) != expected:
        fail(f"expected {expected} tokens (1 + T*W + N), got {len(raw)}")

    pos = 1
    templates = []
    for t in range(T):
        templates.append(raw[pos:pos + W])
        pos += W

    assign = []
    for i in range(N):
        tok = raw[pos]; pos += 1
        try:
            v = int(tok)  # rejects 'nan', 'inf', '3.5', '1e3', ...
        except ValueError:
            fail(f"non-integer assignment token at line {i + 1}: {tok!r}")
        except OverflowError:
            fail(f"assignment token out of representable range: {tok!r}")
        if v < 1 or v > T:
            fail(f"assignment a_{i + 1}={v} out of range [1, {T}]")
        assign.append(v - 1)  # 0-indexed template id

    # ---- strict feasibility: every constant claim must actually hold ----
    counts = [0] * T
    for i in range(N):
        t = assign[i]
        tmpl = templates[t]
        line = lines[i]
        for p in range(W):
            claim = tmpl[p]
            if claim != WILDCARD and claim != line[p]:
                fail(f"line {i + 1} pos {p}: template {t + 1} claims "
                     f"{claim!r} but line has {line[p]!r}")
        counts[t] += 1

    # ---- objective ----
    F = 0
    for t in range(T):
        n_t = counts[t]
        if n_t == 0:
            continue
        v_t = sum(1 for p in range(W) if templates[t][p] == WILDCARD)
        F += W + n_t * v_t

    B = N * W
    if F <= 0:
        fail("degenerate non-positive objective")

    sc = min(1.0, B / (10.0 * F))
    sc = max(0.0, sc)
    print(f"F={F} B={B} T_used={sum(1 for c in counts if c > 0)}")
    print("Ratio: %.6f" % sc)
    sys.exit(0)


if __name__ == "__main__":
    main()
