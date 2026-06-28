#!/usr/bin/env python3
"""
Deterministic local scorer for ale-41: Online Bin Assignment.

This is an *offline-simulated online judge*. It reads an instance file, then
launches the solver as a subprocess and replays the stream one item at a time:
it sends the header, then for each item it sends "s v", flushes, and reads back
exactly one token -- the bin id (1..K) the solver assigns the item to, or 0 to
DROP (reject) the item. Because the solver receives item i only after it has
committed item i-1, it physically cannot peek at the future: the online /
partial-information constraint is enforced by the protocol, not by trust.

Scoring rule (matches context.md):
  * The solver outputs one decision per item, in arrival order.
  * A decision b in 1..K places the item into bin b; b == 0 drops it.
  * Each bin has a fixed capacity; the sum of sizes placed in a bin must never
    exceed its capacity. If ANY placement overflows a bin (or the output is
    malformed: out-of-range id, non-integer, missing/extra token, or the
    process crashes / times out), the score is FLOORED TO 0.
  * Otherwise the raw score is the total value of all *placed* items.

Usage:
  python3 score.py INSTANCE_FILE SOLVER_BINARY   -> prints the raw score (int).
  python3 score.py INSTANCE_FILE --baseline      -> prints the trivial baseline
        ("first bin that currently fits", else drop) raw score.

Both modes use the identical feasibility rules, so scores are comparable.
"""
import sys
import subprocess
import os
import select
import time


DECISION_TIMEOUT_SECONDS = 2.0
EXIT_TIMEOUT_SECONDS = 5.0


def read_instance(path):
    with open(path) as f:
        data = f.read().split()
    idx = 0
    K = int(data[idx]); idx += 1
    N = int(data[idx]); idx += 1
    caps = [int(data[idx + j]) for j in range(K)]; idx += K
    items = []
    for _ in range(N):
        s = int(data[idx]); v = int(data[idx + 1]); idx += 2
        items.append((s, v))
    return K, N, caps, items


def baseline_score(K, N, caps, items):
    """First bin that currently fits; otherwise drop. Always feasible."""
    rem = list(caps)
    total = 0
    for (s, v) in items:
        placed = False
        for b in range(K):
            if rem[b] >= s:
                rem[b] -= s
                total += v
                placed = True
                break
        # else: dropped, contributes nothing
    return total


def solver_score(K, N, caps, items, solver):
    """Run the solver as an interactive online judge. Returns raw score or 0."""
    proc = subprocess.Popen(
        [solver],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    rem = list(caps)
    total = 0

    def write_line(line):
        proc.stdin.write(line.encode())
        proc.stdin.flush()

    def read_line_timeout():
        """Read one solver response line, or None on timeout."""
        fd = proc.stdout.fileno()
        deadline = time.monotonic() + DECISION_TIMEOUT_SECONDS
        buf = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(fd, 1)
            if chunk == b"":
                return b"" if not buf else bytes(buf)
            buf += chunk
            if chunk == b"\n":
                return bytes(buf)
            if len(buf) > 1024:
                return bytes(buf)

    try:
        # Send header: "K N" then capacities line.
        write_line(f"{K} {N}\n")
        write_line(" ".join(str(c) for c in caps) + "\n")
        for (s, v) in items:
            write_line(f"{s} {v}\n")
            line = read_line_timeout()
            if line is None:
                return 0  # solver timed out before answering
            if line == b"":
                return 0  # solver closed/crashed before answering
            tok = line.split()
            if len(tok) != 1:
                return 0  # must emit exactly one token per item
            try:
                b = int(tok[0])
            except ValueError:
                return 0
            if b == 0:
                continue  # dropped
            if b < 1 or b > K:
                return 0  # out-of-range bin id
            if rem[b - 1] < s:
                return 0  # OVERFLOW -> infeasible -> floor 0
            rem[b - 1] -= s
            total += v
        # Clean shutdown.
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=EXIT_TIMEOUT_SECONDS)
        except Exception:
            proc.kill()
            return 0
        if proc.returncode != 0:
            return 0  # solver crashed after emitting decisions
        extra = proc.stdout.read()
        if extra and extra.split():
            return 0  # extra token(s) after the required N decisions
        return total
    except BrokenPipeError:
        return 0
    finally:
        if proc.poll() is None:
            proc.kill()


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py INSTANCE (SOLVER | --baseline)\n")
        sys.exit(1)
    inst = sys.argv[1]
    K, N, caps, items = read_instance(inst)
    if sys.argv[2] == "--baseline":
        print(baseline_score(K, N, caps, items))
    else:
        print(solver_score(K, N, caps, items, sys.argv[2]))


if __name__ == "__main__":
    main()
