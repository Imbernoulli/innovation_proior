#!/usr/bin/env python3
import random
import subprocess
import sys
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOL_CPP = ROOT / "sol.cpp"
SOL_BIN = ROOT / "sol_independent_test"


def oracle(words, s):
    """Reachability oracle over positions, expanding by full words."""
    if not s:
        return "YES"
    seen = [False] * (len(s) + 1)
    seen[0] = True
    q = deque([0])
    words = list(words)
    while q:
        at = q.popleft()
        for w in words:
            nxt = at + len(w)
            if nxt <= len(s) and not seen[nxt] and s.startswith(w, at):
                if nxt == len(s):
                    return "YES"
                seen[nxt] = True
                q.append(nxt)
    return "NO"


def serialize(words, s):
    out = [str(len(words))]
    out.extend(words)
    if s:
        out.append(s)
    return "\n".join(out) + "\n"


def run_sol(inp):
    proc = subprocess.run(
        [str(SOL_BIN)],
        input=inp,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solution exited {proc.returncode}: {proc.stderr}")
    return proc.stdout.strip()


def edge_cases():
    return [
        ([], ""),
        ([], "a"),
        (["a"], ""),
        (["abc"], "abc"),
        (["abc"], "ab"),
        (["a", "b"], "c"),
        (["le", "leet", "code", "etcode", "leetcode"], "leetcode"),
        (["ab", "abc", "cd", "d"], "abcd"),
        (["a", "aab", "ab", "b"], "aab"),
        (["a", "aa", "aaa"], "aaaaaaaaaaa"),
        (["aaaa", "aaa", "a"], "aaaaaaaaaa"),
        (["cat", "cats", "and", "sand", "dog"], "catsanddog"),
        (["cat", "cats", "and", "sand", "dog"], "catsandog"),
        (["x", "xx", "xxx", "xxxx"], "xxxxxxxxxxxxx"),
        (["a", "a", "aa"], "aaa"),
    ]


def random_case(rng):
    alphabet = "abc"[: rng.randint(1, 3)]
    mode = rng.randrange(6)

    if mode == 0:
        return [], "" if rng.random() < 0.5 else "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))

    n = rng.randint(1, 10)
    words = []
    for _ in range(n):
        length = rng.randint(1, 5)
        words.append("".join(rng.choice(alphabet) for _ in range(length)))

    if mode == 1:
        parts = rng.randint(0, 8)
        s = "".join(rng.choice(words) for _ in range(parts))
    elif mode == 2:
        s = "a" * rng.randint(0, 18)
        words.extend("a" * k for k in range(1, rng.randint(2, 8)))
    elif mode == 3:
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 18)))
        if s and rng.random() < 0.7:
            s = s[: rng.randrange(len(s))] + "z" + s[rng.randrange(len(s)) + 1 :]
    elif mode == 4:
        traps = [
            (["ab", "abc", "cd", "d"], "abcd"),
            (["go", "goo", "good", "d", "dog"], "gooddog"),
            (["a", "aab", "ab", "b"], "aab"),
        ]
        return rng.choice(traps)
    else:
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 18)))

    return words, s


def main():
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-pipe", str(SOL_CPP), "-o", str(SOL_BIN)],
        check=True,
    )

    cases = edge_cases()
    rng = random.Random(20260628)
    while len(cases) < 750:
        cases.append(random_case(rng))

    for idx, (words, s) in enumerate(cases):
        inp = serialize(words, s)
        want = oracle(words, s)
        got = run_sol(inp)
        if got != want:
            print(f"Mismatch on case {idx}", file=sys.stderr)
            print(inp, file=sys.stderr)
            print(f"oracle={want} sol={got}", file=sys.stderr)
            return 1

    print(f"PASS {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
