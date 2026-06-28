#!/usr/bin/env python3
import random
import subprocess
import sys


SOL = "/tmp/fcs_sol"


def brute_outputs(points, ops):
    weights = [w for _, _, w in points]
    last = 0
    out = []
    for op in ops:
        if op[0] == 1:
            _, i, d = op
            weights[i] += d
        else:
            _, a, b, c, d = op
            x1 = a ^ last
            y1 = b ^ last
            x2 = c ^ last
            y2 = d ^ last
            ans = 0
            if x1 <= x2 and y1 <= y2:
                for idx, (x, y, _) in enumerate(points):
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        ans += weights[idx]
            out.append(ans)
            last = ans
    return out


def render_case(points, ops):
    lines = [f"{len(points)} {len(ops)}"]
    lines += [f"{x} {y} {w}" for x, y, w in points]
    for op in ops:
        lines.append(" ".join(map(str, op)))
    return "\n".join(lines) + "\n"


def run_sol(case_text):
    proc = subprocess.run(
        [SOL],
        input=case_text.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"solution exited {proc.returncode}\nSTDERR:\n{proc.stderr.decode(errors='replace')}\nINPUT:\n{case_text}"
        )
    return proc.stdout.decode()


def expected_text(points, ops):
    outs = brute_outputs(points, ops)
    if not outs:
        return ""
    return "".join(f"{x}\n" for x in outs)


def check_case(name, points, ops):
    case_text = render_case(points, ops)
    got = run_sol(case_text)
    exp = expected_text(points, ops)
    if got != exp:
        print(f"MISMATCH: {name}", file=sys.stderr)
        print("INPUT:", file=sys.stderr)
        print(case_text, file=sys.stderr)
        print("EXPECTED:", file=sys.stderr)
        print(exp, file=sys.stderr)
        print("GOT:", file=sys.stderr)
        print(got, file=sys.stderr)
        sys.exit(1)


def append_query(points, ops, state_weights, last, rect):
    x1, y1, x2, y2 = rect
    op = (2, x1 ^ last, y1 ^ last, x2 ^ last, y2 ^ last)
    ans = 0
    if x1 <= x2 and y1 <= y2:
        for i, (x, y, _) in enumerate(points):
            if x1 <= x <= x2 and y1 <= y <= y2:
                ans += state_weights[i]
    ops.append(op)
    return ans


def append_update(ops, state_weights, i, d):
    state_weights[i] += d
    ops.append((1, i, d))


def hand_cases():
    cases = []

    points = [(2, 2, 5)]
    ops = [(2, 0, 0, 3, 3), (2, 5, 5, 6, 6)]
    cases.append(("sample", points, ops))

    points = [(0, 0, 7)]
    w = [7]
    ops = []
    last = append_query(points, ops, w, 0, (0, 0, 0, 0))
    append_update(ops, w, 0, -10)
    last = append_query(points, ops, w, last, (0, 0, 0, 0))
    last = append_query(points, ops, w, last, (1, 1, 0, 0))
    cases.append(("single_negative_empty", points, ops))

    points = [(3, 3, 2), (3, 3, 5), (3, 4, -2), (4, 3, 9)]
    w = [p[2] for p in points]
    ops = []
    last = append_query(points, ops, w, 0, (3, 3, 3, 3))
    append_update(ops, w, 2, 7)
    last = append_query(points, ops, w, last, (3, 3, 3, 4))
    append_update(ops, w, 1, -20)
    last = append_query(points, ops, w, last, (3, 3, 4, 4))
    cases.append(("duplicates_same_coordinate", points, ops))

    points = [(-10**9, -10**9, 10**9), (10**9, 10**9, 10**9), (0, 0, -10**9)]
    w = [p[2] for p in points]
    ops = []
    last = append_query(points, ops, w, 0, (-10**9, -10**9, 10**9, 10**9))
    last = append_query(points, ops, w, last, (-10**9, -10**9, -10**9, -10**9))
    append_update(ops, w, 2, 2 * 10**9)
    last = append_query(points, ops, w, last, (-1, -1, 1, 1))
    cases.append(("wide_boundaries_large_sum", points, ops))

    points = [(-5, 2, 4), (-5, -2, 6), (7, 2, -3), (8, 9, 11)]
    w = [p[2] for p in points]
    ops = []
    last = append_query(points, ops, w, 0, (-5, 2, 7, 2))
    last = append_query(points, ops, w, last, (-100, -100, 100, 100))
    last = append_query(points, ops, w, last, (9, 9, 8, 8))
    last = append_query(points, ops, w, last, (-6, -3, -5, 2))
    cases.append(("same_x_same_y_edges", points, ops))

    points = [(i, -i, i - 3) for i in range(7)]
    ops = []
    w = [p[2] for p in points]
    for i in range(len(points)):
        append_update(ops, w, i, i * i - 4)
    cases.append(("updates_only", points, ops))

    return cases


def random_case(rng, case_id):
    n = rng.randint(1, 18)
    q = rng.randint(1, 55)
    points = []
    previous_coords = []
    for _ in range(n):
        if previous_coords and rng.random() < 0.30:
            x, y = rng.choice(previous_coords)
        else:
            x = rng.randint(-12, 12)
            y = rng.randint(-12, 12)
            previous_coords.append((x, y))
        w = rng.randint(-25, 25)
        points.append((x, y, w))

    weights = [p[2] for p in points]
    ops = []
    last = 0
    coord_pool = [-20, -13, -12, -1, 0, 1, 12, 13, 20]
    coord_pool += [p[0] for p in points] + [p[1] for p in points]

    for _ in range(q):
        must_query = not any(op[0] == 2 for op in ops) and len(ops) == q - 1
        if not must_query and rng.random() < 0.45:
            i = rng.randrange(n)
            d = rng.randint(-30, 30)
            append_update(ops, weights, i, d)
            continue

        if rng.random() < 0.55:
            x1, x2 = sorted((rng.choice(coord_pool), rng.choice(coord_pool)))
            y1, y2 = sorted((rng.choice(coord_pool), rng.choice(coord_pool)))
        else:
            x1 = rng.randint(-20, 20)
            x2 = rng.randint(-20, 20)
            y1 = rng.randint(-20, 20)
            y2 = rng.randint(-20, 20)
            if rng.random() < 0.55:
                x1, x2 = sorted((x1, x2))
            if rng.random() < 0.55:
                y1, y2 = sorted((y1, y2))
        last = append_query(points, ops, weights, last, (x1, y1, x2, y2))

    return (f"random_{case_id}", points, ops)


def main():
    rng = random.Random(0xFC506)
    total = 0
    for name, points, ops in hand_cases():
        check_case(name, points, ops)
        total += 1
    for i in range(750):
        name, points, ops = random_case(rng, i)
        check_case(name, points, ops)
        total += 1
    print(f"PASS {total} cases")


if __name__ == "__main__":
    main()
