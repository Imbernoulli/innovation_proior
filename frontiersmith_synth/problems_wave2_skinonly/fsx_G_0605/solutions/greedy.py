# TIER: greedy

import sys


def parse_instance_text(text):
    toks = text.split()
    if len(toks) < 7:
        raise ValueError("short instance")
    it = iter(toks)
    n = int(next(it)); m = int(next(it)); groups = int(next(it))
    budget = int(next(it)); group_cap = int(next(it)); salt = int(next(it)); profile = int(next(it))
    items = []
    for idx in range(1, n + 1):
        cost = int(next(it)); group = int(next(it)); value = int(next(it))
        x = int(next(it)); y = int(next(it)); mask = int(next(it))
        items.append({"idx": idx, "cost": cost, "group": group, "value": value,
                      "x": x, "y": y, "mask": mask})
    return {"n": n, "m": m, "groups": groups, "budget": budget,
            "group_cap": group_cap, "salt": salt, "profile": profile, "items": items}


def feature_weight(j, salt, profile):
    return 7 + ((j * 17 + salt * 5 + profile * 13) % 31)


def conflict(a, b, inst):
    dx = abs(a["x"] - b["x"]); dy = abs(a["y"] - b["y"])
    overlap = (a["mask"] & b["mask"]).bit_count()
    if a["group"] == b["group"] and dx + dy < 18 + inst["profile"]:
        return True
    if overlap >= 5 + (inst["profile"] % 3) and ((dx * 3 + dy * 5 + inst["salt"]) % 11 == 0):
        return True
    return False


def feasible(sel, inst):
    seen = set()
    cost = 0
    group_counts = {}
    items = inst["items"]
    for idx in sel:
        if idx < 1 or idx > inst["n"] or idx in seen:
            return False
        seen.add(idx)
        it = items[idx - 1]
        cost += it["cost"]
        if cost > inst["budget"]:
            return False
        group_counts[it["group"]] = group_counts.get(it["group"], 0) + 1
        if group_counts[it["group"]] > inst["group_cap"]:
            return False
    arr = [items[i - 1] for i in sel]
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if conflict(arr[i], arr[j], inst):
                return False
    return True


def evaluate(sel, inst):
    if not feasible(sel, inst):
        return 0
    items = [inst["items"][i - 1] for i in sel]
    score = sum(it["value"] for it in items)
    cover = [0] * inst["m"]
    for it in items:
        mask = it["mask"]
        for j in range(inst["m"]):
            if (mask >> j) & 1:
                cover[j] += 1
    for j, c in enumerate(cover):
        if c:
            w = feature_weight(j, inst["salt"], inst["profile"])
            score += w * min(3, c) + (w // 4) * max(0, c - 3)
    groups = {}
    for it in items:
        groups[it["group"]] = groups.get(it["group"], 0) + 1
    score += 17 * len(groups)
    for cnt in groups.values():
        score += 3 * min(cnt, inst["group_cap"])
    for i in range(len(items)):
        a = items[i]
        for j in range(i + 1, len(items)):
            b = items[j]
            if a["group"] != b["group"]:
                dist = abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])
                if (dist + inst["salt"] + a["group"] * 3 + b["group"]) % 7 in (0, 1):
                    score += 4 + ((a["group"] + b["group"] + inst["profile"]) % 5)
    return max(1, score)


def can_add(sel, idx, inst):
    return feasible(sel + [idx], inst)


def baseline_select(inst):
    # Deliberately weak calibration anchor: a feasible "minimal deployment" rather than a
    # near-complete greedy packing. The checker uses this same construction for B, so the
    # trivial solution remains exactly calibrated at Ratio ~= 0.1 while better heuristics
    # have enough headroom to separate.
    sel = []
    target = 4 + (inst["profile"] % 2)
    for it in sorted(inst["items"], key=lambda z: (z["cost"], z["idx"])):
        if can_add(sel, it["idx"], inst):
            sel.append(it["idx"])
            if len(sel) >= target:
                break
    return sel


def greedy_select(inst):
    def key(it):
        mask_bonus = sum(feature_weight(j, inst["salt"], inst["profile"])
                         for j in range(inst["m"]) if (it["mask"] >> j) & 1)
        return (it["value"] + mask_bonus + 9 * (it["group"] + 1)) / max(1, it["cost"])
    sel = []
    for it in sorted(inst["items"], key=key, reverse=True):
        if can_add(sel, it["idx"], inst):
            sel.append(it["idx"])
    return sel


def strong_select(inst):
    sel = []
    remaining = {it["idx"] for it in inst["items"]}
    current = 0
    while True:
        best = None
        best_key = 0.0
        for idx in list(remaining):
            cand = sel + [idx]
            if not feasible(cand, inst):
                continue
            val = evaluate(cand, inst)
            gain = val - current
            cost = inst["items"][idx - 1]["cost"]
            key = gain / max(1, cost)
            if best is None or key > best_key + 1e-12 or (abs(key - best_key) <= 1e-12 and gain > best[1]):
                best = (idx, gain, val)
                best_key = key
        if best is None or best[1] <= 0:
            break
        sel.append(best[0])
        remaining.remove(best[0])
        current = best[2]
    return sel


def emit(sel):
    print(len(sel))
    if sel:
        print(" ".join(str(x) for x in sel))



def main():
    inst = parse_instance_text(sys.stdin.read())
    emit(greedy_select(inst))


if __name__ == "__main__":
    main()
