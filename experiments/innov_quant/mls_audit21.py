"""MLS 分母必须是 21(用户 2026-09-16 定)。真 0 记 0 进分母;环境坏的必须重跑。

归因看**最后一条致命 traceback 的异常行**,不是日志正文里模型自己代码的报错 ——
模型写错代码时 mlsbench 会把报错回灌给它继续改,那不是失败;整题死掉只有下面这几种。
"""
import json, os, re, sys
D = "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FATAL = re.compile(r"^(?:\w+\.)*(\w+(?:Error|Exception)): (.*)$", re.M)


def classify(t):
    st = (t.get("status") or "")
    lp = t.get("log") or ""
    if not lp or not os.path.exists(lp):
        return "nolog", "无日志"
    txt = ANSI.sub("", open(lp, encoding="utf-8", errors="replace").read())
    # 最后一段 traceback 的最后一行异常,才是整题的死因
    fatal = None
    i = txt.rfind("Traceback (most recent call last)")
    if i >= 0:
        m = list(FATAL.finditer(txt[i:]))
        if m:
            fatal = (m[-1].group(1), m[-1].group(2)[:90])
    if fatal:
        name, msg = fatal
        if name == "APIConnectionError":
            return "serve", "serve 死"
        if name == "BadRequestError" and "maximum context length" in msg:
            return "ctxlen", "上下文撑爆 40960"
        if name == "ModuleNotFoundError":
            return "pkg", msg
        if "timeout" in st:
            return "timeout", st
        return "other", f"{name}: {msg}"
    if "timeout" in st:
        return "timeout", st
    if "agent_failed" in st or t.get("score") is None:
        return "other", st or "score=None"
    return "ok", ""


def _tasks(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path)); ts = d.get("tasks", d)
    return list(ts.values()) if isinstance(ts, dict) else ts


def audit(tag):
    ts = _tasks(f"{D}/cc_mls21_{tag}/summary.json")
    if ts is None:
        return None
    out = {t["task"]: dict(cls=classify(t)[0], why=classify(t)[1], score=t.get("score"))
           for t in ts}
    # 合并补跑:同名题以 <tag>-fix 里的为准,口径与 year_grid.py 的 mls_cell 一致。
    # 不合并的话审计器读到的是补跑前的旧状态。
    fx = _tasks(f"{D}/cc_mls21_{tag}-fix/summary.json")
    for t in (fx or []):
        out[t["task"]] = dict(cls=classify(t)[0], why=classify(t)[1], score=t.get("score"))
    return out


FULL21 = None
if __name__ == "__main__":
    tags = sys.argv[1:]
    res, allt = {}, set()
    for tg in tags:
        a = audit(tg)
        if a is None:
            continue
        res[tg] = a; allt |= set(a)
    full = sorted(allt)
    CLS = ["ok", "ctxlen", "serve", "timeout", "pkg", "other", "nolog"]
    print(f"# 题集并集 {len(full)} 题。分母一律 {len(full)}。\n")
    print(f"{'tag':42s} " + " ".join(f"{c:>7s}" for c in CLS) + f" {'均分/21':>8s} {'均分/可用':>9s}")
    for tg, a in res.items():
        cnt = dict.fromkeys(CLS, 0)
        for t in full:
            cnt[a[t]["cls"] if t in a else "nolog"] += 1
        ok = [a[t]["score"] for t in full if t in a and a[t]["cls"] == "ok"]
        m21 = sum(ok) / len(full)
        mok = (sum(ok) / len(ok)) if ok else float("nan")
        print(f"{tg:42s} " + " ".join(f"{cnt[c]:7d}" for c in CLS) + f" {m21:8.4f} {mok:9.4f}")
    print("\n# 每个 tag 要重跑的题")
    tot = 0
    for tg, a in res.items():
        bad = [(t, (a[t]["cls"] if t in a else "nolog"), (a[t]["why"] if t in a else "缺")) 
               for t in full if t not in a or a[t]["cls"] != "ok"]
        tot += len(bad)
        if not bad:
            print(f"{tg:42s} 21/21 齐"); continue
        print(f"{tg:42s} {len(bad)} 题  TASKS={','.join(t for t, _, _ in bad)}")
        for t, c, w in bad:
            print(f"      {t:42s} [{c}] {w}")
    print(f"\n# 合计要重跑 {tot} 个 (tag, 题)")
