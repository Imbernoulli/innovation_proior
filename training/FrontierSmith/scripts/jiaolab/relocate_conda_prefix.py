#!/usr/bin/env python3
"""Relocate a conda env moved between machines with different install prefixes.

Used by the jiaolab MLS-Bench port: the 10 `mlsbench-<pkg>` envs are rsynced
from gpublaze (/srv/home/bohanlyu/miniconda3) and their baked-in prefix rewritten
to /home/bohan/miniconda3. This is what `conda-unpack` does, minus the packing:

  python3 scripts/jiaolab/relocate_conda_prefix.py \
      /srv/home/bohanlyu/miniconda3 /home/bohan/miniconda3 \
      /home/bohan/miniconda3/envs/mlsbench-<pkg>

Text files get a plain replace. Binary files get the replace plus NUL padding
back to each embedded C-string's original length, so ELF/section offsets are
preserved -- which is only safe when the NEW prefix is no longer than the OLD
one (asserted below).

Why rsync+rewrite instead of rebuilding the envs on the target: the packages'
install_cmds are mostly unpinned, so a rebuild produces a DIFFERENT package set,
i.e. a different scoring regime. See docs/EVAL_ON_JIAOLAB_zh.md §2.4.4.

Leftover matches in .pyc files are expected and harmless (co_filename strings,
only ever seen in tracebacks); __pycache__ is skipped.
"""
import os, sys, re

OLD = sys.argv[1].encode()
NEW = sys.argv[2].encode()
ROOT = sys.argv[3]
assert len(NEW) <= len(OLD), "new prefix must not be longer than old"

def is_binary(chunk: bytes) -> bool:
    return b"\x00" in chunk

def fix_binary(data: bytes) -> bytes:
    out = bytearray()
    pos = 0
    for m in re.finditer(re.escape(OLD), data):
        s = m.start()
        if s < pos:          # already consumed as part of an earlier C-string
            continue
        # the embedded C string runs from after the previous NUL to the next NUL
        end = data.find(b"\x00", m.end())
        if end == -1:
            end = len(data)
        seg = data[s:end]
        newseg = seg.replace(OLD, NEW)
        newseg = newseg + b"\x00" * (len(seg) - len(newseg))
        out += data[pos:s] + newseg
        pos = end
    out += data[pos:]
    return bytes(out)

n_text = n_bin = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in filenames:
        p = os.path.join(dirpath, fn)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError:
            continue
        if OLD not in data:
            continue
        binary = is_binary(data[:8192]) or is_binary(data)
        new = fix_binary(data) if binary else data.replace(OLD, NEW)
        if new == data:
            continue
        assert not binary or len(new) == len(data), p
        st = os.stat(p)
        with open(p, "wb") as f:
            f.write(new)
        os.chmod(p, st.st_mode)
        if binary:
            n_bin += 1
        else:
            n_text += 1
print(f"[relocate] rewrote text={n_text} binary={n_bin} under {ROOT}")
