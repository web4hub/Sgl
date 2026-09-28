"""Scan IDEDD exception dumps of the HcPreSinkhorn MTE fault.

Each exception_info.* file holds the kernel's ten tensors in dump order:
mixes(196608) rsqrt(8192) hc_scale(12) hc_base(96) x(67108864)
y(16777216) post(32768) comb(131072) workspace(32) tiling(144)
plus 2960 bytes of dump overhead (observed: file size 84257964 =
84255004 + 2960; 2960 = 296 x 10 suggests a per-tensor record header).

The overhead layout is auto-detected per run among candidates
(front header / per-tensor prefix / per-tensor suffix / trailing block)
using a zero-guess anchor: hc_scale and hc_base are MODEL WEIGHTS --
byte-identical in every dump -- so the candidate under which all dumps
agree on those 108 bytes is the true alignment.

At the winning alignment this reports:
  1. cross-dump comparison of the 144B tiling (same shapes must give
     identical tiling; any difference points INSIDE the op package);
  2. workspace(32B) cross-comparison;
  3. mixes/rsqrt fp32 sanity (NaN/Inf/extreme) and a5a5 fill counts
     per section (re-validating the earlier "inputs clean" scan at the
     correct alignment).

Usage:
  python3 test/manual/scan_crash_dumps.py <exception_info files / globs>
"""

import argparse
import glob
import hashlib
import struct
import sys

LAYOUT = [
    ("mixes", 196608), ("rsqrt", 8192), ("hc_scale", 12), ("hc_base", 96),
    ("x", 67108864), ("y", 16777216), ("post", 32768), ("comb", 131072),
    ("workspace", 32), ("tiling", 144),
]
TENSORS_SUM = sum(size for _, size in LAYOUT)
# per-tensor record overhead if the total extra divides evenly
PER_SECTION = 296


def slice_candidate(blob, mode):
    """Return {name: bytes} for one overhead hypothesis, or None."""
    extra = len(blob) - TENSORS_SUM
    if extra < 0:
        return None
    parts = {}
    if mode == "front":
        if extra == 0:
            return None
        off = extra
        for name, size in LAYOUT:
            parts[name] = blob[off:off + size]
            off += size
        return parts if off == len(blob) else None
    if mode == "back":
        off = 0
        for name, size in LAYOUT:
            parts[name] = blob[off:off + size]
            off += size
        return parts if off == TENSORS_SUM and off + extra == len(blob) else None
    if mode == "prefix":
        if PER_SECTION * len(LAYOUT) != extra:
            return None
        off = 0
        for name, size in LAYOUT:
            off += PER_SECTION
            parts[name] = blob[off:off + size]
            off += size
        return parts
    if mode == "suffix":
        if PER_SECTION * len(LAYOUT) != extra:
            return None
        off = 0
        for name, size in LAYOUT:
            parts[name] = blob[off:off + size]
            off += size + PER_SECTION
        return parts
    return None


def anchor_agrees(parsed_files):
    """Weight sections cluster into a few call-site groups under the true
    alignment (attn-side vs FFN-side hc weights differ); a wrong alignment
    scatters into ~one distinct pair per file."""
    pairs = {(p["hc_scale"], p["hc_base"]) for p in parsed_files.values()}
    return 0 < len(pairs) <= 4, len(pairs)


A5X4 = b"\xa5\xa5\xa5\xa5"


def scan_fp32(data, label):
    n = len(data) // 4
    vals = struct.unpack(f"<{n}f", data[: n * 4])
    bad = sum(1 for v in vals if v != v or abs(v) == float("inf") or abs(v) > 1e6)
    amax = max((abs(v) for v in vals if v == v), default=0.0)
    print(f"  {label:8s} fp32 n={n} bad={bad} |v|max={amax:.3e} a5a5={data.count(A5X4)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)))
    if not files:
        sys.exit("no dump files matched")

    blobs = {}
    for path in files:
        with open(path, "rb") as f:
            blobs[path] = f.read()
        extra = len(blobs[path]) - TENSORS_SUM
        if extra < 0:
            print(f"{path}: smaller than tensor sum; skip")

    candidates = [f for f in blobs if len(blobs[f]) >= TENSORS_SUM]
    if not candidates:
        sys.exit("no usable dumps")

    mode = None
    for m in ("prefix", "front", "suffix", "back"):
        parsed = {f: slice_candidate(blobs[f], m) for f in candidates}
        if any(p is None for p in parsed.values()):
            continue
        ok, clusters = anchor_agrees(parsed)
        if ok:
            mode = m
            print(
                f"alignment detected: {m} "
                f"(overhead {len(blobs[candidates[0]]) - TENSORS_SUM}B, "
                f"{clusters} hc weight groups across dumps)"
            )
            break
    if mode is None:
        head = blobs[candidates[0]][:96].hex()
        sys.exit(
            "no overhead hypothesis made hc_scale/hc_base agree across dumps.\n"
            f"first 96 bytes of {candidates[0]}: {head}\n"
            "paste this back for manual layout analysis."
        )

    tilings, workspaces = {}, {}
    for path in candidates:
        parts = slice_candidate(blobs[path], mode)
        tilings[path] = parts["tiling"]
        workspaces[path] = parts["workspace"]
        words = struct.unpack("<36I", parts["tiling"])
        print(f"\n{path}")
        print(f"  tiling u32[0..8] : {' '.join(f'{w:#x}' for w in words[:8])}")
        print(f"  tiling u32[9..17]: {' '.join(f'{w:#x}' for w in words[8:16])}")
        print(f"  tiling sha256    : {hashlib.sha256(parts['tiling']).hexdigest()[:16]}")
        print(f"  workspace bytes  : {parts['workspace'].hex()}")
        scan_fp32(parts["mixes"], "mixes")
        scan_fp32(parts["rsqrt"], "rsqrt")
        for name in ("x", "y", "post", "comb"):
            print(f"  {name:8s} a5a5-hits={parts[name].count(A5X4)}")

    uniq_t = {bytes(v) for v in tilings.values()}
    uniq_w = {bytes(v) for v in workspaces.values()}
    print(f"\n=== cross-dump verdict over {len(tilings)} dumps ===")
    print(f"distinct tiling contents   : {len(uniq_t)}")
    print(f"distinct workspace contents: {len(uniq_w)}")
    if len(uniq_t) > 1:
        print("TILING DIFFERS between same-shape crashes -> nondeterministic/"
              "cross-written tiling: look INSIDE the op package first.")
    elif len(tilings) > 1:
        print("tiling identical across crashes (deterministic; corruption "
              "hypothesis weakened but not excluded).")


if __name__ == "__main__":
    main()
