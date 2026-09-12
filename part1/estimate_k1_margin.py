#!/usr/bin/env python3
# Copyright (C) 2026 the authors (anonymized for review)
#
# This file is part of the artifact for "Cryptanalysis of Deep Neural
# Cryptography: Second Round Key Recovery on the Unprotected Implementation
# and a Floating-Point Attack on the Protected Implementation of AES".
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
estimate_k1_margin.py
=====================
Across-key validation of the K1 collision oracle: over MANY reproducibly seeded
AES-256 keys, check that the TRUE candidate clears tau while a random sample of
WRONG candidates stays at 0. This measures the across-key completeness/soundness of
the oracle (the exhaustive census in aes256_attack_gpu.py instead measures
zero-false-positives over the FULL 2^32 space, but only on a couple of keys).

Per (key, column) it records:
  - the TRUE candidate's collision count out of 32 probes (expect >= tau), and
  - over `nwrong` random WRONG candidates: max / mean collision count and #(>= tau).

Reproducible & shardable
------------------------
Every random draw is derived from --seed via a deterministic SHA-256 seed
(machine-independent, not the process hash), so a run is fully reproducible and any
shard can be recomputed in isolation:
    key i            <- Random(sha256("key",   seed, i))          (256-bit AES key)
    wrong set (i,col)<- Random(sha256("wrong", seed, i, col))     (nwrong distinct 32-bit)
--shard k/N processes only keys i with i % N == k, so N shards (one per core or GPU)
cover the same key set with no overlap. Results stream to a per-shard CSV artifact.

Device / dtype: inherited from aes256_attack_gpu (float16 on Volta+ GPU, else
float32 on CPU); recorded in the artifact header. The committed censuses were run
on CPU (CUDA_VISIBLE_DEVICES=-1): artifacts/fp32/ in float32 and artifacts/fp16/
with FORCE_FP16=1.

Usage
-----
  # full run on one machine (GPU auto-detected):
  python estimate_k1_margin.py --nkeys 10000 --nwrong 1000 --seed 20260903

  # split across 6 cores (CPU), one shard per process, then merge:
  for k in 0 1 2 3 4 5; do \
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \
    python estimate_k1_margin.py --nkeys 10000 --nwrong 1000 --seed 20260903 --shard $k/6 & \
  done; wait
  python estimate_k1_margin.py --merge "artifacts/k1_margin_seed20260903_shard*of6.csv"

  # quick smoke test:
  python estimate_k1_margin.py --nkeys 3 --nwrong 100 --seed 1
"""
import os, sys, glob, json, time, random, hashlib, argparse
import numpy as np
import aes256_attack_gpu as A   # importing sets DEVICE/DTYPE (GPU float16 on Volta+, else CPU float32)

# Optional: measure in the released float16 format even on CPU (default CPU dtype is float32).
# This only overrides the measurement harness; the attack module itself is left untouched.
if os.environ.get("FORCE_FP16") not in (None, "0", "false", "False"):
    A.DTYPE = A.torch.float16
    A.nn_aes.DTYPE = A.torch.float16

DEADZONE = 0x52


def _seed(*parts):
    """Deterministic, machine-independent integer seed from the given parts."""
    return int(hashlib.sha256("_".join(map(str, parts)).encode()).hexdigest(), 16)


def key_for(seed, i):
    return random.Random(_seed("key", seed, i)).getrandbits(256)


def true_c(K1, col):
    return sum((int(K1[r, col]) ^ DEADZONE) << (8 * (3 - r)) for r in range(4))


def wrong_sample(seed, i, col, nwrong, true_ct):
    r = random.Random(_seed("wrong", seed, i, col))
    s = set()
    while len(s) < nwrong:
        x = r.getrandbits(32)
        if x != true_ct:
            s.add(x)
    return sorted(s)


def counts_for(col, cand_ints, K0, other, chunk=256):
    out = []
    for s in range(0, len(cand_ints), chunk):
        out.append(A.enc_collision_counts(col, np.asarray(cand_ints[s:s + chunk], dtype=np.int64), K0, other))
    return np.concatenate(out) if out else np.zeros(0, np.int32)


def run(a):
    cols = tuple(int(x) for x in a.cols.split(","))
    k, N = (int(x) for x in a.shard.split("/"))
    outdir = os.path.dirname(a.out) or "."
    os.makedirs(outdir, exist_ok=True)
    csv_path = f"{a.out}_seed{a.seed}_shard{k}of{N}.csv"
    dtype = str(A.DTYPE).replace("torch.", "")
    device = A.DEVICE.type
    from aes256_core import model_class
    model = model_class()[1]                 # 'base' or 'ttables', from the MODEL environment variable
    new = not os.path.exists(csv_path)
    f = open(csv_path, "a", encoding="utf-8")
    if new:
        f.write(f"# estimate_k1_margin seed={a.seed} nkeys={a.nkeys} nwrong={a.nwrong} "
                f"tau={a.tau} dtype={dtype} device={device} model={model} shard={k}/{N}\n")
        f.write("key_index,col,true_count,wrong_max,wrong_mean,wrong_ge_tau\n")
        f.flush()
    print(f"[margin census] nkeys={a.nkeys} nwrong={a.nwrong} seed={a.seed} shard={k}/{N} "
          f"tau={a.tau} dtype={dtype} device={device} model={model} -> {csv_path}", flush=True)
    t0 = time.time(); done = 0
    worst_true, best_wrong, total_ge = 99, -1, 0
    for i in range(a.nkeys):
        if i % N != k:
            continue
        KEY = key_for(a.seed, i)
        m, rks = A.build_aes256(KEY); A.m = m.to(A.DEVICE)
        K0 = np.array(A.utils.integer_to_bytes_matrix(rks[0]), np.uint8)
        K1 = np.array(A.utils.integer_to_bytes_matrix(rks[1]), np.uint8)
        for col in cols:
            other = A._ZERO44
            ct = true_c(K1, col)
            tcnt = int(counts_for(col, [ct], K0, other)[0])
            wrong = wrong_sample(a.seed, i, col, a.nwrong, ct)
            wc = counts_for(col, wrong, K0, other)
            wmax, wmean, wge = int(wc.max()), float(wc.mean()), int((wc >= a.tau).sum())
            f.write(f"{i},{col},{tcnt},{wmax},{wmean:.4f},{wge}\n")
            worst_true = min(worst_true, tcnt); best_wrong = max(best_wrong, wmax); total_ge += wge
        for _flush_try in range(5):                # tolerate transient flush failures on
            try:                                    # network/cloud-synced filesystems (e.g. Google Drive
                f.flush(); break                    # File Stream throws OSError 22 under concurrent writes)
            except OSError:
                time.sleep(0.3)
        done += 1
        if done % 10 == 0:
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(range(k, a.nkeys, N)) - done) / max(rate, 1e-9)
            print(f"  {done} keys | min_true={worst_true} max_wrong={best_wrong} "
                  f"wrong>=tau={total_ge} | {rate:.3f} keys/s ETA {eta/3600:.2f}h", flush=True)
    f.close()
    print(f"[shard {k}/{N} done] {done} keys | min_true={worst_true} max_wrong={best_wrong} "
          f"wrong>=tau={total_ge} | {time.time()-t0:.0f}s", flush=True)


def merge(pattern, max_key=None):
    """Merge shard CSVs into one summary. max_key restricts the merge to key_index < max_key,
    so a prefix of the census (e.g. 'the first 2500 keys') can be summarized exactly and reproducibly."""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no CSV files match {pattern}"); return 1
    seen = {}
    tau = None; seed = None; nwrong = None; dtype = None; device = None; model = "base"
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#"):
                    for tok in line.split():
                        if tok.startswith("tau="): tau = int(tok[4:])
                        if tok.startswith("seed="): seed = int(tok[5:])
                        if tok.startswith("nwrong="): nwrong = int(tok[7:])
                        if tok.startswith("dtype="): dtype = tok[6:]
                        if tok.startswith("device="): device = tok[7:]
                        if tok.startswith("model="): model = tok[6:]
                    continue
                if line.startswith("key_index") or not line:
                    continue
                ki, col, tc, wmax, wmean, wge = line.split(",")
                if max_key is not None and int(ki) >= max_key:
                    continue
                seen[(int(ki), int(col))] = (int(tc), int(wmax), float(wmean), int(wge))
    if max_key is not None:
        missing = sorted({i for i in range(max_key)} - {ki for (ki, _) in seen})
        if missing:
            print(f"WARNING: {len(missing)} of the first {max_key} keys are missing from the shards "
                  f"(e.g. {missing[:5]}); the summary covers only the keys present", flush=True)
    rows = list(seen.values())
    nkeys = len({ki for (ki, _) in seen})
    trues = np.array([r[0] for r in rows]); wmaxs = np.array([r[1] for r in rows])
    total_ge = int(sum(r[3] for r in rows)); total_wrong = nkeys * len({c for (_, c) in seen}) * (nwrong or 0)
    ci = 1 - 0.05 ** (1.0 / nkeys) if nkeys else float("nan")
    summary = {
        "seed": seed, "nkeys": nkeys, "max_key": max_key, "columns": sorted({c for (_, c) in seen}),
        "nwrong_per_keycol": nwrong, "tau": tau, "dtype": dtype, "device": device, "model": model,
        "true_count": {"min": int(trues.min()), "median": float(np.median(trues)),
                       "max": int(trues.max()), "mean": float(trues.mean())},
        "true_below_tau": int((trues < (tau or 8)).sum()),
        "wrong_max_over_all": int(wmaxs.max()), "wrong_ge_tau_total": total_ge,
        "wrong_candidates_total": total_wrong,
        "per_key_failure_upper_95ci": ci,
    }
    # Derive the summary name from the shard files' own prefix so different runs
    # (e.g. a float16 census named k1_margin_fp16_*) do not overwrite each other.
    import re as _re
    _base = _re.sub(r"_shard\d+of\d+\.csv$", "", os.path.basename(files[0]))
    if max_key is not None:
        _base += f"_keys{max_key}"
    out = os.path.join(os.path.dirname(files[0]) or "artifacts", f"{_base}_summary.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n[written] {out}")
    print(f"[paper]   {nkeys} keys x {len(summary['columns'])} cols: "
          f"true collisions in [{summary['true_count']['min']}, {summary['true_count']['max']}] "
          f"(median {summary['true_count']['median']:.0f}), "
          f"all {total_wrong:,} wrong candidates below tau={tau} "
          f"(max {summary['wrong_max_over_all']}); 0-failure 95% upper CI {ci*100:.2f}% per key.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Reproducible, shardable across-key margin census for the K1 oracle")
    ap.add_argument("--nkeys", type=int, default=10000)
    ap.add_argument("--nwrong", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--cols", type=str, default="0,1,2,3")
    ap.add_argument("--tau", type=int, default=8)
    ap.add_argument("--shard", type=str, default="0/1", help="k/N: process keys i with i%%N==k")
    ap.add_argument("--out", type=str, default="artifacts/k1_margin")
    ap.add_argument("--merge", type=str, default=None, help="glob of shard CSVs to merge into a summary")
    ap.add_argument("--max-key", type=int, default=None,
                    help="with --merge: use only key_index < MAX_KEY (summarize a prefix of the census)")
    a = ap.parse_args()
    if a.merge:
        return merge(a.merge, a.max_key)
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
