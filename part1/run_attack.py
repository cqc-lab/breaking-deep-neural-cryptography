# -*- coding: utf-8 -*-
# Copyright (C) 2026 Sisung Kim, Minjae Lee, and Dongjae Lee
#
# This file is part of the artifact for "Second Round Key Recovery and a
# Floating-Point Attack on DNN-based AES Implementations".
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
run_attack.py — orchestrator that runs the real 2^32 attack on 2 GPUs with 'dynamic assignment'.
  Queues the 4 columns; an idle GPU immediately picks up the next column (load balancing).
  Each column is logged to col{N}.log. Once all finish, K0+K1 are assembled automatically.

Usage:  python run_attack.py                 # default: columns 0-3, GPU 0,1, real blind 2^32 run
       python run_attack.py --gpus 0,1 --cols 0,1,2,3
  For a fast correctness check without a multi-day GPU run, use the CPU demo aes256_attack.py.
Running inside tmux is safe against disconnects:  tmux new -s atk  ->  python run_attack.py
"""
import subprocess, os, sys, time, re, argparse

def launch(col, gpu, script, record_all=False, seed=2026):
    env=dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    env["FULL_COLS"]=str(col)
    env["RECORD_ALL"]="1" if record_all else "0"
    env["SEED"]=str(seed)
    log=open(f"col{col}.log","w",encoding="utf-8")
    p=subprocess.Popen([sys.executable,script],env=env,stdout=log,
                       stderr=subprocess.STDOUT,cwd=os.getcwd())
    print(f"[launch] col{col} -> GPU{gpu}  (pid {p.pid}, log col{col}.log)",flush=True)
    return {"col":col,"gpu":gpu,"p":p,"log":log,"t0":time.time()}

def _fmt_dur(sec):
    h,rem=divmod(int(sec),3600); mnt,s=divmod(rem,60)
    return f"{h}h {mnt}m {s}s" if h else (f"{mnt}m {s}s" if mnt else f"{s}s")

PT_KNOWN=0x00112233445566778899aabbccddeeff      # plaintext for known-plaintext verification (arbitrary)

def _col_candidates(c):
    """Prefer col{c}_hits.txt (record-all: all ciphertext-collision candidates); otherwise a single [colN] from col{c}.log."""
    cands=[]
    if os.path.exists(f"col{c}_hits.txt"):
        for line in open(f"col{c}_hits.txt",encoding="utf-8",errors="ignore"):   # every line = ciphertext-collision candidate
            m=re.search(r"K1_col=\[([^\]]*)\]",line)
            if m: cands.append([int(x.strip().strip("'\"") ,16) for x in m.group(1).split(",")])
    if not cands and os.path.exists(f"col{c}.log"):
        for line in open(f"col{c}.log",encoding="utf-8",errors="ignore"):
            m=re.search(rf"\[col{c}\].*?K1_col=\[([^\]]*)\]",line)
            if m: cands=[[int(x.strip().strip("'\"") ,16) for x in m.group(1).split(",")]]
    seen=set(); uniq=[]
    for cc in cands:
        if tuple(cc) not in seen: seen.add(tuple(cc)); uniq.append(cc)
    return uniq

def _ct(key256, pt):                                       # aes256_core (NIST-verified) exact CPU/float64 encryption
    import torch, nn_aes, aes256_core
    m,_=aes256_core.build_aes256(key256)
    return nn_aes.encrypt_list_of_plaintexts([pt], m, dtype=torch.float64)[0]

def _observed_ct(seed):
    """Return (CT, PT) for the known-plaintext check, preferring a genuinely OBSERVED oracle
    response recorded in oracle_ct.txt (chosen plaintext -> ciphertext, a legitimate black-box
    query written by aes256_attack_gpu.py). Falls back to regenerating the target from the
    public SEED only if that file is absent, warning that the fallback is a harness convenience,
    not a black-box capability."""
    if os.path.exists("oracle_ct.txt"):
        pt=ct=None
        for line in open("oracle_ct.txt",encoding="utf-8",errors="ignore"):
            mm=re.search(r"PT_KNOWN=0x([0-9a-fA-F]+)",line);   pt=int(mm.group(1),16) if mm else pt
            mm=re.search(r"CT_OBSERVED=0x([0-9a-fA-F]+)",line); ct=int(mm.group(1),16) if mm else ct
        if pt is not None and ct is not None:
            return ct, pt
    print("[warn] oracle_ct.txt not found; falling back to SEED regeneration for verification "
          "(harness convenience, NOT a black-box capability).",flush=True)
    try:
        import random as _r
        return _ct(_r.Random(seed).getrandbits(256), PT_KNOWN), PT_KNOWN
    except Exception as e:
        print(f"[!] cannot construct verification CT ({e})")
        return None, PT_KNOWN

def combine(cols,outfile="recovered_key.txt",t_start=None,seed=2026):
    K0=None
    for c in cols:
        if not os.path.exists(f"col{c}.log"): continue
        for line in open(f"col{c}.log",encoding="utf-8",errors="ignore"):
            m=re.search(r"\[K0\].*?0x([0-9a-fA-F]{32})",line)
            if m: K0=int(m.group(1),16)
    per={c:_col_candidates(c) for c in range(4)}
    miss=[c for c in range(4) if not per[c]]
    if K0 is None or miss:
        print(f"[!] assembly impossible — K0={'present' if K0 else 'missing'}, columns not found {miss}"); return
    ncombo=1
    for c in range(4): ncombo*=len(per[c])
    from itertools import product
    verified=None; master=None; K1=None
    CT, pt_known = _observed_ct(seed)                      # OBSERVED (PT,CT) pair from the oracle (see _observed_ct)
    if CT is not None:
        try:                                               # pick the true combination via the observed known-plaintext pair
            for combo in product(*[per[c] for c in range(4)]):
                k1=bytes(combo[c][r] for c in range(4) for r in range(4))
                cand=(K0<<128)|int.from_bytes(k1,"big")
                if _ct(cand,pt_known)==CT:
                    master=cand; K1=k1; verified=True; break
            if master is None: verified=False              # no combination matches CT = recovery failed
        except Exception as e:
            print(f"[!] skipping known-plaintext verification (error: {e}) — assembly only")
    if master is None:                                     # if unverified/failed, assemble from the first candidate only
        combo=tuple(per[c][0] for c in range(4))
        K1=bytes(combo[c][r] for c in range(4) for r in range(4)); master=(K0<<128)|int.from_bytes(K1,"big")
    ts=time.strftime("%Y-%m-%d %H:%M:%S"); dur=_fmt_dur(time.time()-t_start) if t_start else "N/A"
    vstr={True:"VERIFIED (known-plaintext PASS)",False:"FAILED (no combination matches CT!)",None:"NOT CHECKED"}[verified]
    with open(outfile,"w",encoding="utf-8") as f:
        f.write(f"# recovered at {ts}\n")
        f.write(f"# total_time  {dur}\n")
        f.write(f"# verify      {vstr}\n")
        if any(len(per[c])>1 for c in range(4)):
            f.write(f"# candidates  per column {[len(per[c]) for c in range(4)]} (selected by verification among {ncombo} combinations)\n")
        f.write(f"K0_roundkey0 = 0x{K0:032x}\n")
        f.write(f"K1_roundkey1 = 0x{int.from_bytes(K1,'big'):032x}\n")
        f.write(f"master_256   = 0x{master:064x}\n")
    icon={True:"OK",False:"FAIL",None:"?"}[verified]
    print(f"[{icon}] {vstr} -> saved to '{outfile}' (elapsed {dur})")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gpus",default="0,1")
    ap.add_argument("--cols",default="0,1,2,3")
    ap.add_argument("--record-all",dest="record_all",action="store_true",
                    help="exhaustive scan without early exit + record all confirmed candidates to col{N}_hits.txt (guards against false positives)")
    ap.add_argument("--script",default="aes256_attack_gpu.py")
    ap.add_argument("--out",default="recovered_key.txt")
    ap.add_argument("--seed",type=int,default=2026,help="target key SEED (passed to aes256_attack_gpu.py through the environment)")
    ap.add_argument("--outdir",default=None,help="write all logs/artifacts here (use a per-seed dir to run several blind keys without clobbering)")
    a=ap.parse_args()
    a.script=os.path.abspath(a.script)                     # resolve before any chdir so it survives --outdir
    if a.outdir:
        os.makedirs(a.outdir,exist_ok=True); os.chdir(a.outdir)
        print(f"[run] outputs -> {os.path.abspath(a.outdir)}",flush=True)
    gpus=[int(x) for x in a.gpus.split(",")]
    queue=[int(x) for x in a.cols.split(",")]
    print(f"[run] columns {queue} -> GPU {gpus}  full blind 2^32/column",flush=True)
    t_start=time.time()
    running={}                                  # gpu -> job
    for g in gpus:
        if queue: running[g]=launch(queue.pop(0),g,a.script,a.record_all,a.seed)
    while running:
        time.sleep(5)
        for g in list(running):
            job=running[g]
            if job["p"].poll() is not None:
                dt=time.time()-job["t0"]; job["log"].close()
                print(f"[done ] col{job['col']} (GPU{g}) exited rc={job['p'].returncode}  {dt/60:.1f}min",flush=True)
                del running[g]
                if queue: running[g]=launch(queue.pop(0),g,a.script,a.record_all,a.seed)
    combine([int(x) for x in a.cols.split(",")],a.out,t_start,a.seed)

if __name__=="__main__":
    main()
