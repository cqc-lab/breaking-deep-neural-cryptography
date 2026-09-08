# -*- coding: utf-8 -*-
# Copyright (C) 2026 Sisung Kim and Dongjae Lee
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
aes256_attack_gpu.py  —  True attacker model (ciphertext observation only) AES-256 master-key recovery
=====================================================================
Full master-key recovery of the natural AES-256 implementation (K0 symmetric per-bit + K1 collision column).
  RoundKey0(K0) || RoundKey1(K1) = 256-bit master key.

Oracle = final ciphertext only.
  - K0: recovered per-bit from whether the ciphertext of two ±δ perturbed plaintexts differs (enc_full).
  - K1: a candidate is planted into the round-1 output (the plaintext construction needs only K0); after
        perturbing the 32 feeding probes, a candidate is accepted when the number of ciphertext
        collisions >= COLL_MIN (enc_full).
  - DTYPE = float16 on Volta (cc7.0)+, otherwise float32.  Only the round-1 inversion is CPU numpy.

Note: aes256_core sets nn_aes to CPU/float64 on import, so
      nn_aes.DEVICE/DTYPE must be overwritten 'before' calling build_aes256.
=====================================================================
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, random
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, torch
import nn_aes, utils

# ------------------------- Device / precision (auto-detected) -------------------------
FORCE_FP32 = False    # Shamir's original implementation uses FP16 (nn_aes.py). Keep False to attack the target as-is.

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (DEVICE.type == "cuda")
if USE_CUDA:
    _cc    = torch.cuda.get_device_capability()                     # (major, minor)
    _vram  = torch.cuda.get_device_properties(0).total_memory / 1e9 # GB
    _gname = torch.cuda.get_device_name(0)
    # Tensor Cores (fast float16) require Volta (7.0)+.  Pascal (6.x) is slow in FP16, so float32.
    DTYPE  = torch.float32 if FORCE_FP32 else (torch.float16 if _cc[0] >= 7 else torch.float32)
else:
    _cc, _vram, _gname = None, 0, "CPU"
    DTYPE  = torch.float32

# The aes256_core import sets nn_aes to CPU/float64, so overwrite them afterwards (before build_aes256).
from aes256_core import build_aes256
nn_aes.DEVICE = DEVICE
nn_aes.DTYPE  = DTYPE
torch.set_grad_enabled(False)

# ------------------------- Configuration -------------------------
DEADZONE=0x52; DELTA=0.05; K0_BASES=4
SEED=2026

# BATCH: K1 candidate scan batch. enc_collision_counts passes (32+1) rows per candidate through 14 rounds (enc_full). Auto-fitted to VRAM.
# Reduce on OOM; increase for more throughput if VRAM allows (values below can be tuned manually).
if USE_CUDA:
    _b = 2048 if _vram < 8 else (8192 if _vram < 24 else 16384)
    BATCH = _b // 2 if DTYPE==torch.float32 else _b     # float32 uses 2x memory -> halve
else:
    BATCH = 256

# ---- Exhaustive blind streaming per column: a K1 column is 4 bytes = 32 bits, so the search space is exactly 2^32 ----
# The search never references the true key; K1_TRUE is used only for the post-run PASS/FAIL report in main().
COL_BITS     = 32                 # column width in bits (structural constant, not a tunable): full blind scan = 2^COL_BITS
FULL_COLS    = [0,1,2,3]          # Columns to attack (a subset per node when distributed)
PROGRESS_SEC = 10                 # Minimum interval for progress output (seconds)
RECORD_ALL   = False              # True: exhaustive scan without early stop, record all confirmed candidates to col{N}_hits.txt
                                  #   (prevents an extremely rare 2^32 false positive from stopping before the true one. Always full range instead.)

# ---- Oracle decision (final-ciphertext collisions) ----
COLL_MIN = 8                      # Judged a candidate if the number of 'ciphertext collisions' among 32 probes ≥ this value.
                                  #   The true candidate collides on 6-32 of the 32 probes (median 31); a wrong candidate on 0, except the
                                  #   at most 1024 forced candidates per column, which collide on 7 (or 14 for an adjacent pair;
                                  #   paper, Proposition 1 / Appendix A).

# Environment-variable overrides (for multi-GPU distribution).  Assign columns/settings per process without editing the file.
#   e.g.) GPU0: CUDA_VISIBLE_DEVICES=0 FULL_COLS=0,1 python aes256_attack_gpu.py
#       GPU1: CUDA_VISIBLE_DEVICES=1 FULL_COLS=2,3 python aes256_attack_gpu.py
if os.environ.get("FULL_COLS"):    FULL_COLS   = [int(x) for x in os.environ["FULL_COLS"].split(",")]
if os.environ.get("COLL_MIN"):     COLL_MIN    = int(os.environ["COLL_MIN"])
if os.environ.get("RECORD_ALL"):   RECORD_ALL  = os.environ["RECORD_ALL"] not in ("0","false","False")
if os.environ.get("SEED"):         SEED        = int(os.environ["SEED"])   # attack a different key per seed
HIST = os.environ.get("HIST") not in (None, "0", "false", "False")   # accumulate full 0..32 collision-count histogram over the scan

SINV=np.array(utils.SBOX_INV,np.uint8)
_ZERO44=np.zeros((4,4),np.uint8)

# ------------------------- GF & round1 inversion (CPU numpy) -------------------------
def _mk(c):
    t=np.zeros(256,np.uint8)
    for a in range(256):
        p=0;x=a;b=c
        for _ in range(8):
            if b&1:p^=x
            hi=x&0x80;x=(x<<1)&0xff
            if hi:x^=0x1b
            b>>=1
        t[a]=p
    return t
T9,T11,T13,T14=_mk(9),_mk(11),_mk(13),_mk(14)
def inv_mc(S):
    o=np.empty_like(S);x0,x1,x2,x3=S[:,0,:],S[:,1,:],S[:,2,:],S[:,3,:]
    o[:,0,:]=T14[x0]^T11[x1]^T13[x2]^T9[x3];o[:,1,:]=T9[x0]^T14[x1]^T11[x2]^T13[x3]
    o[:,2,:]=T13[x0]^T9[x1]^T14[x2]^T11[x3];o[:,3,:]=T11[x0]^T13[x1]^T9[x2]^T14[x3]
    return o
def inv_sr(S):
    o=S.copy()
    for r in range(4):o[:,r,:]=np.roll(S[:,r,:],r,axis=1)
    return o
_POS=[(p%4,p//4) for p in range(16)]
def feeding(col): return [((col+r)%4)*4+r for r in range(4)]
def mat_to_int(M):
    return sum(int(M[pos%4,pos//4])<<(8*(15-pos)) for pos in range(16))

m=None; K0_TRUE=None; K1_TRUE=None

# ------------------------- GPU neural-network pass -------------------------
def enc_full_gpu(bits_t):            # (N,128) DEVICE tensor -> (N,128) rounded 0/1 DEVICE tensor
    st=bits_t.view(-1,4,4,8).transpose(1,2)
    st=m.ARK(st,m.buff_round_keys[0])
    for r in range(1,m.number_of_rounds):          # 14 rounds (set by aes256_core)
        st=m.AES_round(st);st=m.ARK(st,m.buff_round_keys[r])
    st=m.AES_round(st,final=True);st=m.ARK(st,m.buff_round_keys[m.number_of_rounds])
    return torch.round(st.transpose(1,2).reshape(-1,128))

# ============ Phase 1 : K0 symmetric per-bit (GPU) ============
def recover_K0(rng):
    aval=np.zeros(128,bool)
    for _ in range(K0_BASES):
        base=rng.randint(0,2,128).astype(np.float32)
        rows=np.repeat(base[None],256,0)
        for i in range(128): rows[2*i,i]=+DELTA; rows[2*i+1,i]=-DELTA
        bt=torch.as_tensor(rows,dtype=DTYPE,device=DEVICE)
        C=enc_full_gpu(bt).reshape(128,2,128).cpu().numpy()
        for i in range(128):
            if not np.array_equal(C[i,0],C[i,1]): aval[i]=True
    K0=np.zeros((4,4),np.uint8)
    for byte in range(16):
        v=0
        for b in range(8): v=(v<<1)|int(aval[byte*8+b])
        K0[byte%4,byte//4]=v
    return K0

# ============ Phase 2 : K1 collision column (GPU) ============
def pt_bits(M1,K0):
    X0=inv_sr(SINV[inv_mc(M1)]);P=np.bitwise_xor(X0,K0[None])
    by=np.stack([P[:,r,c] for r,c in _POS],1).astype(np.uint8)
    return np.unpackbits(by,1).astype(np.float32)

def enc_collision_counts(col,cand,K0,other):
    """Oracle: for each candidate, return how many of the 32 probes leave the final ciphertext unchanged
       (a collision with the baseline).  Uses only attacker-observable ciphertext.
       (Plaintext construction pt_bits needs only K0. Round keys K1..K14 are used by the oracle m, which encrypts with its own key.)"""
    B=len(cand)
    cb=np.stack([(cand>>(8*(3-r)))&0xff for r in range(4)],1).astype(np.uint8)
    M1=np.tile(other.astype(np.uint8),(B,1,1));M1[:,:,col]=cb
    base_np=pt_bits(M1,K0)                                 # (B,128) plaintext bits (uses only K0)
    idx=[b*8+k for b in feeding(col) for k in range(8)]    # 32 direction probes
    base=torch.as_tensor(base_np,dtype=DTYPE,device=DEVICE)
    V=base.unsqueeze(0).repeat(len(idx)+1,1,1)             # (P+1,B,128)
    for j,i in enumerate(idx):                             # perturb each probe bit toward the collision value 0x52
        cur=V[j+1,:,i]
        V[j+1,:,i]=torch.where(cur>0.5,cur.new_full(cur.shape,1-DELTA),cur.new_full(cur.shape,DELTA))
    C=enc_full_gpu(V.reshape((len(idx)+1)*B,128)).reshape(len(idx)+1,B,-1)   # full 14R ciphertext (rounded 0/1)
    cnt=torch.zeros(B,dtype=torch.int32,device=DEVICE)
    for k in range(1,len(idx)+1):
        cnt+=(C[k]==C[0]).all(dim=1).to(torch.int32)       # ciphertext collision = exactly equal to baseline
    return cnt.cpu().numpy()

def _fmt_eta(sec):
    return f"{sec/3600:.2f}h" if sec>=3600 else (f"{sec/60:.1f}m" if sec>=60 else f"{sec:.0f}s")

def recover_K1_column_full(col,K0,other,verbose=True,record_all=False,hitfile=None):
    """Exhaustive blind streaming: scan c over the full column space [0, 2^COL_BITS)=2^32.  Oracle = ciphertext collisions (enc_full) only.
       record_all=False -> stop at the first ciphertext-collision candidate.
       record_all=True  -> exhaust to the end. Record every ciphertext-collision candidate to hitfile (census of false positives).
       The scan never references K1_TRUE (true attack)."""
    limit=1<<COL_BITS; t0=time.time(); last=t0
    confirmed=[]                                           # (K1col, c) : ciphertext collisions ≥ COLL_MIN
    hist = np.zeros(33, dtype=np.int64) if HIST else None   # counts of candidates by collision count 0..32 (full space)
    histfile = f"col{col}_hist.txt" if HIST else None
    for s in range(0,limit,BATCH):
        cand=np.arange(s, min(s+BATCH,limit), dtype=np.int64)
        cnt=enc_collision_counts(col,cand,K0,other)        # ciphertext collision count (observable values only)
        if HIST: hist += np.bincount(np.minimum(cnt,32).astype(np.int64), minlength=33)
        scanned=s+len(cand)
        for h in np.where(cnt>=COLL_MIN)[0]:
            c=int(cand[h]); en=int(cnt[h])
            K1col=[((c>>(8*(3-r)))&0xff)^DEADZONE for r in range(4)]
            if hitfile:
                with open(hitfile,"a",encoding="utf-8") as hf:
                    hf.write(f"c=0x{c:08x} scanned={scanned} enc={en} K1_col={[f'{x:02x}' for x in K1col]}\n")
            confirmed.append((K1col,c))
            rate=scanned/max(time.time()-t0,1e-9)
            print(f"    col{col}: collision candidate c=0x{c:08x} (ciphertext collisions {en}/32, {scanned:,} scanned, {rate:,.0f} cand/s)",flush=True)
            if not record_all:
                return K1col,c
        now=time.time()
        if verbose and now-last>=PROGRESS_SEC:
            rate=scanned/max(now-t0,1e-9); eta=(limit-scanned)/max(rate,1e-9)
            extra=f"  candidates={len(confirmed)}" if record_all else ""
            print(f"    col{col}: {scanned:,}/{limit:,}  {rate:,.0f} cand/s  ETA {_fmt_eta(eta)}{extra}",flush=True)
            last=now
            if HIST:
                with open(histfile,"w",encoding="utf-8") as hf:
                    hf.write(f"scanned={scanned} limit={limit}\nhist={hist.tolist()}\n")
    if HIST:
        with open(histfile,"w",encoding="utf-8") as hf:
            hf.write(f"scanned={limit} limit={limit}\nhist={hist.tolist()}\n")
    if not confirmed:
        return None,None
    if len(confirmed)>1:
        print(f"  [col{col}] ⚠ {len(confirmed)} ciphertext-collision candidates (includes false positives)! Check {hitfile}",flush=True)
    return confirmed[0][0],confirmed[0][1]

def recover_K1(K0,rng=None):
    """One full scan per column with the fixed all-zero non-target state (_ZERO44).
       The forced collision counts of the paper's Proposition 1 do not depend on the non-target
       state, so there is nothing to gain from rescanning with another base; a column whose true
       candidate falls below COLL_MIN is handled by the ranking fallback (paper, Appendix B).
       `rng` is accepted for call-site compatibility and unused."""
    K1=np.zeros((4,4),np.uint8);ok=True                   # ===== exhaustive streaming (true attack) =====
    for col in FULL_COLS:
        hitfile=f"col{col}_hits.txt" if RECORD_ALL else None
        if hitfile: open(hitfile,"w",encoding="utf-8").close()   # reset at column start
        K1col,c=recover_K1_column_full(col,K0,_ZERO44,verbose=True,
                                       record_all=RECORD_ALL,hitfile=hitfile)
        if K1col is None:
            ok=False; print(f"  [col{col}] not found (out of range or oracle failure)",flush=True); continue
        K1[:,col]=K1col
        print(f"  [col{col}] c=0x{c:08x} -> K1_col={[f'{x:02x}' for x in K1[:,col]]}",flush=True)
    return K1,ok

PT_KNOWN=0x00112233445566778899aabbccddeeff   # chosen known plaintext for downstream candidate verification

def _record_oracle_ct(model):
    # Submit one chosen binary plaintext to the oracle and record the observed ciphertext,
    # so run_attack.py can verify assembled candidates against an OBSERVED (PT,CT) pair
    # instead of regenerating the secret key. This is a legitimate black-box query.
    try:
        ct=int(nn_aes.encrypt_list_of_plaintexts([PT_KNOWN], model, dtype=DTYPE)[0])
        with open("oracle_ct.txt","w",encoding="utf-8") as f:
            f.write(f"PT_KNOWN=0x{PT_KNOWN:032x}\n")
            f.write(f"CT_OBSERVED=0x{ct:032x}\n")
    except Exception as e:
        print(f"[warn] could not record oracle_ct.txt: {e}",flush=True)

# =====================================================================
def main():
    global m,K0_TRUE,K1_TRUE
    keyrng=random.Random(SEED)
    trials = 1                                            # full attack is expensive: one key per run (vary via SEED)
    _mode=(f"[FULL] exhaustive blind streaming of 2^{COL_BITS} per column  columns={FULL_COLS}")
    print("="*74)
    print(" [GPU] Full master-key recovery of natural AES-256 implementation (K0 symmetric + K1 collision)")
    print(f" device={_gname}"+(f" ({_vram:.1f}GB cc{_cc[0]}.{_cc[1]})" if USE_CUDA else "")+
          f"  dtype={str(DTYPE).replace('torch.','')}  δ={DELTA}  BATCH={BATCH}  model={os.environ.get('MODEL','base')}")
    print(f" torch={torch.__version__}  cuda={torch.version.cuda}")
    print(f" {_mode}")
    print("="*74)
    T0=time.time()
    for t in range(1,trials+1):
        KEY=keyrng.getrandbits(256)
        m,rks=build_aes256(KEY)
        m=m.to(DEVICE)                       # move registered buffers (weights are already on DEVICE at build time)
        _record_oracle_ct(m)                 # one legitimate oracle query: chosen known plaintext -> observed ciphertext
        K0_TRUE=np.array(utils.integer_to_bytes_matrix(rks[0]),np.uint8)
        K1_TRUE=np.array(utils.integer_to_bytes_matrix(rks[1]),np.uint8)
        t0=time.time()
        K0=recover_K0(np.random.RandomState(keyrng.getrandbits(31)))
        ok0=np.array_equal(K0,K0_TRUE)                 # ── Phase1: report RoundKey0 (round-0 key) immediately
        print(f"   [K0] RoundKey0 (round-0 key) symmetric-attack recovery {'OK ' if ok0 else 'X  '} "
              f"0x{mat_to_int(K0):032x}  ({time.time()-t0:.1f}s)",flush=True)
        if not ok0:
            print(f"        actual K0 = 0x{mat_to_int(K0_TRUE):032x}",flush=True)
        K1,ok1=recover_K1(K0,random.Random(keyrng.getrandbits(64)))
        master=(mat_to_int(K0)<<128)|mat_to_int(K1)
        ok=(master==KEY)
        print(f" [{t}] K0={'OK' if np.array_equal(K0,K0_TRUE) else 'X'}  "
              f"K1={'OK' if ok1 and np.array_equal(K1,K1_TRUE) else 'X'}  "
              f"256-bit master-key recovery {'PASS' if ok else 'FAIL'}  ({time.time()-t0:.1f}s)")
        print(f"     recovered = 0x{master:064x}")
        print(f"     actual    = 0x{KEY:064x}")
    print("-"*74)
    print(f" total {time.time()-T0:.1f}s")
    if not USE_CUDA:
        print("[note] CUDA not detected -> running on CPU. With a CUDA-built torch, the GPU is used automatically.")

if __name__=="__main__": main()
