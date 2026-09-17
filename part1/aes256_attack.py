# -*- coding: utf-8 -*-
# Copyright (C) 2026 the authors (anonymized for review)
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
aes256_attack.py  —  full master-key recovery for native AES-256 (CPU reference demo)
  K0 (symmetric per-bit) + K1 (collision column) => RoundKey0 || RoundKey1 = 256-bit master key.
  * nn_aes.py / utils.py are not modified (aes256_core only reuses the gates).
  * Unlike AES-128, in AES-256 K0 is only the first 128 bits -> K1 is a genuinely new key (last 128 bits).

  Oracle: ciphertext-only. collide() reads solely the rounded 14-round ciphertext
  (enc_full); no internal round state is inspected, matching the black-box threat model.

  Scope: this is a REDUCED-SCALE demonstration of the full chain. The true 32-bit
  column value is inserted into a 2^K1_DEMO_BITS candidate pool so the end-to-end
  run completes quickly on CPU. The genuine blind 2^32-per-column search (no
  planting, ciphertext-only) is aes256_attack_gpu.py, with its full census in
  artifacts/.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, random, argparse
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, torch, nn_aes, utils
from aes256_core import build_aes256          # AES-256 builder reusing nn_aes gates
nn_aes.DEVICE = torch.device("cpu"); nn_aes.DTYPE = torch.float64

DEADZONE=0x52; DELTA=0.05; K0_BASES=4; K1_DEMO_BITS=10; COLL_MIN=8
BATCH=4096                                    # candidate scan batch (each candidate = 32+1 rows through enc_full)
SINV=np.array(utils.SBOX_INV,np.uint8)
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
def enc_full(bits):                  # 14 rounds (uses the model's number_of_rounds)
    st=torch.tensor(np.asarray(bits,float),dtype=nn_aes.DTYPE).view(-1,4,4,8).transpose(1,2)
    st=m.ARK(st,m.buff_round_keys[0])
    for r in range(1,m.number_of_rounds):
        st=m.AES_round(st);st=m.ARK(st,m.buff_round_keys[r])
    st=m.AES_round(st,final=True);st=m.ARK(st,m.buff_round_keys[m.number_of_rounds])
    return np.rint(st.transpose(1,2).reshape(-1,128).numpy())
# --- K0: symmetric per-bit (|u-v| symmetry of ARK0) ---
def recover_K0(rng):
    aval=np.zeros(128,bool)
    for _ in range(K0_BASES):
        base=rng.randint(0,2,128).astype(np.float32)
        rows=np.repeat(base[None],256,0)
        for i in range(128): rows[2*i,i]=+DELTA; rows[2*i+1,i]=-DELTA
        C=enc_full(rows).reshape(128,2,128)
        for i in range(128):
            if not np.array_equal(C[i,0],C[i,1]): aval[i]=True
    K0=np.zeros((4,4),np.uint8)
    for byte in range(16):
        v=0
        for b in range(8): v=(v<<1)|int(aval[byte*8+b])
        K0[byte%4,byte//4]=v
    return K0
# --- K1: collision column (invert round1 using K0) ---
def pt_bits(M1,K0):
    X0=inv_sr(SINV[inv_mc(M1)]);P=np.bitwise_xor(X0,K0[None])
    by=np.stack([P[:,r,c] for r,c in _POS],1).astype(np.uint8)
    return np.unpackbits(by,1).astype(np.float32)
def collide(col,cand,K0,other):
    # Ciphertext-only oracle: count, per candidate, how many of the 32 inward probes
    # reproduce the baseline *rounded ciphertext* (enc_full). No internal state is read.
    B=len(cand);cb=np.stack([(cand>>(8*(3-r)))&0xff for r in range(4)],1).astype(np.uint8)
    M1=np.tile(other,(B,1,1));M1[:,:,col]=cb
    base=pt_bits(M1,K0);idx=[b*8+k for b in feeding(col) for k in range(8)]
    rows=[base]
    for i in idx:
        r=base.copy();r[:,i]=np.where(base[:,i]>0.5,1-DELTA,DELTA);rows.append(r)
    C=enc_full(np.concatenate(rows,0)).reshape(len(rows),B,-1)
    cnt=np.zeros(B,int)
    for k in range(1,len(rows)): cnt+=np.all(C[k]==C[0],axis=1)
    return cnt   # collision count over the 32 probes (accept when >= COLL_MIN)
def recover_K1(K0,rng):
    K1=np.zeros((4,4),np.uint8);ok=True
    for col in range(4):
        tt=int.from_bytes(bytes(int(K1_TRUE[r,col])^DEADZONE for r in range(4)),"big")
        pool=np.array([rng.randint(0,(1<<32)-1) for _ in range(1<<K1_DEMO_BITS)],dtype=np.int64)
        pool[rng.randint(0,len(pool)-1)]=tt
        found=None
        other=np.zeros((4,4),np.uint8)                 # fixed non-target state, as in the full attack
        for s in range(0,len(pool),BATCH):
            c=collide(col,pool[s:s+BATCH],K0,other)
            if c.max()>=COLL_MIN: found=int(pool[s:s+BATCH][np.argmax(c)]);break
        if found is None: ok=False;continue
        K1[:,col]=[((found>>(8*(3-r)))&0xff)^DEADZONE for r in range(4)]
    return K1,ok

def main():
    global m,K0_TRUE,K1_TRUE,K1_DEMO_BITS
    ap=argparse.ArgumentParser(description="CPU/float64 reduced-scale AES-256 master-key recovery demo")
    ap.add_argument("--demo-bits",dest="demo_bits",type=int,default=K1_DEMO_BITS,
                    help=f"K1 demo pool size = 2^this per column (default {K1_DEMO_BITS})")
    ap.add_argument("--trials",type=int,default=3,help="number of random keys to recover (default 3)")
    ap.add_argument("--seed",type=int,default=2026,help="PRNG seed for the key stream (default 2026)")
    a=ap.parse_args()
    K1_DEMO_BITS=a.demo_bits
    keyrng=random.Random(a.seed)
    print("="*70); print(f" native AES-256 full master-key recovery (K0 symmetric + K1 collision)  [2^{K1_DEMO_BITS} pool, {a.trials} keys]"); print("="*70)
    for t in range(1,a.trials+1):
        KEY=keyrng.getrandbits(256)
        m,rks=build_aes256(KEY)
        K0_TRUE=np.array(utils.integer_to_bytes_matrix(rks[0]),np.uint8)
        K1_TRUE=np.array(utils.integer_to_bytes_matrix(rks[1]),np.uint8)
        K0=recover_K0(np.random.RandomState(keyrng.getrandbits(31)))
        K1,ok1=recover_K1(K0,random.Random(keyrng.getrandbits(64)))
        master=(mat_to_int(K0)<<128)|mat_to_int(K1)
        ok=(master==KEY)
        print(f" [{t}] K0={'OK' if np.array_equal(K0,K0_TRUE) else 'X'}  K1={'OK' if ok1 and np.array_equal(K1,K1_TRUE) else 'X'}  "
              f"256-bit master key recovery {'PASS' if ok else 'FAIL'}")
        print(f"     recovered = 0x{master:064x}")
        print(f"     actual    = 0x{KEY:064x}")
if __name__=="__main__": main()
