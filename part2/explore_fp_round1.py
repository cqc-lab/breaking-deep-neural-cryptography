#!/usr/bin/env python3
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
explore_fp_round1.py
Can the FP trigger attack reach the Round-1 key (ARK1 / K1) under full Section 9?

Structure of encrypt():  ARK0 -> [AES_round; ARK1] -> [AES_round; ARK2] -> ...
So M1 (state just before ARK1) = AES_round(ARK0(post-STEP input)).
The FP trigger injects value 2 at ONE input coordinate; between that and ARK1 sits
a full nonlinear round (SubBytes) + MixColumns diffusion.

We test two things empirically (unmodified nn_aes, float64 to see finite avalanche):
  (E1) Is the readable (finite/NaN) oracle K1-invariant?  -> if yes, K1 is invisible.
  (E2) When a single trigger's value-2 reaches M1, is it a clean localized probe
       or an already-avalanched, column-coupled mess?
"""
import torch, numpy as np
import nn_aes, utils

nn_aes.DTYPE = torch.float64
dt = torch.float64
KEY  = 0x2B7E151628AED2A6ABF7158809CF4F3C
TRIG = float(2**51 + 1)

def build():
    m = nn_aes.NeuralAESBase(secret_key=KEY, direction='Encryption',
                             c_parameter=1.0, protected=True, epsilon=1/4)
    m = m.to(dt); m.eval()
    return m

K0bits = np.array(utils.integer_to_bitvector(KEY), dtype=np.uint8)  # ARK0 = master key

# ---------- E1: K1-invariance of the trigger oracle ----------
print("="*74)
print("E1: does the finite/NaN oracle change when ONLY K1 (round key 1) changes?")
print("="*74)
m = build()
Q = torch.zeros(128,128,dtype=dt); ii=torch.arange(128); Q[ii,ii]=TRIG

with torch.inference_mode():
    outA = m(Q)
finA = torch.isfinite(outA).all(dim=1)

K1_orig = m.buff_round_keys[1].clone()
torch.manual_seed(0)
m.buff_round_keys[1] = (torch.rand_like(K1_orig) > 0.5).to(dt)     # random new K1, same K0/rest
with torch.inference_mode():
    outB = m(Q)
finB = torch.isfinite(outB).all(dim=1)

same_pattern = bool((finA == finB).all().item())
print(f"  finite/NaN pattern identical across K1 change : {same_pattern}")
print(f"  variant-A finite pattern vs K0 (fp64 naive)   : "
      f"{int((finA.to(torch.uint8).cpu().numpy()==K0bits).sum())}/128"
      f"   (fp64 naive is weak by design; full K0 recovery is the fp16 result, see fp_key_recovery.py)")
clean = finA & finB
val_changed = (outA[clean] != outB[clean]).any(dim=1)
print(f"  clean(finite) rows whose CIPHERTEXT changed   : {int(val_changed.sum())}/{int(clean.sum())}")
print("  => oracle sees only K0; K1 shows up only inside correct AES ciphertexts")
m.buff_round_keys[1] = K1_orig  # restore

# ---------- E2: what does a single trigger look like at M1 (before ARK1)? ----------
print()
print("="*74)
print("E2: a single input trigger -> how localized/clean is it at M1 (ARK1 input)?")
print("="*74)

def state_at_M1(i0):
    """Post-STEP state with value 2 at coord i0, then ARK0 + one AES_round."""
    x = torch.zeros(1,128,dtype=dt); x[0,i0]=TRIG
    with torch.inference_mode():
        s  = m.step(x)                                   # STEP: 2 at i0, else 0
        st = s.view(-1,4,4,8).transpose(1,2)
        u  = m.ARK(st, m.buff_round_keys[0])             # after ARK0
        M1 = m.AES_round(u)                              # state before ARK1
    return M1.transpose(1,2).reshape(-1,128)[0], float(s[0,i0])

for i0 in [0, 1, 5, 8]:
    M1v, stepval = state_at_M1(i0)
    k0 = int(K0bits[i0])
    off = ((M1v < -0.01) | (M1v > 1.01))                 # coords not in [0,1]
    n_off = int(off.sum())
    finite = bool(torch.isfinite(M1v).all())
    mx = float(M1v.abs().max())
    print(f"  trigger@coord {i0:3d}  STEP={stepval:.1f}  k0={k0}  | "
          f"M1 off-domain coords={n_off:3d}/128  finite={finite}  max|M1|={mx:.3g}")
print("  (k0=1 -> ARK0 corrects 2->1 -> M1 fully binary, no probe;")
print("   k0=0 -> value 2 hits round-1 SubBytes -> off-domain spreads, not a clean 2)")
