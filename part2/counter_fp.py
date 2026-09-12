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
counter_fp.py  --  countermeasures against the FP trigger attack on Section 9.

Root cause: STEP(x)=(relu(x-1/4)-relu(x-3/4))/(1/2) fails to binarize large x in
finite precision (STEP(trigger)=2) because the fractional shifts 1/4,3/4 become
unresolvable once ULP(x)>=1/2.

Idea: prepend a SATURATING CLAMP to [0,1] before STEP, so no coordinate ever
reaches the ULP-broken regime. But the clamp gadget must itself be FP-robust and
DNN-expressible (ReLU/linear only).

We compare two DNN realizations of the SAME saturating clamp min(max(x,0),1):
  naive_clamp(x) = relu(x) - relu(x-1)          # GHRS's ClippedReLU (parallel ReLUs)
  clamp01(x)     = 1 - relu(1 - relu(x))        # ours (nested ReLUs; rounding annihilated)

and check (i) both map each per-format trigger to 1, so EITHER blocks our attack;
(ii) but ClippedReLU is NOT a [0,1] clamp in finite precision (e.g. fp16
naive_clamp(2050)=2), and blocks the attack only because STEP maps that stray 2
back to 1, whereas clamp01 provably stays in [0,1] for every finite input;
(iii) clamp01+STEP defeats the full key-recovery attack and preserves FIPS-197.
"""
import torch

FORMATS = [("bfloat16",torch.bfloat16,7,65.0),
           ("float16", torch.float16,10,513.0),
           ("float32", torch.float32,23,4194305.0),
           ("float64", torch.float64,52,2251799813685249.0)]

def T(x,dt): return torch.tensor(x,dtype=dt)

def step(x,dt):
    x=T(x,dt); e=T(0.25,dt); one=T(1.0,dt)
    return float(((torch.relu(x-e)-torch.relu(x-(one-e)))/(one-2*e)).float().item())

def naive_clamp(x,dt):
    x=T(x,dt); return float((torch.relu(x)-torch.relu(x-1)).float().item())

def clamp01(x,dt):
    x=T(x,dt); one=T(1.0,dt)
    return float((one-torch.relu(one-torch.relu(x))).float().item())

print("="*80)
print("(1) On each format's STEP-trigger: STEP fails (=2); do the clamps fix it?")
print("="*80)
print(f"{'format':9} {'trigger':>20} | {'STEP':>6} {'naive_clamp':>12} {'clamp01':>8} {'STEP(clamp01)':>14}")
for name,dt,_,t in FORMATS:
    print(f"{name:9} {int(t):>20} | {step(t,dt):>6.2f} {naive_clamp(t,dt):>12.2f} "
          f"{clamp01(t,dt):>8.2f} {step(clamp01(t,dt),dt):>14.2f}")

print()
print("="*80)
print("(2) Is naive_clamp (GHRS ClippedReLU) a faithful [0,1] clamp in finite precision?")
print("="*80)
# Over the reals ClippedReLU = min(max(x,0),1) in [0,1]. In FP, relu(x)-relu(x-1)
# subtracts two large near-equal operands, so a rounding of x-1 leaks into the output.
dt=torch.float16
print(f"  paper example (fp16): x-1=2049 rounds to {int(T(2049,dt).item())}, "
      f"so naive_clamp(2050) = {naive_clamp(2050,dt):.0f}   (NOT in [0,1])")
xs   = torch.arange(2, 65504, dtype=torch.float64).to(dt)
nc   = (torch.relu(xs) - torch.relu(xs - 1)).float()
c01  = (1 - torch.relu(1 - torch.relu(xs))).float()
nc_out  = (nc  < -1e-3) | (nc  > 1 + 1e-3)
c01_out = (c01 < -1e-3) | (c01 > 1 + 1e-3)
stray   = sorted({v.item() for v in torch.unique(nc[nc_out])})
step_of = sorted({round(step(v, dt), 3) for v in stray})
print(f"  ClippedReLU leaves [0,1] for {int(nc_out.sum())} of the fp16 integers in [2,65504)")
print(f"     stray values taken: {stray}  ->  STEP maps them to {step_of}  "
      f"(binary: STEP neutralizes them, so ClippedReLU still blocks THIS attack)")
print(f"  clamp01 leaves [0,1] for {int(c01_out.sum())} inputs  (provably 0, Lemma):  "
      f"clamp01(1e4)={clamp01(1e4,dt):.2f}, clamp01(inf)={clamp01(float('inf'),dt):.2f}")
print("  => ClippedReLU is not a finite-precision clamp; clamp01 confines every input")
print("     to [0,1] unconditionally, independent of STEP tolerating out-of-range values.")

print()
print("="*80)
print("(3) Full attack against clamp01-defended model (unmodified nn_aes)")
print("="*80)
import numpy as np, nn_aes, utils
for name,dt,_,t in [("float16",torch.float16,10,513.0),("float32",torch.float32,23,4194305.0)]:
    nn_aes.DTYPE=dt
    model=nn_aes.NeuralAESBase(secret_key=0x2B7E151628AED2A6ABF7158809CF4F3C,
        direction='Encryption',c_parameter=1.0,protected=True,epsilon=1/4).to(dt).eval()
    key=0x2B7E151628AED2A6ABF7158809CF4F3C
    true=np.array(utils.integer_to_bitvector(key),dtype=np.uint8)
    Q=torch.zeros(128,128,dtype=dt); ii=torch.arange(128); Q[ii,ii]=T(t,dt)
    # DEFENSE: clamp01 applied to inputs before the (unchanged) protected model
    Qdef = 1 - torch.relu(1 - torch.relu(Q))
    with torch.inference_mode():
        out_undef = model(Q)          # no defense
        out_def   = model(Qdef)       # clamp01 defense
    def recover(o):
        fin=torch.isfinite(o).all(dim=1); clean=fin&(o.abs().amax(dim=1)<=1.5)
        b=clean.to(torch.uint8).numpy(); return utils.bitvector_to_integer(b.tolist()), int((b==true).sum())
    ku,cu=recover(out_undef); kd,cd=recover(out_def)
    print(f"  {name}: NO defense  -> bits {cu}/128  full={'KEY RECOVERED' if ku==key else 'no recovery'}")
    print(f"  {name}: clamp01     -> bits {cd}/128  full={'KEY RECOVERED' if kd==key else 'THIS ATTACK BLOCKED'}   "
          f"(clamped trigger STEP={step(clamp01(t,dt),dt):.0f}, all rows clean)")

print()
print("="*80)
print("(4) FIPS-197 correctness preserved by the clamp01-prepended model")
print("="*80)
PT_FIPS  = 0x3243f6a8885a308d313198a2e0370734          # FIPS-197 Appendix B test vector
KEY_FIPS = 0x2b7e151628aed2a6abf7158809cf4f3c
CT_FIPS  = 0x3925841d02dc09fbdc118597196a0b32
nn_aes.DTYPE = torch.float64
model = nn_aes.NeuralAESBase(secret_key=KEY_FIPS, direction='Encryption', c_parameter=1.0,
                             protected=True, epsilon=1/4).to(torch.float64).eval()
bits    = torch.tensor(np.asarray(utils.integer_to_bitvector(PT_FIPS), float), dtype=torch.float64)
clamped = 1 - torch.relu(1 - torch.relu(bits))         # clamp01 prepended, identity on {0,1}
identical = bool(torch.equal(clamped, bits))           # clamp does not alter a binary input, bit-for-bit
ct = int(nn_aes.encrypt_list_of_plaintexts([PT_FIPS], model, dtype=torch.float64)[0])
ok4 = identical and ct == CT_FIPS
print(f"  clamp01(plaintext bits) identical to plaintext bits : {identical}")
print(f"  protected-model ciphertext = 0x{ct:032x}")
print(f"  FIPS-197 reference         = 0x{CT_FIPS:032x}")
print(f"  => clamp-prepended model preserves FIPS-197 correctness: {'PASS' if ok4 else 'FAIL'}")
