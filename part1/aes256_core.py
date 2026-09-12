# -*- coding: utf-8 -*-
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

# Without modifying nn_aes.py / utils.py, we reuse only their neural gates (round function)
# to construct a native AES-256 (256-bit key schedule + 14 rounds).
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
import nn_aes, utils
nn_aes.DEVICE = torch.device("cpu")      # verify exactly on CPU/float64
nn_aes.DTYPE  = torch.float64

_RCON = [0,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36]
_SB   = list(utils.SBOX)

def aes256_round_key_ints(key256):
    """256-bit master key -> 15 round keys (each a 128-bit integer). RoundKey0||RoundKey1 = master key."""
    kb = int(key256).to_bytes(32, "big")
    W = [list(kb[4*i:4*i+4]) for i in range(8)]         # W[0..7] = master key
    for i in range(8, 60):
        t = W[i-1][:]
        if i % 8 == 0:
            t = t[1:] + t[:1]                            # RotWord
            t = [_SB[b] for b in t]                      # SubWord
            t[0] ^= _RCON[i//8]
        elif i % 8 == 4:
            t = [_SB[b] for b in t]                      # SubWord (extra AES-256 step)
        W.append([W[i-8][k] ^ t[k] for k in range(4)])
    rks = []
    for r in range(15):
        b = bytes(W[4*r] + W[4*r+1] + W[4*r+2] + W[4*r+3])
        rks.append(int.from_bytes(b, "big"))
    return rks

def model_class(model=None):
    """Which of GHRS's two networks to build: 'base' (NeuralAESBase, default) or 'ttables'
    (TTablesNeuralAES). Defaults to the MODEL environment variable."""
    import os
    model = (model or os.environ.get("MODEL", "base")).lower()
    if model in ("ttables", "t-tables", "tt", "ttable"):
        return nn_aes.TTablesNeuralAES, "ttables"
    if model == "base":
        return nn_aes.NeuralAESBase, "base"
    raise ValueError(f"unknown MODEL {model!r}; use 'base' or 'ttables'")

def build_aes256(key256, dtype=None, device=None, model=None):
    """Native AES-256 model reusing nn_aes gates + installing AES-256 round keys / 14 rounds.
    model: 'base' (default) or 'ttables', else taken from the MODEL environment variable."""
    dtype = dtype or nn_aes.DTYPE
    dev   = device or nn_aes.DEVICE
    cls, _ = model_class(model)
    m = cls(0, c_parameter=1.0)                          # build gates (key-independent). dummy 128-bit key.
    rks = aes256_round_key_ints(key256)
    rk_mats = [np.array(utils.integer_to_bytes_matrix(x), np.uint8) for x in rks]  # 15 × (4,4)[row,col]
    m.number_of_rounds = 14
    m.round_keys = rk_mats
    rkt = torch.tensor(utils.bytes_matrix_to_binary_states(rk_mats), dtype=dtype)  # (15,4,4,8)
    m.buff_round_keys = rkt[:, None, :, :].to(dev)       # (15,1,4,4,8)
    return m, rks

if __name__ == "__main__":
    # NIST FIPS-197 AES-256 test vector
    KEY = 0x000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f
    PT  = 0x00112233445566778899aabbccddeeff
    CT_EXP = 0x8ea2b7ca516745bfeafc49904b496089
    m, rks = build_aes256(KEY)
    ct = nn_aes.encrypt_list_of_plaintexts([PT], m)[0]
    print(f"round key count: {len(rks)} (expected 15)")
    print(f"RoundKey0 = 0x{rks[0]:032x}")
    print(f"RoundKey1 = 0x{rks[1]:032x}")
    print(f"K0||K1    = 0x{(rks[0]<<128)|rks[1]:064x}")
    print(f"master key = 0x{KEY:064x}   match? {((rks[0]<<128)|rks[1])==KEY}")
    print(f"ciphertext = 0x{ct:032x}")
    print(f"expected   = 0x{CT_EXP:032x}   AES-256 correct? {ct==CT_EXP}")
