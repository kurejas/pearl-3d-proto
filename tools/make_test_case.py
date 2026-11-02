#!/usr/bin/env python3
"""
Write a small synthetic ToothFairy2-style case (LPS .mha image + label map) to test prepare_scan.py
and the viewer's scan loader before a real scan is available.

It is deliberately asymmetric so orientation mistakes show up:
  · only the LEFT maxillary sinus is present (should appear at +x / screen right in the front 3D view)
  · tooth 16 (upper right first molar) is missing
  · tooth 36 (lower left first molar) carries a metal crown (label 9)
  · the arch opens towards ANTERIOR (incisors in front), the pharynx sits posterior

Usage: .venv/bin/python make_test_case.py [--out test_case]
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "test_case"))
a = ap.parse_args()

sp = 0.4
nL, nP, nS = 240, 250, 150                      # 96 × 100 × 60 mm
L = (np.arange(nL) - nL / 2) * sp              # +L = patient left
P = (np.arange(nP) - nP / 2) * sp              # +P = posterior
S = (np.arange(nS) - nS / 2) * sp              # +S = superior
z, y, x = np.meshgrid(S, P, L, indexing="ij")  # array order [S, P, L]

hu = np.full(x.shape, 0.0, np.float32)          # soft tissue
lab = np.zeros(x.shape, np.uint8)
A, B = 25.0, 30.0                               # arch half-width, depth (anterior is -P)

def arch_dist(x, y):
    t = np.linspace(-1.5, 1.5, 120)
    ax, ay = A * np.sin(t), -(B * np.cos(t)) + 6
    d = np.full(x.shape, 1e9, np.float32)
    for px, py in zip(ax, ay):
        d = np.minimum(d, (x - px) ** 2 + (y - py) ** 2)
    return np.sqrt(d)

da = arch_dist(x, y)
upper = (z > 3) & (z < 22) & (da < 6.5)
lower = (z < -3) & (z > -28) & (da < 7)
hu[upper | lower] = 1200
lab[lower] = 1
lab[upper] = 2

# left sinus only
sin = ((x - 16) / 8) ** 2 + ((y - 0) / 10) ** 2 + ((z - 26) / 8) ** 2 < 1
hu[sin] = -1000; lab[sin] = 5
# pharynx (posterior)
ph = (x ** 2 + (y - 30) ** 2 < 6.5 ** 2)
hu[ph] = -1000; lab[ph] = 7
# canals (inside mandible)
for s, lid in ((1, 3), (-1, 4)):
    t = np.linspace(.6, 1.4, 40)
    cx, cy = s * A * np.sin(t), -(B * np.cos(t)) + 6
    d = np.full(x.shape, 1e9, np.float32)
    for px, py in zip(cx, cy):
        d = np.minimum(d, (x - px) ** 2 + (y - py) ** 2 + (z + 20) ** 2)
    c = d < 1.4 ** 2
    hu[c] = 200; lab[c] = lid

# teeth: cylinders on the arch, FDI label ids
W = [8.6, 6.6, 7.6, 7, 6.8, 10.2, 9.2, 8.6]
tot = sum(W)
for q in (1, 2, 3, 4):
    side = -1 if q in (1, 4) else 1
    upq = q in (1, 2)
    cum = 0
    for i in range(8):
        num = q * 10 + i + 1
        th = side * 1.42 * (cum + W[i] / 2) / tot; cum += W[i]
        if num in (16, 18, 28, 48):
            continue
        tx, ty = A * np.sin(th), -(B * np.cos(th)) + 6
        r = W[i] * .42
        zz = (z > .5) & (z < 21) if upq else (z < -.5) & (z > -23)
        m = ((x - tx) ** 2 + (y - ty) ** 2 < r ** 2) & zz
        hu[m] = 2300; lab[m] = num
        if num == 36:
            c = m & (z > -6)
            hu[c] = 3600; lab[c] = 9

rng = np.random.default_rng(1)
hu += rng.normal(0, 60, hu.shape).astype(np.float32)
air = (x ** 2 / 46 ** 2 + (y + 2) ** 2 / 46 ** 2) > 1
hu[air] = -1000 + rng.normal(0, 20, int(air.sum())); lab[air] = 0
fov = x ** 2 + y ** 2 > 47 ** 2                   # reconstructed cylinder
hu[fov] = -2000; lab[fov] = 0

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for arr, name in ((hu, "case_0000.mha"), (lab, "case.mha")):
    im = sitk.GetImageFromArray(arr)
    im.SetSpacing((sp, sp, sp)); im.SetOrigin((float(L[0]), float(P[0]), float(S[0])))
    sitk.WriteImage(im, str(out / name), useCompression=True)
print("wrote", out)
