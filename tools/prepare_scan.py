#!/usr/bin/env python3
"""
Prepare one ToothFairy2 CBCT case for the Pearl 3D prototype.

Reads an image (.mha / .nii.gz / any SimpleITK format) and its label map, reorients to the
prototype's frame, resamples to a web-sized isotropic grid and writes:

  <out>/scan.json      header: dims, spacing, origin, grey mapping, teeth, licence/attribution
  <out>/gray.u8.gz     uint8 grey volume, gzip
  <out>/labels.u8.gz   uint8 structure labels in the prototype's codes, gzip
  <out>/preview.png    mid axial / coronal / sagittal slices, for a quick check

Prototype frame (mm): +x patient left, +y superior, +z anterior; occlusal plane at y = 0,
dental midline at x = 0. Voxel order: index = i + nx*(k + nz*j)  (x fastest, then z, then y).

Prototype label codes: 0 air · 1 soft tissue · 2 maxilla · 3 mandible · 4 sinus · 5 nasal ·
6 airway · 7 mandibular canal · 8 restoration/implant · 100+n tooth n (order of scan.json teeth) ·
254 outside the field of view.

Usage:
  .venv/bin/python prepare_scan.py IMAGE LABELS [--dataset-json dataset.json] [--out ../v2/data/scan]
                                   [--spacing 0.4] [--max-dim 256] [--auto-window] [--name CASE]
"""
import argparse, gzip, json, re, sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ToothFairy2 label ids as published with the challenge (dataset.json overrides this).
TF2_DEFAULT = {
    1: "Lower Jawbone", 2: "Upper Jawbone", 3: "Left Inferior Alveolar Canal",
    4: "Right Inferior Alveolar Canal", 5: "Left Maxillary Sinus", 6: "Right Maxillary Sinus",
    7: "Pharynx", 8: "Bridge", 9: "Crown", 10: "Implant",
    **{q * 10 + t: f"FDI {q * 10 + t}" for q in (1, 2, 3, 4) for t in range(1, 9)},
}
QUAD = {("upper", "right"): 1, ("upper", "left"): 2, ("lower", "left"): 3, ("lower", "right"): 4}
TOOTH = ["central incisor", "lateral incisor", "canine", "first premolar", "second premolar",
         "first molar", "second molar", "third molar"]

# HU-like value → prototype grey (0..255). Matches the procedural model's grey scale, so the
# viewer's window presets keep working: air 4, soft tissue ~78, bone 150–222, teeth 200–255.
HU_POINTS = [(-1000, 4), (-300, 40), (0, 78), (300, 120), (800, 160), (1500, 205), (2500, 240), (3500, 255)]


def parse_labels(dataset_json):
    """Return {id: name}. Accepts nnU-Net v1 ({"0": "background"}) and v2 ({"background": 0})."""
    if not dataset_json:
        return dict(TF2_DEFAULT)
    d = json.loads(Path(dataset_json).read_text())
    labels = d.get("labels", d)
    out = {}
    for k, v in labels.items():
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
            out[int(v)] = k
        else:
            out[int(k)] = v
    out.pop(0, None)
    return out


def name_to_code(name, lid):
    """Map a dataset label name to (prototype code or 'tooth', fdi)."""
    n = name.lower()
    m = re.search(r"\bfdi\s*(\d\d)\b", n)
    if m:
        return "tooth", int(m.group(1))
    for (ud, lr), q in QUAD.items():
        if ud in n and lr in n:
            for t, tn in enumerate(TOOTH):
                if tn in n:
                    return "tooth", q * 10 + t + 1
    if "canal" in n:
        return 7, None
    if "sinus" in n:
        return 4, None
    if "pharynx" in n or "airway" in n:
        return 6, None
    if "nasal" in n:
        return 5, None
    if "lower jaw" in n or "mandible" in n:
        return 3, None
    if "upper jaw" in n or "maxilla" in n:
        return 2, None
    if any(w in n for w in ("bridge", "crown", "implant", "restoration", "filling")):
        return 8, None
    if 11 <= lid <= 48 and lid % 10 in range(1, 9):
        return "tooth", lid
    return None, None


def resample(img, spacing, interp, default):
    size = [int(round(sz * sp / spacing)) for sz, sp in zip(img.GetSize(), img.GetSpacing())]
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing((spacing,) * 3)
    f.SetSize(size)
    f.SetOutputOrigin(img.GetOrigin())
    f.SetOutputDirection(img.GetDirection())
    f.SetInterpolator(interp)
    f.SetDefaultPixelValue(default)
    if interp == sitk.sitkLinear:   # anti-alias before shrinking
        sig = max(0.0, (spacing - min(img.GetSpacing())) / 2)
        if sig > 0:
            img = sitk.SmoothingRecursiveGaussian(img, sig)
    return f.Execute(img)


def to_proto(arr):
    """SimpleITK array of an LPS image is [S, P, L]; prototype wants [y=S, z=anterior=-P, x=L]."""
    return np.ascontiguousarray(arr[:, ::-1, :])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("labels")
    ap.add_argument("--dataset-json", help="ToothFairy2 dataset.json (exact label names); default table otherwise")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "v2" / "data" / "scan"))
    ap.add_argument("--spacing", type=float, default=0.4, help="target voxel size in mm (default 0.4)")
    ap.add_argument("--max-dim", type=int, default=256, help="cap on any axis; spacing grows to fit")
    ap.add_argument("--auto-window", action="store_true", help="map grey by percentiles instead of HU points")
    ap.add_argument("--name", help="case name for the header (default: image file stem)")
    ap.add_argument("--source", default="ToothFairy2 (UNIMORE, MICCAI 2024) — https://toothfairy2.grand-challenge.org/")
    ap.add_argument("--licence", default="CC BY-SA 4.0 (derived from ToothFairy2; cite the ToothFairy2 challenge papers)")
    ap.add_argument("--anatomy", help="path (relative to the scan folder) of a matching anatomy.json mesh set")
    a = ap.parse_args()

    img = sitk.ReadImage(a.image, sitk.sitkFloat32)
    lab = sitk.ReadImage(a.labels)
    img = sitk.DICOMOrient(img, "LPS")
    lab = sitk.DICOMOrient(lab, "LPS")
    if img.GetSize() != lab.GetSize():
        lab = sitk.Resample(lab, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    print(f"input  size {img.GetSize()}  spacing {tuple(round(s, 3) for s in img.GetSpacing())} mm")

    extent = [sz * sp for sz, sp in zip(img.GetSize(), img.GetSpacing())]
    sp = max(a.spacing, max(extent) / a.max_dim)
    raw = sitk.GetArrayViewFromImage(img)
    pad = float(raw.min())
    img_r = resample(img, sp, sitk.sitkLinear, pad)
    lab_r = resample(lab, sp, sitk.sitkNearestNeighbor, 0)
    ones = sitk.Image(img.GetSize(), sitk.sitkUInt8) + 1
    ones.CopyInformation(img)
    inside = resample(ones, sp, sitk.sitkNearestNeighbor, 0)

    G = to_proto(sitk.GetArrayFromImage(img_r))
    Lsrc = to_proto(sitk.GetArrayFromImage(lab_r)).astype(np.int32)
    FOV = to_proto(sitk.GetArrayFromImage(inside)) > 0
    ny, nz, nx = G.shape
    print(f"output size x{nx} y{ny} z{nz} at {sp:.3f} mm  ({nx * ny * nz / 1e6:.1f} M voxels)")

    # values that equal the scanner's padding are outside the reconstructed field
    FOV &= G > pad + 1e-3
    vals = G[FOV]
    q = np.percentile(vals, [1, 5, 50, 95, 99, 99.9])
    print("grey percentiles  p1 {:.0f}  p5 {:.0f}  p50 {:.0f}  p95 {:.0f}  p99 {:.0f}  p99.9 {:.0f}".format(*q))
    if a.auto_window or q[5] < 1000:
        if not a.auto_window:
            print("  p99.9 < 1000: values don't look like HU, switching to percentile mapping")
        pts = [(q[0], 4), (q[2], 70), (q[3], 150), (q[4], 215), (q[5], 252)]
        mapping = {"mode": "percentile", "points": [[float(x), y] for x, y in pts]}
    else:
        pts = HU_POINTS
        mapping = {"mode": "hu", "points": [[x, y] for x, y in pts]}
    xs, ys = zip(*pts)
    G8 = np.interp(G, xs, ys).astype(np.uint8)

    # labels
    names = parse_labels(a.dataset_json)
    if not a.dataset_json:
        print("  no --dataset-json: using the built-in ToothFairy2 id table (check the mapping below)")
    present = sorted(int(v) for v in np.unique(Lsrc) if v)
    code_of, fdi_of = {}, {}
    for lid in present:
        code, fdi = name_to_code(names.get(lid, f"id {lid}"), lid)
        if code == "tooth":
            fdi_of[lid] = fdi
        elif code is not None:
            code_of[lid] = code
        print(f"  label {lid:3d}  {names.get(lid, '?'):34s} → {('tooth ' + str(fdi)) if code == 'tooth' else code}")
    teeth_ids = sorted(fdi_of, key=lambda l: fdi_of[l])

    L = np.where(G8 < 30, 0, 1).astype(np.uint8)
    for lid, code in code_of.items():
        L[Lsrc == lid] = code
    for n, lid in enumerate(teeth_ids):
        L[Lsrc == lid] = 100 + n
    L[~FOV] = 254
    G8[~FOV] = 0

    # coordinates: centre x on the dental midline and put the occlusal plane at y = 0
    def mm(j, k, i):
        return np.stack([i * sp, j * sp, k * sp], -1)

    tips = {}
    for lid in teeth_ids:
        j, k, i = np.nonzero(Lsrc == lid)
        if len(j) < 20:
            continue
        p = mm(j, k, i)
        up = fdi_of[lid] // 10 in (1, 2)
        y = p[:, 1]
        sel = y <= np.percentile(y, 4) if up else y >= np.percentile(y, 96)
        tips[lid] = p[sel].mean(0)
    if tips:
        allp = np.array(list(tips.values()))
        upper = [tips[l] for l in tips if fdi_of[l] // 10 in (1, 2)]
        lower = [tips[l] for l in tips if fdi_of[l] // 10 in (3, 4)]
        occl = np.mean([np.mean([t[1] for t in upper]) if upper else allp[:, 1].mean(),
                        np.mean([t[1] for t in lower]) if lower else allp[:, 1].mean()])
        shift = np.array([-allp[:, 0].mean(), -occl, -(nz - 1) * sp / 2])
    else:
        shift = -np.array([(nx - 1) * sp / 2, (ny - 1) * sp / 2, (nz - 1) * sp / 2])
    origin = shift.tolist()

    restor = np.isin(Lsrc, [l for l, c in code_of.items() if c == 8])
    teeth = []
    for n, lid in enumerate(teeth_ids):
        j, k, i = np.nonzero(Lsrc == lid)
        if not len(j):
            continue
        p = mm(j, k, i) + shift
        up = fdi_of[lid] // 10 in (1, 2)
        y = p[:, 1]
        lo, hi = np.percentile(y, 4), np.percentile(y, 96)
        tip = p[y <= lo].mean(0) if up else p[y >= hi].mean(0)
        apex = p[y >= hi].mean(0) if up else p[y <= lo].mean(0)
        length = float(np.linalg.norm(apex - tip)) or 1.0
        bb = [int(j.min()), int(j.max()), int(k.min()), int(k.max()), int(i.min()), int(i.max())]
        sub = restor[max(bb[0] - 3, 0):bb[1] + 4, max(bb[2] - 3, 0):bb[3] + 4, max(bb[4] - 3, 0):bb[5] + 4]
        teeth.append({
            "fdi": fdi_of[lid], "label": 100 + n, "voxels": int(len(j)),
            "centroid": [round(float(v), 2) for v in p.mean(0)],
            "tip": [round(float(v), 2) for v in tip], "apex": [round(float(v), 2) for v in apex],
            "length": round(length, 2),
            "radius": round(float(np.sqrt(len(j) * sp ** 3 / (np.pi * length)) * 1.15), 2),
            "restored": bool(sub.sum() > 20),
            "bbox_jkI": bb,
        })

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "gray.u8.gz", "wb", compresslevel=6) as f:
        f.write(G8.tobytes())
    with gzip.open(out / "labels.u8.gz", "wb", compresslevel=6) as f:
        f.write(L.tobytes())
    header = {
        "format": "pearl-proto-scan/1",
        "name": a.name or Path(a.image).name.split(".")[0],
        "source": a.source, "licence": a.licence, **({"anatomy": a.anatomy} if a.anatomy else {}),
        "frame": "+x patient left, +y superior, +z anterior; mm; occlusal plane y=0",
        "layout": "uint8, index = i + nx*(k + nz*j)",
        "dims": [nx, ny, nz], "spacing": sp, "origin": [round(v, 3) for v in origin],
        "grey": mapping,
        "labels": {"0": "air", "1": "soft tissue", "2": "maxilla", "3": "mandible", "4": "sinus", "5": "nasal",
                   "6": "airway", "7": "mandibular canal", "8": "restoration", "100+": "teeth", "254": "outside FOV"},
        "structures": sorted({int(c) for c in code_of.values()}),
        "teeth": teeth,
        "files": {"gray": "gray.u8.gz", "labels": "labels.u8.gz"},
    }
    (out / "scan.json").write_text(json.dumps(header, indent=1))

    try:
        from PIL import Image
        jm, km, im = ny // 2, nz // 2, nx // 2
        if tips:
            jm = int(round(-shift[1] / sp))
        ax = G8[jm][::-1]            # rows: anterior at top
        co = G8[:, km, :][::-1]      # rows: superior at top
        sa = G8[:, :, im][::-1, ::-1]  # anterior on the left
        h = max(ax.shape[0], co.shape[0], sa.shape[0])
        pad_to = lambda a: np.pad(a, ((0, h - a.shape[0]), (0, 4)))
        Image.fromarray(np.hstack([pad_to(ax), pad_to(co), pad_to(sa)])).save(out / "preview.png")
    except Exception as e:  # preview is optional
        print("  preview skipped:", e)

    gz = sum((out / f).stat().st_size for f in ("gray.u8.gz", "labels.u8.gz"))
    print(f"wrote {out}  ({gz / 1e6:.1f} MB gzipped, {len(teeth)} teeth, structures {header['structures']})")


if __name__ == "__main__":
    sys.exit(main())
