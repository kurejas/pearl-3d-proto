#!/usr/bin/env python3
"""
Build the v2 anatomy from BodyParts3D (real skull bones + 28 teeth with roots, one body, one frame).

Input:  a folder of BodyParts3D OBJ files (isa_BP3D_4.0_obj_99, element files FJxxxx.obj).
Output (default ../v2/data/anatomy/):
  anatomy.json      parts: key, group, fdi, name, vertex/index offsets; frame; licence
  anatomy.bin.gz    Float32 positions then Uint32 indices for every part, gzip
Optionally (--volume) also a synthetic CBCT image + ToothFairy2-style label map (.mha) that
prepare_scan.py turns into the slice-view volume, so the 3D and the slices share one anatomy.

Prototype frame (mm): +x patient left, +y superior, +z anterior; occlusal plane at y = 0, midline x = 0.
BodyParts3D frame: +x patient left, +y posterior, +z superior.

Licence: BodyParts3D, (c) The Database Center for Life Science, CC BY-SA 2.1 Japan.

Usage: .venv/bin/python prepare_bp3d.py OBJ_DIR [--out ../v2/data/anatomy] [--volume ../v2/data/bp3d_case]
"""
import argparse, gzip, json, re
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray

BONES = {  # element file → group
    "FJ3269": "maxilla", "FJ3375": "maxilla", "FJ3273": "maxilla", "FJ3379": "maxilla", "FJ3395": "maxilla",
    "FJ3263": "maxilla", "FJ3369": "maxilla",
    "FJ3289": "mandible",
    "FJ3199": "skull", "FJ3200": "skull", "FJ3265": "skull", "FJ3371": "skull", "FJ3272": "skull", "FJ3378": "skull",
    "FJ3274": "skull", "FJ3380": "skull", "FJ3281": "skull", "FJ3386": "skull", "FJ3287": "skull", "FJ3392": "skull",
    "FJ3309": "skull", "FJ3394": "skull",
}
TOOTH_ORDER = ["central secondary incisor", "lateral secondary incisor", "secondary canine",
               "first secondary premolar", "second secondary premolar", "first secondary molar", "second secondary molar"]
QUAD = {("right", "upper"): 1, ("left", "upper"): 2, ("left", "lower"): 3, ("right", "lower"): 4}


def read_obj(path):
    V, F, name = [], [], ""
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
        elif line.startswith("# English name"):
            name = line.split(":", 1)[1].strip()
    return np.array(V, np.float64), np.array(F, np.int64), name


def fdi_of(name):
    m = re.match(r"(Left|Right) (upper|lower) (.+) tooth", name)
    if not m:
        return None
    side, jaw, kind = m.group(1).lower(), m.group(2), m.group(3)
    return QUAD[(side, jaw)] * 10 + TOOTH_ORDER.index(kind) + 1


def to_poly(V, F):
    pts = vtk.vtkPoints(); pts.SetData(numpy_to_vtk(V.astype(np.float64), deep=True))
    cells = np.hstack([np.full((len(F), 1), 3), F]).ravel()
    ca = vtk.vtkCellArray(); ca.SetCells(len(F), numpy_to_vtkIdTypeArray(cells.astype(np.int64), deep=True))
    p = vtk.vtkPolyData(); p.SetPoints(pts); p.SetPolys(ca)
    return p


def from_poly(p):
    V = vtk_to_numpy(p.GetPoints().GetData()).astype(np.float32)
    F = vtk_to_numpy(p.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(np.uint32)
    return V, F


def refine(p, levels, smooth_iters):
    """Clean → subdivide (loop where the mesh allows, else linear) → windowed-sinc smooth."""
    c = vtk.vtkCleanPolyData(); c.SetInputData(p); c.Update(); p = c.GetOutput()
    t = vtk.vtkTriangleFilter(); t.SetInputData(p); t.Update(); p = t.GetOutput()
    for _ in range(levels):
        s = vtk.vtkLoopSubdivisionFilter(); s.SetNumberOfSubdivisions(1); s.SetInputData(p); s.Update()
        q = s.GetOutput()
        if q.GetNumberOfPolys() == 0:
            s = vtk.vtkLinearSubdivisionFilter(); s.SetNumberOfSubdivisions(1); s.SetInputData(p); s.Update(); q = s.GetOutput()
        p = q
    if smooth_iters:
        w = vtk.vtkWindowedSincPolyDataFilter(); w.SetInputData(p); w.SetNumberOfIterations(smooth_iters)
        w.SetPassBand(0.08); w.BoundarySmoothingOff(); w.FeatureEdgeSmoothingOff(); w.NonManifoldSmoothingOn()
        w.NormalizeCoordinatesOn(); w.Update(); p = w.GetOutput()
    return p


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("obj_dir")
    ap.add_argument("--out", default=str(here.parent / "v2" / "data" / "anatomy"))
    ap.add_argument("--volume", default=None, help="also write a synthetic CBCT case (image + labels .mha) here")
    ap.add_argument("--reuse", action="store_true", help="skip meshing; voxelise the existing anatomy.bin.gz")
    a = ap.parse_args()
    src = Path(a.obj_dir)
    if a.reuse:
        out = Path(a.out); meta = json.loads((out / "anatomy.json").read_text())
        buf = gzip.decompress((out / "anatomy.bin.gz").read_bytes())
        pos = np.frombuffer(buf[:meta["nv"] * 12], "<f4"); ind = np.frombuffer(buf[meta["nv"] * 12:], "<u4")
        return write_volume(Path(a.volume), meta["parts"], pos.reshape(-1, 3), ind)

    raw = {}
    for f in sorted(src.glob("FJ*.obj")):
        fid = f.stem
        V, F, name = read_obj(f)
        fdi = fdi_of(name)
        if fid in BONES or fdi:
            raw[fid] = (V, F, name, fdi)

    # frame: midline from the central incisors, occlusal plane between the arches, arch centre in z
    teeth = {r[3]: r[0] for r in raw.values() if r[3]}
    x0 = np.mean([teeth[n][:, 0].mean() for n in (11, 21, 31, 41)])
    up_low = np.mean([teeth[n][:, 2].min() for n in teeth if n // 10 in (1, 2)])
    lo_high = np.mean([teeth[n][:, 2].max() for n in teeth if n // 10 in (3, 4)])
    z0 = (up_low + lo_high) / 2
    y0 = np.mean([teeth[n][:, 1].mean() for n in teeth])

    def frame(V):  # BP3D (x left, y posterior, z superior) → prototype (x left, y superior, z anterior)
        return np.stack([V[:, 0] - x0, V[:, 2] - z0, -(V[:, 1] - y0)], 1)

    parts, blobs_v, blobs_i, nv, ni = [], [], [], 0, 0
    for fid, (V, F, name, fdi) in sorted(raw.items(), key=lambda kv: (kv[1][3] or 0, kv[0])):
        p = to_poly(frame(V), F)
        p = refine(p, levels=2 if fdi else 1, smooth_iters=12 if fdi else 15)
        Vo, Fo = from_poly(p)
        key = f"t{fdi}" if fdi else fid
        parts.append({"key": key, "file": fid, "name": name, "group": "tooth" if fdi else BONES[fid], "fdi": fdi,
                      "v0": nv, "nv": len(Vo), "i0": ni, "ni": Fo.size})
        blobs_v.append(Vo.ravel()); blobs_i.append(Fo.ravel()); nv += len(Vo); ni += Fo.size
        print(f"{key:8s} {name[:44]:44s} {len(V):6d} → {len(Vo):7d} verts")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pos = np.concatenate(blobs_v).astype("<f4"); ind = np.concatenate(blobs_i).astype("<u4")
    with gzip.open(out / "anatomy.bin.gz", "wb", compresslevel=9) as g:
        g.write(pos.tobytes()); g.write(ind.tobytes())
    meta = {"source": "BodyParts3D 4.0 (isa, obj_99)", "frame": "x left, y superior, z anterior (mm)",
            "bp3d_origin": [float(x0), float(y0), float(z0)], "nv": int(nv), "ni": int(ni), "parts": parts,
            "credit": "Anatomy: BodyParts3D, © The Database Center for Life Science, CC BY-SA 2.1 Japan"}
    (out / "anatomy.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}  {nv} verts, {ni // 3} tris, {(out / 'anatomy.bin.gz').stat().st_size / 1e6:.1f} MB")
    if a.volume:
        write_volume(Path(a.volume), parts, pos.reshape(-1, 3), ind)


# ── synthetic CBCT from the meshes ─────────────────────────────────────────────
# Grid in the prototype frame, cropped like a large-FOV dental CBCT (chin to orbit floor).
GRID = dict(x=(-58, 58), y=(-48, 52), z=(-52, 42), s=0.4)
TF2 = {"mandible": 1, "maxilla": 2}


def voxelise(V, F, dims, origin, s):
    """Inside mask of a closed mesh on the grid (array order [y, z, x])."""
    p = to_poly(V.astype(np.float64), F.astype(np.int64))
    img = vtk.vtkImageData(); img.SetSpacing(s, s, s); img.SetOrigin(*origin)
    nx, ny, nz = dims
    img.SetDimensions(nx, nz, ny)  # vtk x=proto x, vtk y=proto z, vtk z=proto y → numpy [y, z, x]
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1); img.GetPointData().GetScalars().Fill(1)
    sw = vtk.vtkTransform(); sw.SetMatrix([1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1])  # (x,y,z)→(x,z,y)
    tp = vtk.vtkTransformPolyDataFilter(); tp.SetInputData(p); tp.SetTransform(sw); tp.Update()
    st = vtk.vtkPolyDataToImageStencil(); st.SetInputData(tp.GetOutput()); st.SetOutputOrigin(*origin)
    st.SetOutputSpacing(s, s, s); st.SetOutputWholeExtent(img.GetExtent()); st.SetTolerance(0); st.Update()
    cut = vtk.vtkImageStencil(); cut.SetInputData(img); cut.SetStencilConnection(st.GetOutputPort())
    cut.ReverseStencilOff(); cut.SetBackgroundValue(0); cut.Update()
    return vtk_to_numpy(cut.GetOutput().GetPointData().GetScalars()).reshape(ny, nz, nx).astype(bool)


def write_volume(dst, parts, P, I):
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    s = GRID["s"]; o = (GRID["x"][0], GRID["z"][0], GRID["y"][0])
    nx, ny, nz = (int((GRID[k][1] - GRID[k][0]) / s) + 1 for k in "xyz")
    lab = np.zeros((ny, nz, nx), np.uint8); bone = np.zeros_like(lab, bool); toothm = np.zeros_like(bone)
    for pt in parts:
        V = P[pt["v0"]:pt["v0"] + pt["nv"]]; F = I[pt["i0"]:pt["i0"] + pt["ni"]].reshape(-1, 3)
        m = voxelise(V, F, (nx, ny, nz), o, s)
        if pt["group"] == "tooth":
            lab[m] = pt["fdi"]; toothm |= m
        else:
            bone |= m
            if pt["group"] in TF2: lab[m & ~toothm & (lab == 0)] = TF2[pt["group"]]
        print(f"  voxelised {pt['key']:8s} {m.sum():8d}")
    bone &= ~toothm
    yy = GRID["y"][0] + np.arange(ny) * s  # proto y per row

    rng = np.random.default_rng(7)
    def fbm(scale):  # band-limited noise for trabecular texture / CBCT grain
        n = rng.standard_normal(lab.shape).astype(np.float32)
        return ndi.gaussian_filter(n, scale / s) * (scale / s) ** 1.5

    # bone: cortical shell, trabecular core with texture
    dB = ndi.distance_transform_edt(bone) * s
    hu = np.full(lab.shape, -1000, np.float32)
    trab = 380 + 260 * fbm(0.8)
    hu[bone] = np.where(dB[bone] < 1.3, 1450, trab[bone])
    # teeth: enamel cap, dentin, pulp; PDL gap where root meets bone
    dT = ndi.distance_transform_edt(toothm) * s
    for pt in parts:
        if pt["group"] != "tooth": continue
        m = lab == pt["fdi"]
        if not m.any(): continue
        ys = yy[np.nonzero(m)[0]]; up = pt["fdi"] // 10 in (1, 2)
        tipy = ys.min() if up else ys.max(); L = ys.max() - ys.min()
        h = (np.abs(yy - tipy))[:, None, None]  # height from the incisal edge
        crown = np.broadcast_to(h < 0.36 * L, m.shape)
        g = np.where(crown, np.where(dT < 1.1, 2600, 1750), 1700).astype(np.float32)
        g[dT > np.maximum(1.5, 0.5 * dT[m].max())] = 160
        hu[m] = g[m]
    pdl = (ndi.distance_transform_edt(~toothm) * s < 0.35) & ~toothm & bone
    hu[pdl] = 120
    # soft tissue envelope + nasal / sinus air inside the maxillary complex
    dOut = ndi.distance_transform_edt(~(bone | toothm)) * s
    soft = (dOut < 13) & ~(bone | toothm)
    hu[soft] = 45 + 25 * fbm(1.5)[soft]
    closed = ndi.binary_closing(bone, iterations=int(7 / s)) & ~bone & ~toothm
    cav = closed & (yy[:, None, None] > 6)
    hu[cav] = -950
    sinus = cav & (np.abs((GRID["x"][0] + np.arange(nx) * s))[None, None, :] > 9)
    xs = GRID["x"][0] + np.arange(nx) * s
    lab[sinus & (xs[None, None, :] > 0)] = 5; lab[sinus & (xs[None, None, :] < 0)] = 6
    # pharynx: airway behind the soft palate
    zz = GRID["z"][0] + np.arange(nz) * s
    ph = ((xs[None, None, :] / 9) ** 2 + ((zz[None, :, None] + 34) / 7) ** 2 < 1) & (yy[:, None, None] < 22) & (yy[:, None, None] > -30) & ~bone
    hu[ph] = -1000; lab[ph] = 7
    write_canals(lab, hu, xs, yy, zz, s)
    # CBCT look: slight blur, correlated grain, cupping, cylindrical FOV
    hu = ndi.gaussian_filter(hu, 0.35 / s) + 55 * fbm(0.5)
    r = np.hypot(xs[None, None, :], (zz + 4)[None, :, None])
    hu += 70 * (r / 58) ** 2
    hu[np.broadcast_to(r > 62, hu.shape)] = -2000; lab[np.broadcast_to(r > 62, lab.shape)] = 0

    dst.mkdir(parents=True, exist_ok=True)
    # store as LPS: L = +x, P = -z, S = +y; numpy array [S, P, L]
    for arr, name, ty in ((hu, "case_0000.mha", sitk.sitkInt16), (lab, "case.mha", sitk.sitkUInt8)):
        a = np.transpose(arr, (0, 1, 2))[:, ::-1, :]  # [y, z, x] → [S, P(= -z), L]
        im = sitk.GetImageFromArray(np.ascontiguousarray(a))
        im = sitk.Cast(im, ty)
        im.SetSpacing((s, s, s)); im.SetOrigin((float(xs[0]), float(-zz[-1]), float(yy[0])))
        sitk.WriteImage(im, str(dst / name), useCompression=True)
    print("wrote", dst)


def write_canals(lab, hu, xs, yy, zz, s):
    """Inferior alveolar canals: through each mandible half, at ~40% height of the body, foramen in the ramus."""
    from scipy import ndimage as ndi
    mand = lab == 1
    for side, lid in ((1, 3), (-1, 4)):
        pts = []
        for zc in np.arange(-46, 6, 1.0):  # posterior → mental foramen region
            kz = int((zc - zz[0]) / s)
            sl = mand[:, kz, :]
            sl = sl & ((xs * side) > 8)[None, :]
            if sl.sum() < 40: continue
            jy, ix = np.nonzero(sl)
            ylo, yhi = yy[jy].min(), yy[jy].max()
            band = (yy[jy] > ylo + 0.28 * (yhi - ylo)) & (yy[jy] < ylo + 0.42 * (yhi - ylo))
            if band.sum() < 5: continue
            pts.append((xs[ix[band]].mean(), yy[jy[band]].mean(), zc))
        if len(pts) < 5: continue
        pts = np.array(pts)
        for k in range(3): pts[1:-1] = (pts[:-2] + 2 * pts[1:-1] + pts[2:]) / 4
        t = np.linspace(0, 1, len(pts)); tt = np.linspace(0, 1, len(pts) * 6)
        dense = np.stack([np.interp(tt, t, pts[:, a]) for a in range(3)], 1)
        for p in dense:
            m = (np.abs(xs - p[0]) < 1.6)[None, None, :] & (np.abs(yy - p[1]) < 1.6)[:, None, None] & (np.abs(zz - p[2]) < 1.6)[None, :, None]
            jj, kk, ii = np.nonzero(m)
            d = (xs[ii] - p[0]) ** 2 + (yy[jj] - p[1]) ** 2 + (zz[kk] - p[2]) ** 2
            keep = d < 1.3 ** 2
            lab[jj[keep], kk[keep], ii[keep]] = lid; hu[jj[keep], kk[keep], ii[keep]] = 90


if __name__ == "__main__":
    main()
