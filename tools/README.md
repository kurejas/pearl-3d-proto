# Loading a real CBCT into the prototype

The viewer loads `data/scan/scan.json` (inside `v2/`) if it exists. Otherwise it falls back to the procedural skull.

## 1. Get one ToothFairy2 case

1. Sign up at https://toothfairy2.grand-challenge.org/ and download the training set.
2. Take one image from `imagesTr/` (for example `ToothFairy2F_001_0000.mha`).
3. Take the label map with the same name from `labelsTr/` (`ToothFairy2F_001.mha`).
4. Also keep `dataset.json`. It gives the exact label names, so the mapping doesn't rely on the built-in table.

## 2. Prepare it

```bash
cd pearl-3d-proto/tools
uv venv .venv && uv pip install --python .venv/bin/python numpy SimpleITK pillow   # once
.venv/bin/python prepare_scan.py /path/imagesTr/ToothFairy2F_001_0000.mha /path/labelsTr/ToothFairy2F_001.mha \
  --dataset-json /path/dataset.json
```

The script writes these files to `../v2/data/scan/`:

| File | Contents |
|---|---|
| `scan.json` | Header: size, voxel spacing, position, teeth and licence |
| `gray.u8.gz` | Image, 8-bit, gzipped |
| `labels.u8.gz` | Labels, 8-bit, gzipped |
| `preview.png` | Axial, coronal and sagittal mid-slices, in that order |

Open `preview.png` to check the scan's orientation. In the axial slice, the incisors should be at the top. In the coronal slice, the patient's left should be on the right.

Options:

| Option | Default | What it does |
|---|---|---|
| `--spacing 0.4` | 0.4 mm | Target voxel size. |
| `--max-dim 256` | 256 | Largest size along any axis. If the scan would be bigger, the spacing grows to fit. |
| `--auto-window` | off | Maps grey levels from the scan's own percentiles instead of Hounsfield units. Use it if the scan comes out too dark or washed out. |
| `--out DIR` | `../v2/data/scan/` | Where to write the files. |

## 3. View it

- `http://localhost:5619/v2/` loads `data/scan/scan.json`.
- `?scan=data/other/scan.json` loads a different prepared case.
- `?scan=none` forces the procedural model.

## Test without a real scan

`make_test_case.py` writes a synthetic ToothFairy2-style case. Only its left sinus is present, tooth 16 is missing, and tooth 36 has a metal crown, so any left/right or front/back mix-up is easy to spot.

```bash
.venv/bin/python make_test_case.py && .venv/bin/python prepare_scan.py test_case/case_0000.mha test_case/case.mha --out ../v2/data/test
```

Then open `http://localhost:5619/v2/?scan=data/test/scan.json`.

## Licence

- ToothFairy2 is CC BY-SA 4.0.
- Anything derived from it must carry the same licence and credit. That includes `data/scan/` and any meshes or screenshots made from it.
- The viewer shows the credit line from `scan.json` in the 3D pane.
- Cite the ToothFairy2 challenge papers wherever the prototype is shown.

## Detailed anatomy without a scan (BodyParts3D)

`prepare_bp3d.py` builds v2's default anatomy from BodyParts3D (CC BY-SA 2.1 JP). It uses real skull bones and all 28 permanent teeth with roots, from one body in one frame.

1. Get the OBJ element files listed in the script (FJ1254–FJ1281 are the teeth; the skull bones are FJ3199–FJ3395) from `isa_BP3D_4.0_obj_99.zip` at https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/.
2. Run these two commands:

```bash
.venv/bin/python prepare_bp3d.py OBJ_DIR --volume bp3d_case        # meshes → ../v2/data/anatomy, synthetic CBCT → bp3d_case
.venv/bin/python prepare_scan.py bp3d_case/case_0000.mha bp3d_case/case.mha --name "Synthetic CBCT" \
  --source "BodyParts3D 4.0 — https://dbarchive.biosciencedbc.jp/en/bodyparts3d/" \
  --licence "CC BY-SA 2.1 JP (© The Database Center for Life Science)" --anatomy ../anatomy/anatomy.json
```

To re-voxelise without remeshing, add `--reuse` to the first command.

In the viewer, the 3D pane shows the meshes and the slice views show the synthetic CBCT. The meshes replace the label-derived surfaces. Press C to switch between the CBCT field-of-view crop and the full skull.
