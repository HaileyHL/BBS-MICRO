#!/usr/bin/env python3
"""
image_reorganize_scipy.py

Re‐organizes the “flat” ImageNet validation JPEGs into val/<WNID>/ layout 
so that torchvision’s ImageFolder can load them, using scipy.io.loadmat to read meta.mat.
"""

import os
import shutil
import argparse
import scipy.io as sio


def load_meta_scipy(meta_mat_path):
    """
    Load meta.mat via scipy.io.loadmat, but only keep entries whose
    ILSVRC2012_ID is in [1..1000]. Returns a dict {ILSVRC2012_ID: WNID}.
    """
    data = sio.loadmat(meta_mat_path)
    if "synsets" not in data:
        raise RuntimeError(f"Expected ‘synsets’ in {meta_mat_path}, but keys are {list(data.keys())}")

    # data["synsets"] might be shape (1, N) or (N,1); we squeeze to 1D
    syn_arr = data["synsets"].squeeze()  
    # syn_arr could have length > 1000 (e.g. 1860). Only some entries have ID in [1..1000].

    id_to_wnid = {}
    for entry in syn_arr:
        # entry["ILSVRC2012_ID"] is a 1×1 array → .item() → int
        cls_id = entry["ILSVRC2012_ID"].item()
        # Only keep entries whose ID is between 1 and 1000 inclusive
        if not (isinstance(cls_id, (int, float)) and 1 <= int(cls_id) <= 1000):
            continue

        cls_id = int(cls_id)
        # entry["WNID"] is a 1×1 array whose .item() is either bytes or str
        raw = entry["WNID"].item()
        if isinstance(raw, bytes):
            wnid = raw.decode("utf-8")
        else:
            wnid = str(raw)

        if cls_id in id_to_wnid:
            # In case of duplicates (shouldn’t normally happen), you can warn or ignore
            # Here we just overwrite, but usually there’s exactly one per ID.
            id_to_wnid[cls_id] = wnid
        else:
            id_to_wnid[cls_id] = wnid

    if len(id_to_wnid) != 1000:
        raise RuntimeError(f"After filtering, loaded {len(id_to_wnid)} synsets but expected 1000.")

    return id_to_wnid



def main():
    parser = argparse.ArgumentParser(
        description="Reorganize flat ImageNet‐1K val JPEGs into val/<WNID>/ folders."
    )
    parser.add_argument(
        "--raw_val_dir", type=str, required=True,
        help="Path to the folder with the 50 000 flat JPEGs (e.g. ILSVRC2012_val_JPEG/)."
    )
    parser.add_argument(
        "--devkit_dir", type=str, required=True,
        help="Path to the unpacked devkit root, so that devkit_dir/data/meta.mat and devkit_dir/data/ILSVRC2012_validation_ground_truth.txt both exist."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path where you want to create val/<WNID>/… (will end up with 1 000 subfolders)."
    )
    args = parser.parse_args()

    raw_val_dir = args.raw_val_dir
    devkit_dir  = args.devkit_dir
    output_dir  = args.output_dir

    # 1. Check that required files exist:
    meta_mat_path = os.path.join(devkit_dir, "data", "meta.mat")
    gt_txt_path   = os.path.join(devkit_dir, "data", "ILSVRC2012_validation_ground_truth.txt")

    if not os.path.isfile(meta_mat_path):
        print(f"ERROR: cannot find {meta_mat_path}")
        return
    if not os.path.isfile(gt_txt_path):
        print(f"ERROR: cannot find {gt_txt_path}")
        return
    if not os.path.isdir(raw_val_dir):
        print(f"ERROR: raw_val_dir {raw_val_dir} does not exist or is not a directory.")
        return

    # 2. Load ID → WNID mapping from meta.mat
    print("Loading meta.mat via scipy.io.loadmat to build ILSVRC2012_ID → WNID map…")
    id2wnid = load_meta_scipy(meta_mat_path)

    # 3. Read the 50 000‐line ground‐truth file
    print("Reading ILSVRC2012_validation_ground_truth.txt (50 000 lines)…")
    with open(gt_txt_path, "r") as f:
        gt_labels = [int(line.strip()) for line in f.readlines()]
    if len(gt_labels) != 50000:
        print(f"WARNING: expected 50000 lines in {gt_txt_path}, but found {len(gt_labels)}")

    # 4. List and sort all JPEG filenames in raw_val_dir
    all_files = sorted(
        fn for fn in os.listdir(raw_val_dir)
        if fn.lower().endswith(".jpeg") or fn.lower().endswith(".jpg")
    )
    if len(all_files) != 50000:
        print(f"WARNING: found {len(all_files)} files under {raw_val_dir}, expected 50000.")

    # 5. Create output_dir if it doesn’t exist
    os.makedirs(output_dir, exist_ok=True)

    # 6. Loop over each image index i = 0…49999
    for i, fname in enumerate(all_files):
        cls_id = gt_labels[i]  # integer in [1…1000]
        if cls_id < 1 or cls_id > 1000:
            print(f"ERROR: ground‐truth index {cls_id} out of range at line {i+1}")
            return

        wnid = id2wnid.get(cls_id)
        if wnid is None:
            print(f"ERROR: no WNID found for ID {cls_id}")
            return

        target_subdir = os.path.join(output_dir, wnid)
        os.makedirs(target_subdir, exist_ok=True)

        src = os.path.join(raw_val_dir, fname)
        dst = os.path.join(target_subdir, fname)
        shutil.move(src, dst)

        if (i + 1) % 5000 == 0:
            print(f"  → Processed {i + 1}/50000 images")

    # 7. Done
    nsub = sum(
        1 for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    )
    print(f"✓ Done. Created {nsub} subfolders under {output_dir} (should be 1000).")


if __name__ == "__main__":
    main()