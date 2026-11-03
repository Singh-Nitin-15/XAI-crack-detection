"""
Download 1000 images (500 positive + 500 negative) from
Kaggle dataset yatata1/crack-dataset using parallel downloads.
Organizes into train/val splits for training.
"""
import os
import sys
import stat
import json
import shutil
import random
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def force_remove(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

DATASET = "yatata1/crack-dataset"
AI_DIR = Path(__file__).parent
DATA_DIR = AI_DIR / "dataset"
TARGET_PER_CLASS = 500
TRAIN_RATIO = 0.8
SEED = 42
MAX_WORKERS = 8

random.seed(SEED)


def list_all_files():
    """List files with pagination to find both Positive and Negative images."""
    all_files = {"Positive": [], "Negative": []}
    page_token = None
    page = 0

    while True:
        cmd = ["kaggle", "datasets", "files", DATASET, "--format", "json", "--page-size", "200"]
        if page_token:
            cmd.extend(["--page-token", page_token])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error: {result.stderr.strip()}")
            break

        stdout = result.stdout.strip()
        # Remove page token line if present
        lines = stdout.split("\n")
        json_start = next((i for i, l in enumerate(lines) if l.strip().startswith("[")), 0)
        json_str = "\n".join(lines[json_start:])

        # Extract next page token
        page_token = None
        for line in lines[:json_start]:
            if "Next Page Token" in line:
                page_token = line.split("=", 1)[1].strip()

        data = json.loads(json_str)
        if not data:
            break

        for f in data:
            name = f["name"]
            if "/Positive/Images/" in name and name.lower().endswith((".jpg", ".jpeg", ".png")):
                all_files["Positive"].append(name)
            elif "/Negative/Images/" in name and name.lower().endswith((".jpg", ".jpeg", ".png")):
                all_files["Negative"].append(name)

        page += 1
        print(f"  Page {page}: found {len(all_files['Positive'])} positive, {len(all_files['Negative'])} negative so far")

        if len(all_files["Positive"]) >= TARGET_PER_CLASS and len(all_files["Negative"]) >= TARGET_PER_CLASS:
            break
        if not page_token:
            break

    return all_files


def download_single(filename, dest_dir):
    """Download a single file."""
    cmd = [
        "kaggle", "datasets", "download", DATASET,
        "-f", filename, "-p", str(dest_dir), "--force", "--quiet"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    basename = Path(filename).name
    filepath = dest_dir / basename
    if filepath.exists():
        return filepath

    # Kaggle sometimes wraps single files in zip
    zip_path = dest_dir / (basename + ".zip")
    if zip_path.exists():
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dest_dir)
        zip_path.unlink()
        if filepath.exists():
            return filepath

    return None


def download_batch(file_list, dest_dir, label):
    """Download files in parallel."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    total = len(file_list)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single, f, dest_dir): f for f in file_list}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                downloaded.append(result)
            if done % 50 == 0 or done == total:
                print(f"  [{label}] {done}/{total} downloaded ({len(downloaded)} successful)")

    return downloaded


def split_files(files, label, base_dir):
    """Split into train/val."""
    random.shuffle(files)
    split = int(len(files) * TRAIN_RATIO)
    splits = {"train": files[:split], "val": files[split:]}

    for split_name, split_files_list in splits.items():
        target = base_dir / split_name / label
        target.mkdir(parents=True, exist_ok=True)
        for f in split_files_list:
            shutil.move(str(f), str(target / f.name))

    return len(splits["train"]), len(splits["val"])


def main():
    print("=" * 60)
    print("Kaggle Crack Dataset Downloader (Parallel)")
    print(f"Dataset: {DATASET}")
    print(f"Target: {TARGET_PER_CLASS} per class, {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%} train/val")
    print("=" * 60)

    # Clean old dataset (only with --force, so an existing dataset is never
    # wiped by accident). The trained model (ai/model.pth) is never touched.
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        if "--force" not in sys.argv:
            print(f"\nERROR: {DATA_DIR} already exists and is not empty.")
            print("  Re-run with --force to delete it and download a fresh copy.")
            sys.exit(1)
        print(f"\nRemoving old dataset (--force)...")
        shutil.rmtree(DATA_DIR, onerror=force_remove)

    # List files
    print("\n[1/3] Listing dataset files (may take a few pages)...")
    all_files = list_all_files()

    pos_files = all_files["Positive"][:TARGET_PER_CLASS]
    neg_files = all_files["Negative"][:TARGET_PER_CLASS]
    print(f"\n  Selected: {len(pos_files)} positive, {len(neg_files)} negative")

    if len(pos_files) == 0 or len(neg_files) == 0:
        print("\nERROR: Could not find enough images. Check authentication:")
        print("  Run: kaggle auth login")
        sys.exit(1)

    # Download
    tmp_pos = DATA_DIR / "_tmp_pos"
    tmp_neg = DATA_DIR / "_tmp_neg"

    print(f"\n[2/3] Downloading images ({MAX_WORKERS} parallel workers)...")
    dl_pos = download_batch(pos_files, tmp_pos, "Positive")
    dl_neg = download_batch(neg_files, tmp_neg, "Negative")

    print(f"\n  Downloaded: {len(dl_pos)} positive, {len(dl_neg)} negative")

    # Organize
    print("\n[3/3] Splitting into train/val...")
    t_pos, v_pos = split_files(dl_pos, "positive", DATA_DIR)
    t_neg, v_neg = split_files(dl_neg, "negative", DATA_DIR)

    # Cleanup
    if tmp_pos.exists():
        shutil.rmtree(tmp_pos, onerror=force_remove)
    if tmp_neg.exists():
        shutil.rmtree(tmp_neg, onerror=force_remove)

    print(f"\n{'=' * 60}")
    print(f"Dataset ready: {DATA_DIR}")
    print(f"  Train: {t_pos} positive + {t_neg} negative = {t_pos + t_neg}")
    print(f"  Val:   {v_pos} positive + {v_neg} negative = {v_pos + v_neg}")
    print(f"{'=' * 60}")
    print(f"\nNext: python ai/train.py")


if __name__ == "__main__":
    main()
