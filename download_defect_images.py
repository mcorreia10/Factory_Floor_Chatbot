import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from factory_floor.config import DEFECT_IMAGE_DIR

RAW_DIR = DEFECT_IMAGE_DIR / "raw"

# Curated subset only — categories physically plausible on/around motors and VFDs.
# See CLAUDE.md's 2026-08-19 note for why the other 11 generic MVTec AD categories
# (bottle, hazelnut, toothbrush, ...) are deliberately excluded.
CATEGORIES = ["cable", "metal_nut", "screw", "transistor"]

# MVTec's own site gates the combined archive behind a license-acceptance page that
# doesn't respond to a plain scripted GET (confirmed: returns 404/HTML, same situation
# the Siemens manuals hit before). Voxel51/mvtec-ad on the Hugging Face Hub is a public,
# no-auth-required re-hosting of the same official images plus per-image metadata
# (category, defect type, train/test split) in samples.json — used here instead.
HF_REPO_ID = "Voxel51/mvtec-ad"
MANUAL_PAGE = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"


def already_downloaded():
    return all((RAW_DIR / category / "test").exists() for category in CATEGORIES)


def print_manual_instructions():
    print(
        "\nAutomated download failed (no network access to huggingface.co?). Manual steps:\n"
        f"  1. Try again later, or open {MANUAL_PAGE} and accept the research-use license to\n"
        "     download the combined archive (mvtec_anomaly_detection.tar.xz, ~5 GB, all 15 categories)\n"
        "     directly from MVTec.\n"
        f"  2. Extract it, then copy ONLY these 4 category folders into {RAW_DIR}:\n"
        f"     {', '.join(CATEGORIES)}\n"
        f"  3. Each should end up as {RAW_DIR}\\<category>\\<split>\\<defect_label>\\... — that's\n"
        "     the structure notebooks/06_computer_vision.ipynb expects.\n"
    )


def download_from_hf():
    print(f"Fetching sample metadata from {HF_REPO_ID}...")
    samples_path = hf_hub_download(repo_id=HF_REPO_ID, filename="samples.json", repo_type="dataset")
    with open(samples_path, encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    rows = [s for s in samples if s["category"]["label"] in CATEGORIES]
    print(f"{len(rows)} images across {CATEGORIES} (out of {len(samples)} total in the dataset).")

    for i, sample in enumerate(rows, 1):
        category = sample["category"]["label"]
        defect_label = sample["defect"]["label"]
        split = sample["split"]
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=sample["filepath"], repo_type="dataset")

        target_dir = RAW_DIR / category / split / defect_label
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / Path(sample["filepath"]).name
        if not target.exists():
            shutil.copyfile(local_path, target)

        if i % 200 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} images in place")

    print(f"OK    {len(rows)} images organized under {RAW_DIR}")


if __name__ == "__main__":
    if already_downloaded():
        print(f"SKIP  all 4 categories already present in {RAW_DIR}")
    else:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        try:
            download_from_hf()
        except Exception as exc:
            print(f"FAIL  {exc}")
            print_manual_instructions()
