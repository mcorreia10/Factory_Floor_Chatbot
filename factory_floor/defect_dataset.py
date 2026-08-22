import csv
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from factory_floor.config import DEFECT_IMAGE_DIR, DEFECT_MANIFEST_CSV, PROJECT_ROOT

# Curated subset only — categories physically plausible on/around motors and VFDs.
# See CLAUDE.md's 2026-08-19 note for why the other 11 generic MVTec AD categories
# (bottle, hazelnut, toothbrush, ...) are deliberately excluded.
MVTEC_CATEGORIES = ["cable", "metal_nut", "screw", "transistor"]

# Generic vocabulary shared across all 4 categories, matching the spec's own wording
# ("a scratch, a crack, contamination, and so on") rather than ~20 MVTec-native
# per-object labels — confirmed with the project owner on 2026-08-19.
COARSE_LABELS = ["good", "scratch", "deformation", "structural_damage", "contamination", "other_defect"]

# (category, native_defect_label) -> coarse_label. Confirmed against the real defect
# subfolder names present in the downloaded dataset (download_defect_images.py), not
# guessed from memory. Every native label seen at scan time must have an entry here.
COARSE_LABEL_MAP = {
    ("cable", "good"): "good",
    ("metal_nut", "good"): "good",
    ("screw", "good"): "good",
    ("transistor", "good"): "good",
    ("metal_nut", "scratch"): "scratch",
    ("screw", "scratch_head"): "scratch",
    ("screw", "scratch_neck"): "scratch",
    ("cable", "bent_wire"): "deformation",
    ("metal_nut", "bent"): "deformation",
    ("metal_nut", "flip"): "deformation",
    ("transistor", "bent_lead"): "deformation",
    ("cable", "cut_inner_insulation"): "structural_damage",
    ("cable", "cut_outer_insulation"): "structural_damage",
    ("cable", "poke_insulation"): "structural_damage",
    ("transistor", "cut_lead"): "structural_damage",
    ("transistor", "damaged_case"): "structural_damage",
    ("screw", "thread_side"): "structural_damage",
    ("screw", "thread_top"): "structural_damage",
    ("metal_nut", "color"): "contamination",
    ("cable", "cable_swap"): "other_defect",
    ("cable", "combined"): "other_defect",
    ("cable", "missing_cable"): "other_defect",
    ("cable", "missing_wire"): "other_defect",
    ("screw", "manipulated_front"): "other_defect",
    ("transistor", "misplaced"): "other_defect",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MANIFEST_FIELDS = ["filepath", "category", "native_defect_label", "coarse_label", "split"]
MIN_IMAGES_PER_CLASS = 4


def scan_raw_images(defect_image_dir=DEFECT_IMAGE_DIR, categories=MVTEC_CATEGORIES) -> list:
    """Walks data/defect_images/raw/<category>/<mvtec_split>/<native_defect_label>/*.png
    (as produced by download_defect_images.py) and returns one row per image, with the
    coarse label already resolved via COARSE_LABEL_MAP. `filepath` is stored relative to
    PROJECT_ROOT so the resulting manifest stays portable across machines."""
    raw_dir = Path(defect_image_dir) / "raw"
    rows = []
    for category in categories:
        category_dir = raw_dir / category
        if not category_dir.exists():
            continue
        for mvtec_split_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            for defect_dir in sorted(p for p in mvtec_split_dir.iterdir() if p.is_dir()):
                native_defect = defect_dir.name
                coarse = COARSE_LABEL_MAP.get((category, native_defect))
                if coarse is None:
                    raise ValueError(
                        f"No coarse-label mapping for ({category!r}, {native_defect!r}) — "
                        "add it to COARSE_LABEL_MAP before scanning."
                    )
                for image_path in sorted(defect_dir.iterdir()):
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    rows.append(
                        {
                            "filepath": str(image_path.relative_to(PROJECT_ROOT).as_posix()),
                            "category": category,
                            "native_defect_label": native_defect,
                            "coarse_label": coarse,
                        }
                    )
    return rows


def build_manifest(rows: list, test_size: float = 0.25, seed: int = 42) -> list:
    """Builds our own stratified train/test split over coarse_label, ignoring MVTec's
    native train/test split (which only has 'good' images in train — unusable for
    supervised classification on its own). Raises if any coarse label has too few
    images to split reliably."""
    labels = [r["coarse_label"] for r in rows]
    counts = Counter(labels)
    too_small = [label for label, n in counts.items() if n < MIN_IMAGES_PER_CLASS]
    if too_small:
        raise ValueError(
            f"Coarse labels with fewer than {MIN_IMAGES_PER_CLASS} images can't be split "
            f"reliably: {too_small}. Widen COARSE_LABEL_MAP or gather more images."
        )
    train_rows, test_rows = train_test_split(
        rows, test_size=test_size, random_state=seed, stratify=labels
    )
    for r in train_rows:
        r["split"] = "train"
    for r in test_rows:
        r["split"] = "test"
    return train_rows + test_rows


def save_manifest(rows: list, path: Path = DEFECT_MANIFEST_CSV) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path = DEFECT_MANIFEST_CSV) -> list:
    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_summary(rows: list) -> Counter:
    return Counter((r["coarse_label"], r["split"]) for r in rows)
