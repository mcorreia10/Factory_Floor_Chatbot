from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUAL_DIR = PROJECT_ROOT / "data" / "manuals"
VECTOR_DIR = PROJECT_ROOT / "data" / "vectorstore"
SOURCES_CSV = PROJECT_ROOT / "manual_sources.csv"
MACHINES_CSV = PROJECT_ROOT / "machines.csv"
HISTORY_CSV = PROJECT_ROOT / "maintenance_history.csv"
COLLECTION_NAME = "factory_floor_manuals"

DEFECT_IMAGE_DIR = PROJECT_ROOT / "data" / "defect_images"
VISION_MODEL_DIR = PROJECT_ROOT / "data" / "vision_model"
DEFECT_MANIFEST_CSV = PROJECT_ROOT / "defect_image_manifest.csv"

EVAL_SCENARIOS_CSV = PROJECT_ROOT / "eval_scenarios.csv"
FAULT_CODES_CSV = PROJECT_ROOT / "fault_codes.csv"
