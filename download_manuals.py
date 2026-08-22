import csv
import requests

from factory_floor.config import MANUAL_DIR, SOURCES_CSV

MANUAL_DIR.mkdir(parents=True, exist_ok=True)

with SOURCES_CSV.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    target = MANUAL_DIR / row["filename"]
    if target.exists() and target.stat().st_size > 50_000:
        print(f"SKIP  {target.name} (already exists)")
        continue
    print(f"GET   {row['manufacturer']} - {row['family']}")
    try:
        r = requests.get(row["url"], timeout=90, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not r.content.startswith(b"%PDF"):
            raise ValueError(f"URL did not return a PDF (content-type={content_type})")
        target.write_bytes(r.content)
        print(f"OK    {target.name} ({len(r.content)/1_000_000:.1f} MB)")
    except Exception as exc:
        print(f"FAIL  {target.name}: {exc}")

print("\nDone. Put any additional official motor/VFD manuals in data/manuals/.")
