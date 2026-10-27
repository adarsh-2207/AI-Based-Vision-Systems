from pathlib import Path
from collections import Counter

DATASET_DIR = Path("data/256_ObjectCategories/256_ObjectCategories")

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

folders = sorted(
    p for p in DATASET_DIR.iterdir()
    if p.is_dir() and not p.name.startswith(".")
)

object_classes = [p for p in folders if p.name != "257.clutter"]
clutter_folder = DATASET_DIR / "257.clutter"

print(f"Dataset root: {DATASET_DIR}")
print(f"Total subfolders: {len(folders)}")
print(f"Object category folders: {len(object_classes)}")
print(f"Clutter folder exists: {clutter_folder.is_dir()}")

extensions = Counter()
total_images = 0
empty_classes = []

for folder in folders:
    files = [
        p for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".")
    ]

    if not files:
        empty_classes.append(folder.name)

    for file in files:
        extensions[file.suffix.lower()] += 1
        if file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            total_images += 1

print(f"Total image files: {total_images}")
print(f"File extensions: {dict(extensions)}")
print(f"Empty folders: {empty_classes}")

print("\nFirst 5 object categories:")
for folder in object_classes[:5]:
    count = sum(
        1 for p in folder.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"{folder.name}: {count} images")

print("\nValidation complete.")
