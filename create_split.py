from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset configuration
DATASET_DIR = Path(
    "data/256_ObjectCategories/256_ObjectCategories"
)
OUTPUT_DIR = Path("data/splits")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_DIR.resolve()}"
    )

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
    raise ValueError("Split ratios must add up to 1.")

records = []

# Use the 256 numbered object categories; exclude clutter.
class_dirs = sorted(
    p for p in DATASET_DIR.iterdir()
    if p.is_dir()
    and p.name != "257.clutter"
    and p.name[:3].isdigit()
)

if len(class_dirs) != 256:
    raise ValueError(
        f"Expected 256 object categories, found {len(class_dirs)}"
    )

for class_dir in class_dirs:
    images = sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(images) < 3:
        raise ValueError(
            f"Not enough images to split: {class_dir.name}"
        )

    # First: reserve 15% for the test set.
    train_val, test = train_test_split(
        images,
        test_size=TEST_RATIO,
        random_state=SEED,
        shuffle=True,
    )

    # Then reserve 15% of the full class for validation.
    val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=SEED,
        shuffle=True,
    )

    for split_name, split_images in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        for image_path in split_images:
            records.append({
                "image_path": str(image_path),
                "class_name": class_dir.name,
                "split": split_name,
            })

df = pd.DataFrame(records)

# Check each class appears in all three splits.
split_counts = df.groupby(["class_name", "split"]).size().unstack(fill_value=0)

if not {"train", "val", "test"}.issubset(split_counts.columns):
    raise ValueError("A required split is missing.")

if (split_counts[["train", "val", "test"]] == 0).any().any():
    raise ValueError("At least one class is missing from a split.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_DIR / "caltech256_splits.csv", index=False)

for split_name in ["train", "val", "test"]:
    df[df["split"] == split_name].to_csv(
        OUTPUT_DIR / f"{split_name}.csv",
        index=False,
    )

print("Dataset split created successfully!")
print(f"Object categories: {df['class_name'].nunique()}")
print(f"Total images: {len(df)}")
print("\nImages per split:")
print(df["split"].value_counts())
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))
print(f"\nCSV files saved in: {OUTPUT_DIR.resolve()}")
