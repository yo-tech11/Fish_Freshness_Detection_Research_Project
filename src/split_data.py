from pathlib import Path
import random
import shutil

# Original dataset location
SOURCE = Path(
    "data/Digital image analysis aided spoilage of puti fish (puntius sophore)"
)

# Output folder for training, validation and testing
OUTPUT = Path("data/split")

# Reproducible random splitting
random.seed(42)

# Supported image formats
extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# Check original dataset
if not SOURCE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {SOURCE.resolve()}"
    )

# Remove previously generated split data, if present
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

# Process both categories
for class_name in ["fresh", "spoiled"]:

    source_class = SOURCE / class_name

    if not source_class.exists():
        raise FileNotFoundError(
            f"Missing class folder: {source_class}"
        )

    # Collect images
    images = [
        p for p in source_class.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
    ]

    # Shuffle images
    random.shuffle(images)

    total = len(images)

    print(f"\n{class_name.upper()} images: {total}")

    # Ensure enough images for all splits
    if total < 7:
        raise ValueError(
            f"At least 7 images required in {class_name}. "
            f"Found only {total}."
        )

    # 70% training, 15% validation, 15% testing
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    splits = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    # Copy images into destination folders
    for split_name, files in splits.items():

        destination = OUTPUT / split_name / class_name

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        for index, image_path in enumerate(files):

            new_name = f"{index}_{image_path.name}"

            shutil.copy2(
                image_path,
                destination / new_name
            )

        print(
            f"{split_name}: {len(files)} images"
        )

print("\nDataset splitting completed successfully!")

print("Original images were not moved or deleted.")