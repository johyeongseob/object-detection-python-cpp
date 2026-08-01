"""Inspect COCO val2017 images, categories, and object annotations."""

import argparse
from pathlib import Path

from pycocotools.coco import COCO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT / "datasets" / "coco" / "annotations" / "instances_val2017.json"
)
DEFAULT_IMAGES = PROJECT_ROOT / "datasets" / "coco" / "val2017"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="Path to the COCO instances annotation JSON file.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGES,
        help="Directory containing COCO images.",
    )
    parser.add_argument(
        "--image-id",
        type=int,
        default=139,
        help="COCO image ID to inspect (default: 139).",
    )
    return parser.parse_args()


def inspect_dataset(annotation_file: Path, image_dir: Path, image_id: int) -> None:
    """Print dataset statistics and annotations for one image."""
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    coco = COCO(str(annotation_file))
    category_ids = coco.getCatIds()
    image_ids = coco.getImgIds()
    annotation_ids = coco.getAnnIds()

    print("\nDataset summary")
    print(f"  Images:      {len(image_ids)}")
    print(f"  Categories:  {len(category_ids)}")
    print(f"  Annotations: {len(annotation_ids)}")

    categories = coco.loadCats(category_ids)
    categories.sort(key=lambda category: category["id"])
    print("\nCategories")
    print("  " + ", ".join(category["name"] for category in categories))

    person_category_ids = coco.getCatIds(catNms=["person"])
    person_annotation_ids = coco.getAnnIds(catIds=person_category_ids)
    print(f"\nPerson annotations: {len(person_annotation_ids)}")

    image_records = coco.loadImgs([image_id])
    if not image_records:
        raise ValueError(f"Unknown COCO image ID: {image_id}")

    image = image_records[0]
    image_path = image_dir / image["file_name"]
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print("\nSelected image")
    print(f"  ID:     {image['id']}")
    print(f"  File:   {image['file_name']}")
    print(f"  Size:   {image['width']} x {image['height']}")
    print(f"  Path:   {image_path}")

    selected_annotation_ids = coco.getAnnIds(imgIds=[image_id])
    selected_annotations = coco.loadAnns(selected_annotation_ids)
    category_names = {
        category["id"]: category["name"] for category in categories
    }

    print(f"\nObjects ({len(selected_annotations)})")
    print("  Bounding-box format: [x, y, width, height]")
    for index, annotation in enumerate(selected_annotations, start=1):
        category_name = category_names[annotation["category_id"]]
        bbox = ", ".join(f"{value:.2f}" for value in annotation["bbox"])
        print(
            f"  {index:>2}. {category_name:<14} "
            f"bbox=[{bbox}] area={annotation['area']:.2f} "
            f"iscrowd={annotation['iscrowd']}"
        )


def main() -> None:
    """Run the COCO inspection command."""
    args = parse_args()
    inspect_dataset(args.annotation_file, args.image_dir, args.image_id)


if __name__ == "__main__":
    main()
