"""Copy COCO val2017 images containing person annotations into a subset."""

import argparse
import shutil
from pathlib import Path

from pycocotools.coco import COCO


COCO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=COCO_ROOT / "annotations" / "instances_val2017.json",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=COCO_ROOT / "val2017",
    )
    parser.add_argument(
        "--destination-dir",
        type=Path,
        default=COCO_ROOT / "val2017_person",
    )
    return parser.parse_args()


def copy_person_images(
    annotation_file: Path, source_dir: Path, destination_dir: Path
) -> None:
    """Copy every image with at least one person annotation."""
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source image directory not found: {source_dir}")

    coco = COCO(str(annotation_file))
    person_category_id = coco.getCatIds(catNms=["person"])[0]
    person_image_ids = sorted(coco.getImgIds(catIds=[person_category_id]))
    image_records = coco.loadImgs(person_image_ids)

    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for image in image_records:
        source = source_dir / image["file_name"]
        destination = destination_dir / image["file_name"]
        if not source.is_file():
            raise FileNotFoundError(f"Source image not found: {source}")
        if destination.exists():
            skipped += 1
            continue
        shutil.copy2(source, destination)
        copied += 1

    print(f"Person images: {len(image_records)}")
    print(f"Copied:       {copied}")
    print(f"Skipped:      {skipped}")
    print(f"Destination:  {destination_dir}")


def main() -> None:
    """Create the person-image subset."""
    args = parse_args()
    copy_person_images(
        args.annotation_file, args.source_dir, args.destination_dir
    )


if __name__ == "__main__":
    main()
