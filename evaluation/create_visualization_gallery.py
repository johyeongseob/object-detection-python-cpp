"""Create aligned GT and runtime visualization galleries for the README."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools.coco import COCO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from visualization import (  # noqa: E402
    GROUND_TRUTH_COLOR,
    draw_person_detection,
)


ANNOTATION_FILE = (
    PROJECT_ROOT / "datasets/coco/annotations/instances_val2017.json"
)
DEFAULT_IMAGE_DIRECTORY = PROJECT_ROOT / "datasets/coco/val2017_person"
GALLERY_DIRECTORY = PROJECT_ROOT / "images"
PERSON_CATEGORY_ID = 1
DEFAULT_IMAGE_COUNT = 20
CANVAS_SIZE = 640
VISUALIZATION_CONFIDENCE = 0.25

PREDICTION_FILES = {
    "python": (
        PROJECT_ROOT / "outputs/yolo11n/python_coco_val2017_person_predictions.json",
        PROJECT_ROOT / "outputs/yolo11n/coco_val2017_person_predictions.json",
    ),
    "cpp": (
        PROJECT_ROOT / "outputs/yolo11n/cpp_coco_val2017_person_predictions.json",
    ),
    "openvino": (
        PROJECT_ROOT
        / "outputs/yolo11n/openvino_gpu_coco_val2017_person_predictions.json",
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse gallery targets, image count, and source directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("gt", "python", "cpp", "openvino"),
        default=("gt", "python", "cpp", "openvino"),
        help="Gallery directories to generate.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_IMAGE_COUNT,
        help="Number of deterministic person-positive images to select.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIRECTORY,
        help="Directory containing the person-positive COCO images.",
    )
    return parser.parse_args()


def existing_file(candidates: tuple[Path, ...]) -> Path:
    """Return the first existing prediction artifact."""
    for path in candidates:
        if path.is_file():
            return path
    paths = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Prediction file not found; checked: {paths}")


def load_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Load person predictions above the visualization threshold by image."""
    with path.open(encoding="utf-8") as file:
        predictions = json.load(file)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if (
            int(prediction["category_id"]) == PERSON_CATEGORY_ID
            and float(prediction["score"]) >= VISUALIZATION_CONFIDENCE
        ):
            grouped[int(prediction["image_id"])].append(prediction)
    for image_predictions in grouped.values():
        image_predictions.sort(key=lambda item: float(item["score"]), reverse=True)
    return grouped


def select_image_ids(coco: COCO, image_directory: Path, count: int) -> list[int]:
    """Select deterministic person-positive images available in the source."""
    if count <= 0:
        raise ValueError("--count must be positive")
    if not image_directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_directory}")

    candidates: list[int] = []
    for image_id in sorted(coco.getImgIds(catIds=[PERSON_CATEGORY_ID])):
        if not (image_directory / f"{image_id:012d}.jpg").is_file():
            continue
        annotation_ids = coco.getAnnIds(
            imgIds=[image_id], catIds=[PERSON_CATEGORY_ID]
        )
        annotations = coco.loadAnns(annotation_ids)
        visible_people = [item for item in annotations if not item["iscrowd"]]
        if 1 <= len(visible_people) <= 8:
            candidates.append(image_id)

    if len(candidates) < count:
        raise RuntimeError("Not enough person-positive images for the gallery")

    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[int(index)] for index in indices]


def letterbox(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Resize an image onto a fixed square canvas and return its transform."""
    height, width = image.shape[:2]
    scale = min(CANVAS_SIZE / width, CANVAS_SIZE / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    horizontal_padding = (CANVAS_SIZE - resized_width) / 2
    vertical_padding = (CANVAS_SIZE - resized_height) / 2
    left = int(round(horizontal_padding - 0.1))
    right = int(round(horizontal_padding + 0.1))
    top = int(round(vertical_padding - 0.1))
    bottom = int(round(vertical_padding + 0.1))
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return canvas, scale, left, top


def transform_xywh(
    box: list[float], scale: float, left: int, top: int
) -> np.ndarray:
    """Transform a COCO xywh box onto the letterboxed canvas."""
    x, y, width, height = (float(value) for value in box)
    return np.array(
        [
            x * scale + left,
            y * scale + top,
            (x + width) * scale + left,
            (y + height) * scale + top,
        ],
        dtype=np.float32,
    )


def save_image(path: Path, image: np.ndarray) -> None:
    """Save one gallery JPEG with consistent encoding settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Failed to save image: {path}")


def create_gallery(args: argparse.Namespace) -> None:
    """Render aligned ground-truth and prediction images."""
    image_directory = (
        args.image_dir
        if args.image_dir.is_absolute()
        else PROJECT_ROOT / args.image_dir
    )
    targets = tuple(dict.fromkeys(args.targets))
    coco = COCO(str(ANNOTATION_FILE))
    image_ids = select_image_ids(coco, image_directory, args.count)
    runtime_predictions = {
        runtime: load_predictions(existing_file(candidates))
        for runtime, candidates in PREDICTION_FILES.items()
        if runtime in targets
    }

    for directory_name in targets:
        (GALLERY_DIRECTORY / directory_name).mkdir(parents=True, exist_ok=True)

    manifest_images: list[dict[str, Any]] = []
    for image_id in image_ids:
        source_path = image_directory / f"{image_id:012d}.jpg"
        source = cv2.imread(str(source_path))
        if source is None:
            raise FileNotFoundError(f"COCO image not found: {source_path}")

        canvas, scale, left, top = letterbox(source)
        annotation_ids = coco.getAnnIds(
            imgIds=[image_id], catIds=[PERSON_CATEGORY_ID]
        )
        annotations = [
            item for item in coco.loadAnns(annotation_ids) if not item["iscrowd"]
        ]

        filename = f"{image_id:012d}.jpg"
        if "gt" in targets:
            ground_truth_image = canvas.copy()
            for annotation in annotations:
                box = transform_xywh(annotation["bbox"], scale, left, top)
                draw_person_detection(
                    ground_truth_image,
                    box,
                    None,
                    GROUND_TRUTH_COLOR,
                )
            save_image(GALLERY_DIRECTORY / "gt" / filename, ground_truth_image)

        prediction_counts: dict[str, int] = {}
        for runtime, grouped_predictions in runtime_predictions.items():
            runtime_image = canvas.copy()
            predictions = grouped_predictions.get(image_id, [])
            for prediction in predictions:
                box = transform_xywh(prediction["bbox"], scale, left, top)
                draw_person_detection(
                    runtime_image,
                    box,
                    float(prediction["score"]),
                )
            save_image(GALLERY_DIRECTORY / runtime / filename, runtime_image)
            prediction_counts[runtime] = len(predictions)

        manifest_images.append(
            {
                "image_id": image_id,
                "file": filename,
                "ground_truth_people": len(annotations),
                "predictions_at_0.25": prediction_counts,
            }
        )

    manifest = {
        "dataset": "COCO val2017",
        "category": "person",
        "image_count": len(image_ids),
        "targets": list(targets),
        "image_directory": str(image_directory.relative_to(PROJECT_ROOT)),
        "canvas_size": [CANVAS_SIZE, CANVAS_SIZE],
        "visualization_confidence": VISUALIZATION_CONFIDENCE,
        "images": manifest_images,
    }
    manifest_path = GALLERY_DIRECTORY / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    print(f"Created {len(image_ids)} images for: {', '.join(targets)}")
    print(f"Gallery:  {GALLERY_DIRECTORY}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    create_gallery(parse_args())
