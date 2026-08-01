"""Evaluate OpenVINO COCO person predictions with the shared COCOeval setup."""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_FILE = PROJECT_ROOT / "datasets/coco/annotations/instances_val2017.json"
COCO_PERSON_CATEGORY_ID = 1


def parse_args() -> argparse.Namespace:
    """Parse the optional smoke-test image limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate predictions for only the first N COCO image IDs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    """Load one required JSON artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def evaluate(args: argparse.Namespace) -> None:
    """Run person-only COCOeval and merge accuracy with runtime metrics."""
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    coco_ground_truth = COCO(str(ANNOTATION_FILE))
    image_ids = sorted(coco_ground_truth.getImgIds())
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    suffix = "" if args.limit is None else f"_{len(image_ids)}_images"

    prediction_file = (
        PROJECT_ROOT
        / "outputs/yolo11n"
        / f"openvino_gpu_coco_val2017_person_predictions{suffix}.json"
    )
    runtime_file = (
        PROJECT_ROOT
        / "results/yolo11n"
        / f"openvino_gpu_coco_person_runtime{suffix}.json"
    )
    metrics_file = (
        PROJECT_ROOT
        / "results/yolo11n"
        / f"openvino_gpu_coco_person_metrics{suffix}.json"
    )

    predictions = load_json(prediction_file)
    runtime = load_json(runtime_file)
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("The OpenVINO prediction file contains no predictions")

    coco_predictions = coco_ground_truth.loadRes(str(prediction_file))
    evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = [COCO_PERSON_CATEGORY_ID]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": runtime["model"],
        "dataset": runtime["dataset"],
        "evaluation": {
            "confidence_threshold": float(
                runtime["evaluation"]["confidence_threshold"]
            ),
            "nms_iou_threshold": float(
                runtime["evaluation"]["nms_iou_threshold"]
            ),
            "max_detections": int(runtime["evaluation"]["max_detections"]),
            "predictions": len(predictions),
            "ap_50_95": float(stats[0]),
            "ap_50": float(stats[1]),
            "ap_75": float(stats[2]),
            "ap_small": float(stats[3]),
            "ap_medium": float(stats[4]),
            "ap_large": float(stats[5]),
            "ar_1": float(stats[6]),
            "ar_10": float(stats[7]),
            "ar_100": float(stats[8]),
            "ar_small": float(stats[9]),
            "ar_medium": float(stats[10]),
            "ar_large": float(stats[11]),
        },
        "performance": runtime["performance"],
        "environment": {
            **runtime["environment"],
            "cocoeval_platform": platform.platform(),
            "pycocotools": getattr(pycocotools, "__version__", "unknown"),
        },
    }

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")
    print(f"\nMetrics summary: {metrics_file}")


def main() -> None:
    """Run the OpenVINO prediction evaluator."""
    evaluate(parse_args())


if __name__ == "__main__":
    main()
