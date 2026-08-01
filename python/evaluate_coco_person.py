"""Evaluate YOLO person detection on all COCO val2017 images."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import ultralytics
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "yolo11n" / "person_detection.yaml"
COCO_PERSON_CATEGORY_ID = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--confidence",
        type=float,
        help="Override the evaluation confidence threshold.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N images for a smoke test.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration mapping."""
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return config


def xyxy_to_xywh(box: np.ndarray) -> list[float]:
    """Convert [x1, y1, x2, y2] to COCO [x, y, width, height]."""
    x1, y1, x2, y2 = (float(value) for value in box)
    return [x1, y1, x2 - x1, y2 - y1]


def cpu_name() -> str:
    """Return a useful CPU name on Linux, with a portable fallback."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    return platform.processor() or "unknown"


def evaluate(args: argparse.Namespace) -> None:
    """Run full-dataset inference and official COCO person evaluation."""
    config = load_config(args.config)
    model_config = config["model"]
    runtime_config = config["runtime"]["python"]
    detection_config = config["detection"]
    data_config = config["data"]
    output_config = config["output"]
    evaluation_config = config["evaluation"]

    confidence_threshold = (
        args.confidence
        if args.confidence is not None
        else float(evaluation_config["confidence_threshold"])
    )
    image_dir = project_path(data_config["image_dir"])
    annotation_file = project_path(data_config["annotation_file"])
    output_dir = project_path(output_config["prediction_directory"])
    results_dir = project_path(output_config["results_directory"])
    progress_interval = int(evaluation_config["progress_interval"])

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Evaluation confidence threshold must be between 0 and 1")

    coco_ground_truth = COCO(str(annotation_file))
    image_ids = sorted(coco_ground_truth.getImgIds())
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        image_ids = image_ids[: args.limit]
    result_suffix = "" if args.limit is None else f"_{len(image_ids)}_images"
    prediction_file = (
        output_dir
        / f"python_coco_val2017_person_predictions{result_suffix}.json"
    )
    metrics_file = results_dir / f"python_coco_person_metrics{result_suffix}.json"
    image_paths = [image_dir / f"{image_id:012d}.jpg" for image_id in image_ids]
    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"COCO image not found: {missing_images[0]}")

    print("\nEvaluation settings", flush=True)
    print(f"  Images:               {len(image_paths)}", flush=True)
    print(f"  Class:                person", flush=True)
    print(f"  Confidence threshold: {confidence_threshold}", flush=True)
    print(f"  NMS IoU threshold:    {detection_config['nms_iou_threshold']}", flush=True)
    print(f"  Input size:           {model_config['image_size']}", flush=True)
    print(f"  Device:               {runtime_config['device']}", flush=True)

    model = YOLO(str(project_path(model_config["paths"]["python"])))
    predictions: list[dict[str, int | float | list[float]]] = []
    inference_times: list[float] = []
    start = time.perf_counter()
    for index, (image_id, image_path) in enumerate(
        zip(image_ids, image_paths), start=1
    ):
        result = model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            iou=float(detection_config["nms_iou_threshold"]),
            imgsz=int(model_config["image_size"]),
            classes=[int(detection_config["class_id"])],
            max_det=int(evaluation_config["max_detections"]),
            device=str(runtime_config["device"]),
            verbose=False,
        )[0]
        boxes = result.boxes
        for box, score in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": COCO_PERSON_CATEGORY_ID,
                    "bbox": xyxy_to_xywh(box),
                    "score": float(score),
                }
            )
        inference_times.append(float(result.speed["inference"]))
        if index % progress_interval == 0 or index == len(image_paths):
            elapsed = time.perf_counter() - start
            rate = index / elapsed
            print(
                f"  Processed {index:>4}/{len(image_paths)} images "
                f"({rate:.2f} images/s)",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    output_dir.mkdir(parents=True, exist_ok=True)
    with prediction_file.open("w", encoding="utf-8") as file:
        json.dump(predictions, file)

    print(f"\nPredictions: {len(predictions)}", flush=True)
    print(f"Prediction file: {prediction_file}", flush=True)
    print(f"Wall time: {elapsed:.2f} s", flush=True)
    print(f"Throughput: {len(image_paths) / elapsed:.2f} images/s", flush=True)
    print(f"Mean model inference: {np.mean(inference_times):.2f} ms", flush=True)
    print(f"P50 model inference:  {np.percentile(inference_times, 50):.2f} ms", flush=True)
    print(f"P95 model inference:  {np.percentile(inference_times, 95):.2f} ms", flush=True)

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
        "model": {
            "name": str(model_config["name"]),
            "path": str(model_config["paths"]["python"]),
            "input_size": int(model_config["image_size"]),
            "runtime": "PyTorch/Ultralytics",
        },
        "dataset": {
            "name": "COCO val2017",
            "images": len(image_ids),
            "category": "person",
            "category_id": COCO_PERSON_CATEGORY_ID,
        },
        "evaluation": {
            "confidence_threshold": confidence_threshold,
            "nms_iou_threshold": float(detection_config["nms_iou_threshold"]),
            "max_detections": int(evaluation_config["max_detections"]),
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
        "performance": {
            "device": str(runtime_config["device"]),
            "wall_time_seconds": elapsed,
            "throughput_images_per_second": len(image_paths) / elapsed,
            "mean_inference_ms": float(np.mean(inference_times)),
            "p50_inference_ms": float(np.percentile(inference_times, 50)),
            "p95_inference_ms": float(np.percentile(inference_times, 95)),
        },
        "environment": {
            "cpu": cpu_name(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "ultralytics": ultralytics.__version__,
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")
    print(f"\nMetrics summary: {metrics_file}", flush=True)


def main() -> None:
    """Run COCO person evaluation."""
    evaluate(parse_args())


if __name__ == "__main__":
    main()
