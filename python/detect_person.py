"""Detect people in a COCO val2017 image with a configured YOLO model."""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "yolo11" / "person_detection.yaml"


@dataclass(frozen=True)
class DetectionConfig:
    """Resolved settings used by the person detection command."""

    model_path: Path
    device: str
    image_size: int
    class_id: int
    class_name: str
    confidence_threshold: float
    nms_iou_threshold: float
    image_id: int
    image_dir: Path
    output_dir: Path
    save_image: bool
    show_labels: bool
    show_confidence: bool


def project_path(value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML configuration file.",
    )
    parser.add_argument("--image-id", type=int, help="Override the COCO image ID.")
    parser.add_argument("--image-dir", type=Path, help="Override the image directory.")
    parser.add_argument("--model", type=Path, help="Override the model path.")
    parser.add_argument("--device", help="Override the inference device.")
    parser.add_argument("--image-size", type=int, help="Override the model input size.")
    parser.add_argument(
        "--confidence", type=float, help="Override the confidence threshold."
    )
    parser.add_argument("--nms-iou", type=float, help="Override the NMS IoU threshold.")
    parser.add_argument("--output-dir", type=Path, help="Override the output directory.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and validate the top level of a YAML configuration file."""
    resolved_path = project_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
    with resolved_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {resolved_path}")
    return data


def resolve_config(args: argparse.Namespace) -> DetectionConfig:
    """Merge CLI overrides with YAML settings."""
    data = load_yaml(args.config)
    try:
        model = data["model"]
        detection = data["detection"]
        dataset = data["data"]
        output = data["output"]
    except KeyError as error:
        raise ValueError(f"Missing configuration section: {error.args[0]}") from error

    config = DetectionConfig(
        model_path=project_path(args.model or model["path"]),
        device=args.device or str(model["device"]),
        image_size=(
            args.image_size
            if args.image_size is not None
            else int(model["image_size"])
        ),
        class_id=int(detection["class_id"]),
        class_name=str(detection["class_name"]),
        confidence_threshold=(
            args.confidence
            if args.confidence is not None
            else float(detection["confidence_threshold"])
        ),
        nms_iou_threshold=(
            args.nms_iou
            if args.nms_iou is not None
            else float(detection["nms_iou_threshold"])
        ),
        image_id=args.image_id if args.image_id is not None else int(dataset["image_id"]),
        image_dir=project_path(args.image_dir or dataset["image_dir"]),
        output_dir=project_path(args.output_dir or output["image_directory"]),
        save_image=bool(output["save_image"]),
        show_labels=bool(output["show_labels"]),
        show_confidence=bool(output["show_confidence"]),
    )
    validate_config(config)
    return config


def validate_config(config: DetectionConfig) -> None:
    """Reject invalid settings before inference starts."""
    if config.image_id < 0:
        raise ValueError("image_id must be non-negative")
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    if not 0.0 <= config.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= config.nms_iou_threshold <= 1.0:
        raise ValueError("nms_iou_threshold must be between 0 and 1")


def detect_people(config: DetectionConfig) -> None:
    """Run person detection and optionally save an annotated image."""
    image_path = config.image_dir / f"{config.image_id:012d}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"COCO image not found: {image_path}")

    model = YOLO(str(config.model_path))
    start = time.perf_counter()
    results = model.predict(
        source=str(image_path),
        conf=config.confidence_threshold,
        iou=config.nms_iou_threshold,
        imgsz=config.image_size,
        classes=[config.class_id],
        device=config.device,
        verbose=False,
    )
    total_ms = (time.perf_counter() - start) * 1_000.0

    result = results[0]
    boxes = result.boxes
    detection_count = 0 if boxes is None else len(boxes)

    print(f"Image: {image_path}")
    print(f"Class: {config.class_name} ({config.class_id})")
    print(f"Person detections: {detection_count}")
    print("Bounding-box format: [x1, y1, x2, y2]")

    if boxes is not None:
        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        for index, (bbox, confidence) in enumerate(
            zip(coordinates, confidences), start=1
        ):
            bbox_text = ", ".join(f"{value:.2f}" for value in bbox)
            print(
                f"  {index:>2}. confidence={confidence:.4f} "
                f"bbox=[{bbox_text}]"
            )

    speed = result.speed
    print("\nLatency")
    print(f"  Preprocess:  {speed['preprocess']:.2f} ms")
    print(f"  Inference:   {speed['inference']:.2f} ms")
    print(f"  Postprocess: {speed['postprocess']:.2f} ms")
    print(f"  Total call:  {total_ms:.2f} ms")

    if config.save_image:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = config.output_dir / f"python_{config.image_id}_person.jpg"
        plotted = result.plot(
            labels=config.show_labels,
            conf=config.show_confidence,
        )
        if not cv2.imwrite(str(output_path), plotted):
            raise RuntimeError(f"Failed to save output image: {output_path}")
        print(f"\nSaved: {output_path}")


def main() -> None:
    """Run the configured person detection command."""
    detect_people(resolve_config(parse_args()))


if __name__ == "__main__":
    main()
