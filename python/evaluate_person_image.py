"""Evaluate YOLO person detections against COCO ground truth for one image."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from pycocotools.coco import COCO
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "yolo11" / "person_detection.yaml"
COCO_PERSON_CATEGORY_ID = 1


@dataclass(frozen=True)
class Match:
    """A matched prediction and ground-truth pair."""

    prediction_index: int
    ground_truth_index: int
    iou: float


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image-id", type=int, help="Override the configured image ID.")
    parser.add_argument(
        "--confidence", type=float, help="Override the configured confidence threshold."
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


def xywh_to_xyxy(box: list[float]) -> np.ndarray:
    """Convert a COCO [x, y, width, height] box to [x1, y1, x2, y2]."""
    x, y, width, height = box
    return np.array([x, y, x + width, y + height], dtype=np.float32)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Calculate intersection over union for two xyxy boxes."""
    intersection_x1 = max(float(box_a[0]), float(box_b[0]))
    intersection_y1 = max(float(box_a[1]), float(box_b[1]))
    intersection_x2 = min(float(box_a[2]), float(box_b[2]))
    intersection_y2 = min(float(box_a[3]), float(box_b[3]))

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0, float(box_a[3] - box_a[1])
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0, float(box_b[3] - box_b[1])
    )
    union_area = area_a + area_b - intersection_area
    return intersection_area / union_area if union_area > 0.0 else 0.0


def match_predictions(
    prediction_boxes: np.ndarray,
    confidences: np.ndarray,
    ground_truth_boxes: np.ndarray,
    iou_threshold: float,
) -> list[Match]:
    """Greedily match predictions to unique ground truths by confidence."""
    matches: list[Match] = []
    matched_ground_truths: set[int] = set()
    prediction_order = np.argsort(-confidences)

    for prediction_index in prediction_order:
        best_ground_truth = -1
        best_iou = 0.0
        for ground_truth_index, ground_truth_box in enumerate(ground_truth_boxes):
            if ground_truth_index in matched_ground_truths:
                continue
            iou = box_iou(prediction_boxes[prediction_index], ground_truth_box)
            if iou > best_iou:
                best_iou = iou
                best_ground_truth = ground_truth_index

        if best_ground_truth >= 0 and best_iou >= iou_threshold:
            matches.append(
                Match(int(prediction_index), best_ground_truth, best_iou)
            )
            matched_ground_truths.add(best_ground_truth)
    return matches


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    confidence: float,
) -> None:
    """Draw one person prediction in the same style as the C++ output."""
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    color = (255, 0, 0)
    label = f"person {confidence:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    label_top = max(y1, text_height + 8)
    cv2.rectangle(
        image,
        (x1, label_top - text_height - 8),
        (x1 + text_width + 8, label_top),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        image,
        label,
        (x1 + 4, label_top - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def evaluate(args: argparse.Namespace) -> None:
    """Run inference, matching, metrics, and visualization for one image."""
    config = load_config(args.config)
    model_config = config["model"]
    detection_config = config["detection"]
    data_config = config["data"]
    output_config = config["output"]
    evaluation_config = config["evaluation"]

    image_id = args.image_id if args.image_id is not None else int(data_config["image_id"])
    confidence_threshold = (
        args.confidence
        if args.confidence is not None
        else float(detection_config["confidence_threshold"])
    )
    match_iou_threshold = float(evaluation_config["match_iou_threshold"])
    ignore_crowd = bool(evaluation_config["ignore_crowd"])

    image_dir = project_path(data_config["image_dir"])
    annotation_file = project_path(data_config["annotation_file"])
    output_dir = project_path(output_config["image_directory"])
    image_path = image_dir / f"{image_id:012d}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"COCO image not found: {image_path}")

    coco = COCO(str(annotation_file))
    annotation_ids = coco.getAnnIds(
        imgIds=[image_id], catIds=[COCO_PERSON_CATEGORY_ID]
    )
    annotations = coco.loadAnns(annotation_ids)
    if ignore_crowd:
        annotations = [annotation for annotation in annotations if not annotation["iscrowd"]]
    ground_truth_boxes = np.array(
        [xywh_to_xyxy(annotation["bbox"]) for annotation in annotations],
        dtype=np.float32,
    ).reshape(-1, 4)

    model = YOLO(str(project_path(model_config["path"])))
    result = model.predict(
        source=str(image_path),
        conf=confidence_threshold,
        iou=float(detection_config["nms_iou_threshold"]),
        imgsz=int(model_config["image_size"]),
        classes=[int(detection_config["class_id"])],
        device=str(model_config["device"]),
        verbose=False,
    )[0]
    prediction_boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    matches = match_predictions(
        prediction_boxes,
        confidences,
        ground_truth_boxes,
        match_iou_threshold,
    )
    true_positives = len(matches)
    false_positives = len(prediction_boxes) - true_positives
    false_negatives = len(ground_truth_boxes) - true_positives
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else None
    recall = true_positives / recall_denominator if recall_denominator else None

    print(f"Image ID: {image_id}")
    print(f"Confidence threshold: {confidence_threshold:.2f}")
    print(f"Match IoU threshold: {match_iou_threshold:.2f}")
    print(f"Ground truths: {len(ground_truth_boxes)}")
    print(f"Predictions:   {len(prediction_boxes)}")
    print("\nMatches")
    if matches:
        for match in matches:
            print(
                f"  prediction {match.prediction_index + 1} -> "
                f"ground truth {match.ground_truth_index + 1}, "
                f"IoU={match.iou:.4f}"
            )
    else:
        print("  None")
    print("\nMetrics")
    print(f"  TP:        {true_positives}")
    print(f"  FP:        {false_positives}")
    print(f"  FN:        {false_negatives}")
    print(f"  Precision: {precision:.4f}" if precision is not None else "  Precision: N/A")
    print(f"  Recall:    {recall:.4f}" if recall is not None else "  Recall:    N/A")
    if not ground_truth_boxes.size and not prediction_boxes.size:
        print("  Result:    Correct negative image")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    for box, confidence in zip(prediction_boxes, confidences):
        draw_box(image, box, float(confidence))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"python_{image_id}_evaluation.jpg"
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to save output image: {output_path}")
    print(f"\nSaved: {output_path}")


def main() -> None:
    """Run the single-image evaluation command."""
    evaluate(parse_args())


if __name__ == "__main__":
    main()
