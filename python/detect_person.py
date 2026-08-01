"""Detect people in a COCO val2017 image with a pretrained YOLO model."""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "datasets" / "coco" / "val2017"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
PERSON_CLASS_ID = 0  # Ultralytics models use zero-based class IDs for COCO.


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-id",
        type=int,
        default=139,
        help="COCO image ID to process (default: 139).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing COCO val2017 images.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model name or path (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence from 0 to 1 (default: 0.25).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which to save the annotated image.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Path:
    """Validate arguments and return the selected image path."""
    if args.image_id < 0:
        raise ValueError("--image-id must be non-negative")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")

    image_path = args.image_dir / f"{args.image_id:012d}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"COCO image not found: {image_path}")
    return image_path


def detect_people(args: argparse.Namespace) -> None:
    """Run person detection and save an annotated image."""
    image_path = validate_args(args)
    model = YOLO(args.model)

    start = time.perf_counter()
    results = model.predict(
        source=str(image_path),
        conf=args.confidence,
        classes=[PERSON_CLASS_ID],
        device="cpu",
        verbose=False,
    )
    total_ms = (time.perf_counter() - start) * 1_000.0

    result = results[0]
    boxes = result.boxes
    detection_count = 0 if boxes is None else len(boxes)

    print(f"Image: {image_path}")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"python_{args.image_id}_person.jpg"
    if not cv2.imwrite(str(output_path), result.plot()):
        raise RuntimeError(f"Failed to save output image: {output_path}")
    print(f"\nSaved: {output_path}")


def main() -> None:
    """Run the person detection command."""
    detect_people(parse_args())


if __name__ == "__main__":
    main()
