"""Join GT, Python, C++, and OpenVINO visualizations into labeled rows."""

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "images"
OUTPUT_DIRECTORY = IMAGE_ROOT / "comparison"
PANEL_SIZE = 640
HEADER_HEIGHT = 64
HEADER_COLOR = (32, 32, 32)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5
FONT_THICKNESS = 3

PANELS = (
    ("gt", "GT"),
    ("python", "Python"),
    ("cpp", "C++"),
    ("openvino", "OpenVINO"),
)


def common_filenames() -> list[str]:
    """Return sorted JPEG names present in every visualization directory."""
    filename_sets = []
    for directory_name, _ in PANELS:
        directory = IMAGE_ROOT / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Visualization directory not found: {directory}")
        filename_sets.append({path.name for path in directory.glob("*.jpg")})

    shared = set.intersection(*filename_sets)
    if not shared:
        raise RuntimeError("No matching visualization images were found")
    return sorted(shared)


def add_header(image: np.ndarray, title: str) -> np.ndarray:
    """Add one centered label above a visualization panel."""
    if image.shape[:2] != (PANEL_SIZE, PANEL_SIZE):
        raise ValueError(
            f"Expected a {PANEL_SIZE}x{PANEL_SIZE} panel, got "
            f"{image.shape[1]}x{image.shape[0]}"
        )

    panel = cv2.copyMakeBorder(
        image,
        HEADER_HEIGHT,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=HEADER_COLOR,
    )
    (text_width, text_height), _ = cv2.getTextSize(
        title,
        FONT,
        FONT_SCALE,
        FONT_THICKNESS,
    )
    text_x = (PANEL_SIZE - text_width) // 2
    text_y = (HEADER_HEIGHT + text_height) // 2
    cv2.putText(
        panel,
        title,
        (text_x, text_y),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return panel


def create_comparison_strips() -> None:
    """Create one four-panel comparison row for every shared image ID."""
    filenames = common_filenames()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        labeled_panels = []
        for directory_name, title in PANELS:
            image_path = IMAGE_ROOT / directory_name / filename
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            labeled_panels.append(add_header(image, title))

        comparison = cv2.hconcat(labeled_panels)
        output_path = OUTPUT_DIRECTORY / filename
        if not cv2.imwrite(
            str(output_path),
            comparison,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        ):
            raise RuntimeError(f"Failed to save image: {output_path}")

    print(f"Created {len(filenames)} comparison strips")
    print(f"Output: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    create_comparison_strips()
