"""Join GT, Python, C++, and OpenVINO visualizations into labeled rows."""

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "images"
OUTPUT_DIRECTORY = IMAGE_ROOT / "comparison"
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


def parse_args() -> argparse.Namespace:
    """Parse visualization input and comparison output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve a path relative to the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def common_filenames(image_root: Path) -> list[str]:
    """Return sorted JPEG names present in every visualization directory."""
    filename_sets = []
    for directory_name, _ in PANELS:
        directory = image_root / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Visualization directory not found: {directory}")
        filename_sets.append({path.name for path in directory.glob("*.jpg")})

    shared = set.intersection(*filename_sets)
    if not shared:
        raise RuntimeError("No matching visualization images were found")
    return sorted(shared)


def add_header(image: np.ndarray, title: str) -> np.ndarray:
    """Add one centered label above a visualization panel."""
    panel_width = image.shape[1]

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
    text_x = (panel_width - text_width) // 2
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


def create_comparison_strips(args: argparse.Namespace) -> None:
    """Create one four-panel comparison row for every shared image ID."""
    image_root = resolve_project_path(args.input_root)
    output_directory = resolve_project_path(args.output_dir)
    filenames = common_filenames(image_root)
    output_directory.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        labeled_panels = []
        expected_shape: tuple[int, ...] | None = None
        for directory_name, title in PANELS:
            image_path = image_root / directory_name / filename
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            if expected_shape is None:
                expected_shape = image.shape
            elif image.shape != expected_shape:
                raise ValueError(
                    f"Panel sizes differ for {filename}: "
                    f"expected {expected_shape}, got {image.shape}"
                )
            labeled_panels.append(add_header(image, title))

        comparison = cv2.hconcat(labeled_panels)
        output_path = output_directory / filename
        if not cv2.imwrite(
            str(output_path),
            comparison,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        ):
            raise RuntimeError(f"Failed to save image: {output_path}")

    print(f"Created {len(filenames)} comparison strips")
    print(f"Output: {output_directory}")


if __name__ == "__main__":
    create_comparison_strips(parse_args())
