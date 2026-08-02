"""Create an animated README GIF from the four-panel comparison images."""

import argparse
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIRECTORY = PROJECT_ROOT / "images/comparison"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "images/comparison.gif"
DEFAULT_WIDTH = 1280
DEFAULT_DURATION_MS = 2000


def parse_args() -> argparse.Namespace:
    """Parse GIF size, speed, and path options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--duration-ms", type=int, default=DEFAULT_DURATION_MS)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """Resolve one path relative to the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def create_gif(args: argparse.Namespace) -> None:
    """Resize comparison images and encode a looping GIF."""
    if args.width <= 0:
        raise ValueError("--width must be positive")
    if args.duration_ms <= 0:
        raise ValueError("--duration-ms must be positive")

    input_directory = project_path(args.input_dir)
    output_file = project_path(args.output)
    image_paths = sorted(input_directory.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(
            f"No comparison JPEG images found: {input_directory}"
        )

    frames: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            height = round(source.height * args.width / source.width)
            resized = source.resize(
                (args.width, height),
                Image.Resampling.LANCZOS,
            )
            frames.append(resized)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )

    cycle_seconds = len(frames) * args.duration_ms / 1000
    print(f"Frames:   {len(frames)}")
    print(f"Size:     {frames[0].width} x {frames[0].height}")
    print(f"Duration: {args.duration_ms} ms/frame ({cycle_seconds:.1f} s/cycle)")
    print(f"Saved:    {output_file}")


if __name__ == "__main__":
    create_gif(parse_args())
