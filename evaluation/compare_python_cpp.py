"""Compare Python and C++ COCO person accuracy and runtime metrics."""

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON_METRICS = (
    PROJECT_ROOT / "results/yolo11n/python_coco_person_metrics.json"
)
DEFAULT_CPP_METRICS = (
    PROJECT_ROOT / "results/yolo11n/cpp_coco_person_metrics.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/yolo11n/python_vs_cpp_comparison.json"
)

METRICS = (
    ("AP50-95", "evaluation", "ap_50_95", True),
    ("AP50", "evaluation", "ap_50", True),
    ("AP75", "evaluation", "ap_75", True),
    ("AR100", "evaluation", "ar_100", True),
    ("Predictions", "evaluation", "predictions", None),
    ("Mean inference (ms)", "performance", "mean_inference_ms", False),
    ("P50 inference (ms)", "performance", "p50_inference_ms", False),
    ("P95 inference (ms)", "performance", "p95_inference_ms", False),
    (
        "Throughput (images/s)",
        "performance",
        "throughput_images_per_second",
        True,
    ),
    ("Wall time (s)", "performance", "wall_time_seconds", False),
)


def parse_args() -> argparse.Namespace:
    """Parse optional metric and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-metrics", type=Path, default=DEFAULT_PYTHON_METRICS)
    parser.add_argument("--cpp-metrics", type=Path, default=DEFAULT_CPP_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_metrics(path: Path) -> dict[str, Any]:
    """Load one required metrics JSON object."""
    resolved_path = project_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {resolved_path}")
    with resolved_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics must be a JSON object: {resolved_path}")
    return metrics


def compare(args: argparse.Namespace) -> None:
    """Print and save aligned Python-versus-C++ metrics."""
    python_metrics = load_metrics(args.python_metrics)
    cpp_metrics = load_metrics(args.cpp_metrics)
    rows: list[dict[str, Any]] = []

    print(f"{'Metric':<25} {'Python':>12} {'C++':>12} {'C++ delta':>12}")
    print("-" * 65)
    for label, section, key, higher_is_better in METRICS:
        try:
            python_value = float(python_metrics[section][key])
            cpp_value = float(cpp_metrics[section][key])
        except KeyError as error:
            raise KeyError(
                f"Missing metric '{section}.{key}'. Complete COCOeval first."
            ) from error

        delta = cpp_value - python_value
        delta_percent = (
            delta / python_value * 100.0 if python_value != 0.0 else None
        )
        if higher_is_better is None or delta == 0.0:
            better = "neutral"
        elif (delta > 0.0) == higher_is_better:
            better = "cpp"
        else:
            better = "python"

        print(
            f"{label:<25} {python_value:>12.4f} {cpp_value:>12.4f} "
            f"{delta:>+12.4f}"
        )
        rows.append(
            {
                "metric": key,
                "label": label,
                "python": python_value,
                "cpp": cpp_value,
                "cpp_delta": delta,
                "cpp_delta_percent": delta_percent,
                "better": better,
            }
        )

    comparison = {
        "python_metrics": str(project_path(args.python_metrics).relative_to(PROJECT_ROOT)),
        "cpp_metrics": str(project_path(args.cpp_metrics).relative_to(PROJECT_ROOT)),
        "metrics": rows,
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)
        file.write("\n")
    print(f"\nComparison summary: {output_path}")


def main() -> None:
    """Run the metrics comparison command."""
    compare(parse_args())


if __name__ == "__main__":
    main()
