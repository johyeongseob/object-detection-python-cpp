"""Compare Python CPU, C++ CPU, and OpenVINO iGPU COCO metrics."""

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON_METRICS = PROJECT_ROOT / "results/yolo11n/python_coco_person_metrics.json"
DEFAULT_CPP_METRICS = PROJECT_ROOT / "results/yolo11n/cpp_coco_person_metrics.json"
DEFAULT_OPENVINO_METRICS = (
    PROJECT_ROOT / "results/yolo11n/openvino_gpu_coco_person_metrics.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/yolo11n/python_cpp_openvino_comparison.json"
)

METRICS = (
    ("AP50-95", "evaluation", "ap_50_95"),
    ("AP50", "evaluation", "ap_50"),
    ("AP75", "evaluation", "ap_75"),
    ("AR100", "evaluation", "ar_100"),
    ("Predictions", "evaluation", "predictions"),
    ("Mean inference (ms)", "performance", "mean_inference_ms"),
    ("P50 inference (ms)", "performance", "p50_inference_ms"),
    ("P95 inference (ms)", "performance", "p95_inference_ms"),
    ("Throughput (images/s)", "performance", "throughput_images_per_second"),
    ("Wall time (s)", "performance", "wall_time_seconds"),
)


def parse_args() -> argparse.Namespace:
    """Parse optional input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-metrics", type=Path, default=DEFAULT_PYTHON_METRICS)
    parser.add_argument("--cpp-metrics", type=Path, default=DEFAULT_CPP_METRICS)
    parser.add_argument(
        "--openvino-metrics", type=Path, default=DEFAULT_OPENVINO_METRICS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_metrics(path: Path) -> dict[str, Any]:
    """Load one metrics JSON object."""
    resolved_path = project_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {resolved_path}")
    with resolved_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics must be a JSON object: {resolved_path}")
    return metrics


def metric_value(metrics: dict[str, Any], section: str, key: str) -> float:
    """Read one required numeric metric."""
    try:
        return float(metrics[section][key])
    except KeyError as error:
        raise KeyError(
            f"Missing metric '{section}.{key}'. Complete COCOeval first."
        ) from error


def compare(args: argparse.Namespace) -> None:
    """Print and save the aligned three-runtime comparison."""
    python_metrics = load_metrics(args.python_metrics)
    cpp_metrics = load_metrics(args.cpp_metrics)
    openvino_metrics = load_metrics(args.openvino_metrics)

    rows: list[dict[str, Any]] = []
    print(
        f"{'Metric':<25} {'Python CPU':>14} {'C++ CPU':>14} "
        f"{'OpenVINO GPU':>14}"
    )
    print("-" * 70)
    for label, section, key in METRICS:
        python_value = metric_value(python_metrics, section, key)
        cpp_value = metric_value(cpp_metrics, section, key)
        openvino_value = metric_value(openvino_metrics, section, key)
        print(
            f"{label:<25} {python_value:>14.4f} {cpp_value:>14.4f} "
            f"{openvino_value:>14.4f}"
        )
        rows.append(
            {
                "metric": key,
                "label": label,
                "python_cpu": python_value,
                "cpp_cpu": cpp_value,
                "openvino_gpu": openvino_value,
            }
        )

    python_mean = metric_value(python_metrics, "performance", "mean_inference_ms")
    cpp_mean = metric_value(cpp_metrics, "performance", "mean_inference_ms")
    openvino_mean = metric_value(
        openvino_metrics, "performance", "mean_inference_ms"
    )
    python_throughput = metric_value(
        python_metrics, "performance", "throughput_images_per_second"
    )
    cpp_throughput = metric_value(
        cpp_metrics, "performance", "throughput_images_per_second"
    )
    openvino_throughput = metric_value(
        openvino_metrics, "performance", "throughput_images_per_second"
    )
    speedups = {
        "openvino_inference_vs_python": python_mean / openvino_mean,
        "openvino_inference_vs_cpp": cpp_mean / openvino_mean,
        "openvino_throughput_vs_python": openvino_throughput / python_throughput,
        "openvino_throughput_vs_cpp": openvino_throughput / cpp_throughput,
    }

    print("\nOpenVINO GPU speedups")
    print(
        f"  Mean inference vs Python CPU: "
        f"{speedups['openvino_inference_vs_python']:.2f}x"
    )
    print(
        f"  Mean inference vs C++ CPU:    "
        f"{speedups['openvino_inference_vs_cpp']:.2f}x"
    )
    print(
        f"  Throughput vs Python CPU:     "
        f"{speedups['openvino_throughput_vs_python']:.2f}x"
    )
    print(
        f"  Throughput vs C++ CPU:        "
        f"{speedups['openvino_throughput_vs_cpp']:.2f}x"
    )

    comparison = {
        "inputs": {
            "python_cpu": str(
                project_path(args.python_metrics).relative_to(PROJECT_ROOT)
            ),
            "cpp_cpu": str(project_path(args.cpp_metrics).relative_to(PROJECT_ROOT)),
            "openvino_gpu": str(
                project_path(args.openvino_metrics).relative_to(PROJECT_ROOT)
            ),
        },
        "metrics": rows,
        "speedups": speedups,
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)
        file.write("\n")
    print(f"\nComparison summary: {output_path}")


def main() -> None:
    """Run the three-runtime comparison."""
    compare(parse_args())


if __name__ == "__main__":
    main()
