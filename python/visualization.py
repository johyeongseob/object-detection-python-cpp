"""Shared visualization style for Python person-detection outputs."""

import cv2
import numpy as np


BOX_COLOR = (255, 0, 0)
GROUND_TRUTH_COLOR = (70, 160, 70)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
LABEL_PADDING = 4


def draw_person_detection(
    image: np.ndarray,
    box: np.ndarray,
    confidence: float | None,
    color: tuple[int, int, int] = BOX_COLOR,
) -> None:
    """Draw one person box and confidence label using the project style."""
    x1, y1, x2, y2 = (int(round(float(value))) for value in box)
    label = "person" if confidence is None else f"person {confidence:.2f}"

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        BOX_THICKNESS,
    )

    (text_width, text_height), _ = cv2.getTextSize(
        label,
        FONT,
        FONT_SCALE,
        FONT_THICKNESS,
    )
    label_height = text_height + 2 * LABEL_PADDING
    label_top = max(y1, label_height)
    cv2.rectangle(
        image,
        (x1, label_top - label_height),
        (x1 + text_width + 2 * LABEL_PADDING, label_top),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        image,
        label,
        (x1 + LABEL_PADDING, label_top - LABEL_PADDING),
        FONT,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
