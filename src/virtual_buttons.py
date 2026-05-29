"""
Virtual Buttons with YOLO11 Pose Estimation.

Run a YOLO11 pose-estimation model on a live webcam stream, track the
right-wrist keypoint and trigger one of two virtual on-screen buttons when the
wrist enters its bounding rectangle.

Controls
--------
q : quit the application.

Usage
-----
    python src/virtual_buttons.py

The YOLO11 pose weights are expected at ``model/yolo11m-pose.pt``. Ultralytics
will download them automatically on the first run if they are not present.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration -----------------------------------------------------------

MODEL_PATH = "model/yolo11m-pose.pt"
WINDOW_NAME = "Virtual Buttons"

# COCO-17 keypoint indices used by Ultralytics pose models.
RIGHT_WRIST_INDEX = 10

# Virtual button rectangles in pixel coordinates: (x1, y1, x2, y2).
LEFT_BUTTON = (30, 30, 160, 120)
RIGHT_BUTTON = (430, 30, 560, 120)

# BGR colors.
LEFT_COLOR = (0, 0, 255)       # red
RIGHT_COLOR = (0, 255, 255)    # yellow
WRIST_COLOR = (0, 255, 0)      # green

# Press 'q' to quit.
QUIT_KEY = ord("q")
WAIT_KEY_DELAY_MS = 5


# --- Helpers -----------------------------------------------------------------


def get_right_wrist(results) -> tuple[int, int] | tuple[None, None]:
    """Return the pixel ``(x, y)`` of the right wrist in the first detection.

    Returns ``(None, None)`` if no person was detected or the keypoint is
    unavailable / NaN.
    """
    if results.keypoints is None or results.keypoints.xy is None:
        return None, None

    keypoints = results.keypoints.xy
    if len(keypoints) == 0 or keypoints.shape[1] <= RIGHT_WRIST_INDEX:
        return None, None

    kp = keypoints[0, RIGHT_WRIST_INDEX]
    if np.isnan(kp[0].item()) or np.isnan(kp[1].item()):
        return None, None

    return int(kp[0].item()), int(kp[1].item())


def point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    """Return ``True`` if ``point`` lies inside the axis-aligned rectangle."""
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_button(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
    pressed: bool,
) -> None:
    """Draw an outlined or filled button rectangle with a label above it."""
    x1, y1, x2, y2 = rect
    thickness = -1 if pressed else 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_wrist_marker(frame: np.ndarray, point: tuple[int, int]) -> None:
    """Draw a green dot and the ``KP10`` label at the wrist position."""
    cv2.circle(frame, point, 5, WRIST_COLOR, -1)
    cv2.putText(
        frame,
        "KP10",
        (point[0] + 6, point[1] - 6),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        WRIST_COLOR,
        1,
        cv2.LINE_AA,
    )


# --- Main loop ---------------------------------------------------------------


def main() -> None:
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        raise RuntimeError("Could not open webcam (device index 0).")

    model = YOLO(MODEL_PATH)

    try:
        while True:
            ret, frame = webcam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            results = model(frame)[0]
            annotated_frame = results.plot()

            wrist_x, wrist_y = get_right_wrist(results)
            wrist_point = (wrist_x, wrist_y) if wrist_x is not None else None

            if wrist_point is not None:
                draw_wrist_marker(annotated_frame, wrist_point)

            left_pressed = wrist_point is not None and point_in_rect(wrist_point, LEFT_BUTTON)
            right_pressed = wrist_point is not None and point_in_rect(wrist_point, RIGHT_BUTTON)

            draw_button(annotated_frame, LEFT_BUTTON, LEFT_COLOR, "Left", left_pressed)
            draw_button(annotated_frame, RIGHT_BUTTON, RIGHT_COLOR, "Right", right_pressed)

            cv2.imshow(WINDOW_NAME, annotated_frame)
            if cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF == QUIT_KEY:
                break
    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
