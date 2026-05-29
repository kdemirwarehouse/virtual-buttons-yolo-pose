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

import argparse
import time
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

# Press 'q' or ESC to quit.
QUIT_KEY = ord("q")
ESC_KEY = 27
WAIT_KEY_DELAY_MS = 5

# Minimum consecutive frames a wrist must stay in a button to count as a press.
DEBOUNCE_FRAMES = 6


# --- Helpers -----------------------------------------------------------------


def get_keypoint(results, index: int) -> tuple[int, int] | tuple[None, None]:
    """Return the pixel ``(x, y)`` of the given COCO-17 keypoint in the first detection.

    Returns ``(None, None)`` if no person was detected, the keypoint index is
    out of range, or the value is NaN.
    """
    if results.keypoints is None or results.keypoints.xy is None:
        return None, None

    keypoints = results.keypoints.xy
    if len(keypoints) == 0 or keypoints.shape[1] <= index:
        return None, None

    kp = keypoints[0, index]
    if np.isnan(kp[0].item()) or np.isnan(kp[1].item()):
        return None, None

    return int(kp[0].item()), int(kp[1].item())


def get_right_wrist(results) -> tuple[int, int] | tuple[None, None]:
    """Convenience wrapper: right wrist is COCO-17 keypoint 10."""
    return get_keypoint(results, RIGHT_WRIST_INDEX)


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


def draw_wrist_marker(frame: np.ndarray, point: tuple[int, int], label: str = "KP10") -> None:
    """Draw a green dot and a keypoint label at the wrist position."""
    cv2.circle(frame, point, 5, WRIST_COLOR, -1)
    cv2.putText(
        frame,
        label,
        (point[0] + 6, point[1] - 6),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        WRIST_COLOR,
        1,
        cv2.LINE_AA,
    )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Render the current FPS in the top-right corner of *frame*."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (w - 110, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)


def draw_counters(frame: np.ndarray, left_count: int, right_count: int) -> None:
    """Render press counters for left and right buttons at the bottom of *frame*."""
    h, w = frame.shape[:2]
    cv2.putText(frame, f"L: {left_count}", (10, h - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"R: {right_count}", (w - 80, h - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)


# --- Main loop ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Virtual Buttons with YOLO11 Pose")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--left-hand", action="store_true", help="Track left wrist (KP9) instead of right (KP10)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracked_index = LEFT_WRIST_INDEX if args.left_hand else RIGHT_WRIST_INDEX
    webcam = cv2.VideoCapture(args.camera)
    if not webcam.isOpened():
        raise RuntimeError(f"Could not open webcam (device index {args.camera}).")

    model = YOLO(MODEL_PATH)
    prev_time = time.time()
    left_hold = 0
    right_hold = 0
    left_count = 0
    right_count = 0
    left_was_pressed = False
    right_was_pressed = False

    try:
        while True:
            ret, frame = webcam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            results = model(frame)[0]
            annotated_frame = results.plot()

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            draw_fps(annotated_frame, fps)

            wrist_x, wrist_y = get_keypoint(results, tracked_index)
            wrist_point = (wrist_x, wrist_y) if wrist_x is not None else None
            kp_label = f"KP{tracked_index}"

            if wrist_point is not None:
                draw_wrist_marker(annotated_frame, wrist_point, label=kp_label)

            in_left = wrist_point is not None and point_in_rect(wrist_point, LEFT_BUTTON)
            in_right = wrist_point is not None and point_in_rect(wrist_point, RIGHT_BUTTON)

            left_hold = left_hold + 1 if in_left else 0
            right_hold = right_hold + 1 if in_right else 0

            left_pressed = left_hold >= DEBOUNCE_FRAMES
            right_pressed = right_hold >= DEBOUNCE_FRAMES

            if left_pressed and not left_was_pressed:
                left_count += 1
            if right_pressed and not right_was_pressed:
                right_count += 1
            left_was_pressed = left_pressed
            right_was_pressed = right_pressed

            draw_button(annotated_frame, LEFT_BUTTON, LEFT_COLOR, "Left", left_pressed)
            draw_button(annotated_frame, RIGHT_BUTTON, RIGHT_COLOR, "Right", right_pressed)
            draw_counters(annotated_frame, left_count, right_count)

            cv2.imshow(WINDOW_NAME, annotated_frame)
            key = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
            if key == QUIT_KEY or key == ESC_KEY:
                break
    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
