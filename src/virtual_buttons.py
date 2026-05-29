"""
Virtual Buttons with YOLO11 Pose Estimation.

This script captures frames from the default webcam, runs the YOLO11 pose model
on every frame, extracts the right-wrist keypoint, and renders two virtual
buttons (LEFT and RIGHT). When the wrist is inside a button rectangle, the
button is filled to indicate that it is being "pressed".
"""

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "model/yolo11m-pose.pt"
RIGHT_WRIST_INDEX = 10

LEFT_BUTTON = (30, 30, 160, 120)
RIGHT_BUTTON = (430, 30, 560, 120)

LEFT_COLOR = (0, 0, 255)
RIGHT_COLOR = (0, 255, 255)
WRIST_COLOR = (0, 255, 0)


def get_right_wrist(results) -> tuple[int, int] | tuple[None, None]:
    if results.keypoints is None or results.keypoints.xy is None:
        return None, None

    keypoints = results.keypoints.xy
    if len(keypoints) == 0 or keypoints.shape[1] <= RIGHT_WRIST_INDEX:
        return None, None

    kp = keypoints[0, RIGHT_WRIST_INDEX]
    if np.isnan(kp[0].item()) or np.isnan(kp[1].item()):
        return None, None

    return int(kp[0].item()), int(kp[1].item())


def point_in_rect(point, rect) -> bool:
    """Return True if `point=(x, y)` is inside the axis-aligned rectangle."""
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_button(frame, rect, color, label, pressed: bool) -> None:
    """Draw a virtual button. If `pressed` is True the rectangle is filled."""
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


def main() -> None:
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        raise RuntimeError("Could not open webcam (device index 0).")

    model = YOLO(MODEL_PATH)

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
            cv2.circle(annotated_frame, wrist_point, 5, WRIST_COLOR, -1)
            cv2.putText(
                annotated_frame,
                "KP10",
                (wrist_point[0] + 6, wrist_point[1] - 6),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                WRIST_COLOR,
                1,
                cv2.LINE_AA,
            )

        left_pressed = wrist_point is not None and point_in_rect(wrist_point, LEFT_BUTTON)
        right_pressed = wrist_point is not None and point_in_rect(wrist_point, RIGHT_BUTTON)

        draw_button(annotated_frame, LEFT_BUTTON, LEFT_COLOR, "Left", left_pressed)
        draw_button(annotated_frame, RIGHT_BUTTON, RIGHT_COLOR, "Right", right_pressed)

        cv2.imshow("Virtual Buttons", annotated_frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
