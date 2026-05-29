"""
Virtual Buttons with YOLO11 Pose Estimation.

This script captures frames from the default webcam, runs the YOLO11 pose model
on every frame, and renders the model's pose skeleton on top of the preview.
"""

import cv2
from ultralytics import YOLO

MODEL_PATH = "model/yolo11m-pose.pt"


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

        # Run pose estimation on the current frame.
        results = model(frame)[0]
        annotated_frame = results.plot()

        cv2.imshow("Virtual Buttons", annotated_frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
