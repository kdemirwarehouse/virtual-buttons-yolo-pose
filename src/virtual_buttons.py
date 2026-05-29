"""
Virtual Buttons with YOLO11 Pose Estimation.

This script captures frames from the default webcam, mirrors them horizontally
so the preview behaves like a mirror, and exits cleanly when the user presses
the 'q' key.
"""

import cv2


def main() -> None:
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        raise RuntimeError("Could not open webcam (device index 0).")

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        # Mirror the frame so it behaves like a real mirror.
        frame = cv2.flip(frame, 1)

        cv2.imshow("Virtual Buttons", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
