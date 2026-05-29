"""Central configuration for Virtual Buttons."""

from __future__ import annotations

MODEL_PATH = "model/yolo11m-pose.pt"
WINDOW_NAME = "Virtual Buttons"

# COCO-17 keypoint indices
RIGHT_WRIST_INDEX = 10
LEFT_WRIST_INDEX = 9

# Virtual button rectangles: (x1, y1, x2, y2)
LEFT_BUTTON = (30, 30, 160, 120)
RIGHT_BUTTON = (430, 30, 560, 120)

# BGR colors
LEFT_COLOR = (0, 0, 255)    # red
RIGHT_COLOR = (0, 255, 255) # yellow
WRIST_COLOR = (0, 255, 0)   # green

QUIT_KEY = ord("q")
ESC_KEY = 27
WAIT_KEY_DELAY_MS = 5

# Debounce: minimum frames between two consecutive presses
DEBOUNCE_FRAMES = 8
