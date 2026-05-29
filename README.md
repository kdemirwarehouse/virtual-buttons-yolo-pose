# Virtual Buttons with YOLO11 Pose

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-purple)
![License](https://img.shields.io/badge/license-MIT-yellow)

A small, beginner-friendly computer-vision demo that turns your webcam into a
touchless interface. The app runs a **YOLO11 pose-estimation** model on every
frame, locks onto the **right-wrist keypoint** and "presses" one of two
on-screen buttons whenever the wrist enters its area.

> Built as a learning project while exploring real-time pose estimation with
> Ultralytics + OpenCV.

## Features

- Real-time pose estimation from any webcam using YOLO11.
- Tracks the right-wrist keypoint (COCO-17 index **10**) and visualises it.
- Two virtual buttons (LEFT / RIGHT) that fill in solid color when "pressed".
- Mirrored preview so movement feels natural.
- Clean shutdown with `q` and `try/finally` guard around the capture loop.

## Demo

> Add screenshots / GIFs under `docs/screenshots/` and reference them here.

```
docs/screenshots/preview.png
docs/screenshots/left_pressed.png
docs/screenshots/right_pressed.png
```

Example embed:

```markdown
![Preview](docs/screenshots/preview.png)
```

## Project Structure

```
virtual-buttons-yolo-pose/
├── docs/
│   └── screenshots/         # Demo images / GIFs (referenced from README)
├── model/
│   ├── README.md            # How to obtain the YOLO11 pose weights
│   └── yolo11m-pose.pt      # (Downloaded automatically, gitignored)
├── src/
│   ├── __init__.py
│   └── virtual_buttons.py   # Main application
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kdemirwarehouse/virtual-buttons-yolo-pose.git
cd virtual-buttons-yolo-pose
```

### 2. Create a virtual environment

Conda:

```bash
conda create -n virtual-buttons python=3.10 -y
conda activate virtual-buttons
```

Or plain venv:

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python src/virtual_buttons.py
```

On the very first launch Ultralytics will download `yolo11m-pose.pt`
automatically. Move it into the `model/` folder afterwards (or change
`MODEL_PATH` in `src/virtual_buttons.py`).

Press **`q`** to quit.

## How It Works

1. **Capture** — `cv2.VideoCapture(0)` reads frames from the default webcam.
2. **Mirror** — `cv2.flip(frame, 1)` flips the frame horizontally so it
   behaves like a mirror.
3. **Inference** — the frame is passed to `model(frame)` and the result's
   skeleton is drawn with `results.plot()`.
4. **Keypoint extraction** — keypoint **10** (right wrist) is read from
   `results.keypoints.xy[0, 10]`, with NaN/length guards so missing
   detections don't crash the loop.
5. **Collision check** — a simple axis-aligned `point_in_rect` test decides
   whether the wrist is inside the LEFT or RIGHT button rectangle.
6. **Feedback** — `cv2.rectangle(..., thickness=-1)` fills the button when it
   is being "pressed"; otherwise only the outline is drawn.

## Keypoint Reference (COCO-17)

| Index | Keypoint        | Index | Keypoint        |
|------:|-----------------|------:|-----------------|
| 0     | Nose            | 9     | Left wrist      |
| 1     | Left eye        | **10**| **Right wrist** |
| 2     | Right eye       | 11    | Left hip        |
| 3     | Left ear        | 12    | Right hip       |
| 4     | Right ear       | 13    | Left knee       |
| 5     | Left shoulder   | 14    | Right knee      |
| 6     | Right shoulder  | 15    | Left ankle      |
| 7     | Left elbow      | 16    | Right ankle     |
| 8     | Right elbow     |       |                 |

To track a different body part, change `RIGHT_WRIST_INDEX` in
`src/virtual_buttons.py`.

## Configuration

All knobs live at the top of `src/virtual_buttons.py`:

| Constant            | Meaning                                          |
|---------------------|--------------------------------------------------|
| `MODEL_PATH`        | Path to the YOLO11 pose weights.                 |
| `RIGHT_WRIST_INDEX` | COCO-17 keypoint index to track.                 |
| `LEFT_BUTTON`       | `(x1, y1, x2, y2)` of the left button.           |
| `RIGHT_BUTTON`      | `(x1, y1, x2, y2)` of the right button.          |
| `LEFT_COLOR`        | BGR color of the left button.                    |
| `RIGHT_COLOR`       | BGR color of the right button.                   |

## Troubleshooting

**Webcam doesn't open**
The script raises `RuntimeError: Could not open webcam (device index 0)`.
Try a different index (`cv2.VideoCapture(1)`) or check OS permissions.

**`ModuleNotFoundError: ultralytics`**
Run `pip install -r requirements.txt` inside your activated environment.
On Spyder, restart the kernel afterwards.

**`torch.cuda.is_available()` is `False`**
The app still works on CPU, just at a lower FPS. For GPU acceleration install
a CUDA-matched torch build manually before installing the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Buttons don't react**
Make sure your right hand (the wrist) is fully inside the rectangle and that
the green `KP10` marker is visible.

## Performance Tips

| Tip | Expected gain |
|-----|--------------|
| Use `yolo11n-pose.pt` (nano) instead of medium | +10–15 FPS on CPU |
| Lower webcam resolution: `webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` | Reduced inference time |
| Install a CUDA-matched PyTorch build | 3–5× FPS on GPU |
| Increase `WAIT_KEY_DELAY_MS` to 15–20 | Frees CPU between frames |
| Disable skeleton overlay: change `results.plot()` to `annotated_frame = frame.copy()` | Removes per-frame drawing cost |

## Tech Stack

- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- Ultralytics YOLO11 Pose

## FAQ

**Can I use a video file instead of a webcam?**
Yes — pass the file path to `cv2.VideoCapture("path/to/video.mp4")` in `main()`.

**Why does the button fill even when I'm not touching it?**
Your wrist is likely within the rectangle coordinates. Increase `x1`/`y1` margins or reduce the button size in `config.py`.

**Can I track a different body part?**
Yes — change `RIGHT_WRIST_INDEX` to any COCO-17 index (0–16). The `--left-hand` flag switches to KP9 automatically.

**Does it work on macOS/Linux?**
Yes — the code is platform-agnostic. Make sure your webcam is accessible and adjust the device index if needed.

## Future Work

- [ ] Add a "click" debounce so the buttons don't trigger every single frame.
- [ ] Map button presses to real OS actions (volume up/down, slide next/prev).
- [ ] Support left wrist (KP9) and let the user pick which hand to track.
- [ ] Configurable layout (number of buttons, positions, labels) via a JSON file.
- [ ] Replace the rectangle hit-test with circular buttons for nicer visuals.
- [ ] Export an `.onnx` / `.engine` variant of the model for faster inference.

## License

MIT — see [LICENSE](LICENSE).
