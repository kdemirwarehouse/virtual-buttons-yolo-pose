# Model Directory

This folder holds the YOLO11 pose-estimation weights used by the application.

## Expected file

```
model/
└── yolo11m-pose.pt
```

## How to obtain the weights

You don't have to download it manually. The first time you run

```bash
python src/virtual_buttons.py
```

Ultralytics will automatically download `yolo11m-pose.pt` from the official
release into your working directory. Just move the downloaded file into this
`model/` folder (or update `MODEL_PATH` in `src/virtual_buttons.py` if you
prefer a different location).

If you want to grab it manually:

* GitHub releases: https://github.com/ultralytics/assets/releases
* Look for the asset named `yolo11m-pose.pt`.

## Other supported variants

You can swap in any other YOLO11 pose checkpoint and update `MODEL_PATH`
accordingly:

| File                 | Size     | Speed  | Accuracy |
|----------------------|----------|--------|----------|
| `yolo11n-pose.pt`    | ~6 MB    | fast   | lower    |
| `yolo11s-pose.pt`    | ~20 MB   | fast   | medium   |
| `yolo11m-pose.pt`    | ~40 MB   | medium | good (default) |
| `yolo11l-pose.pt`    | ~50 MB   | slow   | better   |
| `yolo11x-pose.pt`    | ~110 MB  | slow   | best     |

> The `.pt` files are intentionally listed in `.gitignore` so they don't bloat
> the repository.
