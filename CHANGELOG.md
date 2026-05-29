# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `--camera` CLI argument to select webcam device index.
- Frame-count debounce to avoid accidental button presses.
- Real-time FPS counter overlay in the top-right corner.
- Live press counters (L / R) at the bottom of the frame.
- ESC key as an alternative quit shortcut.
- `src/config.py` centralising all configuration constants.
- `pyproject.toml` with project metadata and `virtual-buttons` entry-point.
- `Makefile` with `install`, `run`, `lint`, `test` and `clean` targets.
- GitHub issue templates for bug reports and feature requests.
- GitHub Actions lint workflow using ruff.
- `CONTRIBUTING.md` with setup and PR guidelines.
- Unit tests for helper functions in `tests/`.

## [0.1.0] - 2026-05-30

### Added
- Initial webcam capture loop with mirror flip.
- YOLO11 pose model loading and skeleton rendering.
- Right-wrist keypoint (KP10) extraction and marker.
- Left and right virtual button rectangles with labels.
- Wrist-over-button collision detection and fill-on-press behaviour.
- `try/finally` cleanup for webcam and OpenCV windows.
- `requirements.txt` with pinned opencv, numpy and ultralytics.
- MIT licence and `.gitignore`.
- Project README with setup, usage and keypoint reference table.
