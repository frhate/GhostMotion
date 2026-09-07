# Ghost Network  Real-Time Privacy-Preserving Camera

## Overview

This application captures live camera feed and applies the **Ghost edge_diff transformation**  a privacy-preserving computer vision technique that strips all identity-revealing information while retaining motion/behavioral signals.

## How Edge Diff Works

```
Frame t-1 ──→ Canny Edges ──┐
                     ├──→ absdiff ──→ Threshold ──→ Ghost Frame
Frame t   ──→ Canny Edges ──┘
```

1. **Canny Edge Detection**: Extract structural edges from consecutive frames
2. **Absolute Difference**: Compute pixel-wise difference between edge maps  captures *where edges moved*
3. **Thresholding**: Apply fixed or Otsu adaptive threshold to produce binary mask
4. **Output**: 3-channel ghost frame showing only motion-induced edge changes

**Privacy guarantee**: The ghost frame contains zero texture, color, or facial feature information  only the temporal change in structural edges. This is sufficient for behavioral analysis but insufficient for face recognition.

## Files

| File | Description |
|------|-------------|
| `ghost_realtime.py` | Lightweight OpenCV-only version (no GUI dependencies beyond OpenCV) |
| `ghost_realtime_gui.py` | Full tkinter GUI with matplotlib plots and recording |
| `ghost_requirements.txt` | Python dependencies |

## Quick Start

### Install dependencies
```bash
sudo apt install python3-venv
sudo apt install python3-tk
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ghost_requirements.txt
```

If you already have a virtual environment, activate it first and run the `python -m pip install -r ghost_requirements.txt` command inside that environment.

### Run lightweight version
```bash
python ghost_realtime.py
python ghost_realtime.py --camera 0 --threshold 20 --adaptive
python ghost_realtime.py --width 1280 --height 720
```

### Run GUI version
```bash
python ghost_realtime_gui.py
```

If the GUI starts with `ModuleNotFoundError: No module named 'tkinter'`, install `python3-tk` with your OS package manager first.

## Controls (Lightweight Version)

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `s` | Save snapshot |
| `a` | Toggle adaptive (Otsu) threshold |
| `SPACE` | Pause/Resume |
| `r` | Reset Ghost state |
| `+` / `-` | Adjust threshold |

## GUI Version Features

- **Real-time parameter adjustment**: threshold, Canny low/high, blur, morphological kernel
- **Display modes**: Side-by-side, Ghost only, RGB only, 4-panel view
- **Live metrics plots**: FPS, Privacy score, Edge density
- **Recording**: Save ghost video to MP4/AVI
- **Snapshots**: Save with JSON metadata

## Privacy Score

The privacy score is estimated using face detection:

```
Privacy Score = 1 - (faces_detected_in_ghost / faces_detected_in_rgb)
```

- **1.0 (100%)**: No faces detected in ghost frames  perfect privacy
- **0.0 (0%)**: Faces still detectable in ghost frames  no privacy

The Haar cascade detector is used by default (ships with OpenCV). For more accurate privacy evaluation, install `facenet-pytorch` for MTCNN or `insightface` for RetinaFace.

If `python ghost_realtime.py` reports that it cannot open camera 0 and no `/dev/video*` devices are available, the machine has no accessible webcam. Use a different camera index if one exists, or connect/enable a camera device before running the app.

## Architecture

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
│ Camera    │────→│ Grayscale   │────→│ Canny Edges │────→│ absdiff    │
│ Frame (BGR) │   │ Conversion  │   │ Detection  │   │ (t vs t-1)  │
└─────────────┘   └──────────────┘   └─────────────┘   └──────┬───────┘
                                             │
                                             ▼
┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
│ Display   │←────│ 3-Channel   │←────│ Binary    │←────│ Threshold   │
│ (RGB+Ghost) │   │ Stack     │   │ Mask     │   │ (Fixed/Otsu) │
└─────────────┘   └──────────────┘   └─────────────┘   └──────────────┘
```

## Citation

If you use this code in your research, please cite the original Ghost-ASD work:

```bibtech
@thesis{ghost_asd_2026,
 title={Privacy-Preserving ASD Behavioral Video Analysis using Ghost Networks},
 author={Ferhat Boulahia},
 year={2026}
}
```

