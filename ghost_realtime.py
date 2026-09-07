"""
Ghost Network — Real-Time Privacy-Preserving Camera Application
================================================================
Captures live camera feed, applies Ghost edge_diff transformation,
and displays side-by-side: original RGB vs Ghost (edge-difference) frames.

Features:
  - Real-time edge_diff Ghost conversion from webcam
  - Adjustable threshold via trackbar (fixed or adaptive/Otsu)
  - FPS counter
  - Privacy score estimation (face detection rate on RGB vs Ghost)
  - Side-by-side display with info overlay
  - Press 'q' to quit, 's' to save snapshot, 'a' to toggle adaptive threshold

Usage:
  python ghost_realtime.py
  python ghost_realtime.py --camera 0 --threshold 15 --adaptive
  python ghost_realtime.py --width 640 --height 480
"""

import os

qt_platform = os.environ.get("QT_QPA_PLATFORM", "")
if ";" in qt_platform or qt_platform.lower() == "wayland;xcb":
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import argparse
import time
from datetime import datetime
from collections import deque


# ============================================================
# GHOST CONVERTER  (edge_diff only, optimized for real-time)
# ============================================================

class GhostConverterRealtime:
    """
    Real-time Ghost converter using edge_diff strategy.

    Pipeline per frame:
      1. Convert current frame to grayscale
      2. Run Canny edge detection on current and previous frame
      3. Compute absolute difference between edge maps
      4. Apply threshold (fixed or Otsu adaptive) → binary mask
      5. Stack mask into 3-channel output

    Privacy mechanism:
      - Only temporal edge changes are preserved
      - All texture, color, and identity information is stripped
      - Sufficient for motion/behavioral analysis
    """

    def __init__(self, threshold=15, use_adaptive=True, canny_low=50, canny_high=150):
        self.threshold = threshold
        self.use_adaptive = use_adaptive
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.prev_frame = None
        self.prev_edges = None

    def reset(self):
        """Reset state (e.g., after camera reconnect)."""
        self.prev_frame = None
        self.prev_edges = None

    def convert_frame(self, frame_bgr):
        """
        Convert a single BGR frame to Ghost representation.

        Args:
            frame_bgr: np.ndarray (H, W, 3) uint8, BGR format from OpenCV

        Returns:
            ghost: np.ndarray (H, W, 3) uint8, 3-channel Ghost frame
            edges_current: np.ndarray (H, W) uint8, current Canny edges (for debug)
            diff: np.ndarray (H, W) uint8, raw edge difference (for debug)
        """
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Step 2: Canny edge detection on current frame
        edges_current = cv2.Canny(gray, self.canny_low, self.canny_high)

        # Step 3: Handle first frame — use itself as previous (zero-diff)
        if self.prev_frame is None or self.prev_edges is None:
            self.prev_frame = frame_bgr.copy()
            self.prev_edges = edges_current.copy()
            # First frame: return black ghost (no motion yet)
            ghost = np.zeros_like(frame_bgr)
            return ghost, edges_current, np.zeros_like(edges_current)

        # Step 4: Compute absolute difference between edge maps
        diff = cv2.absdiff(edges_current, self.prev_edges)

        # Step 5: Apply threshold
        if self.use_adaptive:
            thresh = self._otsu_threshold(diff)
        else:
            thresh = self.threshold

        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

        # Step 6: Stack into 3-channel Ghost frame
        ghost = np.stack([mask, mask, mask], axis=-1)

        # Update state
        self.prev_frame = frame_bgr.copy()
        self.prev_edges = edges_current.copy()

        return ghost, edges_current, diff

    def _otsu_threshold(self, diff):
        """Compute adaptive threshold using Otsu's method."""
        if len(np.unique(diff)) < 2:
            return 10
        t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return max(5, min(int(t), 50))


# ============================================================
# FACE DETECTOR  (for privacy score estimation)
# ============================================================

class SimpleFaceDetector:
    """
    Lightweight face detector using OpenCV Haar Cascade.
    No extra dependencies needed — ships with OpenCV.

    Used to estimate privacy preservation:
      - Detect faces in RGB frame
      - Detect faces in Ghost frame
      - Privacy score = 1 - (ghost_faces / rgb_faces)
    """

    def __init__(self):
        # Try Haar cascade first (always available with OpenCV)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.name = "HaarCascade"

    def detect_count(self, frame_bgr):
        """Return number of faces detected in a BGR frame."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return len(faces)


# ============================================================
# DISPLAY UTILITIES
# ============================================================

class DisplayManager:
    """Manages the display window with side-by-side panels and overlays."""

    # Layout constants
    PANEL_WIDTH = 640
    PANEL_HEIGHT = 480
    INFO_HEIGHT = 120
    PADDING = 5

    # Colors (BGR)
    BG_COLOR = (30, 30, 30)
    TEXT_COLOR = (220, 220, 220)
    ACCENT_COLOR = (0, 200, 100)  # Green
    WARNING_COLOR = (0, 160, 255)  # Orange
    HEADER_BG = (50, 50, 50)

    def __init__(self, title="Ghost Network — Real-Time Privacy-Preserving Camera"):
        self.title = title
        self.fps_history = deque(maxlen=30)
        self.privacy_history = deque(maxlen=60)

    def create_window(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        total_w = self.PANEL_WIDTH * 3 + self.PADDING * 4
        total_h = self.PANEL_HEIGHT + self.INFO_HEIGHT + self.PADDING * 3
        cv2.resizeWindow(self.title, total_w, total_h)

    def render(self, rgb_frame, ghost_frame, edges, diff,
               fps, threshold, use_adaptive, face_rgb, face_ghost,
               frame_count, strategy="edge_diff"):
        """
        Render the full display with 3 panels + info bar.

        Panels:
          [0] Original RGB camera feed
          [1] Canny edges (current frame)
          [2] Ghost output (edge difference, thresholded)
        """
        # Create canvas
        panel_w = self.PANEL_WIDTH
        panel_h = self.PANEL_HEIGHT
        info_h = self.INFO_HEIGHT
        pad = self.PADDING

        total_w = panel_w * 3 + pad * 4
        total_h = panel_h + info_h + pad * 3

        canvas = np.full((total_h, total_w, 3), self.BG_COLOR, dtype=np.uint8)

        # Resize frames to panel size
        rgb_resized = cv2.resize(rgb_frame, (panel_w, panel_h))
        ghost_resized = cv2.resize(ghost_frame, (panel_w, panel_h))

        # Convert edges and diff to 3-channel for display
        edges_color = cv2.cvtColor(cv2.resize(edges, (panel_w, panel_h)), cv2.COLOR_GRAY2BGR)
        diff_color = cv2.applyColorMap(cv2.resize(diff, (panel_w, panel_h)), cv2.COLORMAP_HOT)

        # Place panels
        y0 = pad
        x0 = pad
        canvas[y0:y0 + panel_h, x0:x0 + panel_w] = rgb_resized

        x1 = pad * 2 + panel_w
        canvas[y0:y0 + panel_h, x1:x1 + panel_w] = edges_color

        x2 = pad * 3 + panel_w * 2
        canvas[y0:y0 + panel_h, x2:x2 + panel_w] = ghost_resized

        # Panel labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        for label, x in [("ORIGINAL RGB", x0), ("CANNY EDGES", x1), ("GHOST OUTPUT", x2)]:
            # Label background
            cv2.rectangle(canvas, (x, y0), (x + panel_w, y0 + 30), self.HEADER_BG, -1)
            cv2.putText(canvas, label, (x + 10, y0 + 22), font, 0.6, self.TEXT_COLOR, 1, cv2.LINE_AA)

        # Info bar
        info_y = pad * 2 + panel_h
        cv2.rectangle(canvas, (pad, info_y), (total_w - pad, info_y + info_h), (40, 40, 40), -1)

        # Compute privacy score
        if face_rgb > 0:
            privacy_score = 1.0 - (face_ghost / face_rgb)
        else:
            privacy_score = 1.0  # No faces in RGB = perfect privacy by default

        self.privacy_history.append(privacy_score)
        avg_privacy = np.mean(list(self.privacy_history)) if self.privacy_history else 0

        # Info text
        col1_x = pad + 15
        col2_x = total_w // 2 + 15
        ty = info_y + 25
        line_h = 22

        # Column 1: Performance
        cv2.putText(canvas, f"FPS: {fps:.1f}", (col1_x, ty), font, 0.55, self.ACCENT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame: {frame_count}", (col1_x, ty + line_h), font, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Strategy: {strategy}", (col1_x, ty + line_h * 2), font, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA)

        # Column 2: Privacy metrics
        priv_color = self.ACCENT_COLOR if avg_privacy > 0.8 else self.WARNING_COLOR
        cv2.putText(canvas, f"Privacy Score: {avg_privacy:.1%}", (col2_x, ty), font, 0.55, priv_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Faces RGB: {face_rgb}  Ghost: {face_ghost}", (col2_x, ty + line_h), font, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA)
        mode = "Otsu (adaptive)" if use_adaptive else f"Fixed ({threshold})"
        cv2.putText(canvas, f"Threshold: {mode}", (col2_x, ty + line_h * 2), font, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA)

        # Bottom hint bar
        hint_y = total_h - 8
        hint = "Press: [q]uit  [s]napshot  [a]daptive toggle  [+/-]threshold  [r]eset"
        cv2.putText(canvas, hint, (pad + 10, hint_y), font, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow(self.title, canvas)

    def update_fps(self, dt):
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        return np.mean(list(self.fps_history)) if self.fps_history else 0


# ============================================================
# SNAPSHOT UTILITY
# ============================================================

def save_snapshot(rgb, ghost, edges, diff, output_dir="./ghost_snapshots"):
    """Save a snapshot of all panels."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    cv2.imwrite(os.path.join(output_dir, f"{ts}_rgb.png"), rgb)
    cv2.imwrite(os.path.join(output_dir, f"{ts}_ghost.png"), ghost)
    cv2.imwrite(os.path.join(output_dir, f"{ts}_edges.png"), edges)
    cv2.imwrite(os.path.join(output_dir, f"{ts}_diff.png"), diff)

    # Also save a combined side-by-side
    h, w = rgb.shape[:2]
    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
    combined[:, :w] = rgb
    combined[:, w:w * 2] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined[:, w * 2:] = ghost
    cv2.imwrite(os.path.join(output_dir, f"{ts}_combined.png"), combined)

    return ts


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ghost Network — Real-Time Privacy Camera")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--threshold", type=int, default=15, help="Edge diff threshold (default: 15)")
    parser.add_argument("--adaptive", action="store_true", default=True, help="Use Otsu adaptive threshold")
    parser.add_argument("--no-adaptive", action="store_true", default=False, help="Disable adaptive threshold")
    parser.add_argument("--canny-low", type=int, default=50, help="Canny low threshold (default: 50)")
    parser.add_argument("--canny-high", type=int, default=150, help="Canny high threshold (default: 150)")
    parser.add_argument("--no-faces", action="store_true", default=False, help="Disable face detection (faster)")
    parser.add_argument("--output", type=str, default="./ghost_snapshots", help="Snapshot output directory")
    args = parser.parse_args()

    use_adaptive = args.adaptive and not args.no_adaptive

    print("=" * 60)
    print("  Ghost Network — Real-Time Privacy-Preserving Camera")
    print("=" * 60)
    print(f"  Camera:     /dev/video{args.camera} ({args.width}x{args.height})")
    print(f"  Strategy:   edge_diff")
    print(f"  Threshold:  {'Otsu (adaptive)' if use_adaptive else str(args.threshold)}")
    print(f"  Canny:      low={args.canny_low}, high={args.canny_high}")
    print(f"  Face det:   {'disabled' if args.no_faces else 'HaarCascade'}")
    print(f"  Snapshots:  {args.output}")
    print("=" * 60)
    print()

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        print("Available cameras:")
        for i in range(5):
            test = cv2.VideoCapture(i)
            if test.isOpened():
                print(f"  Camera {i}: available")
                test.release()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_w}x{actual_h}")

    # Initialize components
    ghost_conv = GhostConverterRealtime(
        threshold=args.threshold,
        use_adaptive=use_adaptive,
        canny_low=args.canny_low,
        canny_high=args.canny_high
    )

    face_detector = None if args.no_faces else SimpleFaceDetector()
    display = DisplayManager()
    display.create_window()

    # Create trackbars only when the backend supports them.
    trackbars_enabled = True
    try:
        cv2.createTrackbar("Threshold", display.title, args.threshold, 100, lambda x: None)
        cv2.createTrackbar("Canny Low", display.title, args.canny_low, 255, lambda x: None)
        cv2.createTrackbar("Canny High", display.title, args.canny_high, 255, lambda x: None)
    except cv2.error:
        trackbars_enabled = False
        print("WARNING: OpenCV trackbars are unavailable in this environment; using command-line values only.")

    # State
    frame_count = 0
    running = True
    paused = False

    print("\nControls:")
    print("  q       — Quit")
    print("  s       — Save snapshot")
    print("  a       — Toggle adaptive threshold")
    print("  SPACE   — Pause/Resume")
    print("  r       — Reset Ghost state")
    print()

    try:
        while running:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("WARNING: Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue

                frame_count += 1
                t_start = time.time()

                if trackbars_enabled:
                    # Read trackbar values
                    tb_thresh = cv2.getTrackbarPos("Threshold", display.title)
                    tb_canny_low = cv2.getTrackbarPos("Canny Low", display.title)
                    tb_canny_high = cv2.getTrackbarPos("Canny High", display.title)

                    # Update converter params from trackbars
                    if not ghost_conv.use_adaptive and tb_thresh > 0:
                        ghost_conv.threshold = tb_thresh
                    if tb_canny_low > 0:
                        ghost_conv.canny_low = tb_canny_low
                    if tb_canny_high > 0:
                        ghost_conv.canny_high = tb_canny_high

                # Apply Ghost conversion
                ghost_frame, edges, diff = ghost_conv.convert_frame(frame)

                # Face detection (every 5 frames for performance)
                face_rgb, face_ghost = 0, 0
                if face_detector is not None and frame_count % 5 == 0:
                    face_rgb = face_detector.detect_count(frame)
                    face_ghost = face_detector.detect_count(ghost_frame)

                # FPS
                dt = time.time() - t_start
                fps = display.update_fps(dt)

                # Render
                display.render(
                    rgb_frame=frame,
                    ghost_frame=ghost_frame,
                    edges=edges,
                    diff=diff,
                    fps=fps,
                    threshold=ghost_conv.threshold,
                    use_adaptive=ghost_conv.use_adaptive,
                    face_rgb=face_rgb,
                    face_ghost=face_ghost,
                    frame_count=frame_count,
                    strategy="edge_diff"
                )

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                running = False

            elif key == ord('s'):
                # Ensure variables are defined before use
                if 'frame' in dir() and 'ghost_frame' in dir() and 'edges' in dir() and 'diff' in dir():
                    ts = save_snapshot(frame, ghost_frame, edges, diff, args.output)
                    print(f"  Snapshot saved: {ts}")
                else:
                    print("  No frame available yet for snapshot")

            elif key == ord('a'):
                ghost_conv.use_adaptive = not ghost_conv.use_adaptive
                mode = "Otsu (adaptive)" if ghost_conv.use_adaptive else f"Fixed ({ghost_conv.threshold})"
                print(f"  Threshold mode: {mode}")

            elif key == ord(' '):
                paused = not paused
                print(f"  {'Paused' if paused else 'Resumed'}")

            elif key == ord('r'):
                ghost_conv.reset()
                print("  Ghost state reset")

            elif key == ord('+') or key == ord('='):
                ghost_conv.threshold = min(ghost_conv.threshold + 1, 100)
                print(f"  Threshold: {ghost_conv.threshold}")

            elif key == ord('-') or key == ord('_'):
                ghost_conv.threshold = max(ghost_conv.threshold - 1, 1)
                print(f"  Threshold: {ghost_conv.threshold}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal frames processed: {frame_count}")
        print("Ghost Network stopped.")


if __name__ == "__main__":
    main()

