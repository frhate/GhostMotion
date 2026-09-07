"""
Ghost Network — Real-Time Privacy-Preserving Camera (GUI Version)
===================================================================
Full desktop application with tkinter GUI, matplotlib visualizations,
and comprehensive controls for the Ghost edge_diff pipeline.

Features:
  - Live camera feed with Ghost transformation
  - Matplotlib plots: FPS, privacy score, edge statistics
  - Adjustable parameters: threshold, Canny thresholds, blur
  - Privacy evaluation with face detection
  - Recording capability (save ghost video)
  - Snapshot with metadata
  - Strategy comparison view

Usage:
  python ghost_realtime_gui.py
"""

import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
import json
from datetime import datetime
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: tkinter is not installed. Install it with: sudo apt install python3-tk"
    ) from exc


# ============================================================
# GHOST CONVERTER
# ============================================================

class GhostConverterRealtime:
    """Real-time Ghost converter using edge_diff strategy."""

    def __init__(self, threshold=15, use_adaptive=True, canny_low=50, canny_high=150,
                 blur_kernel=0, morph_kernel=0):
        self.threshold = threshold
        self.use_adaptive = use_adaptive
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel
        self.prev_edges = None

    def reset(self):
        self.prev_edges = None

    def convert_frame(self, frame_bgr):
        """Convert BGR frame to Ghost representation."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Optional Gaussian blur to reduce noise
        if self.blur_kernel > 0:
            k = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        edges_current = cv2.Canny(gray, self.canny_low, self.canny_high)

        if self.prev_edges is None:
            self.prev_edges = edges_current.copy()
            ghost = np.zeros_like(frame_bgr)
            return ghost, edges_current, np.zeros_like(edges_current), 0

        diff = cv2.absdiff(edges_current, self.prev_edges)

        # Optional morphological cleanup
        if self.morph_kernel > 0:
            kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
            diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

        if self.use_adaptive:
            thresh = self._otsu_threshold(diff)
        else:
            thresh = self.threshold

        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        ghost = np.stack([mask, mask, mask], axis=-1)

        # Edge density metric
        edge_density = np.count_nonzero(mask) / max(mask.size, 1)

        self.prev_edges = edges_current.copy()
        return ghost, edges_current, diff, edge_density

    def _otsu_threshold(self, diff):
        if len(np.unique(diff)) < 2:
            return 10
        t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return max(5, min(int(t), 50))


# ============================================================
# FACE DETECTOR
# ============================================================

class SimpleFaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect_count(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return len(faces)


# ============================================================
# VIDEO WRITER
# ============================================================

class GhostVideoWriter:
    def __init__(self, output_path, fps=30, frame_size=(640, 480)):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.output_path = output_path

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


# ============================================================
# MAIN APPLICATION
# ============================================================

class GhostNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ghost Network — Real-Time Privacy-Preserving Camera")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # State
        self.cap = None
        self.running = False
        self.paused = False
        self.recording = False
        self.video_writer = None
        self.frame_count = 0

        # Components
        self.ghost_conv = GhostConverterRealtime()
        self.face_detector = SimpleFaceDetector()

        # Metrics history
        self.fps_history = deque(maxlen=120)
        self.privacy_history = deque(maxlen=120)
        self.edge_density_history = deque(maxlen=120)
        self.face_rgb_history = deque(maxlen=120)
        self.face_ghost_history = deque(maxlen=120)

        # Build UI
        self._build_ui()

        # Bind close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        """Build the complete tkinter UI."""
        # Main layout: left panel (video) | right panel (controls + plots)
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── Left: Video display ────────────────────────────────────
        left_frame = ttk.Frame(self.root)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)

        # Video canvas
        self.video_canvas = tk.Canvas(left_frame, bg="black", highlightthickness=0)
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        # Status bar below video
        status_frame = ttk.Frame(left_frame)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        self.status_label = ttk.Label(status_frame, text="Status: Stopped", font=("Consolas", 10))
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.fps_label = ttk.Label(status_frame, text="FPS: --", font=("Consolas", 10))
        self.fps_label.pack(side=tk.LEFT, padx=20)

        self.privacy_label = ttk.Label(status_frame, text="Privacy: --", font=("Consolas", 10))
        self.privacy_label.pack(side=tk.LEFT, padx=20)

        self.recording_label = ttk.Label(status_frame, text="", font=("Consolas", 10), foreground="red")
        self.recording_label.pack(side=tk.RIGHT, padx=5)

        # ── Right: Controls + Plots ────────────────────────────────
        right_frame = ttk.Frame(self.root)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right_frame.columnconfigure(0, weight=1)

        # Controls section
        controls = ttk.LabelFrame(right_frame, text="Controls", padding=10)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        controls.columnconfigure(0, weight=1)

        # Camera controls
        cam_frame = ttk.Frame(controls)
        cam_frame.grid(row=0, column=0, sticky="ew", pady=2)

        ttk.Label(cam_frame, text="Camera:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="0")
        cam_entry = ttk.Entry(cam_frame, textvariable=self.camera_var, width=5)
        cam_entry.pack(side=tk.LEFT, padx=5)

        self.start_btn = ttk.Button(cam_frame, text="▶ Start", command=self._start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = ttk.Button(cam_frame, text="⏹ Stop", command=self._stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        self.pause_btn = ttk.Button(cam_frame, text="⏸ Pause", command=self._toggle_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=2)

        # Threshold controls
        thresh_frame = ttk.LabelFrame(controls, text="Threshold", padding=5)
        thresh_frame.grid(row=1, column=0, sticky="ew", pady=5)

        self.adaptive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(thresh_frame, text="Adaptive (Otsu)", variable=self.adaptive_var,
                        command=self._update_params).grid(row=0, column=0, sticky="w")

        ttk.Label(thresh_frame, text="Fixed:").grid(row=0, column=1, padx=(10, 2))
        self.threshold_var = tk.IntVar(value=15)
        self.threshold_scale = ttk.Scale(thresh_frame, from_=1, to=100,
                                         variable=self.threshold_var, orient=tk.HORIZONTAL,
                                         command=lambda _: self._update_params())
        self.threshold_scale.grid(row=0, column=2, sticky="ew", padx=2)
        thresh_frame.columnconfigure(2, weight=1)
        self.threshold_val_label = ttk.Label(thresh_frame, text="15", width=3)
        self.threshold_val_label.grid(row=0, column=3)

        # Canny controls
        canny_frame = ttk.LabelFrame(controls, text="Canny Edge Detection", padding=5)
        canny_frame.grid(row=2, column=0, sticky="ew", pady=5)

        ttk.Label(canny_frame, text="Low:").grid(row=0, column=0)
        self.canny_low_var = tk.IntVar(value=50)
        ttk.Scale(canny_frame, from_=10, to=200, variable=self.canny_low_var,
                  orient=tk.HORIZONTAL, command=lambda _: self._update_params()).grid(row=0, column=1, sticky="ew", padx=5)
        canny_frame.columnconfigure(1, weight=1)

        ttk.Label(canny_frame, text="High:").grid(row=1, column=0)
        self.canny_high_var = tk.IntVar(value=150)
        ttk.Scale(canny_frame, from_=50, to=300, variable=self.canny_high_var,
                  orient=tk.HORIZONTAL, command=lambda _: self._update_params()).grid(row=1, column=1, sticky="ew", padx=5)

        # Preprocessing controls
        preproc_frame = ttk.LabelFrame(controls, text="Preprocessing", padding=5)
        preproc_frame.grid(row=3, column=0, sticky="ew", pady=5)

        ttk.Label(preproc_frame, text="Blur:").grid(row=0, column=0)
        self.blur_var = tk.IntVar(value=0)
        ttk.Scale(preproc_frame, from_=0, to=15, variable=self.blur_var,
                  orient=tk.HORIZONTAL, command=lambda _: self._update_params()).grid(row=0, column=1, sticky="ew", padx=5)
        preproc_frame.columnconfigure(1, weight=1)

        ttk.Label(preproc_frame, text="Morph:").grid(row=1, column=0)
        self.morph_var = tk.IntVar(value=0)
        ttk.Scale(preproc_frame, from_=0, to=10, variable=self.morph_var,
                  orient=tk.HORIZONTAL, command=lambda _: self._update_params()).grid(row=1, column=1, sticky="ew", padx=5)

        # Action buttons
        action_frame = ttk.Frame(controls)
        action_frame.grid(row=4, column=0, sticky="ew", pady=5)

        ttk.Button(action_frame, text="📸 Snapshot", command=self._save_snapshot).pack(side=tk.LEFT, padx=2)
        self.record_btn = ttk.Button(action_frame, text="⏺ Record", command=self._toggle_record, state=tk.DISABLED)
        self.record_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🔄 Reset", command=self._reset_ghost).pack(side=tk.LEFT, padx=2)

        # Display mode
        mode_frame = ttk.LabelFrame(controls, text="Display Mode", padding=5)
        mode_frame.grid(row=5, column=0, sticky="ew", pady=5)

        self.display_mode = tk.StringVar(value="side_by_side")
        ttk.Radiobutton(mode_frame, text="Side-by-Side", variable=self.display_mode,
                        value="side_by_side").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Ghost Only", variable=self.display_mode,
                        value="ghost_only").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="RGB Only", variable=self.display_mode,
                        value="rgb_only").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="4-Panel (RGB+Edges+Diff+Ghost)", variable=self.display_mode,
                        value="four_panel").pack(anchor=tk.W)

        # ── Plots section ─────────────────────────────────────────
        plots_frame = ttk.LabelFrame(right_frame, text="Metrics", padding=5)
        plots_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        right_frame.rowconfigure(1, weight=1)

        self.fig = Figure(figsize=(4, 5), dpi=80, facecolor='#f0f0f0')

        self.ax_fps = self.fig.add_subplot(311)
        self.ax_fps.set_title("FPS", fontsize=9)
        self.ax_fps.set_ylim(0, 60)
        self.ax_fps.grid(True, alpha=0.3)

        self.ax_privacy = self.fig.add_subplot(312)
        self.ax_privacy.set_title("Privacy Score", fontsize=9)
        self.ax_privacy.set_ylim(0, 1.1)
        self.ax_privacy.grid(True, alpha=0.3)

        self.ax_edges = self.fig.add_subplot(313)
        self.ax_edges.set_title("Edge Density", fontsize=9)
        self.ax_edges.set_ylim(0, 0.5)
        self.ax_edges.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=2.0)

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_params(self):
        """Update Ghost converter parameters from UI controls."""
        self.ghost_conv.use_adaptive = self.adaptive_var.get()
        self.ghost_conv.threshold = self.threshold_var.get()
        self.ghost_conv.canny_low = self.canny_low_var.get()
        self.ghost_conv.canny_high = self.canny_high_var.get()
        self.ghost_conv.blur_kernel = self.blur_var.get()
        self.ghost_conv.morph_kernel = self.morph_var.get()
        self.threshold_val_label.config(text=str(self.threshold_var.get()))

    def _start_camera(self):
        """Start camera capture."""
        try:
            cam_idx = int(self.camera_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid camera index")
            return

        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {cam_idx}")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        self.frame_count = 0
        self.ghost_conv.reset()

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.NORMAL)
        self.record_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Running")

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _stop_camera(self):
        """Stop camera capture."""
        self.running = False
        if self.recording:
            self._toggle_record()

        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.record_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        self.fps_label.config(text="FPS: --")
        self.privacy_label.config(text="Privacy: --")

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="▶ Resume" if self.paused else "⏸ Pause")
        self.status_label.config(text="Status: Paused" if self.paused else "Status: Running")

    def _toggle_record(self):
        if not self.recording:
            path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")],
                initialfile=f"ghost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            if path:
                self.video_writer = GhostVideoWriter(path, fps=30, frame_size=(640, 480))
                self.recording = True
                self.record_btn.config(text="⏹ Stop Rec")
                self.recording_label.config(text="🔴 REC")
        else:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            self.record_btn.config(text="⏺ Record")
            self.recording_label.config(text="")

    def _reset_ghost(self):
        self.ghost_conv.reset()

    def _save_snapshot(self):
        if not hasattr(self, 'last_rgb') or self.last_rgb is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")],
            initialfile=f"ghost_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        if path:
            # Save combined view
            combined = self._create_display_image(self.last_rgb, self.last_ghost,
                                                   self.last_edges, self.last_diff)
            cv2.imwrite(path, combined)
            # Save metadata
            meta_path = path.rsplit('.', 1)[0] + '_meta.json'
            with open(meta_path, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "frame": self.frame_count,
                    "threshold": self.ghost_conv.threshold,
                    "adaptive": self.ghost_conv.use_adaptive,
                    "canny_low": self.ghost_conv.canny_low,
                    "canny_high": self.ghost_conv.canny_high,
                    "blur": self.ghost_conv.blur_kernel,
                    "morph": self.ghost_conv.morph_kernel,
                }, f, indent=2)

    def _capture_loop(self):
        """Main capture and processing loop (runs in separate thread)."""
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self.frame_count += 1
            t_start = time.time()

            # Ghost conversion
            ghost_frame, edges, diff, edge_density = self.ghost_conv.convert_frame(frame)

            # Face detection (every 10 frames)
            face_rgb, face_ghost = 0, 0
            if self.frame_count % 10 == 0:
                face_rgb = self.face_detector.detect_count(frame)
                face_ghost = self.face_detector.detect_count(ghost_frame)

            # Record
            if self.recording and self.video_writer:
                self.video_writer.write(ghost_frame)

            # Metrics
            dt = time.time() - t_start
            fps = 1.0 / max(dt, 1e-6)
            self.fps_history.append(fps)

            if face_rgb > 0:
                privacy = 1.0 - (face_ghost / face_rgb)
            else:
                privacy = 1.0
            self.privacy_history.append(privacy)
            self.edge_density_history.append(edge_density)
            self.face_rgb_history.append(face_rgb)
            self.face_ghost_history.append(face_ghost)

            # Store for snapshot
            self.last_rgb = frame.copy()
            self.last_ghost = ghost_frame.copy()
            self.last_edges = edges.copy()
            self.last_diff = diff.copy()

            # Update UI (thread-safe via root.after)
            self.root.after(0, self._update_ui, frame, ghost_frame, edges, diff,
                            fps, privacy, edge_density, face_rgb, face_ghost)

            # Cap at ~30 FPS to avoid overwhelming the UI
            elapsed = time.time() - t_start
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)

    def _update_ui(self, rgb, ghost, edges, diff, fps, privacy, edge_density,
                   face_rgb, face_ghost):
        """Update the UI with new frame data."""
        if not self.running:
            return

        # Create display image
        display = self._create_display_image(rgb, ghost, edges, diff)

        # Convert to tkinter
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_rgb)

        # Resize to fit canvas
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            img = img.resize((canvas_w, canvas_h), Image.NEAREST)

        self.photo = ImageTk.PhotoImage(image=img)
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Update labels
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.privacy_label.config(text=f"Privacy: {privacy:.1%}")

        # Update plots
        self._update_plots()

    def _create_display_image(self, rgb, ghost, edges, diff):
        """Create the display image based on current mode."""
        h, w = rgb.shape[:2]
        mode = self.display_mode.get()

        if mode == "side_by_side":
            display = np.zeros((h, w * 2, 3), dtype=np.uint8)
            display[:, :w] = rgb
            display[:, w:] = ghost
            # Labels
            cv2.putText(display, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "GHOST", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif mode == "ghost_only":
            display = ghost.copy()
            cv2.putText(display, "GHOST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif mode == "rgb_only":
            display = rgb.copy()
            cv2.putText(display, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif mode == "four_panel":
            display = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
            display[:h, :w] = rgb
            display[:h, w:] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            display[h:, :w] = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            display[h:, w:] = ghost
            for label, pos in [("RGB", (10, 30)), ("EDGES", (w + 10, 30)),
                               ("DIFF", (10, h + 30)), ("GHOST", (w + 10, h + 30))]:
                cv2.putText(display, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return display

    def _update_plots(self):
        """Update matplotlib plots."""
        x = range(len(self.fps_history))

        self.ax_fps.clear()
        self.ax_fps.plot(list(self.fps_history), 'g-', linewidth=1)
        self.ax_fps.set_title("FPS", fontsize=9)
        self.ax_fps.set_ylim(0, max(60, max(self.fps_history, default=30) * 1.2))
        self.ax_fps.grid(True, alpha=0.3)

        self.ax_privacy.clear()
        self.ax_privacy.plot(list(self.privacy_history), 'b-', linewidth=1)
        self.ax_privacy.set_title("Privacy Score", fontsize=9)
        self.ax_privacy.set_ylim(0, 1.1)
        self.ax_privacy.grid(True, alpha=0.3)

        self.ax_edges.clear()
        self.ax_edges.plot(list(self.edge_density_history), 'r-', linewidth=1)
        self.ax_edges.set_title("Edge Density", fontsize=9)
        self.ax_edges.set_ylim(0, max(0.5, max(self.edge_density_history, default=0.1) * 1.2))
        self.ax_edges.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=1.5)
        self.canvas_plot.draw_idle()

    def _on_close(self):
        self._stop_camera()
        self.root.destroy()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    root = tk.Tk()
    app = GhostNetworkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

