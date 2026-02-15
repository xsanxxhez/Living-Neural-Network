#!/usr/bin/env python3
"""
Living Neural Network - Phase 4 (INTERACTIVE + REAL-TIME) - FIXED

NEW FEATURES:
- Real-time prediction while drawing (updates every 300ms)
- Drag nodes to rearrange visualization
- Hover effects with gradient glow
- Smooth animations between predictions
- Particle flow system (optional - press P to toggle)

Requires: quickdraw_model.pth (from train_quickdraw.py)
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time
from collections import deque

# =====================
# Config & Constants
# =====================

CANVAS_SIZE = 280
PREVIEW_SIZE = 140
DRAW_COLOR = "white"
BG_COLOR = "#050710"  # Deep space
PANEL_BG = "#090b18"
BRUSH_SIZE = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Neon color palette (Jarvis-style)
NEON_CYAN = "#00f6ff"
NEON_BLUE = "#2d6bff"
NEON_PURPLE = "#c455ff"
NEON_GREEN = "#00ffa3"
NEON_ORANGE = "#ff9b3c"
NEON_PINK = "#ff3ba3"

# =====================
# Model Definition
# =====================

class QuickDrawCNN(nn.Module):
    """CNN architecture with activation hooks for visualization"""
    def __init__(self, num_classes):
        super(QuickDrawCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

        # Storage for activations
        self.activations = {}
        self._register_hooks()

    def _hook(self, name):
        def fn(module, input, output):
            self.activations[name] = output.detach().cpu()
        return fn

    def _register_hooks(self):
        self.conv1.register_forward_hook(self._hook('conv1'))
        self.conv2.register_forward_hook(self._hook('conv2'))
        self.conv3.register_forward_hook(self._hook('conv3'))
        self.fc1.register_forward_hook(self._hook('fc1'))

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_model(checkpoint_path='quickdraw_model.pth'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file '{checkpoint_path}' not found. Run train_quickdraw.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    classes = checkpoint.get('classes')
    num_classes = len(classes)
    model = QuickDrawCNN(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, classes

# =====================
# Particle System
# =====================

class Particle:
    def __init__(self, x1, y1, x2, y2, color, speed=3.0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color
        self.progress = 0.0
        self.speed = speed
        self.alive = True

    def update(self):
        self.progress += self.speed * 0.016  # ~60fps
        if self.progress >= 1.0:
            self.alive = False

    def get_pos(self):
        t = self.progress
        x = self.x1 + (self.x2 - self.x1) * t
        y = self.y1 + (self.y2 - self.y1) * t
        return x, y

# =====================
# Main Application
# =====================

class SketchRecognizer:
    def __init__(self, model, classes):
        self.model = model
        self.classes = classes

        self.root = tk.Tk()
        self.root.title("Living Neural Network - Phase 4 (Interactive)")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1400x780")

        # PIL image for drawing
        self.drawing_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.drawing_image)

        self.last_x = None
        self.last_y = None
        self.predictions = [(cat, 0.0) for cat in self.classes]
        self.processed_array = None
        self.layer_node_values = {}

        # Animation state
        self.prev_layer_values = {}
        self.anim_progress = 1.0

        # Real-time inference
        self.last_inference_time = 0
        self.inference_cooldown = 0.3  # 300ms between predictions
        self.canvas_changed = False

        # Interaction state
        self.dragging_node = None
        self.hovered_node = None

        # Particles
        self.particles = []
        self.particles_enabled = True

        # Layout computed flag
        self.layout_initialized = False

        self._setup_ui()

        # Start real-time loop
        self.root.after(50, self._realtime_loop)

    # ---------- UI Setup ----------
    def _setup_ui(self):
        main_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Top title bar
        title_frame = tk.Frame(main_frame, bg=BG_COLOR)
        title_frame.pack(fill=tk.X, pady=(0, 15))

        title = tk.Label(
            title_frame,
            text="LIVING NEURAL NETWORK",
            font=("SF Pro Display", 24, "bold"),
            bg=BG_COLOR,
            fg=NEON_CYAN
        )
        title.pack(side=tk.LEFT)

        subtitle = tk.Label(
            title_frame,
            text="Phase 4 · Real-time Interactive",
            font=("SF Pro Display", 12),
            bg=BG_COLOR,
            fg="#6f7a9f"
        )
        subtitle.pack(side=tk.LEFT, padx=15)

        # Status indicator
        self.status_label = tk.Label(
            title_frame,
            text="● LIVE",
            font=("SF Pro Display", 11, "bold"),
            bg=BG_COLOR,
            fg=NEON_GREEN
        )
        self.status_label.pack(side=tk.RIGHT)

        # Horizontal divider
        divider = tk.Frame(main_frame, bg="#15182a", height=2)
        divider.pack(fill=tk.X, pady=(0, 15))

        # Content: left (drawing), middle (preview + predictions), right (NN viz)
        content = tk.Frame(main_frame, bg=BG_COLOR)
        content.pack(fill=tk.BOTH, expand=True)

        # Left: Drawing
        self._setup_left_panel(content)
        # Middle: Preview + predictions
        self._setup_middle_panel(content)
        # Right: Neural network visualization
        self._setup_viz_panel(content)

        # Keyboard
        self.root.bind("<Key>", self._handle_key)

    def _setup_left_panel(self, parent):
        left = tk.Frame(parent, bg=BG_COLOR)
        left.pack(side=tk.LEFT, padx=(0, 25))

        label = tk.Label(
            left,
            text="INPUT CANVAS",
            font=("SF Pro Display", 12, "bold"),
            bg=BG_COLOR,
            fg="#6f7a9f"
        )
        label.pack(anchor=tk.W)

        # Canvas frame with border
        canvas_frame = tk.Frame(left, bg=PANEL_BG, bd=0, highlightthickness=1, highlightbackground="#20263f")
        canvas_frame.pack(pady=(5, 10))

        self.canvas_widget = tk.Canvas(
            canvas_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="#000000",
            highlightthickness=0
        )
        self.canvas_widget.pack(padx=10, pady=10)

        # Bind drawing
        self.canvas_widget.bind("<Button-1>", self._start_drawing)
        self.canvas_widget.bind("<B1-Motion>", self._draw_motion)
        self.canvas_widget.bind("<ButtonRelease-1>", self._stop_drawing)

        # Controls
        controls = tk.Frame(left, bg=BG_COLOR)
        controls.pack(pady=5, anchor=tk.W)

        btn_clear = tk.Button(
            controls,
            text="CLEAR  (C)",
            command=self._clear_canvas,
            font=("SF Pro Display", 11),
            bg="#1f2438",
            fg="white",
            activebackground="#2c3550",
            activeforeground="white",
            bd=0,
            padx=14,
            pady=6
        )
        btn_clear.pack(side=tk.LEFT, padx=(0, 8))

        info = tk.Label(
            left,
            text="Draw to see real-time predictions\nDrag nodes in visualizer · Press P for particles",
            font=("SF Pro Display", 9),
            bg=BG_COLOR,
            fg="#5a6488",
            justify=tk.LEFT
        )
        info.pack(anchor=tk.W, pady=(8, 0))

    def _setup_middle_panel(self, parent):
        middle = tk.Frame(parent, bg=BG_COLOR)
        middle.pack(side=tk.LEFT, padx=(0, 25))

        # Preview
        preview_label = tk.Label(
            middle,
            text="MODEL INPUT (28×28)",
            font=("SF Pro Display", 12, "bold"),
            bg=BG_COLOR,
            fg="#6f7a9f"
        )
        preview_label.pack(anchor=tk.W)

        preview_frame = tk.Frame(middle, bg=PANEL_BG, highlightthickness=1, highlightbackground="#20263f")
        preview_frame.pack(pady=(5, 15))

        self.preview_widget = tk.Canvas(
            preview_frame,
            width=PREVIEW_SIZE,
            height=PREVIEW_SIZE,
            bg="#000000",
            highlightthickness=0
        )
        self.preview_widget.pack(padx=10, pady=10)

        # Predictions
        pred_label = tk.Label(
            middle,
            text="TOP PREDICTIONS",
            font=("SF Pro Display", 12, "bold"),
            bg=BG_COLOR,
            fg="#6f7a9f"
        )
        pred_label.pack(anchor=tk.W)

        self.prediction_bars = []
        colors = [NEON_GREEN, "#f6ff66", NEON_ORANGE, NEON_BLUE, NEON_PURPLE]

        for i in range(5):
            frame = tk.Frame(middle, bg=PANEL_BG)
            frame.pack(fill=tk.X, pady=5)

            name_label = tk.Label(
                frame,
                text=f"#{i+1}: ...",
                font=("SF Pro Display", 13, "bold"),
                bg=PANEL_BG,
                fg=colors[i]
            )
            name_label.pack(anchor=tk.W, padx=10, pady=(4, 0))

            bar_bg = tk.Frame(frame, bg="#171b2b", height=18)
            bar_bg.pack(fill=tk.X, padx=10, pady=(4, 4))
            bar_bg.pack_propagate(False)

            bar = tk.Frame(bar_bg, bg=colors[i], width=0)
            bar.pack(side=tk.LEFT, fill=tk.Y)

            pct_label = tk.Label(
                frame,
                text="0.0%",
                font=("SF Pro Display", 10),
                bg=PANEL_BG,
                fg="#e0e4ff"
            )
            pct_label.pack(anchor=tk.W, padx=10, pady=(0, 4))

            self.prediction_bars.append({
                'name': name_label,
                'bar': bar,
                'pct': pct_label,
                'bg': bar_bg,
            })

        # Model info
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        model_info = tk.Label(
            middle,
            text=f"CNN · 3×Conv → 256 FC → {len(self.classes)} classes\nDevice: {device_str} · Real-time mode",
            font=("SF Pro Display", 9),
            bg=BG_COLOR,
            fg="#545c7a",
            justify=tk.LEFT
        )
        model_info.pack(anchor=tk.W, pady=(10, 0))

    def _setup_viz_panel(self, parent):
        right = tk.Frame(parent, bg=BG_COLOR)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        header = tk.Label(
            right,
            text="INTERACTIVE NETWORK VISUALIZER",
            font=("SF Pro Display", 12, "bold"),
            bg=BG_COLOR,
            fg="#6f7a9f"
        )
        header.pack(anchor=tk.W)

        viz_frame = tk.Frame(right, bg=PANEL_BG, highlightthickness=1, highlightbackground="#20263f")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        self.viz_canvas = tk.Canvas(
            viz_frame,
            bg="#050712",
            highlightthickness=0
        )
        self.viz_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Bind mouse events for interaction
        self.viz_canvas.bind("<Button-1>", self._viz_click)
        self.viz_canvas.bind("<B1-Motion>", self._viz_drag)
        self.viz_canvas.bind("<ButtonRelease-1>", self._viz_release)
        self.viz_canvas.bind("<Motion>", self._viz_hover)

        # Initialize empty positions dict
        self.viz_positions = {}
        self.viz_layers = ['conv1', 'conv2', 'conv3', 'fc1', 'out']
        self.viz_layer_nodes = {
            'conv1': 8,
            'conv2': 10,
            'conv3': 12,
            'fc1': 16,
            'out': len(self.classes),
        }

    # ---------- Drawing Logic ----------
    def _start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self.canvas_changed = True

    def _draw_motion(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas_widget.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill=DRAW_COLOR,
                width=BRUSH_SIZE,
                capstyle=tk.ROUND,
                smooth=True
            )
            self.draw.line(
                [(self.last_x, self.last_y), (event.x, event.y)],
                fill=255,
                width=BRUSH_SIZE
            )
        self.last_x = event.x
        self.last_y = event.y
        self.canvas_changed = True

    def _stop_drawing(self, event):
        self.last_x = None
        self.last_y = None

    def _clear_canvas(self):
        self.canvas_widget.delete("all")
        self.drawing_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.drawing_image)
        self.processed_array = None
        self.predictions = [(cat, 0.0) for cat in self.classes]
        self._update_prediction_display()
        self.layer_node_values = {}
        self.prev_layer_values = {}
        self._draw_viz_with_activations()
        self.preview_widget.delete("all")
        self.canvas_changed = False

    # ---------- Real-time Loop ----------
    def _realtime_loop(self):
        # Initialize layout on first frame (after canvas has size)
        if not self.layout_initialized:
            self._compute_viz_layout()
            self._draw_viz_base()
            self.layout_initialized = True

        current_time = time.time()

        # Check if we should run inference
        if self.canvas_changed and (current_time - self.last_inference_time) > self.inference_cooldown:
            self._trigger_prediction()
            self.last_inference_time = current_time
            self.canvas_changed = False

        # Update animation
        if self.anim_progress < 1.0:
            self.anim_progress = min(1.0, self.anim_progress + 0.08)
            self._draw_viz_with_activations()

        # Update particles
        if self.particles_enabled and self.particles:
            for p in self.particles:
                p.update()
            self.particles = [p for p in self.particles if p.alive]
            if self.particles:  # Only redraw if there are particles
                self._draw_viz_with_activations()

        # Schedule next frame
        self.root.after(50, self._realtime_loop)

    # ---------- Inference & Preprocessing ----------
    def _preprocess_canvas(self):
        img_28 = self.drawing_image.resize((28, 28), Image.Resampling.LANCZOS)
        arr = np.array(img_28).astype(np.float32) / 255.0
        return arr

    def _run_inference(self, img_array):
        tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        preds = list(zip(self.classes, probs * 100.0))
        preds.sort(key=lambda x: x[1], reverse=True)

        # Store previous for animation
        self.prev_layer_values = self.layer_node_values.copy()

        # Build layer_node_values from model.activations
        layer_values = {}
        acts = self.model.activations

        def conv_summary(key, max_channels):
            if key not in acts:
                return None
            a = acts[key]
            a = a.mean(dim=(0, 2, 3))
            a = a.numpy()
            if len(a) > max_channels:
                a = a[:max_channels]
            if a.max() > 0:
                a = a / a.max()
            return a

        layer_values['conv1'] = conv_summary('conv1', 8)
        layer_values['conv2'] = conv_summary('conv2', 10)
        layer_values['conv3'] = conv_summary('conv3', 12)

        if 'fc1' in acts:
            a = acts['fc1'][0]
            a = a.numpy()
            K = 16
            a = a[:K]
            if a.max() > 0:
                a = a / a.max()
            layer_values['fc1'] = a

        out_vals = probs.copy()
        if out_vals.max() > 0:
            out_vals = out_vals / out_vals.max()
        layer_values['out'] = out_vals

        self.layer_node_values = layer_values
        self.anim_progress = 0.0  # Start animation

        # Spawn particles
        if self.particles_enabled:
            self._spawn_particles()

        return preds

    def _spawn_particles(self):
        """Spawn particles along strong connections"""
        if not self.layer_node_values:
            return

        # Spawn on strongest paths
        for li in range(len(self.viz_layers) - 1):
            layer_a = self.viz_layers[li]
            layer_b = self.viz_layers[li + 1]
            vals_a = self.layer_node_values.get(layer_a)
            vals_b = self.layer_node_values.get(layer_b)

            if vals_a is None or vals_b is None:
                continue

            pos_a = self.viz_positions.get(layer_a, [])
            pos_b = self.viz_positions.get(layer_b, [])

            # Spawn on top 2 connections per layer
            connections = []
            for i, (x1, y1) in enumerate(pos_a):
                if i >= len(vals_a):
                    continue
                va = vals_a[i]
                for j, (x2, y2) in enumerate(pos_b):
                    if j >= len(vals_b):
                        continue
                    vb = vals_b[j]
                    strength = 0.7 * va + 0.3 * vb
                    if strength > 0.4:
                        connections.append((strength, x1, y1, x2, y2, self._value_to_color(strength)))

            connections.sort(reverse=True)
            for strength, x1, y1, x2, y2, color in connections[:2]:
                self.particles.append(Particle(x1, y1, x2, y2, color, speed=2.0))

    # ---------- UI Updates ----------
    def _update_preview(self, arr):
        img_scaled = Image.fromarray((arr * 255).astype(np.uint8))
        img_scaled = img_scaled.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.NEAREST)
        self.tk_preview = ImageTk.PhotoImage(img_scaled)
        self.preview_widget.delete("all")
        self.preview_widget.create_image(
            PREVIEW_SIZE // 2,
            PREVIEW_SIZE // 2,
            image=self.tk_preview
        )

    def _trigger_prediction(self):
        self.processed_array = self._preprocess_canvas()
        self.predictions = self._run_inference(self.processed_array)
        self._update_preview(self.processed_array)
        self._update_prediction_display()

    def _update_prediction_display(self):
        for i, bar in enumerate(self.prediction_bars):
            if i < len(self.predictions):
                category, confidence = self.predictions[i]
                bar['name'].config(text=f"#{i+1}: {category}")
                width = int(260 * (confidence / 100.0))
                bar['bar'].config(width=width)
                bar['pct'].config(text=f"{confidence:.1f}%")
            else:
                bar['name'].config(text=f"#{i+1}: ...")
                bar['bar'].config(width=0)
                bar['pct'].config(text="0.0%")

    def _handle_key(self, event):
        if event.char.lower() == 'c':
            self._clear_canvas()
        elif event.char.lower() == 'p':
            self.particles_enabled = not self.particles_enabled
            status = "ON" if self.particles_enabled else "OFF"
            print(f"Particles: {status}")
        elif event.keysym == 'Escape':
            self.root.quit()

    # ---------- Visualization Interaction ----------
    def _viz_click(self, event):
        # Check if clicking on a node
        for layer, positions in self.viz_positions.items():
            for idx, (x, y) in enumerate(positions):
                dist = ((event.x - x)**2 + (event.y - y)**2)**0.5
                if dist < 12:
                    self.dragging_node = (layer, idx, x, y)
                    return

    def _viz_drag(self, event):
        if self.dragging_node:
            layer, idx, _, _ = self.dragging_node
            # Update position
            self.viz_positions[layer][idx] = (event.x, event.y)
            self._draw_viz_with_activations()

    def _viz_release(self, event):
        self.dragging_node = None

    def _viz_hover(self, event):
        # Check if hovering over a node
        hovered = None
        for layer, positions in self.viz_positions.items():
            for idx, (x, y) in enumerate(positions):
                dist = ((event.x - x)**2 + (event.y - y)**2)**0.5
                if dist < 12:
                    hovered = (layer, idx)
                    break
            if hovered:
                break

        if hovered != self.hovered_node:
            self.hovered_node = hovered
            self._draw_viz_with_activations()

    # ---------- Visualization Layout ----------
    def _compute_viz_layout(self):
        """FIXED: Compute layout after canvas has actual size"""
        # Force canvas to update and get real dimensions
        self.viz_canvas.update_idletasks()
        w = self.viz_canvas.winfo_width()
        h = self.viz_canvas.winfo_height()

        # Fallback if still not sized
        if w < 100:
            w = 600
        if h < 100:
            h = 600

        n_layers = len(self.viz_layers)
        margin_x = 80
        usable_w = w - 2 * margin_x
        col_spacing = usable_w / max(1, n_layers - 1) if n_layers > 1 else 0

        self.viz_positions.clear()

        for idx, layer in enumerate(self.viz_layers):
            x = margin_x + idx * col_spacing
            num_nodes = self.viz_layer_nodes[layer]

            margin_y = 60
            usable_h = h - 2 * margin_y
            if num_nodes > 1:
                node_spacing = usable_h / (num_nodes - 1)
            else:
                node_spacing = 0

            positions = []
            for i in range(num_nodes):
                y = margin_y + i * node_spacing if num_nodes > 1 else h / 2
                positions.append((x, y))
            self.viz_positions[layer] = positions

    def _draw_viz_base(self):
        self.viz_canvas.delete("all")
        w = self.viz_canvas.winfo_width()
        h = self.viz_canvas.winfo_height()

        # Background grid
        for i in range(0, w, 40):
            self.viz_canvas.create_line(i, 0, i, h, fill="#0a0d20", width=1)
        for j in range(0, h, 40):
            self.viz_canvas.create_line(0, j, w, j, fill="#0a0d20", width=1)

        # Draw layer labels
        for layer, positions in self.viz_positions.items():
            if positions:
                x = positions[0][0]
                label = layer.upper()
                if layer == 'out':
                    label = 'OUTPUT'
                self.viz_canvas.create_text(
                    x, 25,
                    text=label,
                    fill="#6f7a9f",
                    font=("SF Pro Display", 10)
                )

        # Draw faint connections
        for li in range(len(self.viz_layers) - 1):
            layer_a = self.viz_layers[li]
            layer_b = self.viz_layers[li + 1]
            for (x1, y1) in self.viz_positions.get(layer_a, []):
                for (x2, y2) in self.viz_positions.get(layer_b, []):
                    self.viz_canvas.create_line(
                        x1, y1, x2, y2,
                        fill="#10152b",
                        width=1
                    )

        # Draw nodes
        for layer, positions in self.viz_positions.items():
            for (x, y) in positions:
                self._draw_node(x, y, 6, intensity=0.05)

    # ---------- Visualization Drawing ----------
    def _value_to_color(self, v):
        """Map activation value 0..1 to neon gradient"""
        v = max(0.0, min(1.0, float(v)))
        if v < 0.5:
            t = v / 0.5
            r = int(0 * (1 - t) + 0 * t)
            g = int(80 * (1 - t) + 246 * t)
            b = int(255 * (1 - t) + 255 * t)
        else:
            t = (v - 0.5) / 0.5
            r = int(0 * (1 - t) + 196 * t)
            g = int(246 * (1 - t) + 85 * t)
            b = int(255 * (1 - t) + 255 * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _draw_node(self, x, y, r, intensity=0.5, is_hovered=False):
        """Draw a glowing node with optional hover effect"""
        color = self._value_to_color(intensity)

        if is_hovered:
            # Rainbow gradient glow when hovered
            self.viz_canvas.create_oval(
                x - r - 8, y - r - 8, x + r + 8, y + r + 8,
                outline=NEON_PINK,
                width=2
            )
            self.viz_canvas.create_oval(
                x - r - 5, y - r - 5, x + r + 5, y + r + 5,
                outline=NEON_CYAN,
                width=1
            )

        # Outer glow
        self.viz_canvas.create_oval(
            x - r - 3, y - r - 3, x + r + 3, y + r + 3,
            outline=color,
            width=1
        )
        # Inner circle
        self.viz_canvas.create_oval(
            x - r, y - r, x + r, y + r,
            outline=color,
            fill=color
        )

    def _draw_viz_with_activations(self):
        self._draw_viz_base()

        if not self.layer_node_values:
            return

        # Interpolate between old and new values
        current_values = {}
        for layer in self.viz_layers:
            old = self.prev_layer_values.get(layer)
            new = self.layer_node_values.get(layer)
            if new is not None:
                if old is not None and len(old) == len(new):
                    current_values[layer] = old * (1 - self.anim_progress) + new * self.anim_progress
                else:
                    current_values[layer] = new

        # Draw connections
        for li in range(len(self.viz_layers) - 1):
            layer_a = self.viz_layers[li]
            layer_b = self.viz_layers[li + 1]
            vals_a = current_values.get(layer_a)
            vals_b = current_values.get(layer_b)

            pos_a = self.viz_positions.get(layer_a, [])
            pos_b = self.viz_positions.get(layer_b, [])

            for i, (x1, y1) in enumerate(pos_a):
                va = vals_a[i] if vals_a is not None and i < len(vals_a) else 0.0
                for j, (x2, y2) in enumerate(pos_b):
                    vb = vals_b[j] if vals_b is not None and j < len(vals_b) else 0.0
                    edge_int = 0.7 * va + 0.3 * vb
                    if edge_int < 0.02:
                        continue
                    color = self._value_to_color(edge_int * 0.85)
                    width = 1 + 2.5 * edge_int
                    self.viz_canvas.create_line(
                        x1, y1, x2, y2,
                        fill=color,
                        width=width
                    )

        # Draw particles
        for p in self.particles:
            px, py = p.get_pos()
            alpha = 1.0 - p.progress
            r = int(3 + 2 * alpha)
            self.viz_canvas.create_oval(
                px - r, py - r, px + r, py + r,
                fill=p.color,
                outline=p.color
            )

        # Draw nodes
        for layer, positions in self.viz_positions.items():
            vals = current_values.get(layer)
            for idx, (x, y) in enumerate(positions):
                v = 0.05
                if vals is not None and idx < len(vals):
                    v = max(0.05, float(vals[idx]))

                is_hovered = self.hovered_node == (layer, idx)
                self._draw_node(x, y, 6, intensity=v, is_hovered=is_hovered)

                # Label output nodes
                if layer == 'out' and idx < len(self.classes):
                    label = self.classes[idx]
                    conf = vals[idx] if vals is not None and idx < len(vals) else 0.0
                    if conf > 0.15:
                        self.viz_canvas.create_text(
                            x + 20, y,
                            text=label,
                            fill="#8896b8",
                            anchor="w",
                            font=("SF Pro Display", 9)
                        )

    # ---------- Main Loop ----------
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    try:
        model, classes = load_model('quickdraw_model.pth')
        print(f"\nLoaded model with {len(classes)} classes: {classes}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("\n🎮 INTERACTIVE MODE")
        print("   - Draw to see real-time predictions")
        print("   - Drag nodes to rearrange")
        print("   - Hover over nodes for effects")
        print("   - Press P to toggle particles")
        print("   - Press C to clear\n")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    app = SketchRecognizer(model, classes)
    app.run()
