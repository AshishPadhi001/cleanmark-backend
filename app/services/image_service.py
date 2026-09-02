import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np
from app.core.config import settings
from app.services.gemini_unblend import gemini_unblend_engine

logger = logging.getLogger("cleanmark.image_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ImageWatermarkService:
    """
    Lightweight Mathematical Watermark Removal Service v4.
    Uses Auto Gain Solver to prevent dark shades and ghost artifacts.
    """
    def __init__(self):
        self.engine = gemini_unblend_engine
        logger.info("Mathematical Watermark Engine initialized (Auto-Gain Solver active)")

    def is_model_ready(self) -> bool:
        return True

    def initialize_on_startup(self):
        logger.info("Mathematical Watermark Engine ready.")

    def load_model(self):
        pass

    def warmup(self):
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        self.engine.unblend_image(dummy)

    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask_np: Optional[np.ndarray] = None,
        preset: Optional[str] = None,
        custom_box: Optional[Dict[str, Any]] = None,
        gain: float = -1.0,       # -1.0 = AUTO GAIN SOLVER (recommended)
        offset_x: int = -24,
        offset_y: int = -24,
        size_scale: float = 1.0
    ) -> Tuple[np.ndarray, str, float, Dict[str, float]]:
        """
        Unblends watermark mathematically from RGB image.
        gain=-1.0 triggers automatic gain detection to prevent dark/ghost artifacts.
        """
        t0 = time.perf_counter()
        h, w = image_rgb.shape[:2]

        resolved_box = None

        if custom_box is not None and custom_box.get("w", custom_box.get("width", 0)) > 0:
            box_w = custom_box.get("w", custom_box.get("width", 0))
            box_h = custom_box.get("h", custom_box.get("height", 0))
            if box_w <= 1.0:  # Normalized
                bx = int(custom_box["x"] * w)
                by = int(custom_box["y"] * h)
                bw = int(box_w * w)
                bh = int(box_h * h)
            else:
                bx = int(custom_box["x"])
                by = int(custom_box["y"])
                bw = int(box_w)
                bh = int(box_h)
            resolved_box = {"x": bx, "y": by, "w": bw, "h": bh, "width": bw, "height": bh}

        elif mask_np is not None:
            pts = np.argwhere(mask_np > 128)
            if len(pts) > 0:
                y_min, x_min = pts.min(axis=0)
                y_max, x_max = pts.max(axis=0)
                bw = max(16, x_max - x_min)
                bh = max(16, y_max - y_min)
                resolved_box = {"x": int(x_min), "y": int(y_min), "w": int(bw), "h": int(bh), "width": int(bw), "height": int(bh)}

        elif preset:
            p = preset.lower().strip()
            min_dim = min(w, h)
            ratio = min_dim / 1536.0
            size = max(16, round(96 * ratio * size_scale))
            margin = max(8, round(64 * ratio))

            if p == "top-left":
                resolved_box = {"x": margin, "y": margin, "w": size, "h": size, "width": size, "height": size}
            elif p == "top-right":
                resolved_box = {"x": w - margin - size, "y": margin, "w": size, "h": size, "width": size, "height": size}
            elif p == "bottom-left":
                resolved_box = {"x": margin, "y": h - margin - size, "w": size, "h": size, "width": size, "height": size}
            elif p == "bottom-right":
                resolved_box = {"x": w - margin - size, "y": h - margin - size, "w": size, "h": size, "width": size, "height": size}

        cleaned_rgb, info = self.engine.unblend_image(
            image_rgb,
            mode="gemini",
            gain=gain,
            offset_x=offset_x,
            offset_y=offset_y,
            size_scale=size_scale,
            custom_box=resolved_box
        )

        elapsed = time.perf_counter() - t0
        timings = {"inference_ms": round(elapsed * 1000, 2), "total_ms": round(elapsed * 1000, 2)}
        logger.info(f"[MATH UNBLEND] {w}x{h} | {elapsed*1000:.2f}ms | items={len(info.get('items', []))}")

        return cleaned_rgb, "Gemini Mathematical Unblend (Auto-Gain, Zero Blur)", elapsed, timings


image_service = ImageWatermarkService()
ImageInpaintingService = ImageWatermarkService
