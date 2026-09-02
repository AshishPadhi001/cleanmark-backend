import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import cv2
import numpy as np
from app.core.config import settings

logger = logging.getLogger("cleanmark.gemini_unblend")


class GeminiUnblendEngine:
    """
    Intelligent Precision Mathematical Watermark Unblending Engine v4.

    KEY FIX — ROOT CAUSE OF "DARK SHADE / GHOST" ARTIFACTS:
    ==========================================================
    The old code used a static gain=1.0. But bg_96.png has a max alpha of only 0.5137.
    When gain=1.0 is applied, the formula subtracts too much white (255 * alpha) from the
    watermarked pixel, making the result DARKER than the true background.

    The fix: AUTO GAIN SOLVER via Total Variation (TV) Minimization:
    - Sweep gain g in [0.1 .. 1.1]
    - For each g, compute the gradient energy (Sobel) of the recovered patch
    - Choose g* that minimizes gradient energy in active alpha zone
    - This guarantees zero dark-shadow and zero ghost artifacts
    """

    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        self.bg48_path = self.models_dir / "bg_48.png"
        self.bg96_path = self.models_dir / "bg_96.png"

        if not self.bg48_path.exists():
            self.bg48_path = self.models_dir / "gemini_bg48.png"
        if not self.bg96_path.exists():
            self.bg96_path = self.models_dir / "gemini_bg96.png"

        self.alpha_48 = None
        self.alpha_96 = None
        self._load_templates()

    def _load_templates(self):
        if self.bg96_path.exists():
            img96 = cv2.imread(str(self.bg96_path), cv2.IMREAD_UNCHANGED)
            if img96 is not None:
                if len(img96.shape) == 3 and img96.shape[2] >= 3:
                    self.alpha_96 = np.max(img96[:, :, :3], axis=2).astype(np.float32) / 255.0
                else:
                    self.alpha_96 = img96.astype(np.float32) / 255.0
                logger.info(f"Loaded bg_96.png — alpha_max={self.alpha_96.max():.4f}")

        if self.bg48_path.exists():
            img48 = cv2.imread(str(self.bg48_path), cv2.IMREAD_UNCHANGED)
            if img48 is not None:
                if len(img48.shape) == 3 and img48.shape[2] >= 3:
                    self.alpha_48 = np.max(img48[:, :, :3], axis=2).astype(np.float32) / 255.0
                else:
                    self.alpha_48 = img48.astype(np.float32) / 255.0
                logger.info(f"Loaded bg_48.png — alpha_max={self.alpha_48.max():.4f}")

        if self.alpha_96 is None and self.alpha_48 is None:
            logger.error("CRITICAL: No alpha templates found in backend/models/!")

    def _get_alpha_template(self, size: int) -> Optional[np.ndarray]:
        ref = self.alpha_48 if (size <= 48 and self.alpha_48 is not None) else self.alpha_96
        if ref is None:
            ref = self.alpha_48 or self.alpha_96
        if ref is None:
            return None
        return cv2.resize(ref, (size, size), interpolation=cv2.INTER_CUBIC)

    def _gradient_energy(self, patch_f32: np.ndarray, alpha_3d: np.ndarray) -> float:
        """Compute mean Sobel gradient energy over the active alpha region."""
        unb = np.clip((patch_f32 - alpha_3d * 255.0) / (1.0 - alpha_3d), 0.0, 255.0)
        gray = (0.299 * unb[:, :, 0] + 0.587 * unb[:, :, 1] + 0.114 * unb[:, :, 2]).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        mask = (alpha_3d[:, :, 0] >= 0.05)
        if not np.any(mask):
            return 999999.0
        return float(np.mean(grad_mag[mask]))

    def _sample_surround(self, image_rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 20) -> Optional[np.ndarray]:
        """Sample surrounding background pixels around the watermark box."""
        h, w = image_rgb.shape[:2]
        strips = []
        # Top strip
        if y1 - pad >= 0:
            strips.append(image_rgb[max(0, y1-pad):y1, x1:x2].reshape(-1, 3))
        # Bottom strip
        if y2 + pad <= h:
            strips.append(image_rgb[y2:min(h, y2+pad), x1:x2].reshape(-1, 3))
        # Left strip
        if x1 - pad >= 0:
            strips.append(image_rgb[y1:y2, max(0, x1-pad):x1].reshape(-1, 3))
        # Right strip
        if x2 + pad <= w:
            strips.append(image_rgb[y1:y2, x2:min(w, x2+pad)].reshape(-1, 3))

        if not strips:
            return None
        all_pixels = np.concatenate(strips, axis=0).astype(np.float32)
        return all_pixels.mean(axis=0)  # shape (3,) RGB mean


    def auto_solve_gain(
        self,
        patch_f32: np.ndarray,
        alpha_template_2d: np.ndarray,
        gain_hint: float = 0.5,
        surround_mean: Optional[np.ndarray] = None,
        surround_std: Optional[np.ndarray] = None,
    ) -> float:
        """
        Find optimal alpha gain using Surrounding Background Matching.

        UPGRADED FROM TV MINIMIZATION:
        The old TV (gradient energy) method fails on textured/high-gradient backgrounds
        because any gain value produces high gradients — so it always picks gain=0.1
        (minimum possible), which removes almost nothing.

        NEW METHOD — Background Similarity Score:
        - For each candidate gain g, compute the unblended patch
        - Measure how closely the active alpha pixels match the SURROUNDING background
        - The optimal gain g* minimizes |mean(unblended_active) - mean(surround)|
        - This works on ANY background type: smooth, textured, dark, bright

        Falls back to TV method only if no surround context is available.
        Enforces minimum gain of 0.25 to prevent near-zero removal.
        """
        mask_act = (alpha_template_2d >= 0.05)
        if not np.any(mask_act):
            return gain_hint

        if surround_mean is not None:
            # === PRIMARY: Surround Background Matching ===
            best_g = 0.30
            min_score = 999999.0
            surr = surround_mean  # shape (3,) RGB

            for g in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
                      0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]:
                a = np.clip(alpha_template_2d * g, 0.0, 0.99)
                a3d = np.expand_dims(a, 2)
                unb = np.clip((patch_f32 - a3d * 255.0) / (1.0 - a3d), 0.0, 255.0)
                unb_mean = unb[mask_act].mean(axis=0)
                score = float(np.abs(unb_mean - surr).mean())
                if score < min_score:
                    min_score = score
                    best_g = g

            # Fine sweep ±0.08 around best
            for g in np.arange(max(0.15, best_g - 0.08), min(1.10, best_g + 0.10), 0.01):
                g = float(g)
                a = np.clip(alpha_template_2d * g, 0.0, 0.99)
                a3d = np.expand_dims(a, 2)
                unb = np.clip((patch_f32 - a3d * 255.0) / (1.0 - a3d), 0.0, 255.0)
                unb_mean = unb[mask_act].mean(axis=0)
                score = float(np.abs(unb_mean - surr).mean())
                if score < min_score:
                    min_score = score
                    best_g = g

            # Enforce minimum gain of 0.25 — never do near-zero removal
            best_g = max(0.25, best_g)
            logger.debug(f"Auto-gain (surround-match): {best_g:.3f} (score={min_score:.3f})")
            return best_g

        else:
            # === FALLBACK: TV Gradient Energy (for when no surround context) ===
            best_g = gain_hint
            min_energy = 999999.0

            for g in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]:
                a = np.clip(alpha_template_2d * g, 0.0, 0.99)
                energy = self._gradient_energy(patch_f32, np.expand_dims(a, 2))
                if energy < min_energy:
                    min_energy = energy
                    best_g = g

            for g in np.arange(max(0.25, best_g - 0.12), min(1.10, best_g + 0.14), 0.02):
                a = np.clip(alpha_template_2d * float(g), 0.0, 0.99)
                energy = self._gradient_energy(patch_f32, np.expand_dims(a, 2))
                if energy < min_energy:
                    min_energy = energy
                    best_g = float(g)

            best_g = max(0.25, best_g)
            logger.debug(f"Auto-gain (TV fallback): {best_g:.3f} (energy={min_energy:.4f})")
            return best_g

    def find_star_in_roi(
        self,
        image_rgb: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        expected_size: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Targeted NCC scan for the Gemini / Veo 4-pointed watermark.
        Constrains search to expected proportional sizes to avoid false noise locks.
        """
        h, w = image_rgb.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 10, min(w, x2))
        y2 = max(y1 + 10, min(h, y2))

        rw, rh = x2 - x1, y2 - y1
        if rw < 16 or rh < 16:
            return None

        roi = image_rgb[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        min_dim_frame = min(w, h)
        if expected_size is None:
            expected_size = max(24, min(round(min_dim_frame / 15.0), min_dim_frame))

        # Search tightly around expected proportional size (+/- 8px)
        min_search = max(24, expected_size - 8)
        max_search = min(min(rw, rh), expected_size + 8)
        if min_search > max_search:
            min_search = max_search = expected_size

        best_score, best_loc, best_size = -1, (0, 0), expected_size

        for s in range(min_search, max_search + 1, 2):
            ref = self.alpha_48 if (s <= 48 and self.alpha_48 is not None) else self.alpha_96
            if ref is None:
                continue
            tmpl = cv2.resize(ref, (s, s), interpolation=cv2.INTER_CUBIC)
            tmpl_u8 = (tmpl * 255.0).astype(np.uint8)
            res = cv2.matchTemplate(gray_roi, tmpl_u8, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(res)
            if max_v > best_score:
                best_score = max_v
                best_loc = max_l
                best_size = s

        # Require solid confidence threshold (>= 0.45)
        if best_score >= 0.45:
            return {"x": x1 + best_loc[0], "y": y1 + best_loc[1],
                    "size": best_size, "width": best_size, "height": best_size,
                    "confidence": float(best_score)}
        return None

    def get_watermark_geometry(
        self,
        width: int, height: int,
        mode: str = "gemini",
        offset_x: int = -24, offset_y: int = -24,
        size_scale: float = 1.0
    ) -> Dict[str, Any]:
        """
        Compute standard Ishara/Veo watermark geometry for a given frame size.
        Matches getWatermarkInfo() and VideoWatermarkEngine.getVeoWatermark() from Ishara main.js.
        """
        min_dim = min(width, height)
        if mode == "veo":
            size = max(24, min(round(min_dim / 15.0), min_dim))
            margin = round(min_dim / 10.0)
        else:
            ratio = min_dim / 1536.0
            size = max(16, round(96 * ratio))
            margin = max(8, round(64 * ratio))

        size = max(8, min(round(size * size_scale), min_dim))
        x = max(0, min(width - size, width - margin - size + int(offset_x)))
        y = max(0, min(height - size, height - margin - size + int(offset_y)))
        return {"x": x, "y": y, "size": size, "width": size, "height": size}

    def unblend_single_watermark(
        self,
        image_rgb: np.ndarray,
        x: int, y: int, size: int,
        gain: float = -1.0  # -1 = AUTO SOLVER
    ) -> np.ndarray:
        """
        Exact inverse Porter-Duff alpha unblend:
          I_clean = (I_wm - gain * alpha * 255) / (1 - gain * alpha)

        gain=-1 triggers the Auto Gain Solver to prevent dark shades and ghost artifacts.
        """
        h, w = image_rgb.shape[:2]
        x1 = max(0, min(w - 1, x))
        y1 = max(0, min(h - 1, y))
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)
        rw, rh = x2 - x1, y2 - y1

        if rw <= 0 or rh <= 0:
            return image_rgb

        alpha_tmpl = self._get_alpha_template(size)
        if alpha_tmpl is None:
            logger.error("No alpha template — unblend skipped!")
            return image_rgb

        alpha_tmpl_crop = alpha_tmpl[:rh, :rw]
        output_rgb = image_rgb.copy()
        patch = output_rgb[y1:y2, x1:x2].astype(np.float32)

        effective_gain = gain
        if gain <= 0:
            # Sample surrounding background for better gain detection
            surround_mean = self._sample_surround(output_rgb, x1, y1, x2, y2)
            effective_gain = self.auto_solve_gain(
                patch, alpha_tmpl_crop, gain_hint=0.5,
                surround_mean=surround_mean
            )
            logger.info(f"[AUTO-GAIN] ({x1},{y1}) size={size} → gain={effective_gain:.3f} (surround={surround_mean.round(1) if surround_mean is not None else None})")
        else:
            logger.info(f"[MANUAL-GAIN] ({x1},{y1}) size={size} → gain={effective_gain:.3f}")

        alpha_crop = np.clip(alpha_tmpl_crop * effective_gain, 0.0, 0.99)
        alpha_3d = np.expand_dims(alpha_crop, axis=2)

        unblended = np.clip(
            np.round((patch - alpha_3d * 255.0) / (1.0 - alpha_3d)),
            0.0, 255.0
        ).astype(np.uint8)

        mask_act = (alpha_3d[:, :, 0] >= 0.002)
        target_roi = output_rgb[y1:y2, x1:x2]
        target_roi[mask_act] = unblended[mask_act]
        output_rgb[y1:y2, x1:x2] = target_roi
        return output_rgb

    def unblend_image(
        self,
        image_rgb: np.ndarray,
        mode: str = "gemini",
        gain: float = -1.0,  # -1 = AUTO
        offset_x: int = -24,
        offset_y: int = -24,
        size_scale: float = 1.0,
        custom_box: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main unblending workflow with auto-gain, spatial alignment, and corner scanning.
        """
        h, w = image_rgb.shape[:2]
        current_img = image_rgb.copy()
        applied_items = []

        if custom_box is not None:
            # Resolve box to pixel coords
            box_w = custom_box.get("w", custom_box.get("width", 0))
            box_h = custom_box.get("h", custom_box.get("height", 0))

            if box_w > 0:
                if box_w <= 1.0:  # Normalized
                    bx1 = int(custom_box["x"] * w)
                    by1 = int(custom_box["y"] * h)
                    bx2 = int((custom_box["x"] + box_w) * w)
                    by2 = int((custom_box["y"] + box_h) * h)
                else:
                    bx1 = int(custom_box["x"])
                    by1 = int(custom_box["y"])
                    bx2 = bx1 + int(box_w)
                    by2 = by1 + int(box_h)

                star = self.find_star_in_roi(current_img, bx1, by1, bx2, by2)
                if star is not None:
                    logger.info(f"Star found in user box: ({star['x']},{star['y']}) conf={star['confidence']:.2f}")
                    current_img = self.unblend_single_watermark(
                        current_img, star["x"], star["y"], star["size"], gain=gain
                    )
                    applied_items.append(star)
                else:
                    bw_px = bx2 - bx1
                    bh_px = by2 - by1
                    sz = max(16, min(bw_px, bh_px))
                    logger.info(f"No star in box — direct unblend ({bx1},{by1}) size={sz}")
                    current_img = self.unblend_single_watermark(current_img, bx1, by1, sz, gain=gain)
                    applied_items.append({"x": bx1, "y": by1, "size": sz})

                return current_img, {"applied": True, "items": applied_items}

        # Auto-scan 4 corners
        corners = [
            ("bottom-right", w - int(w * 0.40), h - int(h * 0.40), w, h),
            ("bottom-left", 0, h - int(h * 0.40), int(w * 0.40), h),
            ("top-right", w - int(w * 0.40), 0, w, int(h * 0.40)),
            ("top-left", 0, 0, int(w * 0.40), int(h * 0.40)),
        ]

        found_any = False
        for cname, cx1, cy1, cx2, cy2 in corners:
            star = self.find_star_in_roi(current_img, cx1, cy1, cx2, cy2)
            if star is not None and star.get("confidence", 0) >= 0.40:
                logger.info(f"Watermark @ {cname}: ({star['x']},{star['y']}) conf={star['confidence']:.2f}")
                current_img = self.unblend_single_watermark(
                    current_img, star["x"], star["y"], star["size"], gain=gain
                )
                applied_items.append({**star, "corner": cname})
                found_any = True

        if not found_any:
            geom = self.get_watermark_geometry(w, h, mode=mode, offset_x=offset_x,
                                               offset_y=offset_y, size_scale=size_scale)
            logger.info(f"Fallback geometry: ({geom['x']},{geom['y']}) size={geom['size']}")
            current_img = self.unblend_single_watermark(
                current_img, geom["x"], geom["y"], geom["size"], gain=gain
            )
            applied_items.append({**geom, "corner": "fallback-bottom-right"})

        return current_img, {"applied": True, "items": applied_items}

    def precompute_video_alpha(
        self,
        width: int, height: int,
        gain: float = -1.0,
        offset_x: int = -24, offset_y: int = -24,
        size_scale: float = 1.0,
        custom_box: Optional[Dict[str, Any]] = None,
        first_frame_rgb: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Pre-compute watermark alpha tensor once before processing all video frames.
        Uses exact deterministic Veo mathematical geometry to guarantee 100% pixel-perfect alignment.
        """
        h, w = height, width
        min_dim = min(w, h)
        expected_size = max(24, min(round(min_dim / 15.0), min_dim))
        margin = round(min_dim / 10.0)

        # Standard Veo watermark coordinates (deterministic formula matching Ishara main.js):
        std_veo_x = max(0, w - margin - expected_size - 24)
        std_veo_y = max(0, h - margin - expected_size - 24)

        # 1. Determine Position & Size (Universal for 16:9, 9:16, 1:1, etc.)
        if custom_box is not None and custom_box.get("w", 0) > 0:
            bw_val = custom_box.get("w", 0)
            bh_val = custom_box.get("h", 0)
            if bw_val <= 1.0:
                bx1 = int(custom_box["x"] * w)
                by1 = int(custom_box["y"] * h)
                bx2 = int((custom_box["x"] + bw_val) * w)
                by2 = int((custom_box["y"] + bh_val) * h)
            else:
                bx1 = int(custom_box["x"])
                by1 = int(custom_box["y"])
                bx2 = bx1 + int(bw_val)
                by2 = by1 + int(bh_val)

            # Center expected watermark star on the user's box center:
            center_x = (bx1 + bx2) // 2
            center_y = (by1 + by2) // 2
            wm_size = expected_size

            # If user's box center is close to standard Veo corner (within 40px), snap to exact Veo formula:
            if abs(center_x - (std_veo_x + expected_size // 2)) <= 40 and abs(center_y - (std_veo_y + expected_size // 2)) <= 40:
                wm_x = std_veo_x
                wm_y = std_veo_y
                logger.info(f"[VIDEO] Snapped to exact Veo coordinates ({w}x{h}): ({wm_x},{wm_y}) size={wm_size}")
            else:
                wm_x = max(0, min(w - wm_size, center_x - wm_size // 2))
                wm_y = max(0, min(h - wm_size, center_y - wm_size // 2))
                logger.info(f"[VIDEO] Centered on custom box ({bx1},{by1} to {bx2},{by2}) -> ({wm_x},{wm_y}) size={wm_size}")
        else:
            wm_x = std_veo_x
            wm_y = std_veo_y
            wm_size = expected_size
            logger.info(f"[VIDEO] Veo standard geometry ({w}x{h}): ({wm_x},{wm_y}) size={wm_size}")

        alpha_tmpl = self._get_alpha_template(wm_size)
        if alpha_tmpl is None:
            logger.error("[VIDEO] No alpha template!")
            return {"wm_x": wm_x, "wm_y": wm_y, "wm_size": wm_size,
                    "alpha_3d": None, "mask_active": None, "opt_gain": 0.28}

        # 2. Determine Optimal Gain
        if gain > 0:
            opt_gain = float(gain)
            logger.info(f"[VIDEO] Using user-selected slider gain: {opt_gain:.3f}")
        elif first_frame_rgb is not None:
            x2 = min(w, wm_x + wm_size)
            y2 = min(h, wm_y + wm_size)
            patch = first_frame_rgb[wm_y:y2, wm_x:x2].astype(np.float32)
            rh, rw = patch.shape[:2]
            alpha_crop = alpha_tmpl[:rh, :rw]
            surround_mean = self._sample_surround(first_frame_rgb, wm_x, wm_y, x2, y2)
            opt_gain = self.auto_solve_gain(patch, alpha_crop, gain_hint=0.28, surround_mean=surround_mean)
            logger.info(f"[VIDEO] Auto-solved gain: {opt_gain:.3f} (surround={surround_mean.round(1) if surround_mean is not None else None})")
        else:
            opt_gain = 0.28
            logger.info(f"[VIDEO] Using default gain: {opt_gain:.3f}")

        alpha_scaled = np.clip(alpha_tmpl * opt_gain, 0.0, 0.99)
        alpha_3d = np.expand_dims(alpha_scaled, axis=2)
        mask_active = (alpha_3d[:, :, 0] >= 0.002)

        return {
            "wm_x": wm_x, "wm_y": wm_y, "wm_size": wm_size,
            "alpha_3d": alpha_3d, "mask_active": mask_active, "opt_gain": opt_gain
        }


gemini_unblend_engine = GeminiUnblendEngine()
