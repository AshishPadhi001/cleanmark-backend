import os
import time
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Generator

import cv2
import numpy as np

import shutil

def find_ffmpeg_executable() -> str:
    """
    Finds a working FFmpeg binary across imageio_ffmpeg, system PATH, or standard Linux/Windows paths.
    """
    # 1. Try imageio_ffmpeg standalone binary
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass

    # 2. Try system PATH
    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        return which_ffmpeg

    # 3. Try common server / OS paths
    common_paths = [
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/var/task/ffmpeg",
        "/tmp/ffmpeg",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p

    return "ffmpeg"

FFMPEG_EXE = find_ffmpeg_executable()

from app.services.gemini_unblend import gemini_unblend_engine
from app.services.image_service import image_service
from app.core.config import settings

logger = logging.getLogger("cleanmark.video_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VideoInpaintingService:
    def __init__(self):
        self.ffmpeg_exe = FFMPEG_EXE
        self.temp_dir = settings.STORAGE_DIR / "videos"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.engine = gemini_unblend_engine

    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        ret, frame_bgr = cap.read()
        cap.release()

        first_frame_b64 = None
        if ret and frame_bgr is not None:
            import base64
            _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            first_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('utf-8')}"

        return {
            "width": width, "height": height,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration": round(duration, 2),
            "file_size": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
            "first_frame_preview": first_frame_b64
        }

    def process_video_generator(
        self,
        input_video_path: str,
        mask_np: Optional[np.ndarray],
        output_video_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        removal_mode: str = "unblend",
        unblend_gain: float = -1.0,   # -1.0 = AUTO GAIN SOLVER
        offset_x: int = -24,
        offset_y: int = -24,
        size_scale: float = 1.0,
        custom_box: Optional[Dict[str, Any]] = None,
        preset: Optional[str] = "bottom-right",
        temporal_smoothing: bool = False
    ) -> Generator[Dict[str, Any], None, str]:
        """
        High-Speed Mathematical Video Watermark Removal Pipeline.

        KEY FIX: Uses precompute_video_alpha() to:
        1. Auto-detect the watermark position once from the first frame
        2. Auto-solve the optimal gain to prevent dark shadows / ghost artifacts
        3. Pre-compute the alpha tensor once for ALL frames (temporal stability)
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        if end_sec is None or end_sec <= 0 or end_sec > duration:
            end_sec = duration

        start_frame = max(0, int(start_sec * fps))
        end_frame = min(total_frames, int(end_sec * fps))
        frames_to_process = max(1, end_frame - start_frame)

        video_temp_raw = str(Path(output_video_path).with_suffix(".temp.mp4"))
        temp_audio_path = str(Path(output_video_path).with_suffix(".audio.aac"))

        # Step 1: Extract audio
        has_audio = False
        try:
            cmd_audio = [self.ffmpeg_exe, "-y", "-i", input_video_path, "-vn", "-c:a", "copy", temp_audio_path]
            res = subprocess.run(cmd_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 100:
                has_audio = True
                logger.info("Audio extracted successfully.")
        except Exception as e:
            logger.warning(f"Audio extraction skipped: {e}")

        # Step 2: Read first frame for auto-detection and gain solving
        ret, first_bgr = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB) if ret and first_bgr is not None else None

        # Step 3: Pre-compute watermark geometry + alpha tensor ONCE
        precomputed = None
        if removal_mode == "unblend":
            precomputed = self.engine.precompute_video_alpha(
                width=width, height=height,
                gain=unblend_gain,
                offset_x=offset_x, offset_y=offset_y,
                size_scale=size_scale,
                custom_box=custom_box,
                first_frame_rgb=first_rgb
            )
            wm_x = precomputed["wm_x"]
            wm_y = precomputed["wm_y"]
            wm_size = precomputed["wm_size"]
            alpha_3d = precomputed["alpha_3d"]
            mask_active = precomputed["mask_active"]
            opt_gain = precomputed["opt_gain"]

            logger.info(f"[VIDEO PRE-COMPUTE] pos=({wm_x},{wm_y}) size={wm_size} gain={opt_gain:.3f}")

        # Step 4: Pre-compute inpainting ROI if needed
        roi_box = None
        crop_mask = None
        mask_feather = None
        if removal_mode == "inpaint" and mask_np is not None:
            if mask_np.shape[:2] != (height, width):
                mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
            pts = np.argwhere(mask_np > 128)
            if len(pts) > 0:
                y_min, x_min = pts.min(axis=0)
                y_max, x_max = pts.max(axis=0)
                pad = 32
                rx1 = max(0, x_min - pad)
                ry1 = max(0, y_min - pad)
                rx2 = min(width, x_max + pad)
                ry2 = min(height, y_max + pad)
                roi_box = (rx1, ry1, rx2, ry2)
                crop_mask = mask_np[ry1:ry2, rx1:rx2]
                mask_norm = crop_mask.astype(np.float32) / 255.0
                mask_feather = cv2.GaussianBlur(mask_norm, (7, 7), 2.0)[:, :, None]

        # Step 5: Open FFmpeg encode pipe
        cmd_encode = [
            self.ffmpeg_exe, "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            video_temp_raw
        ]
        pipe = subprocess.Popen(cmd_encode, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        start_time = time.time()
        processed_count = 0
        current_frame_idx = 0

        logger.info(f"[VIDEO START] {removal_mode.upper()} | {frames_to_process} frames | {width}x{height} @ {fps}fps")

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None or current_frame_idx >= total_frames:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                if start_frame <= current_frame_idx <= end_frame:
                    if removal_mode == "unblend" and precomputed is not None and alpha_3d is not None:
                        # FAST pre-computed unblend — no per-frame recomputation!
                        patch = frame_rgb[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size].astype(np.float32)
                        rh, rw = patch.shape[:2]
                        a3d = alpha_3d[:rh, :rw, :]
                        msk = mask_active[:rh, :rw]

                        unblended = np.clip(
                            np.round((patch - a3d * 255.0) / (1.0 - a3d)),
                            0.0, 255.0
                        ).astype(np.uint8)

                        t_roi = frame_rgb[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size]
                        t_roi[msk] = unblended[msk]
                        frame_rgb[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size] = t_roi

                    elif removal_mode == "inpaint" and roi_box and crop_mask is not None and mask_feather is not None:
                        rx1, ry1, rx2, ry2 = roi_box
                        crop_img = frame_rgb[ry1:ry2, rx1:rx2]
                        inpaint_raw = image_service._infer_core(crop_img, crop_mask)
                        clean_blend = (
                            inpaint_raw.astype(np.float32) * mask_feather +
                            crop_img.astype(np.float32) * (1.0 - mask_feather)
                        ).astype(np.uint8)
                        frame_rgb[ry1:ry2, rx1:rx2] = clean_blend

                pipe.stdin.write(frame_rgb.tobytes())
                processed_count += 1
                current_frame_idx += 1

                if processed_count % 10 == 0 or processed_count == frames_to_process:
                    elapsed = time.time() - start_time
                    fps_proc = processed_count / elapsed if elapsed > 0 else 0
                    pct = min(99.0, round((processed_count / frames_to_process) * 100, 1))
                    yield {
                        "status": "processing",
                        "progress": pct,
                        "frame": processed_count,
                        "total_frames": frames_to_process,
                        "fps_processing": round(fps_proc, 1),
                        "eta_seconds": round((frames_to_process - processed_count) / fps_proc, 1) if fps_proc > 0 else 0
                    }
        finally:
            cap.release()
            if pipe.stdin:
                pipe.stdin.close()
            pipe.wait()

        yield {"status": "remuxing", "progress": 99.0, "message": "Finalizing video..."}

        if has_audio and os.path.exists(temp_audio_path):
            cmd_remux = [
                self.ffmpeg_exe, "-y",
                "-i", video_temp_raw, "-i", temp_audio_path,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                output_video_path
            ]
        else:
            cmd_remux = [self.ffmpeg_exe, "-y", "-i", video_temp_raw, "-c:v", "copy", output_video_path]

        subprocess.run(cmd_remux, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        for tmp in [video_temp_raw, temp_audio_path]:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

        if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
            raise RuntimeError("Video remuxing failed to produce output file.")

        total_elapsed = time.time() - start_time
        avg_fps = total_frames / total_elapsed if total_elapsed > 0 else 0
        logger.info(f"[VIDEO DONE] {total_elapsed:.2f}s | {avg_fps:.1f} FPS avg | {output_video_path}")

        yield {
            "status": "completed",
            "progress": 100.0,
            "message": "Video cleaned successfully!",
            "output_path": output_video_path,
            "output_size": os.path.getsize(output_video_path),
            "elapsed_seconds": round(total_elapsed, 2),
            "fps": round(avg_fps, 1)
        }

        return output_video_path


video_service = VideoInpaintingService()
