import os
import io
import time
import json
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

from app.services.video_service import video_service
from app.utils.mask_utils import decode_mask_bytes
from app.core.config import settings

logger = logging.getLogger("cleanmark.routes_video")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

router = APIRouter(prefix="/api/video", tags=["Video Watermark Removal"])

VIDEO_STORAGE = settings.STORAGE_DIR / "videos"
VIDEO_STORAGE.mkdir(parents=True, exist_ok=True)

# Dedicated ThreadPoolExecutor for true concurrent video processing
VIDEO_EXECUTOR = ThreadPoolExecutor(max_workers=6, thread_name_prefix="video_worker")


@router.post("/info")
async def get_video_info(video: UploadFile = File(...)):
    """Uploads a video and returns its technical metadata and first frame preview."""
    try:
        video_id = str(uuid.uuid4())
        ext = Path(video.filename or "video.mp4").suffix or ".mp4"
        input_path = VIDEO_STORAGE / f"{video_id}_input{ext}"

        content = await video.read()
        with open(input_path, "wb") as f:
            f.write(content)

        metadata = video_service.get_video_metadata(str(input_path))
        metadata["video_id"] = video_id
        metadata["filename"] = video.filename

        return metadata
    except Exception as e:
        logger.error(f"Error reading video info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clean")
async def clean_video_stream(
    video: Optional[UploadFile] = File(None),
    video_id: Optional[str] = Form(None),
    mask: Optional[UploadFile] = File(None),
    mask_base64: Optional[str] = Form(None),
    preset: Optional[str] = Form(None),
    removal_mode: str = Form("unblend"),
    unblend_gain: float = Form(-1.0),  # -1 = AUTO gain solver
    offset_x: int = Form(-24),
    offset_y: int = Form(-24),
    size_scale: float = Form(1.0),
    box_x: Optional[float] = Form(None),
    box_y: Optional[float] = Form(None),
    box_w: Optional[float] = Form(None),
    box_h: Optional[float] = Form(None),
    start_sec: float = Form(0.0),
    end_sec: Optional[float] = Form(None),
    temporal_smoothing: bool = Form(False)
):
    """
    Cleans watermark from video with true parallel multi-threaded SSE streaming.
    Supports ultra-fast 100% mathematical alpha unblending ('unblend') and deep neural inpainting ('inpaint').
    """
    try:
        logger.info(f"➡️ [API VIDEO CLEAN] mode={removal_mode}, unblend_gain={unblend_gain}, box=({box_x}, {box_y}, {box_w}, {box_h}), time={start_sec}..{end_sec}")
        # Determine input video path
        if video is not None:
            vid_id = str(uuid.uuid4())
            ext = Path(video.filename or "video.mp4").suffix or ".mp4"
            input_path = VIDEO_STORAGE / f"{vid_id}_input{ext}"
            content = await video.read()
            with open(input_path, "wb") as f:
                f.write(content)
        elif video_id:
            vid_id = video_id
            matches = list(VIDEO_STORAGE.glob(f"{video_id}_input.*"))
            if not matches:
                raise HTTPException(status_code=404, detail="Uploaded video session not found.")
            input_path = matches[0]
        else:
            raise HTTPException(status_code=400, detail="No video uploaded or video_id provided.")

        metadata = video_service.get_video_metadata(str(input_path))
        width = metadata["width"]
        height = metadata["height"]

        # Build mask numpy array (H, W) if needed for inpaint mode
        mask_np = np.zeros((height, width), dtype=np.uint8)
        custom_box = None

        if box_x is not None and box_y is not None and box_w is not None and box_h is not None:
            custom_box = {"x": float(box_x), "y": float(box_y), "w": float(box_w), "h": float(box_h)}
            x1 = max(0, min(width - 1, int(box_x * width)))
            y1 = max(0, min(height - 1, int(box_y * height)))
            w_px = max(4, int(box_w * width))
            h_px = max(4, int(box_h * height))
            x2 = min(width, x1 + w_px)
            y2 = min(height, y1 + h_px)
            mask_np[y1:y2, x1:x2] = 255
        elif mask is not None:
            mask_bytes = await mask.read()
            uploaded_mask = decode_mask_bytes(mask_bytes)
            mask_np = cv2.resize(uploaded_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        elif mask_base64:
            import base64
            if "," in mask_base64:
                mask_base64 = mask_base64.split(",")[1]
            raw_bytes = base64.b64decode(mask_base64)
            uploaded_mask = decode_mask_bytes(raw_bytes)
            mask_np = cv2.resize(uploaded_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        elif preset:
            p = preset.lower().strip()
            bw = int(width * 0.22)
            bh = int(height * 0.10)
            pad_x = int(width * 0.03)
            pad_y = int(height * 0.03)

            if p == "top-left":
                mask_np[pad_y:pad_y + bh, pad_x:pad_x + bw] = 255
            elif p == "top-right":
                mask_np[pad_y:pad_y + bh, width - pad_x - bw:width - pad_x] = 255
            elif p == "bottom-left":
                mask_np[height - pad_y - bh:height - pad_y, pad_x:pad_x + bw] = 255
            elif p == "bottom-right":
                mask_np[height - pad_y - bh:height - pad_y, width - pad_x - bw:width - pad_x] = 255
            elif p == "bottom-banner":
                mask_np[height - int(height * 0.14):height, :] = 255

        output_video_path = str(VIDEO_STORAGE / f"{vid_id}_cleaned.mp4")

        # Offload synchronous frame processing to background OS ThreadPoolExecutor
        async def parallel_event_stream():
            loop = asyncio.get_running_loop()
            q = asyncio.Queue()

            def sync_worker():
                try:
                    for event in video_service.process_video_generator(
                        input_video_path=str(input_path),
                        mask_np=mask_np,
                        output_video_path=output_video_path,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        removal_mode=removal_mode,
                        unblend_gain=unblend_gain,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        size_scale=size_scale,
                        custom_box=custom_box
                    ):
                        asyncio.run_coroutine_threadsafe(q.put(event), loop)
                except Exception as exc:
                    logger.error(f"Worker video processing error: {exc}", exc_info=True)
                    asyncio.run_coroutine_threadsafe(q.put({"status": "error", "message": str(exc)}), loop)
                finally:
                    # Signal EOF
                    asyncio.run_coroutine_threadsafe(q.put(None), loop)

            loop.run_in_executor(VIDEO_EXECUTOR, sync_worker)

            while True:
                item = await q.get()
                if item is None:
                    break

                if item.get("status") == "completed":
                    item["video_id"] = vid_id
                    item["download_url"] = f"/api/video/download/{vid_id}"
                    item["stream_url"] = f"/api/video/stream/{vid_id}"

                yield f"data: {json.dumps(item)}\n\n"
                await asyncio.sleep(0.005)

                if item.get("status") in ("completed", "error"):
                    break

        return StreamingResponse(
            parallel_event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Error starting video cleaning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{video_id}")
async def download_cleaned_video(video_id: str):
    """Downloads the cleaned MP4 video file."""
    matches = list(VIDEO_STORAGE.glob(f"{video_id}_cleaned.mp4"))
    if not matches:
        raise HTTPException(status_code=404, detail="Cleaned video not found or processing still underway.")

    path = matches[0]
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"cleanmark_{video_id[:8]}.mp4"
    )


@router.get("/stream/{video_id}")
async def stream_cleaned_video(video_id: str):
    """Streams the cleaned MP4 video file for HTML5 video playback."""
    matches = list(VIDEO_STORAGE.glob(f"{video_id}_cleaned.mp4"))
    if not matches:
        raise HTTPException(status_code=404, detail="Cleaned video not found.")

    path = matches[0]
    return FileResponse(path, media_type="video/mp4")
