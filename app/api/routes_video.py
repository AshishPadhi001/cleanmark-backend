import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

from app.core.config import settings
from app.services.video_service import video_service
from app.utils.mask_utils import decode_mask_bytes

logger = logging.getLogger("cleanmark.video")

router = APIRouter(prefix="/video", tags=["video"])

VIDEO_STORAGE = settings.STORAGE_DIR / "videos"
VIDEO_STORAGE.mkdir(parents=True, exist_ok=True)

# Maximum worker threads for parallel multi-core rendering
VIDEO_EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


def get_session_dir(session_id: Optional[str] = None, purge_existing: bool = False) -> Path:
    """
    Returns or creates an isolated session directory under storage/videos/.
    If purge_existing is True, clears any previous files from this session.
    """
    clean_sid = (session_id or "").strip()
    if not clean_sid or "/" in clean_sid or "\\" in clean_sid or ".." in clean_sid:
        clean_sid = str(uuid.uuid4())
    s_dir = VIDEO_STORAGE / clean_sid

    if purge_existing and s_dir.exists():
        try:
            shutil.rmtree(s_dir, ignore_errors=True)
            logger.info(f"🔄 [SESSION RESET] Cleared previous files in session: {clean_sid}")
        except Exception:
            pass

    s_dir.mkdir(parents=True, exist_ok=True)
    return s_dir


def cleanup_session_folder(target_path: Path):
    """Deletes session folder and all temporary video files as soon as user downloads."""
    try:
        if not target_path.exists():
            return
        parent = target_path.parent
        # If inside a session subdirectory under VIDEO_STORAGE, remove the entire session folder
        if parent != VIDEO_STORAGE and parent.parent == VIDEO_STORAGE and parent.is_dir():
            shutil.rmtree(parent, ignore_errors=True)
            logger.info(f"🗑️ [AUTO-CLEANUP] Deleted session directory: {parent.name}")
        else:
            target_path.unlink(missing_ok=True)
            # Remove corresponding input files
            stem_base = target_path.stem.replace("_cleaned", "")
            for inp in target_path.parent.glob(f"{stem_base}_input.*"):
                inp.unlink(missing_ok=True)
            logger.info(f"🗑️ [AUTO-CLEANUP] Deleted video files: {target_path.name}")
    except Exception as e:
        logger.warning(f"Error during auto-cleanup: {e}")


def cleanup_stale_sessions(max_age_seconds: int = 300):
    """Background garbage collector: removes any abandoned session older than 5 minutes."""
    try:
        now = time.time()
        for item in VIDEO_STORAGE.iterdir():
            if item.is_dir():
                try:
                    mtime = item.stat().st_mtime
                    if now - mtime > max_age_seconds:
                        shutil.rmtree(item, ignore_errors=True)
                        logger.info(f"🧹 [TTL CLEANUP] Removed expired session: {item.name}")
                except Exception:
                    pass
            elif item.is_file() and item.suffix == ".mp4":
                try:
                    if now - item.stat().st_mtime > max_age_seconds:
                        item.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"TTL cleanup exception: {e}")



@router.post("/session/init")
@router.post("/sessions/purge-all")
async def init_session_or_purge_all(session_id: Optional[str] = Form(None)):
    """
    Called when frontend mounts / reloads / starts a new session:
    Purges any stale or orphaned sessions in storage/videos/ and registers new clean session.
    """
    deleted = 0
    clean_sid = (session_id or "").strip()
    for item in VIDEO_STORAGE.iterdir():
        if item.is_dir() and item.name != clean_sid:
            try:
                shutil.rmtree(item, ignore_errors=True)
                deleted += 1
            except Exception:
                pass
        elif item.is_file():
            try:
                item.unlink(missing_ok=True)
                deleted += 1
            except Exception:
                pass

    new_session_dir = get_session_dir(clean_sid, purge_existing=True)
    logger.info(f"🔄 [SESSION INIT] Initialized session: {new_session_dir.name} (purged {deleted} old sessions)")
    return {"status": "ok", "session_id": new_session_dir.name, "purged": deleted}

@router.post("/info")
async def get_video_info(
    video: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Extracts video metadata (fps, frame count, dimensions, duration, and first frame preview)
    and saves to an isolated session directory. Automatically clears any previous session files.
    """
    # Trigger background cleanup of stale sessions
    asyncio.get_event_loop().run_in_executor(None, cleanup_stale_sessions)

    # When a new video upload happens, reset/purge previous session video files
    session_dir = get_session_dir(session_id, purge_existing=True)
    vid_id = str(uuid.uuid4())
    ext = Path(video.filename or "video.mp4").suffix or ".mp4"
    temp_path = session_dir / f"{vid_id}_input{ext}"

    try:
        content = await video.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        metadata = video_service.get_video_metadata(str(temp_path))
        metadata["video_id"] = vid_id
        metadata["session_id"] = session_dir.name
        metadata["filename"] = video.filename

        logger.info(f"📹 [VIDEO INFO] session={session_dir.name} id={vid_id} {metadata.get('width')}x{metadata.get('height')} @ {metadata.get('fps')}fps ({metadata.get('frame_count')} frames)")
        return metadata

    except Exception as e:
        logger.error(f"Failed to read video metadata: {e}", exc_info=True)
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted video file: {str(e)}")


@router.post("/clean")
async def clean_video_stream(
    video: Optional[UploadFile] = File(None),
    video_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    mask: Optional[UploadFile] = File(None),
    mask_base64: Optional[str] = Form(None),
    preset: Optional[str] = Form(None),
    removal_mode: str = Form("unblend"),
    unblend_gain: float = Form(-1.0),
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
    Saves in session directory and emits streaming progress events.
    """
    try:
        session_dir = get_session_dir(session_id, purge_existing=False)

        logger.info(f"➡️ [API VIDEO CLEAN] session={session_dir.name}, mode={removal_mode}, unblend_gain={unblend_gain}, box=({box_x}, {box_y}, {box_w}, {box_h}), time={start_sec}..{end_sec}")

        # Determine input video path
        if video_id:
            vid_id = video_id
            matches = list(session_dir.glob(f"{video_id}_input.*"))
            if not matches:
                matches = list(VIDEO_STORAGE.glob(f"**/{video_id}_input.*"))
            if matches:
                input_path = matches[0]
                session_dir = input_path.parent
            elif video is not None:
                ext = Path(video.filename or "video.mp4").suffix or ".mp4"
                input_path = session_dir / f"{vid_id}_input{ext}"
                content = await video.read()
                with open(input_path, "wb") as f:
                    f.write(content)
            else:
                raise HTTPException(status_code=404, detail="Uploaded video session not found.")
        elif video is not None:
            vid_id = str(uuid.uuid4())
            ext = Path(video.filename or "video.mp4").suffix or ".mp4"
            input_path = session_dir / f"{vid_id}_input{ext}"
            content = await video.read()
            with open(input_path, "wb") as f:
                f.write(content)
        else:
            raise HTTPException(status_code=400, detail="No video uploaded or video_id provided.")

        metadata = video_service.get_video_metadata(str(input_path))
        width = metadata["width"]
        height = metadata["height"]

        # Build mask numpy array if needed for inpainting
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

        output_video_path = str(session_dir / f"{vid_id}_cleaned.mp4")

        # Offload frame processing to background OS ThreadPoolExecutor
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
                    asyncio.run_coroutine_threadsafe(q.put(None), loop)

            loop.run_in_executor(VIDEO_EXECUTOR, sync_worker)

            while True:
                item = await q.get()
                if item is None:
                    break

                if item.get("status") == "completed":
                    item["video_id"] = vid_id
                    item["session_id"] = session_dir.name
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
async def download_cleaned_video(video_id: str, background_tasks: BackgroundTasks):
    """
    Downloads the cleaned MP4 video file and immediately clears/deletes
    the session storage files as soon as the download finishes.
    """
    matches = list(VIDEO_STORAGE.glob(f"**/{video_id}_cleaned.mp4"))
    if not matches:
        raise HTTPException(status_code=404, detail="Cleaned video not found or already downloaded and cleared.")

    path = matches[0]

    # Queue immediate background deletion of session files as soon as download stream completes
    background_tasks.add_task(cleanup_session_folder, path)

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"cleanmark_{video_id[:8]}.mp4"
    )


@router.get("/stream/{video_id}")
async def stream_cleaned_video(video_id: str):
    """Streams the cleaned MP4 video file for HTML5 video playback."""
    matches = list(VIDEO_STORAGE.glob(f"**/{video_id}_cleaned.mp4"))
    if not matches:
        raise HTTPException(status_code=404, detail="Cleaned video not found.")

    path = matches[0]
    return FileResponse(path, media_type="video/mp4")


@router.delete("/session/{session_id}/clear")
@router.delete("/cleanup/{video_id}")
async def cleanup_video_endpoint(session_id: Optional[str] = None, video_id: Optional[str] = None):
    """Explicitly deletes all session files when user resets, leaves, or navigates back."""
    deleted = 0
    if session_id:
        s_dir = VIDEO_STORAGE / session_id
        if s_dir.exists() and s_dir != VIDEO_STORAGE:
            shutil.rmtree(s_dir, ignore_errors=True)
            logger.info(f"🗑️ [SESSION CLEAR] Removed session folder: {session_id}")
            deleted += 1

    if video_id:
        matches = list(VIDEO_STORAGE.glob(f"**/{video_id}*.*"))
        for m in matches:
            try:
                cleanup_session_folder(m)
                deleted += 1
            except Exception:
                pass

    return {"status": "ok", "deleted_items": deleted}
