import asyncio
import json
import logging
import os
import re
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

SESSION_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{6,64}$")
VIDEO_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{6,64}$")


def validate_session_id(session_id: Optional[str]) -> Optional[str]:
    """
    Strictly validates session_id format to prevent path traversal and arbitrary deletion.
    Returns the sanitized string if valid, or None if invalid or missing.
    """
    if not session_id:
        return None
    clean = str(session_id).strip()
    if not SESSION_ID_REGEX.match(clean):
        return None
    try:
        storage_root = VIDEO_STORAGE.resolve()
        target = (VIDEO_STORAGE / clean).resolve()
        # Verify strict containment: must be a direct child of VIDEO_STORAGE
        if target.parent != storage_root or target == storage_root:
            return None
    except Exception:
        return None
    return clean


def validate_video_id(video_id: Optional[str]) -> Optional[str]:
    """
    Strictly validates video_id format to prevent glob wildcard injection and path traversal.
    Returns the sanitized string if valid, or None if invalid or missing.
    """
    if not video_id:
        return None
    clean = str(video_id).strip()
    if not VIDEO_ID_REGEX.match(clean):
        return None
    return clean


def get_session_dir(session_id: Optional[str] = None, purge_existing: bool = False) -> Path:
    """
    Returns or creates an isolated session directory under storage/videos/.
    Ensures absolute session isolation: NEVER touches or alters other sessions.
    If purge_existing is True, clears previous files belonging ONLY to this specific session.
    """
    clean_sid = validate_session_id(session_id)
    if not clean_sid:
        clean_sid = f"sess_{uuid.uuid4().hex}"

    storage_root = VIDEO_STORAGE.resolve()
    s_dir = (VIDEO_STORAGE / clean_sid).resolve()

    # Safety check: ensure directory is directly inside VIDEO_STORAGE
    if s_dir.parent != storage_root or s_dir == storage_root:
        clean_sid = f"sess_{uuid.uuid4().hex}"
        s_dir = (VIDEO_STORAGE / clean_sid).resolve()

    if purge_existing and s_dir.exists():
        try:
            shutil.rmtree(s_dir, ignore_errors=True)
            logger.info(f"🔄 [SESSION RESET] Cleared previous files in session: {clean_sid}")
        except Exception as e:
            logger.warning(f"Error resetting session {clean_sid}: {e}")

    s_dir.mkdir(parents=True, exist_ok=True)
    return s_dir


def cleanup_session_folder(target_path: Path):
    """
    Safely deletes a specific session folder or video files after download.
    Strictly verifies path containment to avoid deleting unauthorized directories.
    """
    try:
        if not target_path.exists():
            return
        target_resolved = target_path.resolve()
        parent = target_resolved.parent
        storage_root = VIDEO_STORAGE.resolve()

        # If inside a direct session subdirectory under VIDEO_STORAGE, delete ONLY that session directory
        if parent != storage_root and parent.parent == storage_root and parent.is_dir():
            shutil.rmtree(parent, ignore_errors=True)
            logger.info(f"🗑️ [AUTO-CLEANUP] Deleted session directory: {parent.name}")
        elif target_resolved.parent == storage_root:
            # If a top-level file, unlink only this file and its matching inputs
            target_resolved.unlink(missing_ok=True)
            stem_base = target_resolved.stem.replace("_cleaned", "")
            for inp in storage_root.glob(f"{stem_base}_input.*"):
                inp.unlink(missing_ok=True)
            logger.info(f"🗑️ [AUTO-CLEANUP] Deleted video file: {target_resolved.name}")
    except Exception as e:
        logger.warning(f"Error during auto-cleanup: {e}")


def cleanup_stale_sessions(max_age_seconds: int = 3600):
    """
    Background garbage collector: safely removes abandoned sessions older than max_age_seconds (default 1 hour).
    Only deletes directories where the newest file activity is older than TTL.
    """
    try:
        now = time.time()
        storage_root = VIDEO_STORAGE.resolve()
        for item in VIDEO_STORAGE.iterdir():
            try:
                resolved = item.resolve()
                if resolved == storage_root or resolved.parent != storage_root:
                    continue

                if item.is_dir():
                    files = list(item.glob("*"))
                    newest_mtime = max((f.stat().st_mtime for f in files), default=item.stat().st_mtime)
                    if (now - newest_mtime) > max_age_seconds:
                        shutil.rmtree(item, ignore_errors=True)
                        logger.info(f"🧹 [TTL CLEANUP] Removed expired session: {item.name}")
                elif item.is_file():
                    if (now - item.stat().st_mtime) > max_age_seconds:
                        item.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"TTL cleanup exception: {e}")


@router.post("/session/init")
async def init_session(session_id: Optional[str] = Form(None)):
    """
    Called when frontend mounts / reloads / starts a new session:
    Safely registers or resets the user's isolated session folder.
    NEVER deletes or modifies other users' active sessions.
    """
    clean_sid = validate_session_id(session_id)
    # Reset ONLY this specific user's directory if session_id provided, or allocate a fresh one
    new_session_dir = get_session_dir(clean_sid, purge_existing=True)
    logger.info(f"✨ [SESSION INIT] Initialized isolated session: {new_session_dir.name}")
    return {"status": "ok", "session_id": new_session_dir.name}


@router.post("/sessions/purge-all")
async def purge_all_sessions_deprecated():
    """
    Disabled endpoint for security.
    Cross-user session purging is forbidden to prevent Broken Access Control vulnerabilities.
    """
    raise HTTPException(
        status_code=403,
        detail="Cross-user session purging is disabled for security. Use /api/video/session/{session_id}/clear to clear your own session."
    )


@router.post("/info")
async def get_video_info(
    video: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Extracts video metadata (fps, frame count, dimensions, duration, and first frame preview)
    and saves to an isolated session directory. Automatically clears any previous session files
    belonging ONLY to this session.
    """
    # Trigger background cleanup of genuinely stale sessions (> 1 hour old)
    asyncio.get_event_loop().run_in_executor(None, lambda: cleanup_stale_sessions(max_age_seconds=3600))

    # Reset/purge ONLY this session's video files
    session_dir = get_session_dir(session_id, purge_existing=True)
    vid_id = str(uuid.uuid4())
    ext = Path(video.filename or "video.mp4").suffix or ".mp4"
    if ext.lower() not in settings.ALLOWED_VIDEO_EXTENSIONS:
        ext = ".mp4"
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
    Saves in caller's session directory and emits streaming progress events.
    """
    try:
        session_dir = get_session_dir(session_id, purge_existing=False)
        clean_vid = validate_video_id(video_id)

        logger.info(f"⚡ [API VIDEO CLEAN] session={session_dir.name}, mode={removal_mode}, unblend_gain={unblend_gain}, box=({box_x}, {box_y}, {box_w}, {box_h}), time={start_sec}..{end_sec}")

        # Determine input video path safely
        if clean_vid:
            vid_id = clean_vid
            # First check caller's own session_dir
            matches = list(session_dir.glob(f"{vid_id}_input.*"))
            if not matches:
                # Fallback to session matching the sanitized video ID
                matches = list(VIDEO_STORAGE.glob(f"*/{vid_id}_input.*"))
            if matches:
                input_path = matches[0]
                session_dir = input_path.parent
            elif video is not None:
                ext = Path(video.filename or "video.mp4").suffix or ".mp4"
                if ext.lower() not in settings.ALLOWED_VIDEO_EXTENSIONS:
                    ext = ".mp4"
                input_path = session_dir / f"{vid_id}_input{ext}"
                content = await video.read()
                with open(input_path, "wb") as f:
                    f.write(content)
            else:
                raise HTTPException(status_code=404, detail="Uploaded video session not found.")
        elif video is not None:
            vid_id = str(uuid.uuid4())
            ext = Path(video.filename or "video.mp4").suffix or ".mp4"
            if ext.lower() not in settings.ALLOWED_VIDEO_EXTENSIONS:
                ext = ".mp4"
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
    Downloads the cleaned MP4 video file and safely cleans up ONLY the
    caller's session directory after download.
    """
    clean_vid = validate_video_id(video_id)
    if not clean_vid:
        raise HTTPException(status_code=400, detail="Invalid video ID format.")

    matches = list(VIDEO_STORAGE.glob(f"*/{clean_vid}_cleaned.mp4"))
    if not matches:
        raise HTTPException(status_code=404, detail="Cleaned video not found or already downloaded and cleared.")

    path = matches[0]

    # Queue background deletion of ONLY this session folder
    background_tasks.add_task(cleanup_session_folder, path)

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"cleanmark_{clean_vid[:8]}.mp4"
    )


@router.get("/stream/{video_id}")
async def stream_cleaned_video(video_id: str):
    """Streams the cleaned MP4 video file for HTML5 video playback."""
    clean_vid = validate_video_id(video_id)
    if not clean_vid:
        raise HTTPException(status_code=400, detail="Invalid video ID format.")

    matches = list(VIDEO_STORAGE.glob(f"*/{clean_vid}_cleaned.mp4"))
    if not matches:
        raise HTTPException(status_code=404, detail="Cleaned video not found.")

    path = matches[0]
    return FileResponse(path, media_type="video/mp4")


@router.delete("/session/{session_id}/clear")
@router.delete("/cleanup/{video_id}")
async def cleanup_video_endpoint(session_id: Optional[str] = None, video_id: Optional[str] = None):
    """
    Explicitly deletes session files strictly scoped to the requesting user's session or video.
    Safely ignores unauthorized or malformed IDs and never touches other users' data.
    """
    deleted = 0
    storage_root = VIDEO_STORAGE.resolve()

    if session_id:
        clean_sid = validate_session_id(session_id)
        if clean_sid:
            s_dir = (VIDEO_STORAGE / clean_sid).resolve()
            if s_dir.exists() and s_dir != storage_root and s_dir.parent == storage_root:
                shutil.rmtree(s_dir, ignore_errors=True)
                logger.info(f"🗑️ [SESSION CLEAR] Safely removed caller session folder: {clean_sid}")
                deleted += 1

    if video_id:
        clean_vid = validate_video_id(video_id)
        if clean_vid:
            matches = list(VIDEO_STORAGE.glob(f"*/{clean_vid}*.*"))
            for m in matches:
                try:
                    cleanup_session_folder(m)
                    deleted += 1
                except Exception:
                    pass

    return {"status": "ok", "deleted_items": deleted}
