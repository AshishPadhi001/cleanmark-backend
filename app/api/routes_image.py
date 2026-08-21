import os
import io
import time
import base64
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse

from app.services.image_service import image_service
from app.utils.mask_utils import decode_image_bytes, decode_mask_bytes
from app.core.config import settings

logger = logging.getLogger("cleanmark.routes")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

router = APIRouter()

# Optimal worker pool (2 concurrent CPU workers to prevent thread contention on CPU)
cpu_workers = max(1, min(2, (os.cpu_count() or 4) // 2))
batch_executor = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="cleanmark_worker")

@router.get("/health")
@router.get("/api/health")
async def health_check():
    """Health check endpoint for Docker & frontend connection verification."""
    return {
        "status": "online",
        "device": image_service.device,
        "model_ready": image_service.is_model_ready(),
        "is_downloading": image_service.is_downloading,
        "download_progress": image_service.download_progress,
        "version": settings.VERSION
    }

def _fast_encode_image(image_rgb: np.ndarray, orig_filename: str = "") -> tuple[str, str, float]:
    """High-speed image encoder."""
    t0 = time.perf_counter()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    ext = Path(orig_filename).suffix.lower() if orig_filename else ".jpg"
    if ext in [".png", ".webp"]:
        success, buf = cv2.imencode(".png", image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        mime = "image/png"
    else:
        success, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mime = "image/jpeg"

    if not success:
        raise ValueError("Image encoding failed")

    b64_str = base64.b64encode(buf.tobytes()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64_str}"
    encode_ms = (time.perf_counter() - t0) * 1000
    return data_url, mime, round(encode_ms, 2)

def _sync_inpaint_worker(
    image_bytes: bytes,
    mask_bytes: bytes,
    filename: str,
    dilation: int,
    use_fallback: bool
) -> dict:
    """Synchronous CPU inpaint worker with clean microsecond telemetry."""
    t_start = time.perf_counter()
    breakdown = {}

    # 1. Decode arrays
    t_dec0 = time.perf_counter()
    image_np = decode_image_bytes(image_bytes)
    h, w = image_np.shape[:2]
    mask_np = decode_mask_bytes(mask_bytes, target_shape=(h, w))
    breakdown["decode_ms"] = round((time.perf_counter() - t_dec0) * 1000, 2)

    # 2. Inpaint
    result_np, method, _, service_timings = image_service.inpaint(
        image_np=image_np,
        mask_np=mask_np,
        dilation=dilation,
        use_fallback=use_fallback
    )
    breakdown.update(service_timings)

    # 3. Encode
    data_url, mime, encode_ms = _fast_encode_image(result_np, orig_filename=filename)
    breakdown["base64_encode_ms"] = encode_ms

    total_elapsed = time.perf_counter() - t_start
    breakdown["total_roundtrip_ms"] = round(total_elapsed * 1000, 2)
    infer_sec = round(breakdown.get("total_inference_ms", total_elapsed * 1000) / 1000, 2)

    logger.info(
        f"[Processed: {filename}] AI Inpaint: {infer_sec}s | "
        f"Total: {round(total_elapsed, 2)}s | Method: {method}"
    )

    return {
        "filename": filename,
        "width": w,
        "height": h,
        "method": method,
        "inference_seconds": infer_sec,
        "elapsed_seconds": round(total_elapsed, 2),
        "timings": breakdown,
        "image_data": data_url
    }

async def _process_single_image_task(
    image_file: UploadFile,
    mask_file: UploadFile,
    dilation: int,
    use_fallback: bool
) -> dict:
    """Async wrapper that delegates CPU-bound inpainting to worker pool."""
    image_bytes = await image_file.read()
    mask_bytes = await mask_file.read()
    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(
        batch_executor,
        _sync_inpaint_worker,
        image_bytes,
        mask_bytes,
        image_file.filename or "photo.jpg",
        dilation,
        use_fallback
    )

@router.post("/process/image")
async def process_image(
    image: Optional[UploadFile] = File(None),
    mask: Optional[UploadFile] = File(None),
    dilation: int = Form(3),
    use_fallback: bool = Form(False),
    return_base64: bool = Form(True)
):
    """Single-image inpainting endpoint with high-resolution telemetry."""
    if not image or not mask:
        raise HTTPException(status_code=400, detail="Must provide both 'image' and 'mask' multipart files.")

    try:
        res = await _process_single_image_task(image, mask, dilation, use_fallback)
        return JSONResponse(content={
            "success": True,
            "width": res["width"],
            "height": res["height"],
            "method": res["method"],
            "inference_seconds": res["inference_seconds"],
            "elapsed_seconds": res["elapsed_seconds"],
            "timings": res["timings"],
            "image_data": res["image_data"]
        })
    except Exception as e:
        logger.exception(f"Error in /process/image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/batch")
async def process_batch(
    images: List[UploadFile] = File(...),
    masks: List[UploadFile] = File(...),
    dilation: int = Form(3),
    use_fallback: bool = Form(False)
):
    """
    TRUE PARALLEL Bulk Batch Inpainting Endpoint.
    Dispatches all 1 to 5 images across optimal thread pool.
    """
    t_batch_start = time.perf_counter()
    if len(images) != len(masks):
        raise HTTPException(
            status_code=400,
            detail=f"Mismatched batch: {len(images)} images provided but {len(masks)} masks received."
        )

    logger.info(f"[BATCH PARALLEL] Dispatching {len(images)} images across executor...")
    loop = asyncio.get_running_loop()

    # Read all uploaded bytes concurrently
    read_tasks = [
        (img.read(), msk.read(), img.filename or f"image_{i+1}.jpg")
        for i, (img, msk) in enumerate(zip(images, masks))
    ]
    
    async def inpaint_item(img_task, msk_task, filename):
        img_bytes = await img_task
        msk_bytes = await msk_task
        return await loop.run_in_executor(
            batch_executor,
            _sync_inpaint_worker,
            img_bytes,
            msk_bytes,
            filename,
            dilation,
            use_fallback
        )

    parallel_tasks = [
        inpaint_item(r[0], r[1], r[2])
        for r in read_tasks
    ]

    results = await asyncio.gather(*parallel_tasks, return_exceptions=False)

    total_batch_sec = round(time.perf_counter() - t_batch_start, 2)
    logger.info(
        f"[PARALLEL BATCH COMPLETED] All {len(images)} images finished in {total_batch_sec}s total!"
    )

    return JSONResponse(content={
        "success": True,
        "count": len(results),
        "total_elapsed_seconds": total_batch_sec,
        "results": results
    })
