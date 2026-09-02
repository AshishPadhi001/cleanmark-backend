import os
import io
import time
import base64
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Union
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from app.services.image_service import image_service
from app.utils.mask_utils import decode_image_bytes, decode_mask_bytes
from app.core.config import settings

logger = logging.getLogger("cleanmark.routes_image")

router = APIRouter(tags=["Image Watermark Removal"])
batch_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="cleanmark_worker")


def _fast_encode_image(image_rgb: np.ndarray, orig_filename: str = "") -> tuple[str, str, float]:
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


def _sync_clean_worker(
    image_bytes: bytes,
    mask_bytes: Optional[bytes] = None,
    mask_base64: Optional[str] = None,
    preset: Optional[str] = None,
    box_x: Optional[float] = None,
    box_y: Optional[float] = None,
    box_w: Optional[float] = None,
    box_h: Optional[float] = None,
    gain: float = -1.0,  # -1 = AUTO gain solver
    size_scale: float = 1.0,
    filename: str = "image.jpg"
) -> dict:
    t_start = time.perf_counter()
    breakdown = {}

    t_dec0 = time.perf_counter()
    image_rgb = decode_image_bytes(image_bytes)
    h, w = image_rgb.shape[:2]
    breakdown["decode_ms"] = round((time.perf_counter() - t_dec0) * 1000, 2)

    mask_np = None
    if mask_bytes and len(mask_bytes) > 0:
        mask_np = decode_mask_bytes(mask_bytes, target_shape=(h, w))
    elif mask_base64 and mask_base64 != "undefined":
        if "," in mask_base64:
            mask_base64 = mask_base64.split(",")[1]
        raw = base64.b64decode(mask_base64)
        mask_np = decode_mask_bytes(raw, target_shape=(h, w))

    custom_box = None
    if box_x is not None and box_y is not None and box_w is not None and box_h is not None:
        custom_box = {"x": float(box_x), "y": float(box_y), "w": float(box_w), "h": float(box_h)}

    cleaned_rgb, method, elapsed_sec, service_timings = image_service.inpaint(
        image_rgb=image_rgb,
        mask_np=mask_np,
        preset=preset if preset != "undefined" else None,
        custom_box=custom_box,
        gain=float(gain) if gain is not None else -1.0,
        size_scale=float(size_scale) if size_scale else 1.0
    )
    breakdown.update(service_timings)

    data_url, mime, encode_ms = _fast_encode_image(cleaned_rgb, orig_filename=filename)
    breakdown["base64_encode_ms"] = encode_ms

    total_elapsed = time.perf_counter() - t_start
    breakdown["total_roundtrip_ms"] = round(total_elapsed * 1000, 2)

    logger.info(f"✨ [{filename}] {w}x{h} | Math Unblend: {elapsed_sec*1000:.2f}ms | Total: {total_elapsed*1000:.2f}ms (Zero Blur)")

    return {
        "filename": filename,
        "width": w,
        "height": h,
        "method": method,
        "inference_seconds": round(elapsed_sec, 4),
        "elapsed_seconds": round(total_elapsed, 4),
        "timings": breakdown,
        "image_data": data_url,
        "cleaned_image": data_url,
        "result_b64": data_url
    }


@router.post("/image/clean")
@router.post("/process/image")
async def clean_image_endpoint(request: Request):
    """
    Robust single-image watermark removal endpoint.
    Extracts all form fields dynamically without Pydantic 422 type mismatches.
    """
    form = await request.form()

    # Find image file from form
    target_image = form.get("image") or form.get("file") or form.get("image_file")
    if not target_image or not hasattr(target_image, "read"):
        logger.error("❌ [400] No valid image file found in multipart form-data.")
        raise HTTPException(status_code=400, detail="No image file provided. Please upload an image.")

    target_mask = form.get("mask") or form.get("mask_file")
    mask_base64 = form.get("mask_base64")
    preset = form.get("preset")

    def parse_float(val):
        try:
            return float(val) if val not in (None, "", "undefined", "null") else None
        except Exception:
            return None

    box_x = parse_float(form.get("box_x"))
    box_y = parse_float(form.get("box_y"))
    box_w = parse_float(form.get("box_w"))
    box_h = parse_float(form.get("box_h"))
    gain = (parse_float(form.get("gain")) if form.get("gain") is not None else -1.0)
    size_scale = parse_float(form.get("size_scale")) or 1.0

    try:
        image_bytes = await target_image.read()
        mask_bytes = await target_mask.read() if (target_mask and hasattr(target_mask, "read")) else None
        fname = getattr(target_image, "filename", "image.jpg") or "image.jpg"

        logger.info(f"📥 [SINGLE] Processing '{fname}' ({len(image_bytes)/1024:.1f} KB)...")

        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            batch_executor,
            _sync_clean_worker,
            image_bytes,
            mask_bytes,
            mask_base64,
            preset,
            box_x,
            box_y,
            box_w,
            box_h,
            gain,
            size_scale,
            fname
        )
        return JSONResponse(content={
            "success": True,
            **res
        })
    except Exception as e:
        logger.exception(f"❌ Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image/batch")
@router.post("/process/batch")
async def batch_clean_endpoint(request: Request):
    """
    Robust Bulk Batch Mathematical Watermark Removal.
    Uses request.form().getlist(...) to accept 1, 2, 5, or 20 images seamlessly without 422 list errors.
    """
    form = await request.form()

    # Extract image and mask lists dynamically
    target_images = form.getlist("images") or form.getlist("files") or form.getlist("image")
    if not target_images:
        logger.error("❌ [400] No images provided in batch request.")
        raise HTTPException(status_code=400, detail="No images provided for batch processing.")

    target_masks = form.getlist("masks") or form.getlist("mask")
    preset = form.get("preset")

    def parse_float(val):
        try:
            return float(val) if val not in (None, "", "undefined", "null") else None
        except Exception:
            return None

    gain = (parse_float(form.get("gain")) if form.get("gain") is not None else -1.0)
    size_scale = parse_float(form.get("size_scale")) or 1.0

    t_batch_start = time.perf_counter()
    logger.info(f"📥 [BATCH] Starting batch unblending for {len(target_images)} images...")
    loop = asyncio.get_running_loop()

    image_contents = [
        await img.read() if hasattr(img, "read") else b""
        for img in target_images
    ]
    mask_contents = [
        await m.read() if (m and hasattr(m, "read")) else None
        for m in target_masks
    ] if target_masks else [None] * len(target_images)

    # Pad masks list to match image count
    while len(mask_contents) < len(target_images):
        mask_contents.append(None)

    async def clean_item(img_bytes, msk_bytes, filename):
        return await loop.run_in_executor(
            batch_executor,
            _sync_clean_worker,
            img_bytes,
            msk_bytes,
            None,
            preset,
            None,
            None,
            None,
            None,
            gain,
            size_scale,
            filename
        )

    tasks = [
        clean_item(img_b, msk_b, getattr(img, "filename", f"image_{i+1}.jpg") or f"image_{i+1}.jpg")
        for i, (img, img_b, msk_b) in enumerate(zip(target_images, image_contents, mask_contents))
    ]

    results = await asyncio.gather(*tasks, return_exceptions=False)
    total_batch_sec = round(time.perf_counter() - t_batch_start, 3)
    logger.info(f"✅ [BATCH COMPLETED] All {len(target_images)} images unblended in {total_batch_sec*1000:.1f}ms total!")

    return JSONResponse(content={
        "success": True,
        "count": len(results),
        "total_elapsed_seconds": total_batch_sec,
        "results": results
    })
