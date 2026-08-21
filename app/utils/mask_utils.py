import io
import cv2
import numpy as np
from PIL import Image

def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decodes raw image bytes into an RGB numpy array (H, W, 3).
    """
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image, dtype=np.uint8)

def decode_mask_bytes(mask_bytes: bytes, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    """
    Decodes raw mask bytes into a 1-channel binary mask (H, W) uint8 with values 0 or 255.
    Handles RGBA canvas exports (e.g., painted strokes with alpha).
    """
    mask_img = Image.open(io.BytesIO(mask_bytes))
    
    if mask_img.mode == "RGBA":
        arr = np.array(mask_img)
        r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
        # Pixel is part of mask if it has luminance > 50 AND alpha > 50 (i.e. white painted stroke)
        luminance = 0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32)
        mask = np.where((a > 50) & (luminance > 50), 255, 0).astype(np.uint8)
    else:
        mask_gray = mask_img.convert("L")
        arr = np.array(mask_gray, dtype=np.uint8)
        # Threshold: any bright pixel is treated as watermark mask
        _, mask = cv2.threshold(arr, 50, 255, cv2.THRESH_BINARY)

    if target_shape is not None and (mask.shape[0] != target_shape[0] or mask.shape[1] != target_shape[1]):
        # Resize mask to match target image shape
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask

def dilate_mask(mask: np.ndarray, dilation_radius: int = 3) -> np.ndarray:
    """
    Expands the mask boundary slightly to ensure watermark borders/shadows are completely covered.
    """
    if dilation_radius <= 0:
        return mask
    kernel_size = 2 * dilation_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)

def pad_to_multiple(image: np.ndarray, mask: np.ndarray, multiple: int = 8):
    """
    Pads image (H, W, 3) and mask (H, W) so height and width are divisible by `multiple`.
    Returns padded_image, padded_mask, and original (h, w).
    """
    h, w = image.shape[:2]
    new_h = (h + multiple - 1) // multiple * multiple
    new_w = (w + multiple - 1) // multiple * multiple

    pad_h = new_h - h
    pad_w = new_w - w

    if pad_h == 0 and pad_w == 0:
        return image, mask, (h, w)

    # Pad with edge reflection for seamless boundaries
    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    padded_mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    return padded_img, padded_mask, (h, w)

def unpad_image(image: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """
    Crops padded image back to its original (h, w).
    """
    h, w = original_shape
    return image[:h, :w]

def encode_image_to_bytes(image_np: np.ndarray, format: str = "PNG") -> bytes:
    """
    Encodes an RGB numpy array back to image bytes.
    """
    pil_img = Image.fromarray(image_np)
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=95)
    return buf.getvalue()
