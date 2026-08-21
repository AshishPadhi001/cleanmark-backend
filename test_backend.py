import sys
import numpy as np
import cv2
from pathlib import Path

# Add backend to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.image_service import image_service
from app.utils.mask_utils import encode_image_to_bytes

def run_test():
    print("--- Running CleanMark Inpainting Backend Test ---")
    
    # 1. Create a 400x400 test image with a gradient background and a red "WATERMARK" text
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    for y in range(400):
        for x in range(400):
            img[y, x] = [int(x / 400 * 200 + 30), int(y / 400 * 180 + 40), 160]

    # Draw watermark text
    cv2.putText(img, "WATERMARK", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 30, 30), 4)

    # 2. Create a mask over the text
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.putText(mask, "WATERMARK", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 12)

    print(f"Test image shape: {img.shape}, mask shape: {mask.shape}")
    print(f"Device: {image_service.device}")
    
    # 3. Test OpenCV Inpainting first (fast baseline)
    result_cv, method_cv, time_cv = image_service.inpaint(img, mask, dilation=3, use_fallback=True)
    print(f"[SUCCESS] {method_cv} completed in {time_cv:.4f} seconds.")

    # 4. Test AI Inpainting
    result_ai, method_ai, time_ai = image_service.inpaint(img, mask, dilation=3, use_fallback=False)
    print(f"[SUCCESS] {method_ai} completed in {time_ai:.4f} seconds.")

    # 5. Save test artifacts
    test_out_dir = Path(__file__).resolve().parent / "storage" / "test_results"
    test_out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(test_out_dir / "test_original.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(test_out_dir / "test_mask.png"), mask)
    cv2.imwrite(str(test_out_dir / "test_output.png"), cv2.cvtColor(result_ai, cv2.COLOR_RGB2BGR))

    print(f"Test artifacts saved to {test_out_dir}")
    print("--- Backend Test Passed Successfully ---")

if __name__ == "__main__":
    run_test()
