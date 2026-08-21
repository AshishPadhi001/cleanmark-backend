import os
import time
import logging
import urllib.request
import threading
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
from app.core.config import settings
from app.utils.mask_utils import (
    dilate_mask,
    pad_to_multiple,
    unpad_image,
)

logger = logging.getLogger("cleanmark.image_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ImageInpaintingService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.onnx_session = None
        self.is_downloading = False
        self.download_progress = 0.0
        self.model_path = settings.MODELS_DIR / "big-lama.pt"
        self.onnx_path = settings.MODELS_DIR / "lama.onnx"
        
        # Adaptive inference resolution cap (384px on CPU for 4x speedup, 768px on GPU)
        self.max_infer_dim = 768 if self.device == "cuda" else 384
        
        if self.device == "cpu":
            num_cores = max(1, min(8, os.cpu_count() or 4))
            torch.set_num_threads(num_cores)
            logger.info(f"PyTorch CPU configured with {num_cores} execution threads. Max ROI Patch: {self.max_infer_dim}px")
        else:
            logger.info(f"Initialized Inpainting on GPU: {torch.cuda.get_device_name(0)}")

    def is_model_ready(self) -> bool:
        return self.model is not None or self.onnx_session is not None

    def initialize_on_startup(self):
        if self.model_path.exists() or self.onnx_path.exists():
            logger.info("Found local model weights on startup. Preloading into memory...")
            self.load_model()
            self.warmup()
        else:
            logger.info("No local model weights found on startup. Initiating automatic background download...")
            threading.Thread(target=self._download_and_load_background, daemon=True).start()

    def _download_and_load_background(self):
        success = self.download_model()
        if success:
            self.load_model()
            self.warmup()

    def download_model(self, force: bool = False) -> bool:
        if self.model_path.exists() and not force:
            return True

        if self.is_downloading:
            return False

        self.is_downloading = True
        self.download_progress = 0.0
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        url = settings.LAMA_MODEL_URL
        target_file = self.model_path

        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                self.download_progress = min(100.0, (block_num * block_size / total_size) * 100.0)

        try:
            logger.info(f"Downloading LaMA model from {url} to {target_file}...")
            urllib.request.urlretrieve(url, target_file, reporthook=reporthook)
            self.download_progress = 100.0
            logger.info("LaMA model downloaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Primary model download failed: {e}")
            return False
        finally:
            self.is_downloading = False

    def load_model(self):
        if self.model is not None or self.onnx_session is not None:
            return

        if self.model_path.exists():
            try:
                t0 = time.perf_counter()
                logger.info(f"Loading TorchScript model from {self.model_path} onto {self.device}...")
                self.model = torch.jit.load(str(self.model_path), map_location=self.device)
                self.model.eval()
                load_ms = (time.perf_counter() - t0) * 1000
                logger.info(f"TorchScript model loaded in {load_ms:.1f}ms successfully.")
                return
            except Exception as e:
                logger.warning(f"Error loading TorchScript model: {e}")

        if self.onnx_path.exists():
            try:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
                self.onnx_session = ort.InferenceSession(str(self.onnx_path), providers=providers)
                logger.info("ONNX model loaded successfully.")
                return
            except Exception as e:
                logger.error(f"Error loading ONNX model: {e}")

    def warmup(self):
        if self.model is None and self.onnx_session is None:
            return
        try:
            t0 = time.perf_counter()
            dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
            dummy_mask = np.zeros((256, 256), dtype=np.uint8)
            dummy_mask[50:100, 50:100] = 255
            self._infer_core(dummy_img, dummy_mask)
            warmup_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"LaMA model warmup completed in {warmup_ms:.1f}ms (graphs JIT compiled).")
        except Exception as e:
            logger.warning(f"Warmup notice: {e}")

    def _infer_core(self, image_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        padded_img, padded_mask, orig_shape = pad_to_multiple(image_np, mask_np, multiple=8)

        if self.model is not None:
            img_t = torch.from_numpy(padded_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            mask_t = torch.from_numpy(padded_mask).float().unsqueeze(0).unsqueeze(0) / 255.0
            mask_t = (mask_t > 0).float()

            img_t = img_t.to(self.device)
            mask_t = mask_t.to(self.device)

            with torch.no_grad():
                output_t = self.model(img_t, mask_t)

            output_np = output_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            if output_np.max() <= 1.0 + 1e-4:
                output_np = output_np * 255.0
            output_np = np.clip(output_np, 0, 255).astype(np.uint8)

        elif self.onnx_session is not None:
            img_arr = (padded_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
            mask_arr = (padded_mask.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, ...]
            mask_arr = (mask_arr > 0).astype(np.float32)

            inputs = {
                self.onnx_session.get_inputs()[0].name: img_arr,
                self.onnx_session.get_inputs()[1].name: mask_arr
            }
            outputs = self.onnx_session.run(None, inputs)
            output_np = outputs[0].squeeze(0).transpose(1, 2, 0)
            if output_np.max() <= 1.0 + 1e-4:
                output_np = output_np * 255.0
            output_np = np.clip(output_np, 0, 255).astype(np.uint8)
        else:
            raise RuntimeError("No inpainting model loaded")

        return unpad_image(output_np, orig_shape)

    def inpaint_opencv(self, image: np.ndarray, mask: np.ndarray, radius: int = 3, method: str = "telea") -> np.ndarray:
        flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.inpaint(image_bgr, mask, radius, flags)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    def _get_watermark_clusters(self, mask: np.ndarray, merge_distance: int = 40) -> List[List[int]]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return []

        raw_boxes = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 4:
                raw_boxes.append([x, y, x + w, y + h])

        if not raw_boxes:
            return []

        # Merge overlapping / adjacent boxes
        merged = []
        while raw_boxes:
            cur = raw_boxes.pop(0)
            has_merged = False
            for m in merged:
                if not (cur[2] + merge_distance < m[0] or
                        cur[0] - merge_distance > m[2] or
                        cur[3] + merge_distance < m[1] or
                        cur[1] - merge_distance > m[3]):
                    m[0] = min(m[0], cur[0])
                    m[1] = min(m[1], cur[1])
                    m[2] = max(m[2], cur[2])
                    m[3] = max(m[3], cur[3])
                    has_merged = True
                    break
            if not has_merged:
                merged.append(cur)

        return merged

    def inpaint(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        dilation: int = 3,
        use_fallback: bool = False
    ) -> Tuple[np.ndarray, str, float, dict]:
        """
        High-precision inpainting with microsecond telemetry logging.
        Returns (result_image, method, total_elapsed_sec, timing_breakdown_dict).
        """
        t_start = time.perf_counter()
        timings = {}

        # 1. Mask preprocessing
        t_d0 = time.perf_counter()
        processed_mask = dilate_mask(mask_np, dilation_radius=dilation)
        timings["mask_dilation_ms"] = round((time.perf_counter() - t_d0) * 1000, 2)

        # 2. Fallback check
        if use_fallback or not self.is_model_ready():
            self.load_model()
            if not self.is_model_ready():
                t_cv0 = time.perf_counter()
                result = self.inpaint_opencv(image_np, processed_mask)
                timings["opencv_inpaint_ms"] = round((time.perf_counter() - t_cv0) * 1000, 2)
                elapsed = time.perf_counter() - t_start
                timings["total_pipeline_ms"] = round(elapsed * 1000, 2)
                return result, "OpenCV (Telea)", elapsed, timings

        try:
            h, w = image_np.shape[:2]
            
            # 3. Watermark cluster detection
            t_c0 = time.perf_counter()
            clusters = self._get_watermark_clusters(processed_mask, merge_distance=40)
            timings["clustering_ms"] = round((time.perf_counter() - t_c0) * 1000, 2)
            timings["num_clusters"] = len(clusters)

            if not clusters:
                elapsed = time.perf_counter() - t_start
                timings["total_pipeline_ms"] = round(elapsed * 1000, 2)
                return image_np, "None", elapsed, timings

            working_image = image_np.copy()
            infer_times = []

            # 4. Process each cluster
            for idx, box in enumerate(clusters):
                t_box0 = time.perf_counter()
                min_x, min_y, max_x, max_y = box
                box_w = max_x - min_x
                box_h = max_y - min_y

                margin_x = max(24, int(box_w * 0.30))
                margin_y = max(24, int(box_h * 0.30))

                crop_x1 = max(0, min_x - margin_x)
                crop_x2 = min(w, max_x + margin_x)
                crop_y1 = max(0, min_y - margin_y)
                crop_y2 = min(h, max_y + margin_y)

                crop_img = working_image[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_mask = processed_mask[crop_y1:crop_y2, crop_x1:crop_x2]

                if np.count_nonzero(crop_mask) == 0:
                    continue

                crop_h, crop_w = crop_img.shape[:2]

                # Adaptive scaling to optimal inference dimension (384px on CPU)
                if max(crop_h, crop_w) > self.max_infer_dim:
                    scale = self.max_infer_dim / float(max(crop_h, crop_w))
                    infer_w = int(round(crop_w * scale))
                    infer_h = int(round(crop_h * scale))
                    infer_img = cv2.resize(crop_img, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                    infer_mask = cv2.resize(crop_mask, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST)
                    output_patch = self._infer_core(infer_img, infer_mask)
                    cleaned_patch = cv2.resize(output_patch, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
                else:
                    cleaned_patch = self._infer_core(crop_img, crop_mask)

                patch_mask_3c = np.repeat(crop_mask[:, :, np.newaxis], 3, axis=2)
                working_image[crop_y1:crop_y2, crop_x1:crop_x2] = np.where(
                    patch_mask_3c > 0,
                    cleaned_patch,
                    working_image[crop_y1:crop_y2, crop_x1:crop_x2]
                )
                infer_times.append(round((time.perf_counter() - t_box0) * 1000, 2))

            timings["cluster_infer_times_ms"] = infer_times
            timings["total_inference_ms"] = round(sum(infer_times), 2)

            elapsed = time.perf_counter() - t_start
            timings["total_pipeline_ms"] = round(elapsed * 1000, 2)
            
            logger.info(
                f"[⚡ TIMING] {w}x{h} Image | {len(clusters)} Clusters | "
                f"Inference: {timings['total_inference_ms']}ms | Total: {timings['total_pipeline_ms']}ms"
            )

            return working_image, f"LaMA AI ({self.device.upper()})", elapsed, timings

        except Exception as e:
            logger.error(f"Inpainting error: {e}. Falling back to OpenCV.")
            t_fb0 = time.perf_counter()
            result = self.inpaint_opencv(image_np, processed_mask)
            elapsed = time.perf_counter() - t_start
            timings["fallback_ms"] = round((time.perf_counter() - t_fb0) * 1000, 2)
            timings["total_pipeline_ms"] = round(elapsed * 1000, 2)
            return result, "OpenCV Fallback", elapsed, timings

image_service = ImageInpaintingService()
