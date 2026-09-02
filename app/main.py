import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes_image import router as image_router
from app.api.routes_video import router as video_router
from app.services.image_service import image_service

# Configure root logger with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("cleanmark.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.init_directories()
    (settings.STORAGE_DIR / "videos").mkdir(parents=True, exist_ok=True)
    logger.info("⚡ [STARTUP] CleanMark Mathematical Engine Initialized (0 Blur, Zero-PyTorch)")
    image_service.initialize_on_startup()
    yield
    logger.info("🛑 [SHUTDOWN] CleanMark backend stopped.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Ultra-Fast Mathematical Watermark Removal API (0 Blur, Zero-PyTorch)",
    lifespan=lifespan
)

# Global 422 Request Validation Error Logger
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    logger.error(f"❌ [422 VALIDATION ERROR] on {request.method} {request.url.path}")
    for err in errors:
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "")
        err_type = err.get("type", "")
        logger.error(f"   ⚠️ Field [{loc}]: {msg} (type: {err_type})")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors}
    )

# Detailed Request & Response Logging Middleware
@app.middleware("http")
async def request_logger_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path
    method = request.method

    # Skip logging for health check endpoints (they poll every 15s)
    is_health = path in ("/health", "/api/health")
    if not is_health:
        logger.info(f"➡️  [{method}] {path} from {client_ip}")

    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        code = response.status_code
        status_icon = "✅" if code < 400 else "⚠️" if code < 500 else "❌"
        if not is_health:
            logger.info(f"{status_icon} [{code}] {method} {path} | Finished in {elapsed_ms:.2f}ms")
        logger.info(f"{status_icon} [{code}] {method} {path} | Finished in {elapsed_ms:.2f}ms")
        return response
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"❌ [500 INTERNAL ERROR] {method} {path} failed after {elapsed_ms:.2f}ms: {exc}", exc_info=True)
        raise exc

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static storage
app.mount("/static/uploads", StaticFiles(directory=str(settings.UPLOADS_DIR)), name="uploads")
app.mount("/static/outputs", StaticFiles(directory=str(settings.OUTPUTS_DIR)), name="outputs")
app.mount("/static/videos", StaticFiles(directory=str(settings.STORAGE_DIR / "videos")), name="videos")

# Include Routers (Mount at root and /api for seamless frontend compatibility)
app.include_router(image_router)
app.include_router(image_router, prefix="/api")
app.include_router(video_router)
app.include_router(video_router, prefix="/api")

@app.get("/health")
@app.get("/api/health")
async def health_root():
    return {
        "status": "online",
        "device": "cpu",
        "model_ready": True,
        "engine": "Mathematical Alpha Unblend (0 Blur, Zero-PyTorch)",
        "version": settings.VERSION
    }

@app.get("/")
async def root():
    return {
        "message": "CleanMark Mathematical Watermark Engine is online.",
        "docs_url": "/docs",
        "version": settings.VERSION
    }