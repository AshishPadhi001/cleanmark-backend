import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes_image import router as image_router
from app.services.image_service import image_service

logger = logging.getLogger("cleanmark.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure storage directories exist and preload AI model
    settings.init_directories()
    logger.info("Initializing CleanMark backend services...")
    
    # Pre-download, load and warm up model on startup so first user request is instant!
    image_service.initialize_on_startup()
    
    yield
    # Shutdown
    logger.info("Shutting down CleanMark backend.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="High-performance Local Watermark & Object Remover API",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static storage for direct image previews
app.mount("/static/uploads", StaticFiles(directory=str(settings.UPLOADS_DIR)), name="uploads")
app.mount("/static/outputs", StaticFiles(directory=str(settings.OUTPUTS_DIR)), name="outputs")

# Include Routers
app.include_router(image_router, prefix=settings.API_V1_STR)

@app.get("/health")
@app.get("/api/health")
async def health_root():
    return {
        "status": "online",
        "device": image_service.device,
        "model_ready": image_service.is_model_ready(),
        "is_downloading": image_service.is_downloading,
        "download_progress": image_service.download_progress,
        "version": settings.VERSION
    }

@app.get("/")
async def root():
    return {
        "message": "CleanMark AI Inpainting Backend is running.",
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR
    }
