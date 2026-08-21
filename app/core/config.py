import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    PROJECT_NAME: str = "CleanMark - AI Watermark & Object Remover"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # Server configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True

    # Base paths
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: Path = BACKEND_DIR / "models"
    STORAGE_DIR: Path = BACKEND_DIR / "storage"
    UPLOADS_DIR: Path = STORAGE_DIR / "uploads"
    OUTPUTS_DIR: Path = STORAGE_DIR / "outputs"
    TEMP_DIR: Path = STORAGE_DIR / "temp"

    # Model settings
    LAMA_MODEL_URL: str = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
    LAMA_ONNX_URL: str = "https://huggingface.co/anyisalin/big-lama-onnx/resolve/main/lama.onnx"
    MODEL_NAME: str = "big-lama.pt"
    
    # Upload constraints
    MAX_IMAGE_SIZE_MB: int = 50
    ALLOWED_IMAGE_EXTENSIONS: set = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ALLOWED_VIDEO_EXTENSIONS: set = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"
    ]

    def init_directories(self):
        """Ensure all storage and model directories exist."""
        for path in [self.MODELS_DIR, self.STORAGE_DIR, self.UPLOADS_DIR, self.OUTPUTS_DIR, self.TEMP_DIR]:
            path.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.init_directories()

