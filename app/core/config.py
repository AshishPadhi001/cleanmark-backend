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

    # Detect Serverless (Vercel / AWS Lambda) environment
    IS_SERVERLESS: bool = bool(
        os.environ.get("VERCEL") or
        os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or
        os.environ.get("LAMBDA_TASK_ROOT") or
        os.environ.get("NOW_REGION")
    )

    # Base paths (Adaptive for Vercel / Lambda read-only file systems)
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: Path = BACKEND_DIR / "models"

    # On serverless, use /tmp (the only writable directory in Lambda/Vercel)
    STORAGE_DIR: Path = Path("/tmp/cleanmark_storage") if IS_SERVERLESS else (BACKEND_DIR / "storage")
    UPLOADS_DIR: Path = STORAGE_DIR / "uploads"
    OUTPUTS_DIR: Path = STORAGE_DIR / "outputs"
    TEMP_DIR: Path = STORAGE_DIR / "temp"

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
        "https://*.vercel.app",
        "*"
    ]

    def init_directories(self):
        """Ensure all storage and model directories exist gracefully."""
        try:
            self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        for path in [self.STORAGE_DIR, self.UPLOADS_DIR, self.OUTPUTS_DIR, self.TEMP_DIR]:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

settings = Settings()
settings.init_directories()
