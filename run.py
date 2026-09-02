import multiprocessing
import uvicorn
from app.core.config import settings
from app.main import app

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print(f"Starting CleanMark Backend on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=False)
