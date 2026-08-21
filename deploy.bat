@echo off

echo ==========================================
echo   Docker Deployment Script Starting
echo ==========================================
echo.

echo [1/5] Clearing screen...
cls

echo [2/5] Stopping and removing all containers...
docker compose down --remove-orphans

echo.
echo [3/5] Building images and starting containers...
docker compose up --build -d

echo.
echo [4/5] Showing running containers...
docker ps

echo.
echo [5/5] Streaming container logs (Press CTRL+C to exit)...
docker compose logs -f

echo.
echo ==========================================
echo   Deployment Complete
echo ==========================================
