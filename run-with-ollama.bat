@echo off

echo Building Docker image with Ollama support...
docker build --platform linux/amd64 -t pdf-processor-ollama:latest .

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)

echo Running PDF processor with Ollama...
echo Note: This may take several minutes on first run to download the phi model

docker run --rm -v "%cd%\app\input:/app/input" -v "%cd%\app\output:/app/output" -p 11434:11434 pdf-processor-ollama:latest

if %ERRORLEVEL% neq 0 (
    echo Processing failed!
    exit /b %ERRORLEVEL%
)

echo Processing complete! Check the .\app\output\ directory for results.
pause
