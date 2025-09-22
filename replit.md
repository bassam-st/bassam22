run = "uvicorn main:app --host 0.0.0.0 --port $PORT"
entrypoint = "main.py"

[packager]
language = "python"

[env]
PYTHONUNBUFFERED = "1"
