$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "backend"
uvicorn app.main:app --app-dir backend --reload --port 8000
