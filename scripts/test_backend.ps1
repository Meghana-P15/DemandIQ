$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "backend"
pytest tests -q
