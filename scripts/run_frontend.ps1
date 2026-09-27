$ErrorActionPreference = "Stop"
Set-Location frontend
if (-not (Test-Path ".env.local")) { Copy-Item ".env.example" ".env.local" }
npm run dev
