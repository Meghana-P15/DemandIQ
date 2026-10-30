# Windows + VS Code setup

## Option A — Docker (recommended once Docker Desktop is installed)

1. Open the DemandIQ folder in VS Code.
2. Copy `.env.example` to `.env`.
3. Change `JWT_SECRET` and optionally `POSTGRES_PASSWORD`.
4. Run `docker compose up --build`.
5. Open `http://localhost:3000`.

## Option B — run backend/frontend separately

### Backend

Open PowerShell in the project root:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
$env:PYTHONPATH="backend"
uvicorn app.main:app --app-dir backend --reload --port 8000
```

If you do not create a root `.env`, local development falls back to SQLite. For the full production-style setup, install PostgreSQL and set `DATABASE_URL` from `.env.example`.

### Frontend

Open a second PowerShell:

```powershell
cd frontend
copy .env.example .env.local
npm install
npm run dev
```

Then open `http://localhost:3000`.

### Tests

From the repository root:

```powershell
$env:PYTHONPATH="backend"
pytest tests -q
```

### API docs

Open `http://localhost:8000/docs`.
