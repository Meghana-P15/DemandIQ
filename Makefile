PYTHONPATH := backend

.PHONY: api test frontend docker
api:
	PYTHONPATH=$(PYTHONPATH) uvicorn app.main:app --app-dir backend --reload --port 8000

test:
	PYTHONPATH=$(PYTHONPATH) pytest tests -q

frontend:
	cd frontend && npm run dev

docker:
	docker compose up --build
