from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    app_name: str = "DemandIQ API"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./demandiq_dev.db"
    jwt_secret: str = "change-me-in-production"
    jwt_exp_minutes: int = 1440
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    max_upload_rows: int = 5000
    max_upload_mb: int = 10
    model_path: str = str(ROOT_DIR / "models" / "week10_model_bundle.joblib")
    demo_history_path: str = str(ROOT_DIR / "data" / "demo" / "demo_history.csv")
    reports_results_dir: str = str(ROOT_DIR / "reports" / "results")
    reports_figures_dir: str = str(ROOT_DIR / "reports" / "figures")

    model_config = SettingsConfigDict(env_file=str(ROOT_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

@lru_cache
def get_settings() -> Settings:
    return Settings()
