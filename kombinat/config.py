from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Database
    database_url: str = "postgresql://kombinat:kombinat@localhost:5432/kombinat"

    # GitHub OAuth
    github_client_id: str = ""
    github_client_secret: str = ""

    # JWT
    jwt_secret: str = ""
    jwt_expiry_seconds: int = 604800  # 7 days

    @model_validator(mode="after")
    def _check_jwt_secret(self) -> "Settings":
        if not self.jwt_secret:
            raise ValueError("jwt_secret must be set to a non-empty value")
        return self

    # App
    batch_default_size: int = 100
    batch_max_size: int = 500
    batch_expiry_hours: int = 24
    default_required_annotations: int = 2
    honeypot_ratio: float = 0.05  # 5% of batch pairs are honeypots

    # Server
    host: str = "0.0.0.0"
    port: int = 8000


def get_settings() -> Settings:
    return Settings()
