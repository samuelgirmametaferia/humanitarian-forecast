from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast


@dataclass(frozen=True)
class Settings:
    provider_mode: Literal["demo", "production"]
    database_url: str | None
    ingest_hmac_secret: str | None
    cron_secret: str | None
    allowed_origins: tuple[str, ...]
    redis_url: str | None
    redis_token: str | None
    model_dir: str

    @classmethod
    def from_env(cls) -> Settings:
        raw_mode = os.getenv("HF_PROVIDER_MODE", "demo")
        if raw_mode not in {"demo", "production"}:
            raise RuntimeError("HF_PROVIDER_MODE must be demo or production")
        mode = cast(Literal["demo", "production"], raw_mode)
        origins = tuple(
            origin.strip()
            for origin in os.getenv("HF_ALLOWED_ORIGINS", "").split(",")
            if origin.strip()
        )
        return cls(
            provider_mode=mode,
            database_url=os.getenv("HF_DATABASE_URL"),
            ingest_hmac_secret=os.getenv("HF_INGEST_HMAC_SECRET"),
            # Vercel automatically signs cron requests with CRON_SECRET. Keep the
            # HF-prefixed name for local and non-Vercel schedulers.
            cron_secret=os.getenv("HF_CRON_SECRET") or os.getenv("CRON_SECRET"),
            allowed_origins=origins,
            redis_url=os.getenv("HF_UPSTASH_REDIS_REST_URL"),
            redis_token=os.getenv("HF_UPSTASH_REDIS_REST_TOKEN"),
            model_dir=os.getenv(
                "HF_MODEL_DIR",
                str(Path(__file__).resolve().parents[1] / "models" / "ethiopia_serving"),
            ),
        )

    def validate_production(self) -> None:
        if self.provider_mode != "production":
            return
        missing = [
            name
            for name, value in {
                "HF_DATABASE_URL": self.database_url,
                "HF_INGEST_HMAC_SECRET": self.ingest_hmac_secret,
                "HF_CRON_SECRET": self.cron_secret,
                "HF_UPSTASH_REDIS_REST_URL": self.redis_url,
                "HF_UPSTASH_REDIS_REST_TOKEN": self.redis_token,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Production configuration missing: {', '.join(missing)}")
