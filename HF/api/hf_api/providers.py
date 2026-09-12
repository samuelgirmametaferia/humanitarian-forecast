from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from .repository import ProductionRepository, Repository
from .schemas import ForecastSnapshot


class ForecastProvider(ABC):
    @abstractmethod
    def latest(self) -> ForecastSnapshot:
        raise NotImplementedError

    @abstractmethod
    def history(self, limit: int) -> list[ForecastSnapshot]:
        raise NotImplementedError

    @abstractmethod
    def by_id(self, forecast_id: str) -> ForecastSnapshot | None:
        raise NotImplementedError


class DemoForecastProvider(ForecastProvider):
    def __init__(self) -> None:
        path = Path(__file__).with_name("demo-forecast.json")
        self._snapshot = ForecastSnapshot.model_validate(json.loads(path.read_text()))

    def latest(self) -> ForecastSnapshot:
        return self._snapshot.model_copy(deep=True)

    def history(self, limit: int) -> list[ForecastSnapshot]:
        return [self.latest()][:limit]

    def by_id(self, forecast_id: str) -> ForecastSnapshot | None:
        return self.latest() if forecast_id == self._snapshot.id else None


class ProductionForecastProvider(ForecastProvider):
    def __init__(
        self,
        database_url: str | None = None,
        repository: Repository | None = None,
    ) -> None:
        self.repository = repository or ProductionRepository(database_url)

    def latest(self) -> ForecastSnapshot:
        return self.repository.latest()

    def history(self, limit: int) -> list[ForecastSnapshot]:
        return self.repository.history(limit)

    def by_id(self, forecast_id: str) -> ForecastSnapshot | None:
        return self.repository.by_id(forecast_id)
