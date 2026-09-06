from __future__ import annotations

import csv
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .entities import Agent


@dataclass(slots=True)
class TelemetryRow:
    step: int
    population: int
    plants: int
    births: int
    deaths: int
    mean_energy: float
    mean_generation: float
    mean_diet: float
    mean_speed: float
    mean_vision: float
    dominant_lineage: int | None
    dominant_lineage_size: int


class Telemetry:
    def __init__(self, history: int = 1200) -> None:
        self.rows: deque[TelemetryRow] = deque(maxlen=history)

    def record(
        self,
        *,
        step: int,
        agents: Iterable[Agent],
        plant_count: int,
        births: int,
        deaths: int,
    ) -> None:
        alive = list(agents)
        population = len(alive)
        lineage_counts = Counter(a.lineage_id for a in alive)
        dominant = lineage_counts.most_common(1)

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        self.rows.append(
            TelemetryRow(
                step=step,
                population=population,
                plants=plant_count,
                births=births,
                deaths=deaths,
                mean_energy=mean([a.energy for a in alive]),
                mean_generation=mean([float(a.generation) for a in alive]),
                mean_diet=mean([a.genome.traits.diet for a in alive]),
                mean_speed=mean([a.genome.traits.max_speed for a in alive]),
                mean_vision=mean([a.genome.traits.vision_range for a in alive]),
                dominant_lineage=dominant[0][0] if dominant else None,
                dominant_lineage_size=dominant[0][1] if dominant else 0,
            )
        )

    def export_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = list(self.rows)
        if not rows:
            return

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=asdict(rows[0]).keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
