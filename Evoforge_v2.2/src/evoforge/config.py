from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class WorldConfig:
    width: int = 1400
    height: int = 900
    initial_agents: int = 220
    initial_plants: int = 700
    max_agents: int = 1800
    max_plants: int = 3500
    plant_spawn_rate: float = 3.2
    plant_energy_min: float = 18.0
    plant_energy_max: float = 42.0
    cell_size: int = 64
    wrap_world: bool = True
    day_length: int = 2400
    mutation_rate: float = 0.09
    mutation_scale: float = 0.18
    crossover_rate: float = 0.35
    telemetry_interval: int = 30

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorldConfig":
        return cls(**data)
