from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .genome import Genome


@dataclass(slots=True)
class Plant:
    id: int
    x: float
    y: float
    energy: float
    age: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "energy": self.energy,
            "age": self.age,
        }


@dataclass(slots=True)
class Agent:
    id: int
    lineage_id: int
    parent_id: int | None
    generation: int
    genome: Genome
    x: float
    y: float
    angle: float
    energy: float
    age: int = 0
    children: int = 0
    kills: int = 0
    food_eaten: float = 0.0
    alive: bool = True
    throttle: float = 0.0
    turn: float = 0.0
    attack: float = 0.0
    reproduce_signal: float = 0.0

    @property
    def radius(self) -> float:
        return self.genome.traits.radius

    @property
    def max_energy(self) -> float:
        t = self.genome.traits
        return 150.0 + t.radius * 18.0 + t.efficiency * 35.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "lineage_id": self.lineage_id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "genome": self.genome.to_dict(),
            "x": self.x,
            "y": self.y,
            "angle": self.angle,
            "energy": self.energy,
            "age": self.age,
            "children": self.children,
            "kills": self.kills,
            "food_eaten": self.food_eaten,
            "alive": self.alive,
            "throttle": self.throttle,
            "turn": self.turn,
            "attack": self.attack,
            "reproduce_signal": self.reproduce_signal,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Agent":
        payload = dict(data)
        payload["genome"] = Genome.from_dict(payload["genome"])
        return cls(**payload)
