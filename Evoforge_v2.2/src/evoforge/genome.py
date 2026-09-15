from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .mathutil import clamp


INPUTS = 14
HIDDEN = 12
OUTPUTS = 6


@dataclass(slots=True)
class Traits:
    radius: float
    max_speed: float
    turn_rate: float
    vision_range: float
    metabolism: float
    fertility: float
    aggression: float
    diet: float
    efficiency: float
    color_h: float
    color_s: float
    color_v: float

    def clamp_all(self) -> None:
        self.radius = clamp(self.radius, 2.6, 10.0)
        self.max_speed = clamp(self.max_speed, 0.35, 3.8)
        self.turn_rate = clamp(self.turn_rate, 0.025, 0.34)
        self.vision_range = clamp(self.vision_range, 30.0, 230.0)
        self.metabolism = clamp(self.metabolism, 0.45, 2.0)
        self.fertility = clamp(self.fertility, 0.35, 1.85)
        self.aggression = clamp(self.aggression, 0.0, 1.0)
        self.diet = clamp(self.diet, 0.0, 1.0)
        self.efficiency = clamp(self.efficiency, 0.45, 1.4)
        self.color_h %= 1.0
        self.color_s = clamp(self.color_s, 0.35, 1.0)
        self.color_v = clamp(self.color_v, 0.45, 1.0)


@dataclass(slots=True)
class Genome:
    traits: Traits
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Genome":
        traits = Traits(
            radius=float(rng.uniform(3.2, 7.2)),
            max_speed=float(rng.uniform(0.7, 2.6)),
            turn_rate=float(rng.uniform(0.05, 0.22)),
            vision_range=float(rng.uniform(55.0, 155.0)),
            metabolism=float(rng.uniform(0.7, 1.35)),
            fertility=float(rng.uniform(0.65, 1.35)),
            aggression=float(rng.uniform(0.0, 1.0)),
            diet=float(rng.beta(1.2, 1.2)),
            efficiency=float(rng.uniform(0.7, 1.15)),
            color_h=float(rng.random()),
            color_s=float(rng.uniform(0.55, 0.95)),
            color_v=float(rng.uniform(0.65, 1.0)),
        )
        return cls(
            traits=traits,
            w1=rng.normal(0.0, 0.85, (HIDDEN, INPUTS)).astype(np.float32),
            b1=rng.normal(0.0, 0.35, HIDDEN).astype(np.float32),
            w2=rng.normal(0.0, 0.75, (OUTPUTS, HIDDEN)).astype(np.float32),
            b2=rng.normal(0.0, 0.25, OUTPUTS).astype(np.float32),
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        hidden = np.tanh(self.w1 @ inputs + self.b1)
        return np.tanh(self.w2 @ hidden + self.b2)

    def mutate(
        self,
        rng: np.random.Generator,
        rate: float,
        scale: float,
    ) -> "Genome":
        traits_data = asdict(self.traits)
        for key, value in traits_data.items():
            if rng.random() < rate:
                if key == "color_h":
                    value += float(rng.normal(0.0, scale * 0.2))
                else:
                    value *= float(np.exp(rng.normal(0.0, scale)))
                traits_data[key] = value

        traits = Traits(**traits_data)
        traits.clamp_all()

        def mutate_array(array: np.ndarray) -> np.ndarray:
            result = array.copy()
            mask = rng.random(result.shape) < rate
            noise = rng.normal(0.0, scale, result.shape)
            result += (mask * noise).astype(np.float32)
            return np.clip(result, -5.0, 5.0)

        return Genome(
            traits=traits,
            w1=mutate_array(self.w1),
            b1=mutate_array(self.b1),
            w2=mutate_array(self.w2),
            b2=mutate_array(self.b2),
        )

    @classmethod
    def crossover(
        cls,
        a: "Genome",
        b: "Genome",
        rng: np.random.Generator,
    ) -> "Genome":
        ta = asdict(a.traits)
        tb = asdict(b.traits)
        mixed = {
            key: ta[key] if rng.random() < 0.5 else tb[key]
            for key in ta
        }
        traits = Traits(**mixed)
        traits.clamp_all()

        def mix_arrays(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            mask = rng.random(x.shape) < 0.5
            return np.where(mask, x, y).astype(np.float32)

        return cls(
            traits=traits,
            w1=mix_arrays(a.w1, b.w1),
            b1=mix_arrays(a.b1, b.b1),
            w2=mix_arrays(a.w2, b.w2),
            b2=mix_arrays(a.b2, b.b2),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "traits": asdict(self.traits),
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Genome":
        return cls(
            traits=Traits(**data["traits"]),
            w1=np.asarray(data["w1"], dtype=np.float32),
            b1=np.asarray(data["b1"], dtype=np.float32),
            w2=np.asarray(data["w2"], dtype=np.float32),
            b2=np.asarray(data["b2"], dtype=np.float32),
        )
