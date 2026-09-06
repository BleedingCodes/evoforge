from __future__ import annotations

import math
from typing import Iterable

import numpy as np


TAU = math.tau


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % TAU - math.pi


def angle_delta(target: float, source: float) -> float:
    return wrap_angle(target - source)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -20.0, 20.0))


def normalized(values: Iterable[float]) -> list[float]:
    vals = list(values)
    total = sum(vals)
    if total <= 0:
        return [0.0 for _ in vals]
    return [v / total for v in vals]
