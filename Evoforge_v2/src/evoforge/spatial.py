from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Protocol


class Positioned(Protocol):
    id: int
    x: float
    y: float


class SpatialHash:
    def __init__(self, cell_size: int) -> None:
        self.cell_size = cell_size
        self.cells: dict[tuple[int, int], list[Positioned]] = defaultdict(list)

    def clear(self) -> None:
        self.cells.clear()

    def key(self, x: float, y: float) -> tuple[int, int]:
        return int(x // self.cell_size), int(y // self.cell_size)

    def insert(self, item: Positioned) -> None:
        self.cells[self.key(item.x, item.y)].append(item)

    def rebuild(self, items: Iterable[Positioned]) -> None:
        self.clear()
        for item in items:
            self.insert(item)

    def query(self, x: float, y: float, radius: float) -> list[Positioned]:
        min_x = int((x - radius) // self.cell_size)
        max_x = int((x + radius) // self.cell_size)
        min_y = int((y - radius) // self.cell_size)
        max_y = int((y + radius) // self.cell_size)
        result: list[Positioned] = []

        for cx in range(min_x, max_x + 1):
            for cy in range(min_y, max_y + 1):
                result.extend(self.cells.get((cx, cy), ()))

        return result
