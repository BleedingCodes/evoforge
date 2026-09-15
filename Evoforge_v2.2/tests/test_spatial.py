from dataclasses import dataclass

from evoforge.spatial import SpatialHash


@dataclass
class Item:
    id: int
    x: float
    y: float


def test_query_nearby_cells():
    space = SpatialHash(10)
    items = [Item(1, 5, 5), Item(2, 50, 50)]
    space.rebuild(items)
    found = {item.id for item in space.query(5, 5, 8)}
    assert 1 in found
    assert 2 not in found
