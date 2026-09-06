from evoforge.config import WorldConfig
from evoforge.world import World


def test_world_advances():
    config = WorldConfig(
        width=300,
        height=200,
        initial_agents=8,
        initial_plants=20,
        max_agents=50,
        max_plants=100,
    )
    world = World(config, seed=123)
    world.step()
    assert world.step_count == 1


def test_save_load_roundtrip(tmp_path):
    config = WorldConfig(
        width=300,
        height=200,
        initial_agents=4,
        initial_plants=10,
    )
    world = World(config, seed=5)
    for _ in range(3):
        world.step()

    path = tmp_path / "world.json"
    world.save(path)
    loaded = World.load(path)

    assert loaded.step_count == world.step_count
    assert len(loaded.agents) == len(world.agents)
    assert len(loaded.plants) == len(world.plants)
