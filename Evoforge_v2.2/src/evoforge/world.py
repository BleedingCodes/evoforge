from __future__ import annotations

import colorsys
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import WorldConfig
from .entities import Agent, Plant
from .genome import Genome, INPUTS
from .mathutil import angle_delta, clamp, wrap_angle
from .spatial import SpatialHash
from .telemetry import Telemetry


@dataclass(slots=True)
class Lineage:
    id: int
    founder_id: int
    parent_lineage: int | None
    birth_step: int
    color_h: float
    total_births: int = 1
    total_deaths: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "founder_id": self.founder_id,
            "parent_lineage": self.parent_lineage,
            "birth_step": self.birth_step,
            "color_h": self.color_h,
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
        }


class World:
    def __init__(
        self,
        config: WorldConfig,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.next_agent_id = 1
        self.next_plant_id = 1
        self.next_lineage_id = 1
        self.agents: dict[int, Agent] = {}
        self.plants: dict[int, Plant] = {}
        self.lineages: dict[int, Lineage] = {}
        self.agent_space = SpatialHash(config.cell_size)
        self.plant_space = SpatialHash(config.cell_size)
        self.telemetry = Telemetry()
        self.births_since_sample = 0
        self.deaths_since_sample = 0
        self.plant_spawn_accumulator = 0.0

        # Narrative / spectacle state — purely presentational, not persisted.
        self.event_log: deque[tuple[int, str]] = deque(maxlen=60)
        self.frame_events: list[tuple[str, float, float]] = []
        self.daylight = 1.0
        self.weather = "calm"
        self.weather_timer = 0
        self.records: dict[str, float] = {
            "max_generation": 0,
            "max_kills": 0,
            "max_age": 0,
            "max_radius": 0.0,
        }
        self._last_lowpop_warning = -10_000

        for _ in range(config.initial_plants):
            self.spawn_plant()

        for _ in range(config.initial_agents):
            self.spawn_founder()

        self.rebuild_spatial()

    def spawn_plant(
        self,
        x: float | None = None,
        y: float | None = None,
        energy: float | None = None,
    ) -> Plant:
        plant = Plant(
            id=self.next_plant_id,
            x=float(self.rng.uniform(0, self.config.width) if x is None else x),
            y=float(self.rng.uniform(0, self.config.height) if y is None else y),
            energy=float(
                self.rng.uniform(
                    self.config.plant_energy_min,
                    self.config.plant_energy_max,
                )
                if energy is None
                else energy
            ),
        )
        self.next_plant_id += 1
        self.plants[plant.id] = plant
        return plant

    def spawn_founder(self) -> Agent:
        genome = Genome.random(self.rng)
        agent_id = self.next_agent_id
        lineage_id = self.next_lineage_id
        self.next_agent_id += 1
        self.next_lineage_id += 1

        self.lineages[lineage_id] = Lineage(
            id=lineage_id,
            founder_id=agent_id,
            parent_lineage=None,
            birth_step=self.step_count,
            color_h=genome.traits.color_h,
        )

        agent = Agent(
            id=agent_id,
            lineage_id=lineage_id,
            parent_id=None,
            generation=0,
            genome=genome,
            x=float(self.rng.uniform(0, self.config.width)),
            y=float(self.rng.uniform(0, self.config.height)),
            angle=float(self.rng.uniform(-math.pi, math.pi)),
            energy=float(self.rng.uniform(95.0, 150.0)),
        )
        self.agents[agent.id] = agent
        return agent

    def log(self, text: str) -> None:
        self.event_log.append((self.step_count, text))

    def drain_events(self) -> list[tuple[str, float, float]]:
        """Return and clear the spectacle events queued since the last drain."""
        events = self.frame_events
        self.frame_events = []
        return events

    def rebuild_spatial(self) -> None:
        self.agent_space.rebuild(self.agents.values())
        self.plant_space.rebuild(self.plants.values())

    def nearest_plant(self, agent: Agent) -> tuple[Plant | None, float, float]:
        best = None
        best_d2 = agent.genome.traits.vision_range ** 2
        best_angle = 0.0

        for item in self.plant_space.query(
            agent.x,
            agent.y,
            agent.genome.traits.vision_range,
        ):
            plant = item
            dx, dy = self.displacement(agent.x, agent.y, plant.x, plant.y)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best = plant
                best_d2 = d2
                best_angle = math.atan2(dy, dx)

        return best, math.sqrt(best_d2), best_angle

    def nearest_agent(
        self,
        agent: Agent,
        *,
        prefer_prey: bool,
    ) -> tuple[Agent | None, float, float]:
        best = None
        best_score = float("inf")
        best_distance = agent.genome.traits.vision_range
        best_angle = 0.0

        for item in self.agent_space.query(
            agent.x,
            agent.y,
            agent.genome.traits.vision_range,
        ):
            other = item
            if other.id == agent.id or not other.alive:
                continue

            dx, dy = self.displacement(agent.x, agent.y, other.x, other.y)
            distance = math.hypot(dx, dy)
            if distance <= 0 or distance > agent.genome.traits.vision_range:
                continue

            size_bias = other.radius / max(agent.radius, 0.1)
            if prefer_prey:
                score = distance * (0.55 + size_bias)
            else:
                score = distance / max(size_bias, 0.25)

            if score < best_score:
                best = other
                best_score = score
                best_distance = distance
                best_angle = math.atan2(dy, dx)

        return best, best_distance, best_angle

    def local_density(self, agent: Agent, radius: float = 60.0) -> float:
        count = 0
        for item in self.agent_space.query(agent.x, agent.y, radius):
            if item.id != agent.id:
                dx, dy = self.displacement(agent.x, agent.y, item.x, item.y)
                if dx * dx + dy * dy <= radius * radius:
                    count += 1
        return clamp(count / 12.0, 0.0, 1.0)

    def displacement(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> tuple[float, float]:
        dx = x2 - x1
        dy = y2 - y1

        if self.config.wrap_world:
            if abs(dx) > self.config.width / 2:
                dx -= math.copysign(self.config.width, dx)
            if abs(dy) > self.config.height / 2:
                dy -= math.copysign(self.config.height, dy)

        return dx, dy

    def boundary_signal(self, agent: Agent) -> tuple[float, float]:
        margin = 90.0
        sx = 0.0
        sy = 0.0

        if agent.x < margin:
            sx = 1.0 - agent.x / margin
        elif agent.x > self.config.width - margin:
            sx = -(1.0 - (self.config.width - agent.x) / margin)

        if agent.y < margin:
            sy = 1.0 - agent.y / margin
        elif agent.y > self.config.height - margin:
            sy = -(1.0 - (self.config.height - agent.y) / margin)

        return sx, sy

    def sense(self, agent: Agent) -> np.ndarray:
        plant, plant_dist, plant_angle = self.nearest_plant(agent)
        prey, prey_dist, prey_angle = self.nearest_agent(agent, prefer_prey=True)
        threat, threat_dist, threat_angle = self.nearest_agent(agent, prefer_prey=False)
        vision = agent.genome.traits.vision_range
        bx, by = self.boundary_signal(agent)
        day_phase = (self.step_count % self.config.day_length) / self.config.day_length

        inputs = np.zeros(INPUTS, dtype=np.float32)
        inputs[0] = agent.energy / agent.max_energy * 2.0 - 1.0
        inputs[1] = 1.0 - min(1.0, plant_dist / vision) if plant else -1.0
        inputs[2] = math.sin(angle_delta(plant_angle, agent.angle)) if plant else 0.0
        inputs[3] = math.cos(angle_delta(plant_angle, agent.angle)) if plant else 0.0
        inputs[4] = 1.0 - min(1.0, prey_dist / vision) if prey else -1.0
        inputs[5] = math.sin(angle_delta(prey_angle, agent.angle)) if prey else 0.0
        inputs[6] = 1.0 - min(1.0, threat_dist / vision) if threat else -1.0
        inputs[7] = math.sin(angle_delta(threat_angle, agent.angle)) if threat else 0.0
        inputs[8] = self.local_density(agent) * 2.0 - 1.0
        inputs[9] = bx
        inputs[10] = by
        inputs[11] = math.sin(day_phase * math.tau)
        inputs[12] = math.cos(day_phase * math.tau)
        inputs[13] = self.rng.uniform(-1.0, 1.0)
        return inputs

    def think(self, agent: Agent) -> None:
        outputs = agent.genome.forward(self.sense(agent))
        agent.turn = float(outputs[0])
        agent.throttle = float((outputs[1] + 1.0) * 0.5)
        agent.attack = float((outputs[2] + 1.0) * 0.5)
        agent.reproduce_signal = float((outputs[3] + 1.0) * 0.5)

    def move(self, agent: Agent) -> None:
        traits = agent.genome.traits
        agent.angle = wrap_angle(
            agent.angle + agent.turn * traits.turn_rate
        )
        speed = traits.max_speed * agent.throttle
        agent.x += math.cos(agent.angle) * speed
        agent.y += math.sin(agent.angle) * speed

        if self.config.wrap_world:
            agent.x %= self.config.width
            agent.y %= self.config.height
        else:
            agent.x = clamp(agent.x, 0.0, self.config.width)
            agent.y = clamp(agent.y, 0.0, self.config.height)

        movement_cost = (
            0.008
            + traits.metabolism * 0.007
            + speed * speed * (0.0045 + traits.radius * 0.0009)
            + traits.vision_range * 0.000006
        )
        agent.energy -= movement_cost

    def eat_plants(self, agent: Agent) -> None:
        traits = agent.genome.traits
        herbivory = 1.0 - traits.diet
        if herbivory < 0.08:
            return

        reach = traits.radius + 4.0
        for item in self.plant_space.query(agent.x, agent.y, reach):
            plant = self.plants.get(item.id)
            if plant is None:
                continue
            dx, dy = self.displacement(agent.x, agent.y, plant.x, plant.y)
            if dx * dx + dy * dy <= reach * reach:
                gained = plant.energy * herbivory * traits.efficiency
                agent.energy = min(agent.max_energy, agent.energy + gained)
                agent.food_eaten += gained
                self.plants.pop(plant.id, None)
                break

    def attack_agents(self, agent: Agent) -> None:
        traits = agent.genome.traits
        carnivory = traits.diet
        if carnivory < 0.12 or agent.attack < 0.42:
            return

        reach = traits.radius + 5.0
        candidates = self.agent_space.query(agent.x, agent.y, reach)

        for item in candidates:
            prey = self.agents.get(item.id)
            if prey is None or prey.id == agent.id or not prey.alive:
                continue

            dx, dy = self.displacement(agent.x, agent.y, prey.x, prey.y)
            if dx * dx + dy * dy > (reach + prey.radius) ** 2:
                continue

            advantage = (
                traits.aggression
                * carnivory
                * (agent.radius / max(prey.radius, 0.1))
                * agent.attack
            )
            defense = 0.35 + prey.genome.traits.aggression * 0.35

            damage = max(0.0, (advantage - defense) * 9.0 + 0.8)
            attack_cost = 0.12 + traits.radius * 0.015
            agent.energy -= attack_cost
            prey.energy -= damage

            if prey.energy <= 0 and prey.alive:
                prey.alive = False
                agent.kills += 1
                gained = (
                    prey.max_energy
                    * 0.32
                    * carnivory
                    * traits.efficiency
                )
                agent.energy = min(agent.max_energy, agent.energy + gained)
                agent.food_eaten += gained
                self.frame_events.append(("kill", agent.x, agent.y))
                if agent.kills > self.records["max_kills"]:
                    self.records["max_kills"] = agent.kills
                    self.log(
                        f"new apex predator: #{agent.id} (lineage {agent.lineage_id}) "
                        f"reaches {agent.kills} kills"
                    )
            break

    def maybe_reproduce(self, agent: Agent) -> Agent | None:
        if len(self.agents) >= self.config.max_agents:
            return None

        traits = agent.genome.traits
        threshold = agent.max_energy * (0.68 / traits.fertility)
        minimum_age = int(90 / traits.fertility)

        if (
            agent.age < minimum_age
            or agent.energy < threshold
            or agent.reproduce_signal < 0.52
        ):
            return None

        mate = None
        if self.rng.random() < self.config.crossover_rate:
            nearby = [
                item
                for item in self.agent_space.query(agent.x, agent.y, 35.0)
                if item.id != agent.id and item.alive
            ]
            if nearby:
                mate = nearby[int(self.rng.integers(0, len(nearby)))]

        if mate:
            base = Genome.crossover(agent.genome, mate.genome, self.rng)
            parent_lineage = agent.lineage_id
        else:
            base = agent.genome
            parent_lineage = agent.lineage_id

        child_genome = base.mutate(
            self.rng,
            self.config.mutation_rate,
            self.config.mutation_scale,
        )

        child_id = self.next_agent_id
        self.next_agent_id += 1

        lineage_id = parent_lineage
        divergence = abs(
            child_genome.traits.color_h - agent.genome.traits.color_h
        )
        if divergence > 0.085 or self.rng.random() < 0.008:
            lineage_id = self.next_lineage_id
            self.next_lineage_id += 1
            self.lineages[lineage_id] = Lineage(
                id=lineage_id,
                founder_id=child_id,
                parent_lineage=parent_lineage,
                birth_step=self.step_count,
                color_h=child_genome.traits.color_h,
            )
            self.log(f"new lineage #{lineage_id} branches from #{parent_lineage}")

        birth_cost = min(agent.energy * 0.46, child_genome.traits.radius * 12.0 + 38.0)
        agent.energy -= birth_cost
        agent.children += 1

        child = Agent(
            id=child_id,
            lineage_id=lineage_id,
            parent_id=agent.id,
            generation=agent.generation + 1,
            genome=child_genome,
            x=(agent.x + float(self.rng.normal(0, agent.radius * 2.2))) % self.config.width,
            y=(agent.y + float(self.rng.normal(0, agent.radius * 2.2))) % self.config.height,
            angle=wrap_angle(agent.angle + float(self.rng.normal(0, 0.8))),
            energy=birth_cost * 0.88,
        )
        self.lineages[lineage_id].total_births += 1
        self.births_since_sample += 1
        self.frame_events.append(("birth", child.x, child.y))

        if child.generation > self.records["max_generation"]:
            self.records["max_generation"] = child.generation
            if child.generation % 10 == 0 or child.generation < 5:
                self.log(f"lineage {lineage_id} reaches generation {child.generation}")

        return child

    def step(self) -> None:
        self.step_count += 1
        self.rebuild_spatial()

        newborns: list[Agent] = []

        for agent in list(self.agents.values()):
            if not agent.alive:
                continue

            agent.age += 1
            self.think(agent)
            self.move(agent)
            self.eat_plants(agent)
            self.attack_agents(agent)

            if agent.energy <= 0 or agent.age > 14000:
                agent.alive = False
                continue

            child = self.maybe_reproduce(agent)
            if child:
                newborns.append(child)

        for child in newborns:
            self.agents[child.id] = child

        dead_ids = [
            agent.id
            for agent in self.agents.values()
            if not agent.alive or agent.energy <= 0
        ]

        for agent_id in dead_ids:
            agent = self.agents.pop(agent_id)
            lineage = self.lineages.get(agent.lineage_id)
            if lineage:
                lineage.total_deaths += 1
                if lineage.total_deaths >= lineage.total_births and not any(
                    a.lineage_id == agent.lineage_id for a in self.agents.values()
                ):
                    self.log(f"lineage #{agent.lineage_id} has gone extinct")
            self.deaths_since_sample += 1
            self.frame_events.append(("death", agent.x, agent.y))

            if agent.age > self.records["max_age"]:
                self.records["max_age"] = agent.age
                self.log(
                    f"longevity record: #{agent.id} (lineage {agent.lineage_id}) "
                    f"survived {agent.age:,} steps"
                )

            if len(self.plants) < self.config.max_plants and self.rng.random() < 0.38:
                self.spawn_plant(
                    x=agent.x,
                    y=agent.y,
                    energy=min(45.0, agent.max_energy * 0.12),
                )

        for plant in self.plants.values():
            plant.age += 1

        self.daylight = 0.55 + 0.45 * math.sin(
            (self.step_count % self.config.day_length)
            / self.config.day_length
            * math.tau
        )

        self._update_weather()
        weather_multiplier = {"calm": 1.0, "bloom": 2.4, "blight": 0.3}[self.weather]

        self.plant_spawn_accumulator += (
            self.config.plant_spawn_rate
            * (0.45 + self.daylight)
            * weather_multiplier
            * max(0.0, 1.0 - len(self.plants) / self.config.max_plants)
        )

        if len(self.agents) <= 3 and self.step_count - self._last_lowpop_warning > 300:
            self._last_lowpop_warning = self.step_count
            self.log(f"population critical: only {len(self.agents)} organisms left")

        while self.plant_spawn_accumulator >= 1.0:
            if len(self.plants) < self.config.max_plants:
                self.spawn_plant()
            self.plant_spawn_accumulator -= 1.0

        if self.step_count % self.config.telemetry_interval == 0:
            self.telemetry.record(
                step=self.step_count,
                agents=self.agents.values(),
                plant_count=len(self.plants),
                births=self.births_since_sample,
                deaths=self.deaths_since_sample,
            )
            self.births_since_sample = 0
            self.deaths_since_sample = 0

            biggest = max(
                (a for a in self.agents.values()),
                key=lambda a: a.radius,
                default=None,
            )
            if biggest is not None and biggest.radius > self.records["max_radius"]:
                self.records["max_radius"] = biggest.radius
                self.log(
                    f"size record: #{biggest.id} (lineage {biggest.lineage_id}) "
                    f"grew to radius {biggest.radius:.1f}"
                )

    def _update_weather(self) -> None:
        if self.weather_timer > 0:
            self.weather_timer -= 1
            if self.weather_timer == 0:
                self.weather = "calm"
                self.log("weather settles back to calm")
            return

        if self.rng.random() < 0.0006:
            self.weather = "bloom" if self.rng.random() < 0.5 else "blight"
            self.weather_timer = int(self.rng.integers(400, 900))
            if self.weather == "bloom":
                self.log("a resource bloom sweeps the world — plants surge")
            else:
                self.log("a blight settles in — food grows scarce")

    def stats(self) -> dict[str, Any]:
        agents = list(self.agents.values())
        lineages = Counter(a.lineage_id for a in agents)
        return {
            "step": self.step_count,
            "population": len(agents),
            "plants": len(self.plants),
            "lineages": len(lineages),
            "max_generation": max((a.generation for a in agents), default=0),
            "dominant_lineage": lineages.most_common(1)[0] if lineages else None,
        }

    def save(self, path: Path) -> None:
        payload = {
            "config": self.config.to_dict(),
            "seed": self.seed,
            "step_count": self.step_count,
            "next_agent_id": self.next_agent_id,
            "next_plant_id": self.next_plant_id,
            "next_lineage_id": self.next_lineage_id,
            "agents": [a.to_dict() for a in self.agents.values()],
            "plants": [p.to_dict() for p in self.plants.values()],
            "lineages": [l.to_dict() for l in self.lineages.values()],
            "rng_state": self.rng.bit_generator.state,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "World":
        data = json.loads(path.read_text(encoding="utf-8"))
        world = cls.__new__(cls)
        world.config = WorldConfig.from_dict(data["config"])
        world.seed = data["seed"]
        world.rng = np.random.default_rng()
        world.rng.bit_generator.state = data["rng_state"]
        world.step_count = data["step_count"]
        world.next_agent_id = data["next_agent_id"]
        world.next_plant_id = data["next_plant_id"]
        world.next_lineage_id = data["next_lineage_id"]
        world.agents = {
            item["id"]: Agent.from_dict(item)
            for item in data["agents"]
        }
        world.plants = {
            item["id"]: Plant(**item)
            for item in data["plants"]
        }
        world.lineages = {
            item["id"]: Lineage(**item)
            for item in data["lineages"]
        }
        world.agent_space = SpatialHash(world.config.cell_size)
        world.plant_space = SpatialHash(world.config.cell_size)
        world.telemetry = Telemetry()
        world.births_since_sample = 0
        world.deaths_since_sample = 0
        world.plant_spawn_accumulator = 0.0
        world.rebuild_spatial()
        return world
