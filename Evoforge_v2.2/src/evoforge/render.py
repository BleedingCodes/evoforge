from __future__ import annotations

import colorsys
import math
from collections import deque
from dataclasses import dataclass

import pygame

from .entities import Agent
from .mathutil import clamp
from .world import World


NIGHT_BG = (5, 6, 12)
DAY_BG = (13, 16, 22)

PARTICLE_STYLE = {
    "birth": ((140, 230, 255), 26),
    "kill": ((235, 70, 70), 22),
    "death": ((120, 120, 130), 18),
}


@dataclass(slots=True)
class Camera:
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0

    def world_to_screen(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        return (
            int((x - self.x) * self.zoom + width / 2),
            int((y - self.y) * self.zoom + height / 2),
        )

    def screen_to_world(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        return (
            (x - width / 2) / self.zoom + self.x,
            (y - height / 2) / self.zoom + self.y,
        )


def hsv_color(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)
        self.title_font = pygame.font.SysFont("consolas", 22, bold=True)
        self.trails: dict[int, deque[tuple[float, float]]] = {}
        self.particles: list[dict] = []

    def _update_trails(self, world: World) -> None:
        live_ids = set()
        for agent in world.agents.values():
            live_ids.add(agent.id)
            trail = self.trails.setdefault(agent.id, deque(maxlen=14))
            trail.append((agent.x, agent.y))
        stale = [aid for aid in self.trails if aid not in live_ids]
        for aid in stale:
            del self.trails[aid]

    def _ingest_events(self, events: list[tuple[str, float, float]]) -> None:
        for kind, x, y in events:
            color, ttl = PARTICLE_STYLE.get(kind, ((200, 200, 200), 16))
            self.particles.append(
                {"kind": kind, "x": x, "y": y, "ttl": ttl, "max_ttl": ttl, "color": color}
            )

    def _draw_particles(self, camera: Camera, width: int, height: int) -> None:
        alive = []
        for particle in self.particles:
            particle["ttl"] -= 1
            if particle["ttl"] <= 0:
                continue
            alive.append(particle)

            progress = 1.0 - particle["ttl"] / particle["max_ttl"]
            sx, sy = camera.world_to_screen(particle["x"], particle["y"], width, height)
            radius = int((4 + progress * 22) * camera.zoom)
            if radius <= 0:
                continue
            alpha = max(0, int(255 * (1.0 - progress)))
            ring = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(
                ring,
                (*particle["color"], alpha),
                (radius + 1, radius + 1),
                radius,
                width=2,
            )
            self.screen.blit(ring, (sx - radius - 1, sy - radius - 1))
        self.particles = alive

    def background_color(self, world: World) -> tuple[int, int, int]:
        t = world.daylight  # roughly 0.1..1.0
        return tuple(
            int(night + (day - night) * t) for night, day in zip(NIGHT_BG, DAY_BG)
        )

    def draw(
        self,
        world: World,
        camera: Camera,
        selected_id: int | None,
        *,
        show_graph: bool,
        show_help: bool,
        paused: bool,
        speed: int,
        events: list[tuple[str, float, float]] | None = None,
    ) -> None:
        width, height = self.screen.get_size()
        self.screen.fill(self.background_color(world))

        self._ingest_events(events or [])
        self._update_trails(world)

        self.draw_grid(camera, width, height)

        for plant in world.plants.values():
            sx, sy = camera.world_to_screen(
                plant.x,
                plant.y,
                width,
                height,
            )
            if -10 <= sx <= width + 10 and -10 <= sy <= height + 10:
                radius = max(1, int((1.3 + plant.energy / 30.0) * camera.zoom))
                pygame.draw.circle(
                    self.screen,
                    (76, 180, 82),
                    (sx, sy),
                    radius,
                )

        self.draw_trails(camera, width, height)

        for agent in world.agents.values():
            self.draw_agent(agent, camera, width, height, agent.id == selected_id)

        self._draw_particles(camera, width, height)

        self.draw_hud(world, selected_id, paused, speed)

        if show_graph:
            self.draw_graph(world, width, height)

        if show_help:
            self.draw_help(width, height)

        self.draw_ticker(world, width, height)

    def draw_ticker(self, world: World, width: int, height: int) -> None:
        recent = list(world.event_log)[-4:]
        if not recent:
            return

        panel_h = 20 * len(recent) + 12
        panel = pygame.Surface((width - 24, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.screen.blit(panel, (12, height - panel_h - 12))

        y = height - panel_h - 6
        for step, text in recent:
            surface = self.small_font.render(f"[{step:,}] {text}", True, (200, 220, 235))
            self.screen.blit(surface, (22, y))
            y += 20

    def draw_trails(self, camera: Camera, width: int, height: int) -> None:
        # Cheap fading trail: draw directly on screen, dimming toward the
        # background rather than using per-segment alpha surfaces (would be
        # far too slow with hundreds of agents each frame).
        bg = self.screen.get_at((0, 0))[:3]
        for trail in self.trails.values():
            if len(trail) < 2:
                continue
            points = list(trail)
            n = len(points)
            for i in range(n - 1):
                weight = (i + 1) / n
                color = tuple(
                    int(bg[c] + (200 - bg[c]) * weight * 0.55) for c in range(3)
                )
                p1 = camera.world_to_screen(*points[i], width, height)
                p2 = camera.world_to_screen(*points[i + 1], width, height)
                pygame.draw.line(self.screen, color, p1, p2, 1)

    def draw_grid(self, camera: Camera, width: int, height: int) -> None:
        spacing = max(30, int(100 * camera.zoom))
        offset_x = int((-camera.x * camera.zoom + width / 2) % spacing)
        offset_y = int((-camera.y * camera.zoom + height / 2) % spacing)

        for x in range(offset_x, width, spacing):
            pygame.draw.line(self.screen, (18, 22, 28), (x, 0), (x, height))
        for y in range(offset_y, height, spacing):
            pygame.draw.line(self.screen, (18, 22, 28), (0, y), (width, y))

    def draw_agent(
        self,
        agent: Agent,
        camera: Camera,
        width: int,
        height: int,
        selected: bool,
    ) -> None:
        sx, sy = camera.world_to_screen(agent.x, agent.y, width, height)
        if not (-30 <= sx <= width + 30 and -30 <= sy <= height + 30):
            return

        traits = agent.genome.traits
        color = hsv_color(traits.color_h, traits.color_s, traits.color_v)
        radius = max(2, int(agent.radius * camera.zoom))
        energy_ratio = clamp(agent.energy / max(agent.max_energy, 1.0), 0.0, 1.0)

        # Aggression reads as spiky/predatory outward jabs around the body;
        # a purely peaceful herbivore stays a smooth circle.
        if traits.aggression > 0.35 and radius >= 2:
            spikes = 3 + int(traits.aggression * 5)
            spike_len = radius * (0.7 + traits.aggression * 0.9)
            for i in range(spikes):
                spike_angle = agent.angle + (i / spikes) * math.tau
                tip = (
                    sx + math.cos(spike_angle) * (radius + spike_len),
                    sy + math.sin(spike_angle) * (radius + spike_len),
                )
                pygame.draw.line(self.screen, color, (sx, sy), tip, 1)

        # A dim, low-energy organism looks visibly starved; a full one glows.
        body_color = tuple(int(c * (0.45 + 0.55 * energy_ratio)) for c in color)
        pygame.draw.circle(self.screen, body_color, (sx, sy), radius)

        if energy_ratio > 0.85:
            glow = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(
                glow, (*color, 70), (radius * 2, radius * 2), radius * 2
            )
            self.screen.blit(glow, (sx - radius * 2, sy - radius * 2))

        # Eyes: bigger and more forward for high-vision agents, so a keen
        # sniper visually reads differently from a near-blind grazer.
        eye_offset = radius * 0.55
        eye_size = max(1, int(radius * clamp(traits.vision_range / 230.0, 0.0, 1.0) * 0.6))
        for side in (-1, 1):
            eye_angle = agent.angle + side * 0.6
            ex = sx + math.cos(eye_angle) * eye_offset
            ey = sy + math.sin(eye_angle) * eye_offset
            pygame.draw.circle(self.screen, (15, 15, 20), (int(ex), int(ey)), eye_size)

        direction = (
            sx + int(math.cos(agent.angle) * radius * 1.5),
            sy + int(math.sin(agent.angle) * radius * 1.5),
        )
        pygame.draw.line(self.screen, (240, 240, 240), (sx, sy), direction, 1)

        if traits.diet > 0.67:
            pygame.draw.circle(self.screen, (235, 70, 70), (sx, sy), radius, 1)
        elif traits.diet < 0.33:
            pygame.draw.circle(self.screen, (80, 235, 110), (sx, sy), radius, 1)
        else:
            pygame.draw.circle(self.screen, (220, 190, 90), (sx, sy), radius, 1)

        # Elder ring: a visible reward for long-lived, prolific lineages.
        if agent.age > 4000 or agent.children >= 4:
            pygame.draw.circle(self.screen, (255, 215, 90), (sx, sy), radius + 2, 1)

        if selected:
            pygame.draw.circle(
                self.screen,
                (255, 230, 90),
                (sx, sy),
                radius + 5,
                2,
            )
            vision = int(traits.vision_range * camera.zoom)
            pygame.draw.circle(
                self.screen,
                (70, 80, 95),
                (sx, sy),
                max(1, vision),
                1,
            )

    def draw_hud(
        self,
        world: World,
        selected_id: int | None,
        paused: bool,
        speed: int,
    ) -> None:
        stats = world.stats()
        weather_tag = {
            "calm": "",
            "bloom": "  🌱 BLOOM",
            "blight": "  ☠ BLIGHT",
        }[world.weather]
        panel = pygame.Surface((390, 232), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 175))
        self.screen.blit(panel, (12, 12))

        lines = [
            ("EVOFORGE", self.title_font),
            (
                f"step {stats['step']:,}  "
                f"{'PAUSED' if paused else f'{speed}x'}{weather_tag}",
                self.font,
            ),
            (
                f"agents {stats['population']:,}   plants {stats['plants']:,}",
                self.font,
            ),
            (
                f"lineages {stats['lineages']:,}   "
                f"max generation {stats['max_generation']}",
                self.font,
            ),
            (
                f"records: gen {world.records['max_generation']}  "
                f"kills {int(world.records['max_kills'])}  "
                f"age {int(world.records['max_age']):,}  "
                f"size {world.records['max_radius']:.1f}",
                self.small_font,
            ),
        ]

        y = 22
        for text, font in lines:
            surface = font.render(text, True, (225, 230, 238))
            self.screen.blit(surface, (24, y))
            y += 28

        agent = world.agents.get(selected_id) if selected_id else None
        if agent:
            t = agent.genome.traits
            details = [
                f"selected #{agent.id} lineage {agent.lineage_id}",
                f"gen {agent.generation} age {agent.age} children {agent.children}",
                f"energy {agent.energy:.1f}/{agent.max_energy:.1f} kills {agent.kills}",
                f"diet {t.diet:.2f} speed {t.max_speed:.2f} vision {t.vision_range:.0f}",
            ]
            for text in details:
                surface = self.small_font.render(text, True, (255, 225, 130))
                self.screen.blit(surface, (24, y))
                y += 20

    def draw_graph(self, world: World, width: int, height: int) -> None:
        rows = list(world.telemetry.rows)
        if len(rows) < 2:
            return

        graph_w = min(620, width - 40)
        graph_h = 180
        x0 = width - graph_w - 18
        y0 = height - graph_h - 18

        panel = pygame.Surface((graph_w, graph_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 185))
        self.screen.blit(panel, (x0, y0))

        max_pop = max(max(r.population, r.plants // 3) for r in rows) or 1

        pop_points = []
        plant_points = []
        for i, row in enumerate(rows):
            x = x0 + int(i / max(1, len(rows) - 1) * (graph_w - 20)) + 10
            pop_y = y0 + graph_h - 10 - int(row.population / max_pop * (graph_h - 30))
            plant_y = y0 + graph_h - 10 - int((row.plants / 3) / max_pop * (graph_h - 30))
            pop_points.append((x, pop_y))
            plant_points.append((x, plant_y))

        if len(pop_points) > 1:
            pygame.draw.lines(self.screen, (245, 165, 75), False, pop_points, 2)
            pygame.draw.lines(self.screen, (80, 200, 100), False, plant_points, 2)

        label = self.small_font.render(
            "orange: agents   green: plants / 3",
            True,
            (210, 215, 220),
        )
        self.screen.blit(label, (x0 + 10, y0 + 7))

    def draw_help(self, width: int, height: int) -> None:
        lines = [
            "CONTROLS",
            "Space pause   . step   1-5 speed",
            "Left click inspect   right click plant",
            "Mouse wheel zoom   middle-drag pan",
            "F follow   X spotlight (auto-follow drama)   G graph   W wrap",
            "S save   L load   T export telemetry   M mute",
            "R restart   H help   Esc quit",
        ]
        panel_w = 470
        panel_h = 195
        x = width // 2 - panel_w // 2
        y = height // 2 - panel_h // 2
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((2, 4, 7, 230))
        self.screen.blit(panel, (x, y))
        pygame.draw.rect(self.screen, (90, 110, 140), (x, y, panel_w, panel_h), 1)

        for i, text in enumerate(lines):
            font = self.title_font if i == 0 else self.font
            rendered = font.render(text, True, (235, 238, 242))
            self.screen.blit(rendered, (x + 20, y + 15 + i * 22))
