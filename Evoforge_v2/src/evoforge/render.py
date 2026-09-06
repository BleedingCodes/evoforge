from __future__ import annotations

import colorsys
import math
from collections import Counter, deque
from dataclasses import dataclass

import pygame

from .entities import Agent
from .genome import species_name
from .mathutil import clamp as clampf
from .world import World


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


def lerp_color(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


NIGHT_BG = (5, 6, 11)
DAY_BG = (15, 17, 24)
TRAIL_LENGTH = 12


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)
        self.title_font = pygame.font.SysFont("consolas", 22, bold=True)
        self.trails: dict[int, deque[tuple[int, int]]] = {}

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
        caption: str = "",
    ) -> None:
        width, height = self.screen.get_size()
        daylight = world.daylight()
        self.screen.fill(lerp_color(NIGHT_BG, DAY_BG, daylight))

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

        live_ids = set()
        for agent in world.agents.values():
            live_ids.add(agent.id)
            self.draw_agent(agent, camera, width, height, agent.id == selected_id)

        if len(self.trails) > len(live_ids) + 64:
            for stale_id in list(self.trails.keys() - live_ids):
                del self.trails[stale_id]

        self.draw_flashes(world, camera, width, height)
        self.draw_hud(world, selected_id, paused, speed)
        self.draw_events(world, width, height)

        if caption:
            self.draw_caption(caption, width, height)

        if show_graph:
            self.draw_graph(world, width, height)

        if show_help:
            self.draw_help(width, height)

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
            # Still record ground truth so a trail resumes cleanly on-screen.
            trail = self.trails.setdefault(agent.id, deque(maxlen=TRAIL_LENGTH))
            trail.append((sx, sy))
            return

        traits = agent.genome.traits
        energy_ratio = clampf(agent.energy / max(agent.max_energy, 1.0), 0.0, 1.0)
        value = clampf(traits.color_v * (0.35 + 0.65 * energy_ratio), 0.15, 1.0)
        color = hsv_color(traits.color_h, traits.color_s, value)
        radius = max(2, int(agent.radius * camera.zoom))

        # --- motion trail: a dim comet-tail of recent positions ---
        trail = self.trails.setdefault(agent.id, deque(maxlen=TRAIL_LENGTH))
        trail.append((sx, sy))
        if len(trail) >= 2:
            trail_color = (
                max(0, int(color[0] * 0.32)),
                max(0, int(color[1] * 0.32)),
                max(0, int(color[2] * 0.32)),
            )
            pygame.draw.lines(self.screen, trail_color, False, list(trail), 1)

        # --- tail: streamlined for fast agents, drawn opposite travel ---
        tail_len = (2.0 + traits.max_speed * 4.5) * camera.zoom
        tail_x = sx - math.cos(agent.angle) * tail_len
        tail_y = sy - math.sin(agent.angle) * tail_len
        pygame.draw.line(
            self.screen,
            tuple(max(0, c - 60) for c in color),
            (sx, sy),
            (int(tail_x), int(tail_y)),
            max(1, int(radius * 0.5)),
        )

        # --- spikes: aggressive predators look visibly dangerous ---
        if traits.aggression > 0.45:
            spike_count = 3 + int(traits.aggression * 5)
            spike_len = radius * (0.5 + traits.aggression * 1.1)
            for i in range(spike_count):
                spike_angle = agent.angle + (i / spike_count) * math.tau
                bx = sx + math.cos(spike_angle) * radius
                by = sy + math.sin(spike_angle) * radius
                tx = sx + math.cos(spike_angle) * (radius + spike_len)
                ty = sy + math.sin(spike_angle) * (radius + spike_len)
                pygame.draw.line(
                    self.screen,
                    tuple(min(255, c + 30) for c in color),
                    (int(bx), int(by)),
                    (int(tx), int(ty)),
                    1,
                )

        # --- reproduction glow: a soft pulse when ready to breed ---
        if agent.reproduce_signal > 0.55:
            pulse = radius + 3 + int(2 * math.sin(agent.age * 0.3))
            pygame.draw.circle(self.screen, (200, 255, 170), (sx, sy), pulse, 1)

        pygame.draw.circle(self.screen, color, (sx, sy), radius)

        # --- eye: bigger for sharper-eyed organisms, set toward heading ---
        eye_radius = max(1, int((radius * 0.28 + traits.vision_range * 0.01) * 1.0))
        eye_offset = radius * 0.5
        ex = sx + math.cos(agent.angle) * eye_offset
        ey = sy + math.sin(agent.angle) * eye_offset
        pygame.draw.circle(self.screen, (15, 15, 20), (int(ex), int(ey)), eye_radius)

        if traits.diet > 0.67:
            pygame.draw.circle(self.screen, (235, 70, 70), (sx, sy), radius, 1)
        elif traits.diet < 0.33:
            pygame.draw.circle(self.screen, (80, 235, 110), (sx, sy), radius, 1)

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
            label = self.small_font.render(
                species_name(traits),
                True,
                (255, 230, 140),
            )
            self.screen.blit(label, (sx + radius + 8, sy - radius - 6))

    def draw_flashes(
        self,
        world: World,
        camera: Camera,
        width: int,
        height: int,
    ) -> None:
        max_age = 18
        for step, x, y, kind in world.flashes:
            age = world.step_count - step
            if age < 0 or age > max_age:
                continue
            sx, sy = camera.world_to_screen(x, y, width, height)
            if not (-20 <= sx <= width + 20 and -20 <= sy <= height + 20):
                continue
            fade = 1.0 - age / max_age
            ring_radius = max(1, int((3 + age * 1.6) * camera.zoom))
            base_color = (110, 235, 140) if kind == "birth" else (235, 90, 70)
            surface = pygame.Surface((ring_radius * 2 + 4, ring_radius * 2 + 4), pygame.SRCALPHA)
            color = (*base_color, max(0, int(220 * fade)))
            pygame.draw.circle(
                surface,
                color,
                (ring_radius + 2, ring_radius + 2),
                ring_radius,
                2,
            )
            self.screen.blit(surface, (sx - ring_radius - 2, sy - ring_radius - 2))

    def draw_caption(self, caption: str, width: int, height: int) -> None:
        panel = pygame.Surface((width, 40), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.screen.blit(panel, (0, height - 40))
        text = self.font.render(f"🎥 {caption}", True, (255, 235, 200))
        self.screen.blit(text, (width // 2 - text.get_width() // 2, height - 30))

    def draw_events(self, world: World, width: int, height: int) -> None:
        rows = list(world.events)
        if not rows:
            return
        rows = rows[-6:]
        panel_w = 420
        panel_h = 20 + 18 * len(rows)
        x = width - panel_w - 12
        y = 12
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.screen.blit(panel, (x, y))

        for i, (step, text) in enumerate(reversed(rows)):
            fade = 255 - i * 30
            color = (min(255, max(80, fade)), min(255, max(80, fade)), min(255, max(90, fade)))
            surface = self.small_font.render(f"t{step:,}  {text}", True, color)
            self.screen.blit(surface, (x + 10, y + 8 + i * 18))

    def draw_hud(
        self,
        world: World,
        selected_id: int | None,
        paused: bool,
        speed: int,
    ) -> None:
        stats = world.stats()
        panel = pygame.Surface((390, 250), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 175))
        self.screen.blit(panel, (12, 12))

        lines = [
            ("EVOFORGE", self.title_font),
            (
                f"step {stats['step']:,}  "
                f"{'PAUSED' if paused else f'{speed}x'}",
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
                f"{species_name(t)}",
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
            "F follow   G graph   W wrap   D documentary mode",
            "S save   L load   T export telemetry",
            "R restart   H help   Esc quit",
        ]
        panel_w = 470
        panel_h = 175
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
