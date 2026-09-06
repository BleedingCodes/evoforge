from __future__ import annotations

import colorsys
import math
from collections import Counter
from dataclasses import dataclass

import pygame

from .entities import Agent
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


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)
        self.title_font = pygame.font.SysFont("consolas", 22, bold=True)

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
    ) -> None:
        width, height = self.screen.get_size()
        self.screen.fill((8, 10, 14))

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

        for agent in world.agents.values():
            self.draw_agent(agent, camera, width, height, agent.id == selected_id)

        self.draw_hud(world, selected_id, paused, speed)

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
            return

        traits = agent.genome.traits
        color = hsv_color(traits.color_h, traits.color_s, traits.color_v)
        radius = max(2, int(agent.radius * camera.zoom))
        pygame.draw.circle(self.screen, color, (sx, sy), radius)

        direction = (
            sx + int(math.cos(agent.angle) * radius * 1.5),
            sy + int(math.sin(agent.angle) * radius * 1.5),
        )
        pygame.draw.line(self.screen, (240, 240, 240), (sx, sy), direction, 1)

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

    def draw_hud(
        self,
        world: World,
        selected_id: int | None,
        paused: bool,
        speed: int,
    ) -> None:
        stats = world.stats()
        panel = pygame.Surface((390, 208), pygame.SRCALPHA)
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
            "F follow   G graph   W wrap",
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
