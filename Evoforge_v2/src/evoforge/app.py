from __future__ import annotations

import math
from pathlib import Path

import pygame

from .config import WorldConfig
from .render import Camera, Renderer
from .world import World


SAVE_PATH = Path("evoforge_save.json")
TELEMETRY_PATH = Path("evoforge_telemetry.csv")


class App:
    def __init__(
        self,
        config: WorldConfig,
        *,
        seed: int | None = None,
        load_path: Path | None = None,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("EvoForge — Artificial Life Laboratory")
        self.screen = pygame.display.set_mode(
            (min(config.width, 1600), min(config.height, 1000)),
            pygame.RESIZABLE,
        )
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.base_config = config
        self.seed = seed
        self.world = World.load(load_path) if load_path else World(config, seed)
        self.camera = Camera(
            x=self.world.config.width / 2,
            y=self.world.config.height / 2,
            zoom=1.0,
        )
        self.selected_id: int | None = None
        self.follow_selected = False
        self.show_graph = True
        self.show_help = True
        self.paused = False
        self.speed = 1
        self.running = True
        self.dragging = False
        self.last_mouse = (0, 0)
        self.documentary = False
        self.doc_timer = 0
        self.doc_caption = ""

    def nearest_agent_at(self, sx: int, sy: int) -> int | None:
        width, height = self.screen.get_size()
        wx, wy = self.camera.screen_to_world(sx, sy, width, height)
        best_id = None
        best_d2 = (22.0 / self.camera.zoom) ** 2

        for agent in self.world.agent_space.query(
            wx,
            wy,
            30.0 / self.camera.zoom,
        ):
            dx = agent.x - wx
            dy = agent.y - wy
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_id = agent.id

        return best_id

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_PERIOD and self.paused:
                self.world.step()
            elif pygame.K_1 <= event.key <= pygame.K_5:
                self.speed = 2 ** (event.key - pygame.K_1)
            elif event.key == pygame.K_f:
                self.follow_selected = not self.follow_selected
            elif event.key == pygame.K_g:
                self.show_graph = not self.show_graph
            elif event.key == pygame.K_h:
                self.show_help = not self.show_help
            elif event.key == pygame.K_w:
                self.world.config.wrap_world = not self.world.config.wrap_world
            elif event.key == pygame.K_d:
                self.documentary = not self.documentary
                if self.documentary:
                    self.follow_selected = True
                    self.doc_timer = 0
                else:
                    self.doc_caption = ""
            elif event.key == pygame.K_s:
                self.world.save(SAVE_PATH)
            elif event.key == pygame.K_l and SAVE_PATH.exists():
                self.world = World.load(SAVE_PATH)
                self.selected_id = None
            elif event.key == pygame.K_t:
                self.world.telemetry.export_csv(TELEMETRY_PATH)
            elif event.key == pygame.K_r:
                self.world = World(self.base_config, self.seed)
                self.selected_id = None

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.selected_id = self.nearest_agent_at(*event.pos)
            elif event.button == 2:
                self.dragging = True
                self.last_mouse = event.pos
            elif event.button == 3:
                width, height = self.screen.get_size()
                wx, wy = self.camera.screen_to_world(
                    *event.pos,
                    width,
                    height,
                )
                self.world.spawn_plant(
                    x=wx % self.world.config.width,
                    y=wy % self.world.config.height,
                    energy=35.0,
                )
            elif event.button == 4:
                self.camera.zoom = min(4.0, self.camera.zoom * 1.12)
            elif event.button == 5:
                self.camera.zoom = max(0.22, self.camera.zoom / 1.12)

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            dx = event.pos[0] - self.last_mouse[0]
            dy = event.pos[1] - self.last_mouse[1]
            self.camera.x -= dx / self.camera.zoom
            self.camera.y -= dy / self.camera.zoom
            self.last_mouse = event.pos

        elif event.type == pygame.MOUSEWHEEL:
            factor = 1.12 ** event.y
            self.camera.zoom = max(0.22, min(4.0, self.camera.zoom * factor))

    def update(self) -> None:
        if not self.paused:
            for _ in range(self.speed):
                self.world.step()

        if self.documentary and not self.paused:
            self.doc_timer -= 1
            agent = self.world.agents.get(self.selected_id) if self.selected_id else None
            if self.doc_timer <= 0 or agent is None or not agent.alive:
                pick = self.world.pick_notable_agent()
                if pick:
                    notable, caption = pick
                    self.selected_id = notable.id
                    self.doc_caption = caption
                    self.doc_timer = int(self.world.rng.integers(180, 360))

        if self.follow_selected and self.selected_id:
            agent = self.world.agents.get(self.selected_id)
            if agent:
                self.camera.x += (agent.x - self.camera.x) * 0.08
                self.camera.y += (agent.y - self.camera.y) * 0.08
            else:
                self.selected_id = None

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)

            self.update()
            self.renderer.draw(
                self.world,
                self.camera,
                self.selected_id,
                show_graph=self.show_graph,
                show_help=self.show_help,
                paused=self.paused,
                speed=self.speed,
                caption=self.doc_caption if self.documentary else "",
            )
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
