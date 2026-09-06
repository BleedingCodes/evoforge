from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygame

from .config import WorldConfig
from .render import Camera, Renderer
from .world import World


SAVE_PATH = Path("evoforge_save.json")
TELEMETRY_PATH = Path("evoforge_telemetry.csv")


def _make_tone(frequency: float, duration: float, volume: float) -> "pygame.mixer.Sound | None":
    """Generate a short procedural chime — no asset files needed."""
    try:
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(frequency * math.tau * t) * np.exp(-3.0 * t / duration)
        stereo = np.repeat((wave * volume * 32767).astype(np.int16).reshape(-1, 1), 2, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))
    except Exception:
        return None


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
        self.pending_events: list[tuple[str, float, float]] = []

        self.spotlight = False
        self.spotlight_timer = 0

        self.muted = False
        self.sounds: dict[str, "pygame.mixer.Sound | None"] = {}
        try:
            pygame.mixer.init()
            self.sounds = {
                "birth": _make_tone(660.0, 0.12, 0.18),
                "kill": _make_tone(160.0, 0.22, 0.25),
            }
        except Exception:
            self.muted = True

    def play_sound(self, kind: str) -> None:
        if self.muted:
            return
        sound = self.sounds.get(kind)
        if sound is not None:
            sound.play()

    def pick_notable_agent(self) -> int | None:
        """Choose whichever living agent is currently most worth watching."""
        agents = list(self.world.agents.values())
        if not agents:
            return None
        # Rotate the definition of "notable" so the spotlight doesn't just
        # camp on one apex predator forever.
        criteria = [
            lambda a: a.kills,
            lambda a: a.age,
            lambda a: a.radius,
            lambda a: a.children,
        ]
        key = criteria[(self.world.step_count // 240) % len(criteria)]
        return max(agents, key=key).id

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
            elif event.key == pygame.K_x:
                self.spotlight = not self.spotlight
                self.spotlight_timer = 0
            elif event.key == pygame.K_m:
                self.muted = not self.muted
            elif event.key == pygame.K_g:
                self.show_graph = not self.show_graph
            elif event.key == pygame.K_h:
                self.show_help = not self.show_help
            elif event.key == pygame.K_w:
                self.world.config.wrap_world = not self.world.config.wrap_world
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

        self.pending_events = self.world.drain_events()
        for kind, _x, _y in self.pending_events:
            if kind in ("birth", "kill"):
                self.play_sound(kind)

        if self.spotlight:
            self.spotlight_timer -= 1
            if self.spotlight_timer <= 0 or self.selected_id not in self.world.agents:
                self.selected_id = self.pick_notable_agent()
                self.spotlight_timer = 240
            self.follow_selected = True

        if self.follow_selected and self.selected_id:
            agent = self.world.agents.get(self.selected_id)
            if agent:
                self.camera.x = agent.x
                self.camera.y = agent.y
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
                events=self.pending_events,
            )
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
