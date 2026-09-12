# EvoForge

A 2D artificial life simulator where neural-network agents evolve movement, vision, diet, and aggression across generations — with zero scripted behavior. Every behavioral pattern that emerges is a product of selection pressure: energy, food, predation, reproduction cost.

---

## What It Does

EvoForge drops a population of agents into a 2D environment and runs selection pressure until behavioral strategies emerge on their own.

- **Neural-network brains** — each agent is controlled by a small feedforward network. Weights are the genome.
- **Evolved sensory systems** — agents develop vision range, field of view, and input sensitivity through selection, not design.
- **Diet and aggression as traits** — herbivorous, omnivorous, and predatory strategies emerge from energy math, not rules.
- **Reproduction with mutation** — successful agents reproduce, passing mutated weights to offspring. No crossover — single-parent heredity keeps lineage traceable.
- **Real-time visualization** — watch the population evolve live. Agent color encodes diet bias; size encodes energy state.
- **Three versioned architectures** — the repo documents three distinct design iterations, each solving problems introduced by the last.

There is no fitness function being optimized. There is no target behavior. Agents that fail to acquire energy die. Agents that acquire energy reproduce. That's the entire engine.

---

## How the Simulation Works

### The Environment
A bounded 2D world populated with plants that respawn at a configurable rate. World wrapping is toggleable at runtime.

### Agents
Each agent has:
- A position and velocity vector
- A neural network (input → hidden → output)
- An energy level that decays every tick
- A genome (the network's weights)
- A generation counter, kill count, and child count

### Sensory Inputs (network inputs)
- Nearest food: angle, distance
- Nearest agent: angle, distance, energy delta (relative size)
- Self: current energy level, current velocity

### Motor Outputs (network outputs)
- Turn rate
- Speed (throttle)
- Attack flag (binary — initiates energy transfer from target to self)

### Selection Pressure
- Moving costs energy (proportional to speed)
- Existing costs energy (baseline metabolic drain)
- Eating plants restores energy
- Attacking an agent transfers energy if the attacker wins (resolved by size differential)
- Agents with energy above threshold reproduce (one offspring, mutated weights, half energy transferred)
- Agents at zero energy are removed

No fitness function. No tournament selection. No epochs. The simulation runs continuously and the population self-regulates.

---

## Three Architecture Versions

EvoForge was built in three documented passes. Each version is preserved in the repo.

### Evoforge_v1 — Component Decomposition
Separated into distinct modules: `world`, `entities`, `genome`, `render`, `config`, `spatial`, `telemetry`. State passed explicitly. Configuration externalized to a dataclass. Proved the core loop and validated the neural-network + mutation approach.

**What it revealed:** Agent-world interaction needed cleaner authority boundaries as predation complexity grew.

### Evoforge_v2 — Refined Internals
Iterative improvement on v1. Bug fixes, tightened module boundaries, expanded test coverage (`test_genome`, `test_spatial`, `test_world`). Same module structure, cleaner execution.

### Evoforge_v2.2 — Current
The production version. `App` owns the game loop and event handling. `World` owns physics and state resolution. `genome.py` handles mutation. `spatial.py` handles spatial queries. `telemetry.py` handles stats export. Brain is a pure function — no side effects, fully testable in isolation.

**What this enables:** Deterministic replay via seed, save/load, CSV telemetry export, and headless benchmarking without touching rendering code.

---

## Requirements

- Python 3.11+
- `pygame` — rendering
- `numpy` — matrix operations for neural network forward pass

Install from the repo root (editable install, includes all dependencies):

```bash
pip install -e .
```

---

## Installation

```bash
git clone https://github.com/BleedingCodes/EvoForge
cd EvoForge/Evoforge_v2.2
pip install -e .
```

---

## Running the Simulation

**Launch the interactive simulation:**
```bash
evoforge run
```

**With custom parameters:**
```bash
evoforge run --width 1400 --height 900 --agents 220 --plants 700 --seed 42
```

**Load a saved state:**
```bash
evoforge run --load evoforge_save.json
```

**Headless benchmark (no renderer — maximum speed):**
```bash
evoforge benchmark --steps 5000 --agents 500 --plants 1500 --seed 1
```

Benchmark outputs a JSON summary: step count, elapsed time, steps/second, and final population stats.

---

### Runtime Controls

| Key / Button | Action |
|---|---|
| `Space` | Pause / resume |
| `1`–`5` | Set simulation speed (1×, 2×, 4×, 8×, 16×) |
| `.` (period) | Step one tick while paused |
| `F` | Follow selected agent |
| `X` | Toggle spotlight mode (auto-follows notable agents) |
| `G` | Toggle population graph |
| `H` | Toggle help overlay |
| `W` | Toggle world wrapping |
| `M` | Mute / unmute |
| `S` | Save world state to `evoforge_save.json` |
| `L` | Load saved world state |
| `T` | Export telemetry to `evoforge_telemetry.csv` |
| `R` | Reset simulation with same config and seed |
| `Esc` | Quit |
| Left click | Select nearest agent |
| Middle click + drag | Pan camera |
| Right click | Spawn plant at cursor |
| Scroll wheel | Zoom in / out |

---

## Configuration

All parameters are set via CLI flags on `evoforge run` or `evoforge benchmark`.

| Flag | Default | Description |
|---|---|---|
| `--width` | 1400 | World width in pixels |
| `--height` | 900 | World height in pixels |
| `--agents` | 220 | Starting agent count |
| `--plants` | 700 | Starting plant count |
| `--seed` | (random) | RNG seed for reproducible runs |
| `--load` | — | Path to a saved `.json` world state |
| `--steps` | 5000 | Steps to run (benchmark only) |

---

## What Makes This Architecturally Interesting

**The brain is a pure function.** No state, no side effects. Brains are serializable, testable in isolation, and swappable without touching agent or world logic.

**The world owns resolution.** Agents submit intent each tick; the world resolves everything in a single pass — movement, collisions, feeding, combat, reproduction. No agent has direct knowledge of another agent. This eliminates update-order bugs and makes the simulation fully deterministic given the same seed.

**`App` and `World` are cleanly separated.** `World` has no knowledge of pygame, rendering, or input. `App` drives the loop and handles all I/O. This is what makes headless benchmarking (`evoforge benchmark`) possible with zero code changes — just skip the `App` and drive `World.step()` directly.

**Mutation without crossover is a deliberate choice.** Single-parent heredity keeps the evolutionary tree traceable. You can follow a lineage from generation 1 forward and see exactly which mutations accumulated. Sexual recombination would make lineage analysis intractable.

**No fitness function means no Goodhart's Law.** Systems optimized against a fixed fitness function find ways to maximize the metric rather than solve the underlying problem. EvoForge has no metric to exploit — only the environment.

---

## License

MIT License — see `LICENSE` file.

---

## Case Study

A full architectural walkthrough — covering design decisions, the v1→v2→v2.2 evolution, and lessons learned — is documented in the MainbyteLabs technical portfolio:

[EvoForge Case Study → MR-MainbyteLabs/technical-docs-portfolio](https://github.com/MR-MainbyteLabs/technical-docs-portfolio/tree/main/evoforge-case-study)

---

*Built by [MainbyteLabs](https://github.com/MR-MainbyteLabs) — Python tooling and documentation for electronics labs and hardware teams.*
