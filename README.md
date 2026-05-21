# Spatial Prisoner's Dilemma

A cellular automaton simulation of prisoner's dilemma on a 2D grid. This was my final project for EN.605.716 Modeling and Simulation of Complex Systems. It replicates and extends the results from [Nowak & May (1992)](https://www.nature.com/articles/359826a0), which showed that spatial structure alone can sustain cooperation indefinitely.

---

## The Game

In the classic Prisoner's Dilemma, two players each choose to **cooperate** or **defect**. Mutual cooperation yields a modest reward; mutual defection yields nothing; but a defector exploiting a cooperator earns a bonus `b > 1`.

The spatial version asks: what happens when thousands of agents play repeatedly with their *neighbors*, and copy the strategy of whichever neighbor scored highest?

| | Cooperator | Defector |
|---|---|---|
| **Cooperator** | 1 / 1 | 0 / b |
| **Defector** | b / 0 | 0 / 0 |

The `b` parameter (temptation to defect) is the key variable. Small `b` means cooperation dominates. Large `b` means defection dominates. Near the critical threshold (1.8 < b < 2.0), complex spatial patterns emerge.

---

## Quickstart

```bash
pip install numpy matplotlib
python spatialpd.py
```

The simulation is controlled via four experiment functions at the bottom of `spatialpd.py`:

```python
# Single run: visualize one b value
experiment_single_b(b=1.9, n=100, steps=200)

# Phase diagram: sweep across b values
experiment_vary_b(b_values=[1.5, 1.7, 1.9, 2.1], n=100, steps=200)

# Compare boundary conditions
experiment_boundary_conditions(b=1.9, n=100, steps=200)

# Compare Moore vs Von Neumann neighborhoods  
experiment_neighborhoods(b=1.9, n=100, steps=200)
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `n` | 100 | Grid size (n × n cells) |
| `steps` | 200 | Simulation timesteps |
| `b` | 1.9 | Temptation-to-defect bonus |
| `p_cooperator` | 0.9 | Initial fraction of cooperators |
| `neighborhood` | `'moore'` | `'moore'` (8-cell) or `'von_neumann'` (4-cell) |
| `boundary` | `'periodic'` | `'periodic'`, `'cutoff'`, or `'fixed'` |
| `self_interaction` | `True` | Whether cells include themselves in payoff |

---

## How It Works

In the beginning of the simulation, cell is initialized to be a cooperator or a defector. Long-term behavior is independent of initial percentage of cooperators, so I chose `p_cooperator = 0.9`, which is what Nowak and May did.

Each timestep:
1. Every cell accumulates a payoff from playing PD with all neighbors
2. Every cell surveys its neighborhood and finds the highest-scoring cell
3. Every cell adopts the strategy of that highest-scoring cell (ties broken randomly)
4. All updates happen synchronously

---

## References

- Nowak, M. A. & May, R. M. (1992). [Evolutionary games and spatial chaos](https://www.nature.com/articles/359826a0). *Nature*, 359, 826–829.
