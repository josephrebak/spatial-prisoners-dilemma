# Spatial Prisoner's Dilemma

A cellular automaton simulation of evolutionary game theory's most famous puzzle: can cooperation survive in a world that rewards defection?

This project implements the **Spatial Prisoner's Dilemma** as a synchronous 2D grid simulation, complete with configurable neighborhoods, boundary conditions, payoff parameters, and publication-quality visualizations. It replicates and extends the landmark results from [Nowak & May (1992)](https://www.nature.com/articles/359826a0), which showed that spatial structure alone can sustain cooperation indefinitely.

---

## The Game

In the classic Prisoner's Dilemma, two players each choose to **cooperate** or **defect**. Mutual cooperation yields a modest reward; mutual defection yields nothing; but a defector exploiting a cooperator earns a bonus `b > 1`. Rational self-interest always favors defection — yet cooperation persists everywhere in nature.

The spatial version asks: what happens when thousands of agents play repeatedly with their *neighbors*, and copy the strategy of whichever neighbor scored highest?

| | Cooperator | Defector |
|---|---|---|
| **Cooperator** | 1 / 1 | 0 / b |
| **Defector** | b / 0 | 0 / 0 |

The `b` parameter (temptation to defect) is the key control variable. Small `b` → cooperation dominates. Large `b` → defection sweeps. Near the critical threshold, complex spatial patterns emerge.

---

## Emergent Dynamics

The simulation reveals behaviors invisible in non-spatial models:

- **Cooperation clusters** form protective enclaves — defectors on the boundary exhaust their cooperator neighbors while the interior remains shielded
- **Phase transitions** occur at critical `b` values where cooperation abruptly collapses
- **Boundary conditions matter** — periodic (toroidal) grids show different steady states than open or fixed-border grids
- **Neighborhood topology** (Moore 8-cell vs Von Neumann 4-cell) shifts the critical threshold

The transition coloring scheme (C→C blue, D→D red, C→D yellow, D→C green) makes these dynamics visually legible at a glance.

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

Each timestep:
1. Every cell accumulates a payoff from interactions with all neighbors
2. Every cell surveys its neighborhood and finds the highest-scoring cell
3. Every cell adopts the strategy of that highest-scoring cell (ties broken randomly)
4. All updates happen synchronously

This **best-neighbor imitation** rule is what produces the rich spatial dynamics. Clusters of cooperators are stable because interior cells score higher than the defectors preying on their edges.

---

## References

- Nowak, M. A. & May, R. M. (1992). [Evolutionary games and spatial chaos](https://www.nature.com/articles/359826a0). *Nature*, 359, 826–829.
- Axelrod, R. (1984). *The Evolution of Cooperation*. Basic Books.
- Szabó, G. & Fáth, G. (2007). [Evolutionary games on graphs](https://www.sciencedirect.com/science/article/pii/S0370157307000124). *Physics Reports*, 446(4–6), 97–216.
