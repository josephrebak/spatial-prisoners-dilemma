import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================
# Spatial Prisoner's Dilemma as a Cellular Automaton
#
# States:
#   1 = C (cooperator)
#   0 = D (defector)
#
# Payoff matrix:
#   C vs C -> 1
#   C vs D -> 0
#   D vs C -> b
#   D vs D -> 0
#
# Update rule:
#   Each cell computes its total payoff from local interactions.
#   Then each cell copies the strategy of the highest-scoring
#   cell in its neighborhood (including itself).
#
# Paper-style transition coloring:
#   C -> C : blue
#   D -> D : red
#   C -> D : yellow
#   D -> C : green
# ============================================================


@dataclass
class SimulationResult:
    initial_grid: np.ndarray
    final_grid: np.ndarray
    fc_history: np.ndarray
    score_history: Optional[List[np.ndarray]] = None
    transition_frames: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None


def make_initial_grid(
    n: int,
    p_cooperator: float = 0.9,
    seed: Optional[int] = None,
    mode: str = "random"
) -> np.ndarray:
    """
    Create initial grid.

    mode:
        - "random": each cell is C with probability p_cooperator
        - "single_defector": all C except one D at center
        - "single_cooperator": all D except one C at center
    """
    rng = np.random.default_rng(seed)

    if mode == "random":
        return (rng.random((n, n)) < p_cooperator).astype(np.int8)

    if mode == "single_defector":
        grid = np.ones((n, n), dtype=np.int8)
        grid[n // 2, n // 2] = 0
        return grid

    if mode == "single_cooperator":
        grid = np.zeros((n, n), dtype=np.int8)
        grid[n // 2, n // 2] = 1
        return grid

    raise ValueError(f"Unknown mode: {mode}")


def get_neighbor_offsets(neighborhood: str, include_self: bool = True) -> List[Tuple[int, int]]:
    """
    Return offsets for neighborhood used in both interactions and imitation.

    neighborhood:
        - "moore": 8 neighbors
        - "von_neumann": 4 orthogonal neighbors
    """
    if neighborhood == "moore":
        offsets = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                   if not (di == 0 and dj == 0)]
    elif neighborhood == "von_neumann":
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("neighborhood must be 'moore' or 'von_neumann'")

    if include_self:
        offsets = offsets + [(0, 0)]

    return offsets


def get_state_for_interaction(
    grid: np.ndarray,
    i: int,
    j: int,
    boundary: str,
    fixed_boundary_state: int
) -> Optional[int]:
    """
    Return state at (i,j) under chosen boundary condition.

    boundary:
        - "periodic": wrap around
        - "cutoff": outside cells do not exist -> return None
        - "fixed": outside cells have fixed_boundary_state
    """
    nrows, ncols = grid.shape

    if 0 <= i < nrows and 0 <= j < ncols:
        return int(grid[i, j])

    if boundary == "periodic":
        return int(grid[i % nrows, j % ncols])

    if boundary == "cutoff":
        return None

    if boundary == "fixed":
        return int(fixed_boundary_state)

    raise ValueError("boundary must be 'periodic', 'cutoff', or 'fixed'")


def interaction_payoff(state_a: int, state_b: int, b: float) -> float:
    """
    Return payoff to A when A interacts with B.
    """
    if state_a == 1 and state_b == 1:   # C vs C
        return 1.0
    if state_a == 1 and state_b == 0:   # C vs D
        return 0.0
    if state_a == 0 and state_b == 1:   # D vs C
        return b
    if state_a == 0 and state_b == 0:   # D vs D
        return 0.0
    raise ValueError("States must be 0 or 1")


def compute_scores(
    grid: np.ndarray,
    b: float,
    neighborhood: str = "moore",
    boundary: str = "cutoff",
    include_self_interaction: bool = True,
    fixed_boundary_state: int = 1
) -> np.ndarray:
    """
    Compute total payoff for each cell from interactions with neighbors.
    """
    nrows, ncols = grid.shape
    scores = np.zeros((nrows, ncols), dtype=float)

    interaction_offsets = get_neighbor_offsets(
        neighborhood=neighborhood,
        include_self=include_self_interaction
    )

    for i in range(nrows):
        for j in range(ncols):
            my_state = int(grid[i, j])
            total = 0.0

            for di, dj in interaction_offsets:
                ni, nj = i + di, j + dj
                nbr_state = get_state_for_interaction(
                    grid, ni, nj, boundary, fixed_boundary_state
                )
                if nbr_state is None:
                    continue
                total += interaction_payoff(my_state, nbr_state, b)

            scores[i, j] = total

    return scores


def update_grid(
    grid: np.ndarray,
    scores: np.ndarray,
    neighborhood: str = "moore",
    boundary: str = "cutoff",
    fixed_boundary_state: int = 1
) -> np.ndarray:
    """
    Update synchronously:
    each cell copies the state of the highest-scoring cell
    in its neighborhood (including itself).
    """
    nrows, ncols = grid.shape
    new_grid = np.zeros_like(grid)

    imitation_offsets = get_neighbor_offsets(
        neighborhood=neighborhood,
        include_self=True
    )

    for i in range(nrows):
        for j in range(ncols):
            best_score = -np.inf
            best_state = int(grid[i, j])

            for di, dj in imitation_offsets:
                ni, nj = i + di, j + dj

                if boundary == "periodic":
                    ci, cj = ni % nrows, nj % ncols
                    candidate_score = scores[ci, cj]
                    candidate_state = grid[ci, cj]
                else:
                    if not (0 <= ni < nrows and 0 <= nj < ncols):
                        continue
                    candidate_score = scores[ni, nj]
                    candidate_state = grid[ni, nj]

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_state = int(candidate_state)

            new_grid[i, j] = best_state

    return new_grid


def simulate_spatial_pd(
    n: int = 100,
    steps: int = 200,
    b: float = 1.9,
    p_cooperator: float = 0.9,
    neighborhood: str = "moore",
    boundary: str = "cutoff",
    include_self_interaction: bool = True,
    fixed_boundary_state: int = 1,
    seed: Optional[int] = None,
    init_mode: str = "random",
    keep_score_history: bool = False
) -> SimulationResult:
    """
    Run simulation and return full result.
    """
    grid = make_initial_grid(n, p_cooperator=p_cooperator, seed=seed, mode=init_mode)
    initial_grid = grid.copy()

    fc_history = [grid.mean()]
    score_history = [] if keep_score_history else None
    transition_frames = []

    for _ in range(steps):
        prev_grid = grid.copy()

        scores = compute_scores(
            grid=grid,
            b=b,
            neighborhood=neighborhood,
            boundary=boundary,
            include_self_interaction=include_self_interaction,
            fixed_boundary_state=fixed_boundary_state
        )

        if keep_score_history:
            score_history.append(scores.copy())

        grid = update_grid(
            grid=grid,
            scores=scores,
            neighborhood=neighborhood,
            boundary=boundary,
            fixed_boundary_state=fixed_boundary_state
        )

        transition_frames.append((prev_grid.copy(), grid.copy()))
        fc_history.append(grid.mean())

    return SimulationResult(
        initial_grid=initial_grid,
        final_grid=grid,
        fc_history=np.array(fc_history),
        score_history=score_history,
        transition_frames=transition_frames
    )


# ============================================================
# Transition coloring (paper convention)
# ============================================================

def compute_transition_colors(prev_grid: np.ndarray, curr_grid: np.ndarray) -> np.ndarray:
    """
    Returns an RGB image encoding transitions:
        C -> C : blue
        D -> D : red
        C -> D : yellow
        D -> C : green
    """
    nrows, ncols = prev_grid.shape
    rgb = np.zeros((nrows, ncols, 3), dtype=float)

    cc_mask = (prev_grid == 1) & (curr_grid == 1)   # C -> C
    dd_mask = (prev_grid == 0) & (curr_grid == 0)   # D -> D
    cd_mask = (prev_grid == 1) & (curr_grid == 0)   # C -> D
    dc_mask = (prev_grid == 0) & (curr_grid == 1)   # D -> C

    rgb[cc_mask] = [0.0, 0.0, 1.0]   # blue
    rgb[dd_mask] = [1.0, 0.0, 0.0]   # red
    rgb[cd_mask] = [1.0, 1.0, 0.0]   # yellow
    rgb[dc_mask] = [0.0, 1.0, 0.0]   # green

    return rgb


# ============================================================
# Plotting utilities
# ============================================================

def plot_grid(grid: np.ndarray, title: str = "") -> None:
    """
    Plot raw state grid.
    1 = C, 0 = D
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="State (0=D, 1=C)")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()


def plot_transition_frame(prev_grid: np.ndarray, curr_grid: np.ndarray, title: str = "") -> None:
    """
    Plot transition-colored grid using the paper's convention.
    """
    rgb = compute_transition_colors(prev_grid, curr_grid)

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()


def plot_fc(fc_history: np.ndarray, title: str = "") -> None:
    """
    Plot fraction of cooperators over time.
    """
    plt.figure(figsize=(8, 4.5))
    plt.plot(fc_history)
    plt.xlabel("Time step")
    plt.ylabel("Fraction of cooperators, $f_C(t)$")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_initial_and_final(result: SimulationResult, title_prefix: str = "") -> None:
    """
    Plot initial raw grid and final raw grid.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(result.initial_grid, cmap="viridis", interpolation="nearest")
    axes[0].set_title(f"{title_prefix} Initial")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    axes[1].imshow(result.final_grid, cmap="viridis", interpolation="nearest")
    axes[1].set_title(f"{title_prefix} Final")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    plt.tight_layout()
    plt.show()


def plot_initial_and_final_transition(result: SimulationResult, title_prefix: str = "") -> None:
    """
    Plot initial raw grid and final transition-colored frame.
    """
    if not result.transition_frames:
        raise ValueError("No transition frames available.")

    prev_grid, curr_grid = result.transition_frames[-1]
    rgb = compute_transition_colors(prev_grid, curr_grid)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(result.initial_grid, cmap="viridis", interpolation="nearest")
    axes[0].set_title(f"{title_prefix} Initial")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    axes[1].imshow(rgb, interpolation="nearest")
    axes[1].set_title(f"{title_prefix} Final Transition")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    plt.tight_layout()
    plt.show()


def plot_final_transition(result: SimulationResult, title: str = "") -> None:
    """
    Plot the final transition frame only.
    """
    if not result.transition_frames:
        raise ValueError("No transition frames available.")

    prev_grid, curr_grid = result.transition_frames[-1]
    plot_transition_frame(prev_grid, curr_grid, title=title)


# ============================================================
# Experiments
# ============================================================

def experiment_single_b(
    n: int = 100,
    steps: int = 200,
    b: float = 1.9,
    p_cooperator: float = 0.9,
    neighborhood: str = "moore",
    boundary: str = "cutoff",
    include_self_interaction: bool = True,
    fixed_boundary_state: int = 1,
    seed: Optional[int] = 0,
    init_mode: str = "random"
) -> SimulationResult:
    """
    Run one simulation and show:
      - initial / final raw grid
      - final transition-colored grid
      - f_C(t)
    """
    result = simulate_spatial_pd(
        n=n,
        steps=steps,
        b=b,
        p_cooperator=p_cooperator,
        neighborhood=neighborhood,
        boundary=boundary,
        include_self_interaction=include_self_interaction,
        fixed_boundary_state=fixed_boundary_state,
        seed=seed,
        init_mode=init_mode
    )

    title_base = (
        f"b={b}, neighborhood={neighborhood}, boundary={boundary}, "
        f"self_interaction={include_self_interaction}"
    )

    plot_initial_and_final(result, title_prefix=title_base)
    plot_final_transition(result, title=f"Final transition (paper colors)\n{title_base}")
    plot_fc(result.fc_history, title=f"$f_C(t)$ for {title_base}")

    return result


def experiment_vary_b(
    b_values: List[float],
    n: int = 100,
    steps: int = 200,
    p_cooperator: float = 0.9,
    neighborhood: str = "moore",
    boundary: str = "cutoff",
    include_self_interaction: bool = True,
    fixed_boundary_state: int = 1,
    seed: Optional[int] = 0,
    init_mode: str = "random"
) -> List[SimulationResult]:
    """
    Run simulations across multiple b values.
    Shows:
      - f_C(t) curves on one plot
      - final f_C vs b
      - final transition-colored grids in a panel
    """
    results = []
    final_fc = []

    for b in b_values:
        result = simulate_spatial_pd(
            n=n,
            steps=steps,
            b=b,
            p_cooperator=p_cooperator,
            neighborhood=neighborhood,
            boundary=boundary,
            include_self_interaction=include_self_interaction,
            fixed_boundary_state=fixed_boundary_state,
            seed=seed,
            init_mode=init_mode
        )
        results.append(result)
        final_fc.append(result.fc_history[-1])

    # Plot fc(t) for all b
    plt.figure(figsize=(9, 5))
    for b, result in zip(b_values, results):
        plt.plot(result.fc_history, label=f"b={b}")
    plt.xlabel("Time step")
    plt.ylabel("Fraction of cooperators, $f_C(t)$")
    plt.title(
        f"$f_C(t)$ across b values\n"
        f"neighborhood={neighborhood}, boundary={boundary}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot final fc vs b
    plt.figure(figsize=(7, 4.5))
    plt.plot(b_values, final_fc, marker="o")
    plt.xlabel("b")
    plt.ylabel("Final fraction of cooperators")
    plt.title("Final $f_C$ versus b")
    plt.tight_layout()
    plt.show()

    # Plot final transition-colored grids
    ncols = min(3, len(b_values))
    nrows = int(np.ceil(len(b_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, b, result in zip(axes, b_values, results):
        prev_grid, curr_grid = result.transition_frames[-1]
        rgb = compute_transition_colors(prev_grid, curr_grid)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"b={b}\nfinal $f_C$={result.fc_history[-1]:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(b_values):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return results


def experiment_boundary_conditions(
    boundaries: List[str],
    b: float = 1.9,
    n: int = 100,
    steps: int = 200,
    p_cooperator: float = 0.9,
    neighborhood: str = "moore",
    include_self_interaction: bool = True,
    fixed_boundary_state: int = 1,
    seed: Optional[int] = 0,
    init_mode: str = "random"
) -> dict:
    """
    Compare periodic / cutoff / fixed boundary conditions.
    """
    results = {}

    plt.figure(figsize=(9, 5))
    for boundary in boundaries:
        result = simulate_spatial_pd(
            n=n,
            steps=steps,
            b=b,
            p_cooperator=p_cooperator,
            neighborhood=neighborhood,
            boundary=boundary,
            include_self_interaction=include_self_interaction,
            fixed_boundary_state=fixed_boundary_state,
            seed=seed,
            init_mode=init_mode
        )
        results[boundary] = result
        plt.plot(result.fc_history, label=boundary)

    plt.xlabel("Time step")
    plt.ylabel("Fraction of cooperators, $f_C(t)$")
    plt.title(f"Boundary condition comparison at b={b}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, len(boundaries), figsize=(5 * len(boundaries), 5))
    if len(boundaries) == 1:
        axes = [axes]

    for ax, boundary in zip(axes, boundaries):
        result = results[boundary]
        prev_grid, curr_grid = result.transition_frames[-1]
        rgb = compute_transition_colors(prev_grid, curr_grid)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"{boundary}\nfinal $f_C$={result.fc_history[-1]:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    return results


def experiment_neighborhoods(
    neighborhoods: List[str],
    b: float = 1.9,
    n: int = 100,
    steps: int = 200,
    p_cooperator: float = 0.9,
    boundary: str = "cutoff",
    include_self_interaction: bool = True,
    fixed_boundary_state: int = 1,
    seed: Optional[int] = 0,
    init_mode: str = "random"
) -> dict:
    """
    Compare Moore vs Von Neumann neighborhoods.
    """
    results = {}

    plt.figure(figsize=(9, 5))
    for neighborhood in neighborhoods:
        result = simulate_spatial_pd(
            n=n,
            steps=steps,
            b=b,
            p_cooperator=p_cooperator,
            neighborhood=neighborhood,
            boundary=boundary,
            include_self_interaction=include_self_interaction,
            fixed_boundary_state=fixed_boundary_state,
            seed=seed,
            init_mode=init_mode
        )
        results[neighborhood] = result
        plt.plot(result.fc_history, label=neighborhood)

    plt.xlabel("Time step")
    plt.ylabel("Fraction of cooperators, $f_C(t)$")
    plt.title(f"Neighborhood comparison at b={b}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, len(neighborhoods), figsize=(5 * len(neighborhoods), 5))
    if len(neighborhoods) == 1:
        axes = [axes]

    for ax, neighborhood in zip(axes, neighborhoods):
        result = results[neighborhood]
        prev_grid, curr_grid = result.transition_frames[-1]
        rgb = compute_transition_colors(prev_grid, curr_grid)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"{neighborhood}\nfinal $f_C$={result.fc_history[-1]:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    return results


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # 1) Base experiment at one b
    # --------------------------------------------------------
    base_result = experiment_single_b(
        n=100,
        steps=200,
        b=1.9,
        p_cooperator=0.9,
        neighborhood="moore",
        boundary="cutoff",
        include_self_interaction=True,
        fixed_boundary_state=1,
        seed=1,
        init_mode="random"
    )

    # --------------------------------------------------------
    # 2) Vary b
    # Try values below, inside, and above the interesting regime
    # --------------------------------------------------------
    b_values = [1.6, 1.75, 1.85, 1.9, 1.95, 2.1]
    vary_b_results = experiment_vary_b(
        b_values=b_values,
        n=100,
        steps=200,
        p_cooperator=0.9,
        neighborhood="moore",
        boundary="cutoff",
        include_self_interaction=True,
        fixed_boundary_state=1,
        seed=1,
        init_mode="random"
    )

    # --------------------------------------------------------
    # 3) Boundary condition experiment
    # --------------------------------------------------------
    boundary_results = experiment_boundary_conditions(
        boundaries=["cutoff", "periodic", "fixed"],
        b=1.9,
        n=100,
        steps=200,
        p_cooperator=0.9,
        neighborhood="moore",
        include_self_interaction=True,
        fixed_boundary_state=1,
        seed=1,
        init_mode="random"
    )

    # --------------------------------------------------------
    # 4) Neighborhood experiment
    # --------------------------------------------------------
    neighborhood_results = experiment_neighborhoods(
        neighborhoods=["moore", "von_neumann"],
        b=1.9,
        n=100,
        steps=200,
        p_cooperator=0.9,
        boundary="cutoff",
        include_self_interaction=True,
        fixed_boundary_state=1,
        seed=1,
        init_mode="random"
    )