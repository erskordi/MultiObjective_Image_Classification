"""Regular-grid multi-objective optimization demo on a 4D parameter grid.

The workflow is:
1) create a regular 4D parameter grid,
2) assign 4 objective fields on that grid,
3) compute Pareto-optimal (non-dominated) points,
4) create a deterministic ordering for Pareto front traversal,
5) visualize objectives in a 3D projection with color as the 4th objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class MOStruct:
    """Container for 4 design variables and 4 objective values."""

    x1: np.ndarray
    x2: np.ndarray
    x3: np.ndarray
    x4: np.ndarray
    o1: np.ndarray
    o2: np.ndarray
    o3: np.ndarray
    o4: np.ndarray
    model_file: np.ndarray


def make_mo_struct(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x4: np.ndarray,
    o1: np.ndarray,
    o2: np.ndarray,
    o3: np.ndarray,
    o4: np.ndarray,
    model_file: np.ndarray,
) -> MOStruct:
    """Create a uniform multi-objective structure."""
    return MOStruct(
        x1=x1,
        x2=x2,
        x3=x3,
        x4=x4,
        o1=o1,
        o2=o2,
        o3=o3,
        o4=o4,
        model_file=model_file,
    )


def _parse_model_file_to_4d(file_name: str) -> tuple[float, float, float, float]:
    """Extract 4 design coordinates from model file stem.

    Example stems:
    - vit_model_12_16_32_0.001
    - cnn_model_64_4_3_0.0005
    - gnn_model_3_64_3_64_0.001
    """
    stem = Path(file_name).stem
    tokens = stem.split("_")
    numeric_tokens = []
    for tok in tokens:
        try:
            numeric_tokens.append(float(tok))
        except ValueError:
            continue

    if len(numeric_tokens) >= 4:
        return numeric_tokens[0], numeric_tokens[1], numeric_tokens[2], numeric_tokens[3]

    # Fallback for unexpected file names.
    return 0.0, 0.0, 0.0, 0.0


def setup_multi_obj_from_csv(
    csv_path: str | Path = "model_evaluation_results_FashionMNIST.csv",
    model_type: str | None = None,
) -> MOStruct:
    """Build the 4D multi-objective structure from evaluation CSV results.

    Objective columns are read from:
    - avg_flops_log (or vg_flops_log for compatibility with typoed headers)
    - energy_log
    - mem_utilization_log
    - auc
    """
    df = pd.read_csv(csv_path)

    if model_type is not None:
        df = df[df["model_type"] == model_type]

    if df.empty:
        raise ValueError("No rows found in CSV after applying filters.")

    required_cols = ["avg_flops_log", "energy_log", "mem_utilization_log", "auc", "file"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    x_coords = np.array([_parse_model_file_to_4d(name) for name in df["file"].astype(str)])

    return make_mo_struct(
        x1=x_coords[:, 0],
        x2=x_coords[:, 1],
        x3=x_coords[:, 2],
        x4=x_coords[:, 3],
        o1=df["avg_flops_log"].to_numpy(dtype=float),
        o2=df["energy_log"].to_numpy(dtype=float),
        o3=df["mem_utilization_log"].to_numpy(dtype=float),
        o4=df["auc"].to_numpy(dtype=float),
        model_file=df["file"].to_numpy(dtype=str),
    )


def _to_objective_matrix(mo4d: MOStruct) -> np.ndarray:
    """Flatten objective tensors to [num_points, 4]."""
    return np.stack(
        [
            mo4d.o1.ravel(),
            mo4d.o2.ravel(),
            mo4d.o3.ravel(),
            mo4d.o4.ravel(),
        ],
        axis=1,
    )


def _subset_mo_struct(mo4d: MOStruct, mask: np.ndarray) -> MOStruct:
    """Return a filtered multi-objective structure."""
    return make_mo_struct(
        x1=mo4d.x1.ravel()[mask],
        x2=mo4d.x2.ravel()[mask],
        x3=mo4d.x3.ravel()[mask],
        x4=mo4d.x4.ravel()[mask],
        o1=mo4d.o1.ravel()[mask],
        o2=mo4d.o2.ravel()[mask],
        o3=mo4d.o3.ravel()[mask],
        o4=mo4d.o4.ravel()[mask],
        model_file=mo4d.model_file.ravel()[mask],
    )


def _constraint_mask(
    mo4d: MOStruct,
    flops_max: float = np.inf,
    energy_max: float = np.inf,
    memory_max: float = np.inf,
    auc_min: float = -np.inf,
) -> np.ndarray:
    """Return the feasible-set mask for user-specified objective constraints."""
    return (
        (mo4d.o1.ravel() <= flops_max)
        & (mo4d.o2.ravel() <= energy_max)
        & (mo4d.o3.ravel() <= memory_max)
        & (mo4d.o4.ravel() >= auc_min)
    )


def _pareto_mask(objectives: np.ndarray, maximize: tuple[bool, bool, bool, bool]) -> np.ndarray:
    """Return a boolean mask for non-dominated points.

    A point is dominated if another point is at least as good in all objectives
    and strictly better in at least one objective.
    """
    signs = np.where(np.asarray(maximize, dtype=bool), 1.0, -1.0)
    scaled = objectives * signs
    n_points = scaled.shape[0]
    dominated = np.zeros(n_points, dtype=bool)

    for i in range(n_points):
        if dominated[i]:
            continue
        dominates_i = np.all(scaled >= scaled[i], axis=1) & np.any(scaled > scaled[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            dominated[i] = True

    return ~dominated


def pareto_opt4d(
    mo4d: MOStruct,
    maximize: tuple[bool, bool, bool, bool] = (False, False, False, True),
) -> MOStruct:
    """Compute Pareto-optimal points from a 4D parameter grid and 4 objectives."""
    objectives = _to_objective_matrix(mo4d)
    keep = _pareto_mask(objectives, maximize=maximize)

    return make_mo_struct(
        x1=mo4d.x1.ravel()[keep],
        x2=mo4d.x2.ravel()[keep],
        x3=mo4d.x3.ravel()[keep],
        x4=mo4d.x4.ravel()[keep],
        o1=mo4d.o1.ravel()[keep],
        o2=mo4d.o2.ravel()[keep],
        o3=mo4d.o3.ravel()[keep],
        o4=mo4d.o4.ravel()[keep],
        model_file=mo4d.model_file.ravel()[keep],
    )


def pareto_front_conts4d(
    mo4d: MOStruct,
    maximize: tuple[bool, bool, bool, bool] = (False, False, False, True),
) -> MOStruct:
    """Create a deterministic ordering of Pareto points.

    In 4 objectives, there is no single geometric curve to trace. This helper
    sorts Pareto points lexicographically by objective priority O1 -> O2 -> O3 -> O4.
    """
    points = pareto_opt4d(mo4d, maximize=maximize)
    if points.o1.size == 0:
        return points

    objective_stack = np.column_stack((points.o1, points.o2, points.o3, points.o4))
    sort_stack = np.column_stack((
        objective_stack[:, 3],
        objective_stack[:, 2],
        objective_stack[:, 1],
        objective_stack[:, 0],
    ))
    signs = np.where(np.asarray(maximize, dtype=bool), -1.0, 1.0)
    sort_stack = sort_stack * signs[::-1]
    order = np.lexsort((sort_stack[:, 0], sort_stack[:, 1], sort_stack[:, 2], sort_stack[:, 3]))

    return make_mo_struct(
        x1=points.x1[order],
        x2=points.x2[order],
        x3=points.x3[order],
        x4=points.x4[order],
        o1=points.o1[order],
        o2=points.o2[order],
        o3=points.o3[order],
        o4=points.o4[order],
        model_file=points.model_file[order],
    )


def constrained_pairwise_front(
    mo4d: MOStruct,
    objective_i: int,
    objective_j: int,
    maximize: tuple[bool, bool, bool, bool] = (False, False, False, True),
    flops_max: float = np.inf,
    energy_max: float = np.inf,
    memory_max: float = np.inf,
    auc_min: float = -np.inf,
) -> MOStruct:
    """Compute the 2D Pareto front on a feasible subset for a chosen objective pair.

    The feasible subset is defined by the user thresholds. On that subset, this
    returns the non-dominated points for the visible objective pair only.
    """
    feasible_mask = _constraint_mask(
        mo4d,
        flops_max=flops_max,
        energy_max=energy_max,
        memory_max=memory_max,
        auc_min=auc_min,
    )
    feasible = _subset_mo_struct(mo4d, feasible_mask)
    if feasible.o1.size == 0:
        return feasible

    objectives = _to_objective_matrix(feasible)
    pair_objectives = objectives[:, [objective_i, objective_j]]
    pair_directions = (maximize[objective_i], maximize[objective_j])
    keep = _pareto_mask(pair_objectives, pair_directions)
    pair_points = _subset_mo_struct(feasible, keep)

    visible = np.column_stack(
        (pair_points.o1, pair_points.o2, pair_points.o3, pair_points.o4)
    )[:, [objective_i, objective_j]]
    sort_signs = np.where(np.asarray(pair_directions, dtype=bool), -1.0, 1.0)
    sort_key = visible * sort_signs
    order = np.lexsort((sort_key[:, 1], sort_key[:, 0]))
    return _subset_mo_struct(pair_points, order)


_OBJ_LABELS = [
    "FLOPs (log)",
    "Energy (log)",
    "Memory Utilization (log)",
    "AUC",
]


def plot_mo_all(figure_num: int, mo_points: MOStruct, mo_pareto_front: MOStruct) -> None:
    """Plot Pareto points in 3D projection (FLOPs / Energy / Memory) colored by AUC."""
    fig = plt.figure(figure_num)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Pareto Solutions on a 4D Objective Space")
    ax.set_xlabel(_OBJ_LABELS[0])
    ax.set_ylabel(_OBJ_LABELS[1])
    ax.set_zlabel(_OBJ_LABELS[2])
    ax.zaxis.labelpad = 12

    scatter_all = ax.scatter(
        mo_points.o1.ravel(),
        mo_points.o2.ravel(),
        mo_points.o3.ravel(),
        c=mo_points.o4.ravel(),
        cmap="viridis",
        alpha=0.85,
        s=80,
        edgecolors="white",
        linewidths=0.4,
        label="Pareto points",
    )

    ax.plot(
        mo_pareto_front.o1.ravel(),
        mo_pareto_front.o2.ravel(),
        mo_pareto_front.o3.ravel(),
        color="crimson",
        linewidth=2.5,
        label="Ordered Pareto traversal",
    )
    fig.colorbar(
        scatter_all,
        ax=ax,
        label=_OBJ_LABELS[3],
        pad=0.2,
        fraction=0.045,
        shrink=0.82,
    )

    # Add compact point indices; use a separate table for index -> model mapping.
    for idx, (o1, o2, o3) in enumerate(zip(
        mo_pareto_front.o1.ravel(),
        mo_pareto_front.o2.ravel(),
        mo_pareto_front.o3.ravel(),
    ), start=1):
        ax.text(o1, o2, o3, str(idx), fontsize=8, alpha=0.95)

    ax.legend(loc="best")


def plot_pairwise_objectives(
    mo_all: MOStruct,
    figure_num: int = 2,
    maximize: tuple[bool, bool, bool, bool] = (False, False, False, True),
    flops_max: float = np.inf,
    energy_max: float = np.inf,
    memory_max: float = np.inf,
    auc_min: float = -np.inf,
) -> None:
    """Plot all 6 pairwise 2D scatter plots of the 4 objectives.

    Each subplot shows all candidate points, the feasible subset under the
    user constraints, and the constrained 2D Pareto front for the visible pair.
    Returns a dictionary of index tables, one per subplot pair.
    """
    objectives_all = [mo_all.o1, mo_all.o2, mo_all.o3, mo_all.o4]
    feasible_mask = _constraint_mask(
        mo_all,
        flops_max=flops_max,
        energy_max=energy_max,
        memory_max=memory_max,
        auc_min=auc_min,
    )
    mo_feasible = _subset_mo_struct(mo_all, feasible_mask)
    objectives_feasible = [mo_feasible.o1, mo_feasible.o2, mo_feasible.o3, mo_feasible.o4]

    pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3),
    ]

    fig, axes = plt.subplots(2, 3, num=figure_num, figsize=(15, 9))
    fig.suptitle("Pairwise Objective Scatter Plots", fontsize=14)
    axes = axes.ravel()

    pair_tables: dict[str, pd.DataFrame] = {}

    for ax, (i, j) in zip(axes, pairs):
        # All candidate points in background.
        ax.scatter(
            objectives_all[i].ravel(),
            objectives_all[j].ravel(),
            c="lightgrey",
            s=70,
            alpha=0.45,
            edgecolors="grey",
            linewidths=0.35,
            label="All points",
        )
        # Feasible points under objective constraints.
        sc = ax.scatter(
            objectives_feasible[i].ravel(),
            objectives_feasible[j].ravel(),
            c=mo_feasible.o4.ravel(),
            cmap="viridis",
            s=120,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.45,
            zorder=3,
            label="Feasible points",
        )

        pair_front = constrained_pairwise_front(
            mo_all,
            i,
            j,
            maximize=maximize,
            flops_max=flops_max,
            energy_max=energy_max,
            memory_max=memory_max,
            auc_min=auc_min,
        )

        front_x = [pair_front.o1, pair_front.o2, pair_front.o3, pair_front.o4][i].ravel()
        front_y = [pair_front.o1, pair_front.o2, pair_front.o3, pair_front.o4][j].ravel()

        # Constrained 2D Pareto front for the visible pair.
        ax.plot(
            front_x,
            front_y,
            color="crimson",
            linewidth=2.2,
            marker="o",
            markersize=7,
            markeredgecolor="darkred",
            zorder=4,
            label="Constrained front",
        )
        ax.set_xlabel(_OBJ_LABELS[i])
        ax.set_ylabel(_OBJ_LABELS[j])
        ax.set_title(f"{_OBJ_LABELS[i]} vs {_OBJ_LABELS[j]}")

        for idx, (x_val, y_val) in enumerate(zip(front_x, front_y), start=1):
            ax.annotate(
                str(idx),
                (x_val, y_val),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                alpha=0.9,
            )

        pair_key = f"{_OBJ_LABELS[i]} vs {_OBJ_LABELS[j]}"
        pair_tables[pair_key] = pd.DataFrame(
            {
                "Index": np.arange(1, pair_front.model_file.size + 1),
                "Model File": pair_front.model_file,
                _OBJ_LABELS[i]: front_x,
                _OBJ_LABELS[j]: front_y,
            }
        )

        fig.colorbar(sc, ax=ax, label=_OBJ_LABELS[3])

    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    return pair_tables


def plot_pair_index_tables(
    pair_tables: dict[str, pd.DataFrame],
    figure_num: int = 3,
) -> None:
    """Plot index-to-model mapping tables for all pairwise constrained fronts."""
    fig, axes = plt.subplots(num=figure_num, nrows=3, ncols=2, figsize=(18, 14))
    axes = axes.ravel()

    for ax, (pair_name, table_df) in zip(axes, pair_tables.items()):
        ax.axis("off")
        ax.set_title(pair_name, fontsize=11)
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.1)

    fig.suptitle("Constrained Front Index Tables by Subplot", fontsize=14)
    fig.tight_layout()


def run_reg_mo_obj(
    csv_path: str | Path = "model_evaluation_results_FashionMNIST.csv",
    model_type: str | None = None,
    maximize: tuple[bool, bool, bool, bool] = (False, False, False, True),
    flops_max: float = np.inf,
    energy_max: float = np.inf,
    memory_max: float = np.inf,
    auc_min: float = -np.inf,
) -> tuple[MOStruct, MOStruct, MOStruct]:
    """End-to-end 4D Pareto extraction from CSV metrics.

    The maximize tuple controls each objective direction. Example:
    (False, False, False, True) means minimize O1/O2/O3 and maximize O4.
    """
    mo4d = setup_multi_obj_from_csv(csv_path=csv_path, model_type=model_type)
    mo4d_pareto_points = pareto_opt4d(mo4d, maximize=maximize)
    mo4d_pareto_front = pareto_front_conts4d(mo4d, maximize=maximize)

    plot_mo_all(1, mo4d_pareto_points, mo4d_pareto_front)
    pair_tables = plot_pairwise_objectives(
        mo4d,
        figure_num=2,
        maximize=maximize,
        flops_max=flops_max,
        energy_max=energy_max,
        memory_max=memory_max,
        auc_min=auc_min,
    )
    plot_pair_index_tables(pair_tables, figure_num=3)

    for pair_name, table_df in pair_tables.items():
        print(f"\n{pair_name}")
        if table_df.empty:
            print("No constrained-front points for this subplot.")
        else:
            print(table_df.to_string(index=False))

    return mo4d, mo4d_pareto_points, mo4d_pareto_front


if __name__ == "__main__":
    run_reg_mo_obj()
    plt.show()
