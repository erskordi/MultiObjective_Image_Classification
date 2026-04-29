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
from matplotlib.lines import Line2D


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


def constrained_single_objective_optima(
    mo4d: MOStruct,
    flops_max: float = np.inf,
    energy_max: float = np.inf,
    memory_max: float = np.inf,
    auc_min: float = -np.inf,
) -> pd.DataFrame:
    """Compute one constrained optimum for each objective.

    - maximize AUC with FLOPs/Energy/Memory upper bounds
    - minimize Energy with FLOPs/Memory upper bounds and AUC lower bound
    - minimize FLOPs with Energy/Memory upper bounds and AUC lower bound
    - minimize Memory with FLOPs/Energy upper bounds and AUC lower bound
    """
    rows: list[dict[str, object]] = []

    flops = mo4d.o1.ravel()
    energy = mo4d.o2.ravel()
    memory = mo4d.o3.ravel()
    auc = mo4d.o4.ravel()
    model_file = mo4d.model_file.ravel()

    def _append_best(label: str, mask: np.ndarray, values: np.ndarray, maximize: bool) -> None:
        idx_candidates = np.where(mask)[0]
        if idx_candidates.size == 0:
            return
        best_local = np.argmax(values[idx_candidates]) if maximize else np.argmin(values[idx_candidates])
        best_idx = int(idx_candidates[best_local])
        rows.append(
            {
                "Optimization": label,
                "Model File": model_file[best_idx],
                "FLOPs (log)": float(flops[best_idx]),
                "Energy (log)": float(energy[best_idx]),
                "Memory Utilization (log)": float(memory[best_idx]),
                "AUC": float(auc[best_idx]),
            }
        )

    auc_mask = (flops <= flops_max) & (energy <= energy_max) & (memory <= memory_max)
    _append_best("Max AUC", auc_mask, auc, maximize=True)

    energy_mask = (flops <= flops_max) & (memory <= memory_max) & (auc >= auc_min)
    _append_best("Min Energy", energy_mask, energy, maximize=False)

    flops_mask = (energy <= energy_max) & (memory <= memory_max) & (auc >= auc_min)
    _append_best("Min FLOPs", flops_mask, flops, maximize=False)

    memory_mask = (flops <= flops_max) & (energy <= energy_max) & (auc >= auc_min)
    _append_best("Min Memory", memory_mask, memory, maximize=False)

    return pd.DataFrame(rows)


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
        s=52,
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
    flops_max: float = np.inf,
    energy_max: float = np.inf,
    memory_max: float = np.inf,
    auc_min: float = -np.inf,
) -> dict[str, pd.DataFrame]:
    """Plot all 6 pairwise 2D objective plots with constrained single-objective optima."""
    objectives_all = [mo_all.o1, mo_all.o2, mo_all.o3, mo_all.o4]

    optima_df = constrained_single_objective_optima(
        mo_all,
        flops_max=flops_max,
        energy_max=energy_max,
        memory_max=memory_max,
        auc_min=auc_min,
    )

    marker_map = {
        "Max AUC": "*",
        "Min Energy": "s",
        "Min FLOPs": "^",
        "Min Memory": "D",
    }

    pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3),
    ]

    fig, axes = plt.subplots(2, 3, num=figure_num, figsize=(15, 9))
    fig.suptitle("Pairwise Objective Plots with Constrained Single-Objective Optima", fontsize=14)
    axes = axes.ravel()

    pair_tables: dict[str, pd.DataFrame] = {}

    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(
            objectives_all[i].ravel(),
            objectives_all[j].ravel(),
            c="lightgrey",
            s=24,
            alpha=0.45,
            edgecolors="grey",
            linewidths=0.35,
            label="All points",
        )

        pair_rows: list[dict[str, object]] = []
        for _, row in optima_df.iterrows():
            x_val = float(row[_OBJ_LABELS[i]])
            y_val = float(row[_OBJ_LABELS[j]])
            opt_label = str(row["Optimization"])
            marker = marker_map.get(opt_label, "o")

            ax.scatter(
                x_val,
                y_val,
                s=150 if marker == "*" else 50,
                marker=marker,
                c="crimson",
                edgecolors="black",
                linewidths=0.7,
                zorder=2,
                label=opt_label,
            )

            pair_rows.append(
                {
                    "Optimization": opt_label,
                    "Model File": row["Model File"],
                    _OBJ_LABELS[i]: x_val,
                    _OBJ_LABELS[j]: y_val,
                }
            )

        pair_table_df = pd.DataFrame(pair_rows)

        # Stagger labels for duplicate points to avoid complete overlap.
        point_label_counts: dict[tuple[float, float], int] = {}
        for idx, (_, row) in enumerate(pair_table_df.iterrows(), start=1):
            x_val = float(row[_OBJ_LABELS[i]])
            y_val = float(row[_OBJ_LABELS[j]])

            # Round keys to keep numerically equivalent points grouped together.
            point_key = (round(x_val, 8), round(y_val, 8))
            dup_count = point_label_counts.get(point_key, 0)
            point_label_counts[point_key] = dup_count + 1

            # Cycle through small offsets so coincident labels remain readable.
            base_offsets = [(4, 4), (4, 14), (14, 4), (-10, 10), (10, -8), (-12, -6)]
            dx, dy = base_offsets[dup_count % len(base_offsets)]

            ax.annotate(
                str(idx),
                (x_val, y_val),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=7,
                alpha=0.9,
            )

        ax.set_xlabel(_OBJ_LABELS[i])
        ax.set_ylabel(_OBJ_LABELS[j])
        ax.set_title(f"{_OBJ_LABELS[i]} vs {_OBJ_LABELS[j]}")

        unique_labels = [str(v) for v in pd.unique(pair_table_df["Optimization"])] if not pair_table_df.empty else []
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="lightgrey",
                markeredgecolor="grey",
                markeredgewidth=0.35,
                markersize=2,
                linestyle="None",
                label="All points",
            )
        ]
        legend_labels = ["All points"]
        for label in unique_labels:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker_map.get(label, "o"),
                    color="none",
                    markerfacecolor="crimson",
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    markersize=4,
                    linestyle="None",
                    label=label,
                )
            )
            legend_labels.append(label)

        ax.legend(
            legend_handles,
            legend_labels,
            fontsize=6,
            loc="best",
            handlelength=1.0,
            borderpad=0.25,
            labelspacing=0.25,
        )

        pair_key = f"{_OBJ_LABELS[i]} vs {_OBJ_LABELS[j]}"
        if pair_table_df.empty:
            pair_tables[pair_key] = pd.DataFrame(
                columns=["Index", "Optimization", "Model File", _OBJ_LABELS[i], _OBJ_LABELS[j]]
            )
        else:
            pair_tables[pair_key] = pd.DataFrame(
                {
                    "Index": np.arange(1, pair_table_df.shape[0] + 1),
                    "Optimization": pair_table_df["Optimization"],
                    "Model File": pair_table_df["Model File"],
                    _OBJ_LABELS[i]: pair_table_df[_OBJ_LABELS[i]],
                    _OBJ_LABELS[j]: pair_table_df[_OBJ_LABELS[j]],
                }
            )

    fig.tight_layout()
    return pair_tables


def plot_pair_index_tables(
    pair_tables: dict[str, pd.DataFrame],
    figure_num: int = 3,
) -> None:
    """Plot index-to-model mapping tables for all pairwise single-objective overlays."""
    fig, axes = plt.subplots(num=figure_num, nrows=3, ncols=2, figsize=(18, 14))
    axes = axes.ravel()

    for ax, (pair_name, table_df) in zip(axes, pair_tables.items()):
        ax.axis("off")
        ax.set_title(pair_name, fontsize=10, pad=8)
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.1)

    fig.suptitle("Constrained Single-Objective Optima: Index Tables by Subplot", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])


def run_reg_mo_obj(
    csv_path: str | Path = "model_evaluation_results_FashionMNIST.csv",
    model_type: str | None = None,
    maximize: tuple[bool, bool, bool, bool] = (False, False, False, True),
    flops_max: float = np.inf,
    energy_max: float = np.inf,
    memory_max: float = np.inf,
    auc_min: float = -np.inf,
) -> tuple[MOStruct, MOStruct, MOStruct]:
    """End-to-end 4D analysis from CSV metrics."""
    mo4d = setup_multi_obj_from_csv(csv_path=csv_path, model_type=model_type)
    mo4d_pareto_points = pareto_opt4d(mo4d, maximize=maximize)
    mo4d_pareto_front = pareto_front_conts4d(mo4d, maximize=maximize)

    plot_mo_all(1, mo4d_pareto_points, mo4d_pareto_front)
    pair_tables = plot_pairwise_objectives(
        mo4d,
        figure_num=2,
        flops_max=flops_max,
        energy_max=energy_max,
        memory_max=memory_max,
        auc_min=auc_min,
    )
    plot_pair_index_tables(pair_tables, figure_num=3)

    for pair_name, table_df in pair_tables.items():
        print(f"\n{pair_name}")
        if table_df.empty:
            print("No feasible single-objective optima under the provided constraints.")
        else:
            print(table_df.to_string(index=False))

    return mo4d, mo4d_pareto_points, mo4d_pareto_front


if __name__ == "__main__":
    run_reg_mo_obj()
    plt.show()
