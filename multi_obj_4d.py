"""Regular-grid multi-objective optimization demo on a 5D objective space.

The workflow is:
1) build a 4D parameter grid from model file metadata,
2) assign 5 objective fields (FLOPs, Energy, Memory, AUC, Params),
3) compute Pareto-optimal (non-dominated) points,
4) create a deterministic ordering for Pareto front traversal,
5) visualize objectives in a 3D projection colored by DNN family (vit/cnn/gat),
6) plot all C(5,2)=10 pairwise 2D objective plots.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull, QhullError
from sklearn.svm import SVC

try:
    go = importlib.import_module("plotly.graph_objects")
except ModuleNotFoundError:
    go = None


@dataclass
class MOStruct:
    """Container for 4 design variables, 5 objective values, and DNN family labels."""

    x1: np.ndarray
    x2: np.ndarray
    x3: np.ndarray
    x4: np.ndarray
    o1: np.ndarray  # FLOPs (log)
    o2: np.ndarray  # Energy (log)
    o3: np.ndarray  # Memory (log)
    o4: np.ndarray  # AUC
    o5: np.ndarray  # Params (log)
    model_file: np.ndarray
    model_type: np.ndarray  # DNN family: vit, cnn, gat


def make_mo_struct(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x4: np.ndarray,
    o1: np.ndarray,
    o2: np.ndarray,
    o3: np.ndarray,
    o4: np.ndarray,
    o5: np.ndarray,
    model_file: np.ndarray,
    model_type: np.ndarray,
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
        o5=o5,
        model_file=model_file,
        model_type=model_type,
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
    """Build the 5D multi-objective structure from evaluation CSV results.

    Objective columns read from CSV:
    - avg_flops_log       → o1 (FLOPs)
    - energy_log          → o2 (Energy)
    - mem_utilization_log → o3 (Memory)
    - auc                 → o4
    - params              → o5
    - model_type          → DNN family label (vit / cnn / gat)
    """
    df = pd.read_csv(csv_path)

    if model_type is not None:
        df = df[df["model_type"] == model_type]

    if df.empty:
        raise ValueError("No rows found in CSV after applying filters.")

    required_cols = [
        "avg_flops_log", "energy_log", "mem_utilization_log",
        "auc", "params", "file", "model_type",
    ]
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
        o5=df["params"].to_numpy(dtype=float),
        model_file=df["file"].to_numpy(dtype=str),
        model_type=df["model_type"].to_numpy(dtype=str),
    )


def _to_objective_matrix(mo4d: MOStruct) -> np.ndarray:
    """Flatten objective tensors to [num_points, 5]."""
    return np.stack(
        [
            mo4d.o1.ravel(),
            mo4d.o2.ravel(),
            mo4d.o3.ravel(),
            mo4d.o4.ravel(),
            mo4d.o5.ravel(),
        ],
        axis=1,
    )


def _subset_mo_struct(mo4d: MOStruct, mask: np.ndarray) -> MOStruct:
    """Return a filtered (or reordered) multi-objective structure."""
    return make_mo_struct(
        x1=mo4d.x1.ravel()[mask],
        x2=mo4d.x2.ravel()[mask],
        x3=mo4d.x3.ravel()[mask],
        x4=mo4d.x4.ravel()[mask],
        o1=mo4d.o1.ravel()[mask],
        o2=mo4d.o2.ravel()[mask],
        o3=mo4d.o3.ravel()[mask],
        o4=mo4d.o4.ravel()[mask],
        o5=mo4d.o5.ravel()[mask],
        model_file=mo4d.model_file.ravel()[mask],
        model_type=mo4d.model_type.ravel()[mask],
    )


def _constraint_mask(
    mo4d: MOStruct,
    flops_max: float = np.inf,
    energy_max: float = np.inf,
    memory_max: float = np.inf,
    auc_min: float = -np.inf,
    params_max: float = np.inf,
) -> np.ndarray:
    """Return the feasible-set boolean mask for user-specified objective constraints."""
    return (
        (mo4d.o1.ravel() <= flops_max)
        & (mo4d.o2.ravel() <= energy_max)
        & (mo4d.o3.ravel() <= memory_max)
        & (mo4d.o4.ravel() >= auc_min)
        & (mo4d.o5.ravel() <= params_max)
    )


def _pareto_mask(objectives: np.ndarray, maximize: tuple[bool, ...]) -> np.ndarray:
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
    maximize: tuple[bool, ...] = (False, False, False, True, False),
) -> MOStruct:
    """Compute Pareto-optimal points from 5 objectives."""
    objectives = _to_objective_matrix(mo4d)
    keep = _pareto_mask(objectives, maximize=maximize)

    return _subset_mo_struct(mo4d, keep)


def pareto_front_conts4d(
    mo4d: MOStruct,
    maximize: tuple[bool, ...] = (False, False, False, True, False),
) -> MOStruct:
    """Create a deterministic lexicographic ordering of Pareto points.

    Points are sorted with o1 (FLOPs) as the primary key and o5 (Params) as the
    least-significant key, respecting each objective's optimization direction.
    """
    points = pareto_opt4d(mo4d, maximize=maximize)
    if points.o1.size == 0:
        return points

    objective_stack = np.column_stack(
        (points.o1, points.o2, points.o3, points.o4, points.o5)
    )
    # Flip sign for maximized objectives so lexsort always sorts ascending.
    signs = np.where(np.asarray(maximize, dtype=bool), -1.0, 1.0)
    sort_stack = objective_stack * signs  # shape (n, 5)

    # np.lexsort: last key is primary. Reverse columns so o1 becomes primary.
    order = np.lexsort(sort_stack[:, ::-1].T)

    return _subset_mo_struct(points, order)


def constrained_single_objective_optima(
    mo4d: MOStruct,
    flops_max: float = np.inf,
    energy_max: float = np.inf,
    memory_max: float = np.inf,
    auc_min: float = -np.inf,
    params_max: float = np.inf,
) -> pd.DataFrame:
    """Compute one constrained optimum for each of the 5 objectives.

    - maximize AUC   under FLOPs / Energy / Memory / Params upper bounds
    - minimize Energy with remaining resource bounds and AUC lower bound
    - minimize FLOPs  with remaining resource bounds and AUC lower bound
    - minimize Memory with remaining resource bounds and AUC lower bound
    - minimize Params with remaining resource bounds and AUC lower bound
    """
    rows: list[dict[str, object]] = []

    flops = mo4d.o1.ravel()
    energy = mo4d.o2.ravel()
    memory = mo4d.o3.ravel()
    auc = mo4d.o4.ravel()
    params = mo4d.o5.ravel()
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
                "Params (log)": float(params[best_idx]),
            }
        )

    auc_mask = (
        (flops <= flops_max) & (energy <= energy_max)
        & (memory <= memory_max) & (params <= params_max)
    )
    _append_best("Max AUC", auc_mask, auc, maximize=True)

    energy_mask = (
        (flops <= flops_max) & (memory <= memory_max)
        & (auc >= auc_min) & (params <= params_max)
    )
    _append_best("Min Energy", energy_mask, energy, maximize=False)

    flops_mask = (
        (energy <= energy_max) & (memory <= memory_max)
        & (auc >= auc_min) & (params <= params_max)
    )
    _append_best("Min FLOPs", flops_mask, flops, maximize=False)

    memory_mask = (
        (flops <= flops_max) & (energy <= energy_max)
        & (auc >= auc_min) & (params <= params_max)
    )
    _append_best("Min Memory", memory_mask, memory, maximize=False)

    params_mask = (
        (flops <= flops_max) & (energy <= energy_max)
        & (memory <= memory_max) & (auc >= auc_min)
    )
    _append_best("Min Params", params_mask, params, maximize=False)

    return pd.DataFrame(rows)


_OBJ_LABELS = [
    "FLOPs (log)",
    "Energy (log)",
    "Memory Utilization (log)",
    "AUC",
    "Params (log)",
]

# Visual identity per DNN family (consistent across all plots).
_FAMILY_COLORS: dict[str, str] = {
    "vit": "steelblue",
    "cnn": "darkorange",
    "gat": "seagreen",
}
_FAMILY_MARKERS: dict[str, str] = {
    "vit": "o",
    "cnn": "s",
    "gat": "^",
}


def plot_mo_all(figure_num: int, mo_points: MOStruct, mo_pareto_front: MOStruct) -> None:
    """Plot Pareto points in 3D (FLOPs / Energy / Memory) colored by DNN family."""
    fig = plt.figure(figure_num)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Pareto Solutions on a 5D Objective Space")
    ax.set_xlabel(_OBJ_LABELS[0])
    ax.set_ylabel(_OBJ_LABELS[1])
    ax.set_zlabel(_OBJ_LABELS[2])
    ax.zaxis.labelpad = 12

    families = np.unique(mo_points.model_type.ravel())
    for family in families:
        fmask = mo_points.model_type.ravel() == family
        color = _FAMILY_COLORS.get(family, "grey")
        fmarker = _FAMILY_MARKERS.get(family, "o")
        ax.scatter(
            mo_points.o1.ravel()[fmask],
            mo_points.o2.ravel()[fmask],
            mo_points.o3.ravel()[fmask],
            c=color,
            marker=fmarker,
            alpha=0.85,
            s=52,
            edgecolors="white",
            linewidths=0.4,
            label=family.upper(),
        )

    ax.plot(
        mo_pareto_front.o1.ravel(),
        mo_pareto_front.o2.ravel(),
        mo_pareto_front.o3.ravel(),
        color="crimson",
        linewidth=2.5,
        label="Ordered Pareto traversal",
    )

    # Compact point indices; full index→model mapping is in the table figure.
    for idx, (o1, o2, o3) in enumerate(
        zip(
            mo_pareto_front.o1.ravel(),
            mo_pareto_front.o2.ravel(),
            mo_pareto_front.o3.ravel(),
        ),
        start=1,
    ):
        ax.text(o1, o2, o3, str(idx), fontsize=8, alpha=0.95)

    ax.legend(loc="best")


def save_mo_all_interactive_html(
    mo_points: MOStruct,
    mo_pareto_front: MOStruct,
    out_html: str | Path = "pareto_3d_interactive.html",
) -> None:
    """Save an interactive 3D Pareto figure to an HTML file.

    Requires Plotly. If Plotly is unavailable, the function prints a hint and exits.
    """
    if go is None:
        print("Plotly is not installed. Run 'pip install plotly' to enable interactive HTML export.")
        return

    marker_map = {
        "o": "circle",
        "s": "square",
        "^": "diamond",
    }

    fig = go.Figure()

    families = np.unique(mo_points.model_type.ravel())
    for family in families:
        fmask = mo_points.model_type.ravel() == family
        color = _FAMILY_COLORS.get(family, "grey")
        fmarker = marker_map.get(_FAMILY_MARKERS.get(family, "o"), "circle")
        fig.add_trace(
            go.Scatter3d(
                x=mo_points.o1.ravel()[fmask],
                y=mo_points.o2.ravel()[fmask],
                z=mo_points.o3.ravel()[fmask],
                mode="markers",
                name=family.upper(),
                marker={"size": 5, "color": color, "symbol": fmarker, "opacity": 0.85},
                text=mo_points.model_file.ravel()[fmask],
                hovertemplate=(
                    "Model: %{text}<br>"
                    "FLOPs: %{x:.4f}<br>"
                    "Energy: %{y:.4f}<br>"
                    "Memory: %{z:.4f}<extra></extra>"
                ),
            )
        )

    n_pareto = mo_pareto_front.o1.ravel().size
    fig.add_trace(
        go.Scatter3d(
            x=mo_pareto_front.o1.ravel(),
            y=mo_pareto_front.o2.ravel(),
            z=mo_pareto_front.o3.ravel(),
            mode="lines+markers+text",
            name="Ordered Pareto traversal",
            line={"color": "crimson", "width": 5},
            marker={"size": 4, "color": "crimson"},
            text=[str(i) for i in range(1, n_pareto + 1)],
            textposition="top center",
        )
    )

    fig.update_layout(
        title="Pareto Solutions on a 5D Objective Space",
        scene={
            "xaxis_title": _OBJ_LABELS[0],
            "yaxis_title": _OBJ_LABELS[1],
            "zaxis_title": _OBJ_LABELS[2],
        },
        template="plotly_white",
        legend={"title": {"text": "DNN Family"}},
    )

    out_path = Path(out_html)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"Saved interactive Pareto figure to {out_path}")


def plot_pairwise_objectives(
    mo_all: MOStruct,
    figure_num: int = 2,
    flops_max: float = np.inf,
    energy_max: float = np.inf,
    memory_max: float = np.inf,
    auc_min: float = -np.inf,
    params_max: float = np.inf,
) -> dict[str, pd.DataFrame]:
    """Plot all C(5,2)=10 pairwise 2D objective plots in a 2×5 grid.

    Background points are colored by DNN family (vit/cnn/gat). Constrained
    single-objective optima are overlaid with distinct crimson markers.
    """
    objectives_all = [mo_all.o1, mo_all.o2, mo_all.o3, mo_all.o4, mo_all.o5]

    optima_df = constrained_single_objective_optima(
        mo_all,
        flops_max=flops_max,
        energy_max=energy_max,
        memory_max=memory_max,
        auc_min=auc_min,
        params_max=params_max,
    )

    marker_map = {
        "Max AUC": "*",
        "Min Energy": "s",
        "Min FLOPs": "^",
        "Min Memory": "D",
        "Min Params": "P",
    }

    def _plot_svm_separator(ax: plt.Axes, x_vals: np.ndarray, y_vals: np.ndarray) -> None:
        """Plot a linear SVM separator for low/high regime split in (x, y)."""
        x = np.asarray(x_vals, dtype=float)
        y = np.asarray(y_vals, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        if x.size < 4:
            return

        # Define low/high regime by normalized combined load in this objective pair.
        x_std = np.std(x)
        y_std = np.std(y)
        x_norm = (x - np.median(x)) / (x_std if x_std > 0 else 1.0)
        y_norm = (y - np.median(y)) / (y_std if y_std > 0 else 1.0)
        regime_score = x_norm + y_norm
        labels = (regime_score >= np.median(regime_score)).astype(int)

        if np.unique(labels).size < 2:
            return

        features = np.column_stack((x, y))
        clf = SVC(kernel="linear", C=1.0)
        clf.fit(features, labels)

        w = clf.coef_[0]
        b = float(clf.intercept_[0])
        eps = 1e-12
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        x_half_span = 0.30 * (x_max - x_min)
        y_half_span = 0.30 * (y_max - y_min)

        if abs(w[1]) > eps:
            # Draw only the central segment to avoid visually overwhelming boundaries.
            xs = np.linspace(x_mid - x_half_span, x_mid + x_half_span, 120)
            ys = -(w[0] * xs + b) / w[1]
            ax.plot(xs, ys, color="black", linestyle="-.", linewidth=1.1, alpha=0.65, zorder=1)
        elif abs(w[0]) > eps:
            x0 = -b / w[0]
            ax.plot(
                [x0, x0],
                [y_mid - y_half_span, y_mid + y_half_span],
                color="black",
                linestyle="-.",
                linewidth=1.1,
                alpha=0.65,
                zorder=1,
            )

    # All C(5,2) pairs in row-major order.
    pairs = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 4),
    ]

    fig, axes = plt.subplots(2, 5, num=figure_num, figsize=(27, 9))
    fig.suptitle("Pairwise Objective Plots with Constrained Single-Objective Optima", fontsize=14)
    axes = axes.ravel()

    families = np.unique(mo_all.model_type.ravel())
    pair_tables: dict[str, pd.DataFrame] = {}

    # Build the shared legend handles once — same for every subplot.
    shared_handles: list[Line2D] = []
    shared_labels: list[str] = []
    for family in families:
        color = _FAMILY_COLORS.get(family, "grey")
        fmarker = _FAMILY_MARKERS.get(family, "o")
        shared_handles.append(
            Line2D(
                [0], [0],
                marker=fmarker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.3,
                markersize=6,
                linestyle="None",
                label=family.upper(),
            )
        )
        shared_labels.append(family.upper())
    for opt_label, opt_marker in marker_map.items():
        shared_handles.append(
            Line2D(
                [0], [0],
                marker=opt_marker,
                color="none",
                markerfacecolor="crimson",
                markeredgecolor="black",
                markeredgewidth=0.7,
                markersize=7 if opt_marker == "*" else 5,
                linestyle="None",
                label=opt_label,
            )
        )
        shared_labels.append(opt_label)
    shared_handles.append(
        Line2D(
            [0], [0],
            color="black",
            linestyle="-.",
            linewidth=1.1,
            label="SVM separator\n(low/high regime)",
        )
    )
    shared_labels.append("SVM separator\n(low/high regime)")

    for ax, (i, j) in zip(axes, pairs):
        # --- background: all points colored by DNN family + convex hull ---
        pair_x_all = objectives_all[i].ravel()
        pair_y_all = objectives_all[j].ravel()
        for family in families:
            fmask = mo_all.model_type.ravel() == family
            color = _FAMILY_COLORS.get(family, "grey")
            fmarker = _FAMILY_MARKERS.get(family, "o")
            fx = objectives_all[i].ravel()[fmask]
            fy = objectives_all[j].ravel()[fmask]
            ax.scatter(
                fx, fy,
                c=color,
                marker=fmarker,
                s=24,
                alpha=0.55,
                edgecolors="white",
                linewidths=0.3,
                label=family.upper(),
            )
            # Draw family grouping envelope only for 3+ distinct points.
            # Collinear sets are skipped when convex hull construction fails.
            pts = np.column_stack((fx, fy))
            unique_pts = np.unique(pts, axis=0)
            if unique_pts.shape[0] >= 3:
                try:
                    hull = ConvexHull(unique_pts)
                    hull_verts = unique_pts[hull.vertices]
                    patch = MplPolygon(
                        hull_verts,
                        closed=True,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.10,
                        linewidth=1.2,
                        linestyle="--",
                        zorder=0,
                    )
                    ax.add_patch(patch)
                except (QhullError, ValueError):
                    pass  # degenerate geometry — skip hull silently

            # Optimization-based regime boundary across all models for this pair.
            _plot_svm_separator(ax, pair_x_all, pair_y_all)

        # --- overlay: constrained single-objective optima ---
        pair_rows: list[dict[str, object]] = []
        for _, row in optima_df.iterrows():
            x_val = float(row[_OBJ_LABELS[i]])
            y_val = float(row[_OBJ_LABELS[j]])
            opt_label = str(row["Optimization"])
            marker = marker_map.get(opt_label, "o")

            ax.scatter(
                x_val,
                y_val,
                s=180 if marker == "*" else 60,
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

        # Stagger labels for duplicate (overlapping) points.
        point_label_counts: dict[tuple[float, float], int] = {}
        for idx, (_, row) in enumerate(pair_table_df.iterrows(), start=1):
            x_val = float(row[_OBJ_LABELS[i]])
            y_val = float(row[_OBJ_LABELS[j]])
            point_key = (round(x_val, 8), round(y_val, 8))
            dup_count = point_label_counts.get(point_key, 0)
            point_label_counts[point_key] = dup_count + 1

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

        # Keep a zoomed AUC scale whenever AUC appears on an axis.
        if i == 3:
            ax.set_xlim(0.9, 1.0)
            ax.set_xticks(np.linspace(0.9, 1.0, 6))
        if j == 2:
            mem_vals = pair_y_all[np.isfinite(pair_y_all)]
            if mem_vals.size > 0:
                mem_min = float(np.min(mem_vals))
                mem_max = float(np.max(mem_vals))
                mem_span = mem_max - mem_min
                pad = 0.08 * mem_span if mem_span > 0 else 0.02
                ax.set_ylim(mem_min - pad, mem_max + pad)
                ax.set_yticks(np.linspace(mem_min - pad, mem_max + pad, 6))
        if j == 3:
            ax.set_ylim(0.9, 1.0)
            ax.set_yticks(np.linspace(0.9, 1.0, 6))

        pair_key = f"{_OBJ_LABELS[i]} vs {_OBJ_LABELS[j]}"
        if pair_table_df.empty:
            pair_tables[pair_key] = pd.DataFrame(
                columns=["Index", "Optimization", "Model File", _OBJ_LABELS[i], _OBJ_LABELS[j]]
            )
        else:
            pair_tables[pair_key] = pd.DataFrame(
                {
                    "Index": np.arange(1, pair_table_df.shape[0] + 1),
                    "Optimization": pair_table_df["Optimization"].to_numpy(),
                    "Model File": pair_table_df["Model File"].to_numpy(),
                    _OBJ_LABELS[i]: pair_table_df[_OBJ_LABELS[i]].to_numpy(),
                    _OBJ_LABELS[j]: pair_table_df[_OBJ_LABELS[j]].to_numpy(),
                }
            )

    fig.legend(
        shared_handles,
        shared_labels,
        loc="center right",
        fontsize=9,
        markerscale=1.2,
        handlelength=1.2,
        borderpad=0.6,
        labelspacing=0.5,
        framealpha=0.9,
        title="DNN Family / Objective",
        title_fontsize=9,
    )
    # Extra spacing reduces label overlap in the dense 2x5 layout.
    fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0], w_pad=2.0, h_pad=2.0)
    return pair_tables


def plot_correlation_analysis(mo_all: MOStruct, figure_num: int = 4) -> None:
    """Figure showing Pearson correlation between all 5 objectives.

    Layout: 1 overall heatmap (left) + 3 per-family heatmaps (right), 1×4 grid.
    AUC correlation with every other objective is highlighted by its row/column.
    """
    families = np.unique(mo_all.model_type.ravel())
    obj_arrays = np.column_stack(
        [mo_all.o1, mo_all.o2, mo_all.o3, mo_all.o4, mo_all.o5]
    )

    def _corr_matrix(arr: np.ndarray) -> np.ndarray:
        """Pearson r matrix; NaN-safe for constant columns."""
        n = arr.shape[1]
        mat = np.full((n, n), np.nan)
        for a in range(n):
            for b in range(n):
                x, y = arr[:, a], arr[:, b]
                if x.std() == 0 or y.std() == 0:
                    mat[a, b] = np.nan
                else:
                    mat[a, b] = float(np.corrcoef(x, y)[0, 1])
        return mat

    ncols = 1 + len(families)
    fig, axes = plt.subplots(1, ncols, num=figure_num, figsize=(5 * ncols, 5))
    fig.suptitle(
        "Pearson Correlation Between Objectives (Overall and Per DNN Family)",
        fontsize=13,
    )

    short_labels = ["FLOPs", "Energy", "Memory", "AUC", "Params"]
    auc_idx = 3  # highlight AUC row/col

    panels = [(axes[0], "All models", obj_arrays)]
    for family in families:
        fmask = mo_all.model_type.ravel() == family
        panels.append((axes[1 + list(families).index(family)], family.upper(), obj_arrays[fmask]))

    for ax, title, arr in panels:
        mat = _corr_matrix(arr)

        im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

        # Annotate every cell with the r value.
        for a in range(len(short_labels)):
            for b in range(len(short_labels)):
                val = mat[a, b]
                txt = f"{val:.2f}" if not np.isnan(val) else "n/a"
                text_color = "black" if abs(val) < 0.75 else "white"
                ax.text(b, a, txt, ha="center", va="center", fontsize=8,
                        color=text_color, fontweight="bold")

        # Highlight the AUC row and column with a border.
        for spine_pos in [auc_idx - 0.5, auc_idx + 0.5]:
            ax.axhline(spine_pos, color="navy", linewidth=1.5, alpha=0.6)
            ax.axvline(spine_pos, color="navy", linewidth=1.5, alpha=0.6)

        ax.set_xticks(range(len(short_labels)))
        ax.set_yticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title(title, fontsize=10)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="r")

    fig.tight_layout()


def plot_pair_index_tables(
    pair_tables: dict[str, pd.DataFrame],
    figure_num: int = 3,
) -> None:
    """Plot index-to-model mapping tables for all 10 pairwise subplots (5×2 grid)."""
    fig, axes = plt.subplots(num=figure_num, nrows=5, ncols=2, figsize=(18, 22))
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
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])


def run_reg_mo_obj(
    csv_path: str | Path = "model_evaluation_results_FashionMNIST.csv",
    model_type: str | None = None,
    maximize: tuple[bool, ...] = (False, False, False, True, False),
    flops_max: float = np.inf,
    energy_max: float = np.inf,
    memory_max: float = np.inf,
    auc_min: float = -np.inf,
    params_max: float = np.inf,
    interactive_html_path: str | Path | None = None,
) -> tuple[MOStruct, MOStruct, MOStruct]:
    """End-to-end 5D multi-objective analysis from CSV metrics."""
    mo5d = setup_multi_obj_from_csv(csv_path=csv_path, model_type=model_type)
    mo5d_pareto_points = pareto_opt4d(mo5d, maximize=maximize)
    mo5d_pareto_front = pareto_front_conts4d(mo5d, maximize=maximize)

    plot_mo_all(1, mo5d_pareto_points, mo5d_pareto_front)
    if interactive_html_path is not None:
        save_mo_all_interactive_html(
            mo5d_pareto_points,
            mo5d_pareto_front,
            out_html=interactive_html_path,
        )
    pair_tables = plot_pairwise_objectives(
        mo5d,
        figure_num=2,
        flops_max=flops_max,
        energy_max=energy_max,
        memory_max=memory_max,
        auc_min=auc_min,
        params_max=params_max,
    )
    plot_pair_index_tables(pair_tables, figure_num=3)
    plot_correlation_analysis(mo5d, figure_num=4)

    for pair_name, table_df in pair_tables.items():
        print(f"\n{pair_name}")
        if table_df.empty:
            print("No feasible single-objective optima under the provided constraints.")
        else:
            print(table_df.to_string(index=False))

    return mo5d, mo5d_pareto_points, mo5d_pareto_front


if __name__ == "__main__":
    csv_files = list(Path(".").glob("model_evaluation_results_*.csv"))
    dataset_choice = input(
        "Available datasets:\n" + "\n".join(f"- {csv.stem}" for csv in csv_files) + "\n\n"
        "Enter the dataset you want to analyze (e.g., 'FashionMNIST'): "
    ).strip()
    run_reg_mo_obj(
        csv_path=f"model_evaluation_results_{dataset_choice}.csv",
        interactive_html_path=f"pareto_3d_interactive_{dataset_choice}.html",
    )
    plt.show()
