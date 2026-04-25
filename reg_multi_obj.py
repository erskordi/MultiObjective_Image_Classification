"""Regular-grid multi-objective optimization demo translated from MATLAB.

This module reproduces the behavior of ``reg_multi_obj.m``:
1) create a regular 2D parameter grid,
2) assign random objective values,
3) compute Pareto-optimal (non-dominated) points for maximization,
4) sort points into a simple Pareto front ordering,
5) visualize points and front.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MOStruct:
    """Container matching the MATLAB struct fields x1, x2, o1, o2."""

    x1: np.ndarray
    x2: np.ndarray
    o1: np.ndarray
    o2: np.ndarray


def make_mo_struct(x1: np.ndarray, x2: np.ndarray, o1: np.ndarray, o2: np.ndarray) -> MOStruct:
    """Create a uniform multi-objective structure."""
    return MOStruct(x1=x1, x2=x2, o1=o1, o2=o2)


def setup_multi_obj_rand2d(n: int, m: int, seed: int | None = None) -> MOStruct:
    """Sample a regular parameter grid and assign random objective values."""
    x1_range = np.linspace(0.0, 1.0, n)
    x2_range = np.linspace(0.0, 1.0, m)
    x1, x2 = np.meshgrid(x1_range, x2_range)

    rng = np.random.default_rng(seed)
    f1 = rng.random((n, m))
    f2 = rng.random((n, m))

    return make_mo_struct(x1, x2, f1, f2)


def pareto_opt2d(mo2d: MOStruct) -> MOStruct:
    """Compute Pareto-optimal points for 2 objectives (both maximized)."""
    x1_vals: list[float] = []
    x2_vals: list[float] = []
    o1_vals: list[float] = []
    o2_vals: list[float] = []

    n, m = mo2d.x1.shape
    for i in range(n):
        for j in range(m):
            o1_val = mo2d.o1[i, j]
            o2_val = mo2d.o2[i, j]

            better_or_equal_in_o2 = (o1_val > mo2d.o1) & (o2_val >= mo2d.o2)
            better_or_equal_in_o1 = (o1_val >= mo2d.o1) & (o2_val > mo2d.o2)
            opt_size = np.count_nonzero(better_or_equal_in_o2) + np.count_nonzero(better_or_equal_in_o1)

            if opt_size == 0:
                x1_vals.append(float(mo2d.x1[i, j]))
                x2_vals.append(float(mo2d.x2[i, j]))
                o1_vals.append(float(o1_val))
                o2_vals.append(float(o2_val))

    return make_mo_struct(
        x1=np.asarray(x1_vals),
        x2=np.asarray(x2_vals),
        o1=np.asarray(o1_vals),
        o2=np.asarray(o2_vals),
    )


def pareto_front_conts2d(mo2d: MOStruct) -> MOStruct:
    """Create a simple ordered Pareto front by descending objective 1."""
    points = pareto_opt2d(mo2d)

    order = np.argsort(points.o1)[::-1]
    o1_sorted = points.o1[order]
    o2_vals = points.o2[order]
    x1_vals = points.x1[order]
    x2_vals = points.x2[order]

    return make_mo_struct(x1=x1_vals, x2=x2_vals, o1=o1_sorted, o2=o2_vals)


def plot_mo_all(figure_num: int, mo_points: MOStruct, mo_pareto_front: MOStruct) -> None:
    """Plot sampled objective points and connected Pareto front."""
    plt.figure(figure_num)
    plt.title("Multi-objective Optimization Demonstration in 2D")
    plt.xlabel("O1")
    plt.ylabel("O2")
    plt.plot(mo_points.o1.ravel(), mo_points.o2.ravel(), ".")
    plt.plot(mo_pareto_front.o1.ravel(), mo_pareto_front.o2.ravel())


def run_reg_mo_obj(
        n: int = 10,
        m: int = 10,
        seed: int | None = None
        ) -> tuple[MOStruct, MOStruct, MOStruct]:
    """End-to-end demo matching MATLAB `run_reg_mo_obj`."""
    mo2d = setup_multi_obj_rand2d(n, m, seed=seed)
    mo2d_pareto_points = pareto_opt2d(mo2d)
    mo2d_pareto_front = pareto_front_conts2d(mo2d)

    plot_mo_all(1, mo2d_pareto_points, mo2d_pareto_front)
    plt.plot(mo2d.o1, mo2d.o2, ".")

    return mo2d, mo2d_pareto_points, mo2d_pareto_front


if __name__ == "__main__":
    run_reg_mo_obj()
    plt.show()
