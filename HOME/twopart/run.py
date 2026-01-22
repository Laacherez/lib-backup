import os
import argparse
import numpy as np
import parameters as params
import libMobility as lm
from datetime import datetime

# ---------------------- interface ----------------------

BASE_OUTDIR = "results_long"
RUN_STAMP = datetime.now().strftime("%Y-%m-%d")
RUN_OUTDIR = os.path.join(BASE_OUTDIR, RUN_STAMP)

MAX_RUNS = 5000
CHECK_EVERY = 10
TOL = 1e-3
PATIENCE = 3

BASE_SEED = 12345

MSD_LAGS = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500], dtype=int)
BINS = 51
DMAX_FACTOR = 8.0 

# ------------------------------------------------------


def unwrap_periodic_xy(xyz_wrapped: np.ndarray, L: float) -> np.ndarray:
    u = xyz_wrapped.astype(np.float64, copy=True)
    for dim in (0, 1):
        d = np.diff(u[..., dim], axis=0)
        d -= L * np.round(d / L)
        u[1:, :, dim] = u[0, :, dim] + np.cumsum(d, axis=0)
    return u


def run_one_replica(rep: int, d: float, z0: float) -> np.ndarray:
    np.random.seed(BASE_SEED + rep)

    dt = params.dt
    kbt = params.kbT
    L = params.periodic_length
    eta = params.eta
    a = params.a

    pos = np.array(
        [
            [L * 0.5, L * 0.5, z0], # correct this
            [L * 0.5 + d, L * 0.5, z0],
        ],
        dtype=np.float32,
        order="C",
    )

    solver = lm.DPStokes("periodic", "periodic", "single_wall")
    solver.setParameters(Lx=L, Ly=L, zmin=0.0, zmax=100 * a)
    solver.initialize(viscosity=eta, hydrodynamicRadius=a)

    T = 100_000
    N = pos.shape[0]
    frames = np.empty((T * N, 4), dtype=np.float32)

    for t in range(T):
        i0 = t * N
        frames[i0 : i0 + N, :3] = pos
        frames[i0 : i0 + N, 3] = t

        solver.setPositions(pos)

        F = np.zeros_like(pos)
        dX_det, _ = solver.Mdot(F)
        dX_sto, _ = solver.sqrtMdotW()
        dX_drift, _ = solver.divM()

        pos += dX_det * dt
        pos += dX_sto * np.sqrt(2.0 * kbt * dt)
        pos += dt * kbt * dX_drift

        pos[:, 0] = np.mod(pos[:, 0], L)
        pos[:, 1] = np.mod(pos[:, 1], L)
        pos[:, 2] = np.maximum(pos[:, 2], 1.1 * a) # is that correct ?

    return frames


def convergence_metric(msd_mean: np.ndarray, prev_msd_mean: np.ndarray, eps: float = 1e-15) -> float:
    """Relative L2 change of MSD mean tensor."""
    num = np.linalg.norm(msd_mean - prev_msd_mean)
    den = np.linalg.norm(prev_msd_mean) + eps
    return float(num / den)


def precompute_edges_by_lag(lags: np.ndarray, dt: float, kbt: float, eta: float, a: float) -> np.ndarray:
    D = kbt / (6.0 * np.pi * eta * a)
    edges = np.empty((len(lags), BINS + 1), dtype=np.float64)

    for i, m in enumerate(lags):
        std_rc = np.sqrt(4.0 * D * m * dt)
        dmax = float(DMAX_FACTOR * std_rc)
        edges[i] = np.linspace(-dmax, dmax, BINS + 1, dtype=np.float64)

    return edges


def update_msd_and_hists(
    xyz_wrapped: np.ndarray,
    L: float,
    lags: np.ndarray,
    edges_by_lag: np.ndarray,
    msd_sum: np.ndarray,
    msd_sumsq: np.ndarray,
    hist_counts: np.ndarray,
) -> None:
    """
    Updates:
      msd_sum, msd_sumsq : (K,6)
      hist_counts        : (K,6,BINS)
    for components [rx, ry, rz, cx, cy, cz], at each lag in 'lags'.

    xyz_wrapped: (T,2,3)
    edges_by_lag: (K,BINS+1), aligned with MSD_LAGS (not just active lags)
    """
    xyz = unwrap_periodic_xy(xyz_wrapped, L=L)

    x1 = xyz[:, 0, :]
    x2 = xyz[:, 1, :]
    r = x1 - x2
    c = x1 + x2

    Tsteps = xyz.shape[0]

    for i, m in enumerate(lags):
        if m >= Tsteps:
            continue


        dr = r[m:] - r[:-m]  # (T-m,3)
        dc = c[m:] - c[:-m]  # (T-m,3)

        msd_r = np.mean(dr * dr, axis=0)
        msd_c = np.mean(dc * dc, axis=0)
        msd_vec = np.concatenate([msd_r, msd_c])  # (6,)

        msd_sum[i] += msd_vec
        msd_sumsq[i] += msd_vec * msd_vec

        edges = edges_by_lag[i]
        for comp in range(3):
            vals = dr[:, comp]
            vals = np.clip(vals, edges[0], edges[-1])
            h, _ = np.histogram(vals, bins=edges)
            hist_counts[i, comp, :] += h

        for comp in range(3):
            vals = dc[:, comp]
            vals = np.clip(vals, edges[0], edges[-1])
            h, _ = np.histogram(vals, bins=edges)
            hist_counts[i, 3 + comp, :] += h


def save_histograms_per_lag(outdir: str, lags: np.ndarray, edges_by_lag: np.ndarray, hist_counts: np.ndarray) -> None:
    np.save(os.path.join(outdir, "hist_edges_by_lag.npy"), edges_by_lag)
    np.save(os.path.join(outdir, "hist_counts_by_lag.npy"), hist_counts)

    with open(os.path.join(outdir, "hist_metadata.txt"), "w") as f:
        f.write(f"MSD_LAGS {lags.tolist()}\n")
        f.write("hist_counts shape (K,6,BINS)\n")
        f.write("component order: [rx, ry, rz, cx, cy, cz]\n")
        f.write("edges_by_lag shape (K,BINS+1), aligned with MSD_LAGS\n")
        f.write("histograms are for lag-m displacements used in MSD:\n")
        f.write("  dr(t;m)=r(t+m)-r(t), dc(t;m)=c(t+m)-c(t)\n")
        f.write("x/y are unwrapped before computing r,c and displacements.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, required=True, help="colloid-colloid distance (apex-to-apex)")
    ap.add_argument("--z", type=float, required=True, help="height above wall (apex-to-wall)")
    ap.add_argument("--outdir", type=str, default=None, help="override output directory")
    ap.add_argument("--max-runs", type=int, default=MAX_RUNS)
    args = ap.parse_args()

    d = float(args.d)
    z0 = float(args.z)

    outdir = args.outdir or os.path.join(RUN_OUTDIR, f"d_{d:.6g}_z_{z0:.6g}")
    os.makedirs(outdir, exist_ok=True)

    dt = params.dt
    L = params.periodic_length
    kbt = params.kbT
    eta = params.eta
    a = params.a


    edges_by_lag = precompute_edges_by_lag(MSD_LAGS, dt=dt, kbt=kbt, eta=eta, a=a)

    K = len(MSD_LAGS)
    msd_sum = np.zeros((K, 6), dtype=np.float64)
    msd_sumsq = np.zeros((K, 6), dtype=np.float64)
    msd_n = 0
    hist_counts = np.zeros((K, 6, BINS), dtype=np.int64)

    prev_ckpt_mean = None
    ok_streak = 0
    converged = False

    for rep in range(args.max_runs):
        frames = run_one_replica(rep, d=d, z0=z0)

        N = 2
        xyz_wrapped = frames[:, :3].reshape(-1, N, 3)
        Tsteps = xyz_wrapped.shape[0]
        active_lags = MSD_LAGS[MSD_LAGS < Tsteps]

        update_msd_and_hists(
            xyz_wrapped=xyz_wrapped,
            L=L,
            lags=active_lags,
            edges_by_lag=edges_by_lag,
            msd_sum=msd_sum,
            msd_sumsq=msd_sumsq,
            hist_counts=hist_counts,
        )
        msd_n += 1

        if (rep + 1) % CHECK_EVERY == 0:
            msd_mean = msd_sum / msd_n

            if prev_ckpt_mean is None:
                dist = np.nan
                ok_streak = 0
            else:
                dist = convergence_metric(msd_mean, prev_ckpt_mean)
                ok_streak = ok_streak + 1 if dist < TOL else 0

            prev_ckpt_mean = msd_mean.copy()

            print(
                f"[check] d={d:.6g} z={z0:.6g} reps={rep+1:5d} "
                f"MSD_relL2={dist} streak={ok_streak}/{PATIENCE}"
            )

            if ok_streak >= PATIENCE:
                print(f"[done] Converged after {rep+1} replicas.")
                converged = True
                break

    save_histograms_per_lag(outdir, MSD_LAGS, edges_by_lag, hist_counts)

    msd_mean = msd_sum / msd_n
    if msd_n >= 2:
        msd_var = (msd_sumsq - (msd_sum * msd_sum) / msd_n) / (msd_n - 1)
        msd_stderr = np.sqrt(msd_var / msd_n)
    else:
        msd_stderr = np.full_like(msd_mean, np.nan)

    tau = MSD_LAGS * dt
    out = np.column_stack(
        [
            tau,
            msd_mean[:, 0], msd_mean[:, 1], msd_mean[:, 2],
            msd_mean[:, 3], msd_mean[:, 4], msd_mean[:, 5],
            msd_stderr[:, 0], msd_stderr[:, 1], msd_stderr[:, 2],
            msd_stderr[:, 3], msd_stderr[:, 4], msd_stderr[:, 5],
        ]
    )

    header = (
        "tau "
        "msd_rx msd_ry msd_rz msd_cx msd_cy msd_cz "
        "stderr_rx stderr_ry stderr_rz stderr_cx stderr_cy stderr_cz"
    )
    np.savetxt(os.path.join(outdir, "msd_vs_tau.txt"), out, header=header)
    np.save(os.path.join(outdir, "msd_vs_tau.npy"), out)

    with open(os.path.join(outdir, "msd_metadata.txt"), "w") as f:
        f.write(f"msd_n_replicas {msd_n}\n")
        f.write(f"MSD_LAGS {MSD_LAGS.tolist()}\n")
        f.write(f"dt {dt}\n")
        f.write(f"L {L}\n")
        f.write(f"d {d}\n")
        f.write(f"z {z0}\n")
        f.write("relative mode r = x1 - x2; cooperative mode c = x1 + x2 (use 0.5*(x1+x2) for COM)\n")
        f.write("columns in msd_vs_tau.txt: " + header + "\n")

    if converged:
        with open(os.path.join(outdir, "converged.txt"), "w") as f:
            f.write(f"converged True\nreps {msd_n}\n")
    else:
        with open(os.path.join(outdir, "converged.txt"), "w") as f:
            f.write(f"converged False\nreps {msd_n}\n")


if __name__ == "__main__":
    main()

