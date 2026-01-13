#!/usr/bin/env python3
import os
import argparse
import numpy as np

import parameters as params


def index_to_pair(k: int, dbins: int, hbins: int):
    """
    Flat index k in [ 0, dbins*hbins-1 ] -> (i, j)
      i = index in d
      j = index in h
    """
    if k < 0 or k >= dbins * hbins:
        raise ValueError(f"k={k} out of range (0..{dbins*hbins-1})")
    i = k // hbins
    j = k % hbins
    return i, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Override SLURM_ARRAY_TASK_ID (for local tests).",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Base output directory. If omitted, uses /scratch/$USER/<project>/results.",
    )
    args = ap.parse_args()

    # Determine task id
    if args.task_id is not None:
        k = int(args.task_id)
    else:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            raise RuntimeError(
                "SLURM_ARRAY_TASK_ID not set. Use --task-id for local testing."
            )
        k = int(env)

    dbins = int(params.dbins)
    hbins = int(params.hbins)

    i, j = index_to_pair(k, dbins, hbins)
    d_val = float(params.d[i])
    h_val = float(params.h[j])

    if args.outdir:
        base = args.outdir
    else:
        base = os.path.join("/scratch", os.environ.get("USER"))  # add the date

    pair_dir = os.path.join(base, f"d_{d_val:.6e}", f"h_{h_val:.6e}")
    os.makedirs(pair_dir, exist_ok=True)

    np.savez(
        os.path.join(pair_dir, "meta.npz"),
        task_id=k,
        d_index=i,
        h_index=j,
        d=d_val,
        h=h_val,
        a=float(params.a),
        rho_fluid=float(params.rho_fluid),
        rho_bead=float(params.rho_bead),
        eta=float(params.eta),
        kbT=float(params.kbT),
        g=float(params.g),
    )

    # ---- run ----
    # Example placeholder:
    with open(os.path.join(pair_dir, "run.txt"), "w") as f:
        f.write(f"SLURM_ARRAY_TASK_ID={k}\n")
        f.write(f"d[{i}]={d_val:.6e}\n")
        f.write(f"h[{j}]={h_val:.6e}\n")
        f.write("Replace this section with your actual simulation call.\n")


if __name__ == "__main__":
    main()
