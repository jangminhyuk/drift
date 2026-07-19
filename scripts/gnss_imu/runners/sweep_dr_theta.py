"""
Parallel DR theta_w × theta_v grid sweep on KF-GINS.

Runs N_PARALLEL evaluations concurrently using subprocess. Each eval
calls main_kfgins.py --theta_w --theta_v on one (sequence, theta_w, theta_v)
triple. Results are saved as individual JSON files under the output dir.

Usage:
    python scripts/gnss_imu/runners/sweep_dr_theta.py \
        --seqs playground00 building01 parking00 \
        --output_dir results_gnss/i2nav/theta_sweep_v2 \
        --n_parallel 4
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time


THETA_W_GRID = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
THETA_V_GRID = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

ALL_SEQS = [
    'building00', 'building01', 'building02', 'parking00',
    'playground00', 'street00', 'street01', 'street02',
]


def run_one(seq, tw, tv, output_dir, gnss_source='F9P',
            checkpoint=None, no_R=True, timeout=600, mu_v_clip=None):
    """Run one (seq, theta_w, theta_v) config via subprocess.

    If checkpoint is provided, the adapter is applied with --no_R by default
    (matches the legacy kfgins_muv sweep pattern: μv adaptation only + DR).
    Set no_R=False to let the adapter's R head fire too.

    mu_v_clip overrides main_kfgins.py's default 2.0 m injection clip when
    given. Use ~5-10 m for UrbanNav-style multipath regimes.
    """
    out_file = os.path.join(output_dir, f'{seq}_tw{tw}_tv{tv}.json')
    if os.path.exists(out_file):
        return out_file  # skip if already done (resume support)

    cmd = [
        sys.executable,
        'scripts/gnss_imu/runners/main_kfgins.py',
        '--seq', seq,
        '--gnss_source', gnss_source,
        '--theta_w', str(tw),
        '--theta_v', str(tv),
        '--output', out_file,
    ]
    if checkpoint is not None:
        cmd += ['--checkpoint', checkpoint]
        if no_R:
            cmd += ['--no_R']
    if mu_v_clip is not None:
        cmd += ['--mu_v_clip', str(mu_v_clip)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd='.',
        )
        if result.returncode != 0:
            print(f'  FAIL {seq} tw={tw} tv={tv}: {result.stderr[-200:]}',
                  flush=True)
            return None
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT {seq} tw={tw} tv={tv}', flush=True)
        return None
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=ALL_SEQS)
    parser.add_argument('--gnss_source', default='F9P')
    parser.add_argument('--output_dir', default='results_gnss/i2nav/theta_sweep_v2')
    parser.add_argument('--n_parallel', type=int, default=4)
    parser.add_argument('--theta_w_grid', nargs='+', type=float, default=THETA_W_GRID)
    parser.add_argument('--theta_v_grid', nargs='+', type=float, default=THETA_V_GRID)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Adapter checkpoint (enables adapter injection; '
                             'use_R defaults to False for μv-only sweep).')
    parser.add_argument('--use_R', action='store_true',
                        help='If set with --checkpoint, allow the adapter R '
                             'head to fire too (default: --no_R, μv only).')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Per-config subprocess timeout in seconds.')
    parser.add_argument('--mu_v_clip', type=float, default=None,
                        help='Override per-axis mu_v injection clip (m). '
                             'Default leaves main_kfgins.py at its 2.0 m '
                             'safety rail. Pass 5-10 for UrbanNav.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build task list: all (seq, tw, tv) triples
    tasks = list(itertools.product(args.seqs, args.theta_w_grid, args.theta_v_grid))
    n_total = len(tasks)
    print(f'DR theta sweep: {n_total} configs '
          f'({len(args.seqs)} seqs × {len(args.theta_w_grid)} tw × {len(args.theta_v_grid)} tv)')
    print(f'Parallelism: {args.n_parallel}')
    print(f'Output: {args.output_dir}')
    print(flush=True)

    # Run with bounded parallelism via subprocess pool
    from concurrent.futures import ProcessPoolExecutor, as_completed

    t0 = time.time()
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.n_parallel) as pool:
        futures = {
            pool.submit(run_one, seq, tw, tv, args.output_dir,
                        args.gnss_source,
                        checkpoint=args.checkpoint,
                        no_R=(not args.use_R),
                        timeout=args.timeout,
                        mu_v_clip=args.mu_v_clip): (seq, tw, tv)
            for seq, tw, tv in tasks
        }
        for future in as_completed(futures):
            seq, tw, tv = futures[future]
            completed += 1
            result = future.result()
            if result is None:
                failed += 1
            elapsed = time.time() - t0
            eta = elapsed / completed * (n_total - completed) if completed > 0 else 0
            if completed % 10 == 0 or completed == n_total:
                print(f'  [{completed}/{n_total}] {elapsed:.0f}s elapsed, '
                      f'~{eta:.0f}s remaining, {failed} failed', flush=True)

    wall = time.time() - t0
    print(f'\nSweep done: {completed}/{n_total} in {wall:.0f}s ({failed} failed)')

    # Aggregate results into a summary table
    print(f'\n{"seq":14} {"theta_w":>8} {"theta_v":>8} {"3D_RMSE":>10}')
    for seq in args.seqs:
        best_rmse = float('inf')
        best_tw, best_tv = None, None
        for tw in args.theta_w_grid:
            for tv in args.theta_v_grid:
                jf = os.path.join(args.output_dir, f'{seq}_tw{tw}_tv{tv}.json')
                if os.path.exists(jf):
                    data = json.load(open(jf))[0]
                    r3d = data['rmse']['rmse_3d']
                    if r3d < best_rmse:
                        best_rmse = r3d
                        best_tw, best_tv = tw, tv
        if best_tw is not None:
            print(f'{seq:14} {best_tw:>8} {best_tv:>8} {best_rmse:>10.3f}')

    # Save full grid as CSV for easy analysis
    csv_path = os.path.join(args.output_dir, 'sweep_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('seq,theta_w,theta_v,rmse_n,rmse_e,rmse_d,rmse_h,rmse_3d\n')
        for seq in args.seqs:
            for tw in args.theta_w_grid:
                for tv in args.theta_v_grid:
                    jf = os.path.join(args.output_dir, f'{seq}_tw{tw}_tv{tv}.json')
                    if os.path.exists(jf):
                        data = json.load(open(jf))[0]
                        r = data['rmse']
                        f.write(f'{seq},{tw},{tv},{r["rmse_n"]:.4f},{r["rmse_e"]:.4f},'
                                f'{r["rmse_d"]:.4f},{r["rmse_h"]:.4f},{r["rmse_3d"]:.4f}\n')
    print(f'\nCSV: {csv_path}')


if __name__ == '__main__':
    main()
