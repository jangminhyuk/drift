import os
"""Aggregate the per-shard UWB eval (results/eval_table_v1/<name>__b{0,1}/) into:
  1. a per-sequence RMSE data file in the dataset-paper Table-4 form (4 methods);
  2. two headline-improvement plots (same style as improvement.pdf), one per TDOA
     mode, over ALL sequences;
  3. an RA-L LaTeX table of avg +/- std RMSE per (constellation, TDOA mode).

Run after the eval_table_v1_fast array completes.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import plot_8way as P

SHARD_ROOT = os.environ.get('UWB_TABLE_SHARD_ROOT',
                            os.path.join(os.environ.get('DRIFT_SHARD_ROOT','shards'), 'eval_table_v1'))  # bulk per-shard outputs
OUT_ROOT = os.environ.get('UWB_TABLE_OUT_ROOT',
                          'results/eval_table_v1')    # small deliverables (csv/tex/fig)
FIG = os.path.join(OUT_ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

METHODS = [('eskf', 'ESKF'), ('dr_eskf', 'DR-only'),
           ('mamba_eskf', 'Adapter-only'), ('mamba_dr', 'DRiFt')]
if os.environ.get('UWB_TABLE_ALL_BACKBONES'):
    METHODS += [('transformer_eskf', 'Transformer-only'), ('transformer_dr', 'Transformer-DRiFt'),
                ('gru_eskf', 'GRU-only'), ('gru_dr', 'GRU-DRiFt')]
MKEYS = [m for m, _ in METHODS]                          # 4 main: CSV + LaTeX table
# Also collect Transformer/GRU so the improvement plot can populate all backbones.
ALL_MKEYS = MKEYS + ['transformer_eskf', 'transformer_dr', 'gru_eskf', 'gru_dr']


def load_shards():
    """name -> {const_name, tdoa, methods:{mkey: rms_all}, grids:{mkey: {theta: rms}}}"""
    combined = {}
    for sj in sorted(glob.glob(f'{SHARD_ROOT}/*__*/eval_8way_summary.json')):
        try:
            s = json.load(open(sj))
        except Exception:
            continue
        for d in s.get('datasets', []):
            name = d['dataset_name']
            tdoa = 2 if 'tdoa2' in name else 3
            e = combined.setdefault(name, dict(const_name=d['const_name'],
                                               tdoa=tdoa, methods={}, grids={}))
            for mk, md in d['methods'].items():
                if mk in ALL_MKEYS:
                    r = md.get('rms_all')
                    if isinstance(r, (int, float)) and np.isfinite(r):
                        e['methods'][mk] = float(r)
                    g = md.get('grid')
                    if isinstance(g, dict) and g:
                        e['grids'][mk] = {
                            k: float(v['rms_all']) for k, v in g.items()
                            if isinstance(v.get('rms_all'), (int, float))
                            and np.isfinite(v['rms_all'])}
    return combined


def apply_val_theta(combined):
    """Validation-theta protocol (UWB_TABLE_THETA_PROTOCOL=val): for each
    DR method, select ONE theta per (constellation, tdoa) minimizing the mean
    ESKF-normalized RMSE over the validation trials (4, 5) — the same trials
    used for adapter model selection — then apply it to every sequence.
    Replaces the default per-sequence best-theta (oracle) numbers."""
    import re
    selection = {}
    dr_methods = [mk for mk in ALL_MKEYS if mk.endswith('_dr') or mk == 'dr_eskf']
    groups = {}
    for name, e in combined.items():
        groups.setdefault((e['const_name'], e['tdoa']), []).append(name)
    for (cn, tdoa), names in sorted(groups.items()):
        val = [n for n in names
               if int(re.search(r'trial(\d+)', n).group(1)) in (4, 5)]
        for mk in dr_methods:
            keys = None
            for n in val:
                g = combined[n]['grids'].get(mk)
                if g:
                    keys = sorted(g.keys()) if keys is None else [k for k in keys if k in g]
            if not keys:
                continue
            scores = {}
            for k in keys:
                vals = [combined[n]['grids'][mk][k] / combined[n]['methods']['eskf']
                        for n in val if mk in combined[n]['grids']
                        and 'eskf' in combined[n]['methods']]
                if vals:
                    scores[k] = float(np.mean(vals))
            if not scores:
                continue
            best_k = min(scores, key=scores.get)
            selection[f'{cn}_tdoa{tdoa}_{mk}'] = best_k
            for n in names:
                g = combined[n]['grids'].get(mk)
                if g and best_k in g:
                    combined[n]['methods'][mk] = g[best_k]
    with open(os.path.join(OUT_ROOT, 'valtheta_selection.json'), 'w') as f:
        json.dump(selection, f, indent=1, sort_keys=True)
    print('validation-theta protocol: selections written to '
          f'{OUT_ROOT}/valtheta_selection.json')
    for k, v in sorted(selection.items()):
        print(f'  {k}: {v}')
    return combined


def summary_for_tdoa(combined, tdoa):
    """Build a plot_8way-compatible summary for one TDOA mode."""
    datasets = []
    for name, e in sorted(combined.items()):
        if e['tdoa'] != tdoa:
            continue
        datasets.append(dict(dataset_name=name, const_name=e['const_name'],
                             methods={mk: {'rms_all': e['methods'][mk]}
                                      for mk in e['methods']}))
    return {'datasets': datasets}


def write_per_seq_csv(combined, path):
    rows = ['sequence,constellation,tdoa,' + ','.join(lbl for _, lbl in METHODS)]
    for name, e in sorted(combined.items()):
        vals = ['%.4f' % e['methods'][mk] if mk in e['methods'] else 'NaN'
                for mk in MKEYS]
        rows.append(f"{name},{e['const_name']},{e['tdoa']}," + ','.join(vals))
    open(path, 'w').write('\n'.join(rows) + '\n')
    print('wrote', path, f'({len(combined)} sequences)')


def write_latex(combined, path):
    # group (tdoa, const) -> per-method list of rms
    groups = {}
    for e in combined.values():
        key = (e['tdoa'], e['const_name'])
        g = groups.setdefault(key, {mk: [] for mk in MKEYS})
        for mk in MKEYS:
            if mk in e['methods']:
                g[mk].append(e['methods'][mk])

    def cell(vs, best=False):
        if not vs:
            return '--'
        m, s = np.mean(vs), np.std(vs)
        txt = '%.3f\\,$\\pm$\\,%.3f' % (m, s)
        return '\\textbf{%s}' % txt if best else txt

    if os.environ.get('UWB_TABLE_THETA_PROTOCOL') == 'val':
        theta_txt = ('DR radii $(\\theta_w,\\theta_v)$ are selected once per '
                     'constellation on the validation trials and held fixed '
                     'across sequences.')
    else:
        theta_txt = 'DR uses per-sequence best $(\\theta_w,\\theta_v)$.'
    scope_txt = ('all autonomous sequences'
                 if os.environ.get('UWB_TABLE_SCOPE') == 'autonomous'
                 else 'all sequences')
    tdoa_only = os.environ.get('UWB_TABLE_TDOA')
    mode_txt = {None: ', for centralized (TDOA2) and decentralized (TDOA3) modes',
                '2': ' (centralized TDOA2 mode)',
                '3': ' (decentralized TDOA3 mode)'}[tdoa_only]
    L = []
    L.append('\\begin{table}[t]')
    L.append('\\centering')
    L.append('\\caption{UWB localization RMSE (m), mean\\,$\\pm$\\,std over ' +
             scope_txt + ' per anchor constellation' + mode_txt +
             '. Adapter = Mamba; ' + theta_txt + '}')
    L.append('\\label{tab:uwb_rmse}')
    L.append('\\setlength{\\tabcolsep}{3pt}')
    L.append('\\resizebox{\\linewidth}{!}{%')
    L.append('\\begin{tabular}{ll' + 'c' * len(METHODS) + '}')
    L.append('\\toprule')
    L.append('Mode & Const. & ' + ' & '.join(lbl for _, lbl in METHODS) + ' \\\\')
    L.append('\\midrule')
    modes = sorted({t for (t, c) in groups})
    for tdoa in modes:
        consts = sorted({c for (t, c) in groups if t == tdoa})
        for i, cn in enumerate(consts):
            g = groups[(tdoa, cn)]
            means = {mk: (np.mean(g[mk]) if g[mk] else np.inf) for mk in MKEYS}
            best_mk = min(MKEYS, key=lambda mk: means[mk])
            cells = ' & '.join(cell(g[mk], best=(mk == best_mk)) for mk in MKEYS)
            lead = ('\\multirow{%d}{*}{TDOA%d}' % (len(consts), tdoa)) if i == 0 else ''
            label = cn.replace('const', '\\#')
            L.append(f'{lead} & {label} & {cells} \\\\')
        if tdoa != modes[-1]:
            L.append('\\midrule')
    L.append('\\bottomrule')
    L.append('\\end{tabular}}')
    L.append('\\end{table}')
    open(path, 'w').write('\n'.join(L) + '\n')
    print('wrote', path)


def main():
    combined = load_shards()
    if os.environ.get('UWB_TABLE_SCOPE') == 'autonomous':
        combined = {n: e for n, e in combined.items() if 'manual' not in n}
        print('scope: autonomous sequences only (manual flights excluded)')
    tdoa_only = os.environ.get('UWB_TABLE_TDOA')
    if tdoa_only:
        combined = {n: e for n, e in combined.items()
                    if e['tdoa'] == int(tdoa_only)}
        print(f'mode filter: TDOA{tdoa_only} only')
    if os.environ.get('UWB_TABLE_THETA_PROTOCOL') == 'val':
        combined = apply_val_theta(combined)
    print(f'aggregated {len(combined)} sequences')
    n2 = sum(1 for e in combined.values() if e['tdoa'] == 2)
    n3 = sum(1 for e in combined.values() if e['tdoa'] == 3)
    full = sum(1 for e in combined.values() if len(e['methods']) == 4)
    print(f'  tdoa2={n2} tdoa3={n3}; with all 4 methods={full}')

    write_per_seq_csv(combined, os.path.join(OUT_ROOT, 'uwb_rmse_per_sequence.csv'))
    write_latex(combined, os.path.join(OUT_ROOT, 'uwb_rmse_table.tex'))

    for tdoa in (2, 3):
        summ = summary_for_tdoa(combined, tdoa)
        if summ['datasets']:
            P.plot_headline_improvement(summ, FIG, out_name=f'improvement_tdoa{tdoa}.pdf')


if __name__ == '__main__':
    main()
