"""External validation: is the measurability result a property of sci-Plex, or of the field?

The central claim in `docs/dose_regime_findings.md` rests on 188 compounds in one assay. The
obvious objection is that sci-Plex is shallow by construction and the result says more about
sci-Plex than about chemical perturbation benchmarks. LINCS L1000 Phase II is the test: a
different platform (bulk L1000 rather than single-cell combinatorial indexing), a different lab,
1,796 compounds instead of 188, 30 cell lines instead of 3, and a median of 3 replicates per
condition instead of our 2.

What transfers and what does not:

  transfers      the replicate floor, snr, and the median-group claim. Replication is per well,
                 so a floor is estimable exactly as it is for us.
  does NOT       the depth law (section 2). L1000 is bulk, so there is no cell count per
                 condition and 1/sqrt(cells) cannot be tested here. Stated, not fudged.

Level 3 is per-instance normalised expression, not a response, so each condition is referenced to
the vehicle controls ON ITS OWN DETECTION PLATE. That is the plate-matched convention, which our
own ablation found slightly worse than pooling but which is the only honest choice here: LINCS
plates are not interchangeable.

Paths are arguments because the download directory holds untrusted third-party files.
"""
import argparse, json
from pathlib import Path
import h5py
import numpy as np, pandas as pd
from cell2compound.measurability import noise_metrics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('gctx', type=Path, help='GSE70138 Level 3 .gctx')
parser.add_argument('inst_info', type=Path)
parser.add_argument('gene_info', type=Path)
parser.add_argument('--cell-lines', nargs='+', default=['MCF7', 'PC3', 'A549', 'HA1E', 'VCAP'])
parser.add_argument('--time', type=float, default=24.)
parser.add_argument('--min-replicates', type=int, default=2)
parser.add_argument('--min-controls', type=int, default=4, help='vehicle wells needed on a plate')
parser.add_argument('--out', type=Path, default=Path('results/lincs'))
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

inst = pd.read_csv(args.inst_info, sep='\t', low_memory=False)
genes = pd.read_csv(args.gene_info, sep='\t', low_memory=False)
lm_col = next((c for c in ('pr_is_lm', 'is_lm') if c in genes.columns), None)
landmark = (genes[genes[lm_col] == 1] if lm_col else genes).pr_gene_id.astype(str).to_numpy()
print(f'landmark genes in gene_info: {len(landmark):,}')

with h5py.File(args.gctx, 'r') as f:
    row_ids = np.array([s.decode() if isinstance(s, bytes) else str(s)
                        for s in f['0/META/ROW/id'][:]])
    col_ids = np.array([s.decode() if isinstance(s, bytes) else str(s)
                        for s in f['0/META/COL/id'][:]])
    dset = f['0/DATA/0/matrix']
    print(f'matrix {dset.shape}  rows(ids) {len(row_ids):,}  cols(ids) {len(col_ids):,}')
    # gctx stores the matrix transposed relative to its id vectors in some releases; resolve it
    # by shape rather than trusting a convention, because guessing would silently transpose data.
    if dset.shape == (len(col_ids), len(row_ids)):
        instances_are_rows = True
    elif dset.shape == (len(row_ids), len(col_ids)):
        instances_are_rows = False
    else:
        raise ValueError(f'matrix shape {dset.shape} matches neither id vector layout')
    print(f'instances are {"rows" if instances_are_rows else "columns"} of the stored matrix')

    gene_index = {g: i for i, g in enumerate(row_ids)}
    keep_genes = np.array([gene_index[g] for g in landmark if g in gene_index])
    print(f'landmark genes present in matrix: {len(keep_genes):,}')

    inst = inst[inst.inst_id.isin(set(col_ids))]
    position = {c: i for i, c in enumerate(col_ids)}
    records, per_dose_rows = [], []

    for cell in args.cell_lines:
        sub = inst[(inst.cell_id == cell) & (inst.pert_time == args.time)]
        if not len(sub):
            print(f'{cell}: no instances at {args.time} h'); continue
        treated = sub[sub.pert_type == 'trt_cp']
        vehicle = sub[sub.pert_type == 'ctl_vehicle']
        if not len(vehicle):
            print(f'{cell}: no vehicle controls at {args.time} h'); continue

        # Load only this cell line's wells, landmark genes only, in row chunks. The full matrix is
        # 345,976 x 12,328 float32 (~17 GB), and slicing a whole cell line at once would
        # materialise every gene before the landmark subset is taken -- several GB, which is more
        # than is free when chemCPA training is resident.
        wanted = pd.concat([treated, vehicle])
        cols = np.array([position[i] for i in wanted.inst_id])
        order = np.argsort(cols)
        picked = cols[order]
        block = np.empty((len(picked), len(keep_genes)), dtype=np.float32)
        step = 2048
        for start in range(0, len(picked), step):
            chunk = picked[start:start + step]
            block[start:start + len(chunk)] = (dset[chunk, :][:, keep_genes] if instances_are_rows
                                               else dset[:, chunk][keep_genes, :].T)
        frame_index = {inst_id: k for k, inst_id in enumerate(wanted.inst_id.to_numpy()[order])}
        print(f'{cell}: loaded {block.shape[0]:,} wells x {block.shape[1]:,} landmark genes')

        plate_control = {}
        for plate, part in vehicle.groupby('det_plate'):
            rows = [frame_index[i] for i in part.inst_id if i in frame_index]
            if len(rows) >= args.min_controls:
                plate_control[plate] = block[rows].mean(0)

        full, diff, meta = [], [], []
        for (pert, dose), part in treated.groupby(['pert_id', 'pert_dose']):
            usable = part[part.det_plate.isin(plate_control)]
            if len(usable) < args.min_replicates:
                continue
            deltas = np.stack([block[frame_index[r.inst_id]] - plate_control[r.det_plate]
                               for r in usable.itertuples() if r.inst_id in frame_index])
            if len(deltas) < args.min_replicates:
                continue
            # Split replicates into two halves so the floor is the disagreement of two independent
            # means, matching our rep1/rep2 definition rather than a within-group variance.
            half = len(deltas) // 2
            a, b = deltas[:half].mean(0), deltas[half:2 * half].mean(0)
            full.append(deltas.mean(0)); diff.append(a - b)
            meta.append({'cell_id': cell, 'pert_id': pert, 'dose_um': float(dose),
                         'replicates': int(len(deltas)), 'plates': int(usable.det_plate.nunique())})
        if not meta:
            print(f'{cell}: no conditions survived'); continue

        metrics = noise_metrics(np.stack(full), np.stack(diff))
        table = pd.DataFrame(meta)
        for k, v in metrics.items():
            table[k] = v
        records.append(table)
        print(f'{cell}: {len(table):,} conditions  median snr {np.nanmedian(table.snr):.3f}  '
              f'>= snr 1: {np.mean(table.snr >= 1):.1%}')

if not records:
    raise SystemExit('no cell line produced usable conditions')

groups = pd.concat(records, ignore_index=True)
groups.to_csv(args.out / 'conditions.csv', index=False)

# Dose is a continuous micromolar value in LINCS, so bin by decade to mirror our four-dose design.
groups['dose_decade'] = np.floor(np.log10(groups.dose_um.clip(lower=1e-4))).astype(int)
summary = {'source': args.gctx.name, 'time_h': args.time, 'conditions': int(len(groups)),
           'compounds': int(groups.pert_id.nunique()),
           'cell_lines': sorted(groups.cell_id.unique()),
           'median_replicates': float(groups.replicates.median()),
           'median_snr': float(np.nanmedian(groups.snr)),
           'fraction_snr_above_1': float(np.mean(groups.snr >= 1)),
           'fraction_signal_exactly_zero': float(np.mean(groups.signal == 0))}

by_cell = (groups.groupby('cell_id')
           .agg(conditions=('snr', 'size'), median_snr=('snr', 'median'),
                frac_above_1=('snr', lambda s: float(np.mean(s >= 1))),
                median_floor=('floor', 'median'))).round(4)
by_dose = (groups.groupby('dose_decade')
           .agg(conditions=('snr', 'size'), median_snr=('snr', 'median'),
                frac_above_1=('snr', lambda s: float(np.mean(s >= 1))),
                median_floor=('floor', 'median'))).round(4)
by_cell.to_csv(args.out / 'summary_by_cell.csv')
by_dose.to_csv(args.out / 'summary_by_dose.csv')
(args.out / 'summary.json').write_text(json.dumps(summary, indent=1))

print()
print('=== LINCS L1000 Phase II, external check of the measurability claim ===')
print(json.dumps(summary, indent=1))
print()
print('--- by cell line ---'); print(by_cell.to_string())
print()
print('--- by dose decade (log10 micromolar) ---'); print(by_dose.to_string())
print()
verdict = ('REPLICATES: the median condition is below its own replicate noise here too'
           if summary['median_snr'] < 1 else
           'DOES NOT REPLICATE: the median condition clears its noise floor in LINCS')
print(verdict)
