"""Cross-dataset summary of the two headline contrasts, with per-seed spread."""
import json
from pathlib import Path
import numpy as np, pandas as pd

out = Path('results/dose_regime')
labels = ['A549_24h', 'K562_24h', 'MCF7_24h', 'A549_72h']
KEY = 'tanimoto_kernel_ridge'

snr_rows, contrast_rows, measurability = [], [], []
mediation, calibrated = [], []
for lab in labels:
    s1 = out / f'stage1_{lab}.json'
    if s1.exists():
        rep = json.loads(s1.read_text())
        for r in rep['per_dose']:
            snr_rows.append({'dataset': lab, **{k: r[k] for k in
                            ('dose_nM', 'groups', 'floor_median', 'signal_median', 'snr_median',
                             'frac_groups_snr_above_1', 'cells_median')}})
    raw = out / f'stage2_raw_{lab}.csv'
    if not raw.exists():
        continue
    r = pd.read_csv(raw)
    for model in sorted(r.model.unique()):
        k = r[(r.model == model) & (r.stratum == 'all')]
        if k.empty:
            continue
        piv = k.pivot_table(index='seed', columns='arm', values='rmse')
        if not {'full', 'class_onehot'}.issubset(piv.columns):
            continue
        d_oh = piv.class_onehot - piv.full
        d_wc = (piv.within_class_scramble - piv.full) if 'within_class_scramble' in piv else pd.Series([np.nan])
        d_gs = (piv.global_scramble - piv.full) if 'global_scramble' in piv else pd.Series([np.nan])
        contrast_rows.append({'dataset': lab, 'model': model,
                              'full_rmse': round(piv.full.mean(), 4),
                              'onehot_rmse': round(piv.class_onehot.mean(), 4),
                              'onehot_minus_full': round(d_oh.mean(), 5),
                              'sd': round(d_oh.std(), 5),
                              'sd_units': round(d_oh.mean() / d_oh.std(), 2) if d_oh.std() else np.nan,
                              'withinclass_minus_full': round(d_wc.mean(), 5),
                              'globalscramble_minus_full': round(d_gs.mean(), 5)})
    # measurability by SNR quartile
    sm = pd.read_csv(out / f'stage2_summary_{lab}.csv')
    q = sm[(sm.model == KEY) & (sm.arm == 'full') & sm.stratum.str.startswith('snr_q')]
    for _, row in q.iterrows():
        measurability.append({'dataset': lab, 'stratum': row.stratum, 'groups': int(row.groups),
                              'floor': round(row.floor, 4), 'headroom': round(row.headroom, 4),
                              'measurable': bool(row.measurable),
                              'attainable': None if pd.isna(row.attainable) else round(row.attainable, 3)})

    # Stage 3: is dose anything beyond measurability, and does an outside metric agree?
    s3 = out / f'stage3_{lab}.json'
    if s3.exists():
        m = json.loads(s3.read_text())
        for arm in ('full', 'class_onehot'):
            a = m[arm]
            lo, hi = a['dose_given_snr_ci95']
            mediation.append({'dataset': lab, 'arm': arm,
                              'dose_alone': round(a['dose_coefficient_marginal'], 5),
                              'dose_given_snr': round(a['dose_coefficient_given_snr'], 5),
                              'ci_lo': round(lo, 5), 'ci_hi': round(hi, 5),
                              'ci_includes_zero': bool(lo < 0 < hi),
                              'attenuation': round(a['attenuation'], 3),
                              'r2_dose_only': round(a['r2_dose_only'], 3),
                              'r2_snr_only': round(a['r2_snr_only'], 3),
                              'r2_both': round(a['r2_both'], 3)})
    cal = out / f'stage3_calibrated_{lab}.csv'
    if cal.exists():
        c = pd.read_csv(cal)
        for _, row in c[c.arm == 'full'].iterrows():
            drf = row.get(f'drf_{KEY}')
            calibrated.append({'dataset': lab, 'stratum': row.stratum, 'groups': int(row.groups),
                               'negative_baseline': row.negative_baseline_rmse,
                               'positive_control': row.positive_control_rmse,
                               'dynamic_range': row.dynamic_range,
                               # Must use the same guard dynamic_range_fraction() applies, or a
                               # stratum can read range_exists=True with an undefined DRF beside it.
                               'range_exists': bool(row.dynamic_range >
                                                    max(1e-9, .05 * row.positive_control_rmse)),
                               'drf': None if pd.isna(drf) else round(float(drf), 3)})

snr = pd.DataFrame(snr_rows); contrasts = pd.DataFrame(contrast_rows); meas = pd.DataFrame(measurability)
print('================ SNR vs dose, all datasets ================')
print(snr.to_string(index=False))
print('\n================ class one-hot vs full fingerprint ================')
print(contrasts.to_string(index=False))
print('\n================ measurability by SNR quartile ================')
print(meas.to_string(index=False))
snr.to_csv(out / 'summary_snr_by_dose.csv', index=False)
contrasts.to_csv(out / 'summary_onehot_vs_fingerprint.csv', index=False)
meas.to_csv(out / 'summary_measurability.csv', index=False)

med = pd.DataFrame(mediation); cal_frame = pd.DataFrame(calibrated)
if not med.empty:
    print('\n================ stage 3: does dose survive conditioning on SNR? ================')
    print(med.to_string(index=False))
    med.to_csv(out / 'summary_mediation.csv', index=False)
if not cal_frame.empty:
    print('\n================ stage 3: calibrated metric (interpolated duplicate) ================')
    print(cal_frame.to_string(index=False))
    cal_frame.to_csv(out / 'summary_calibrated.csv', index=False)

oh = contrasts[contrasts.model == KEY]
print('\n================ headline ================')
print(f'class one-hot vs 2048-bit fingerprint, {KEY}, across {len(oh)} datasets:')
for _, r in oh.iterrows():
    print(f'  {r.dataset:10s} delta {r.onehot_minus_full:+.5f}  sd {r.sd:.5f}  = {r.sd_units:+.2f} sd')
