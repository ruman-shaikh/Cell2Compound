"""Re-summarise a docking validation run from its saved runs.csv, without re-docking.

Adds the per-chemotype breakdown to summary.json: hydroxamates chelate the catalytic zinc
bidentately through a flexible linker, benzamides do not, and the protocol does not perform
equally on both.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from cell2compound.docking import pick_scoring, summarise_runs

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dir',type=Path,default=Path('results/docking/validation'))
parser.add_argument('--threshold',type=float,default=2.)
args=parser.parse_args()
runs=pd.read_csv(args.dir/'runs.csv')
summary=summarise_runs(runs,args.threshold)
report=json.loads((args.dir/'summary.json').read_text()) if (args.dir/'summary.json').exists() else {}
report['rmsd_threshold_angstrom']=args.threshold
report['success_rates_rmsd_le_2A']={f'{m}/{s}':v for (m,s),v in summary['overall'].to_dict('index').items()}
report['success_rates_by_chemotype']={f'{c}/{m}/{s}':v for (c,m,s),v in summary['by_chemotype'].to_dict('index').items()}
report['median_top1_rmsd_by_chemotype']={f'{c}/{m}/{s}':v for (c,m,s),v in summary['median_top1_rmsd_by_chemotype'].items()}
report['median_top1_rmsd']={f'{p}/{m}':v for (p,m),v in summary['per_complex'].to_dict('index').items()}
report['selection_rule']=('Scoring function for candidate docking = best crossdock top-1 success '
                          '(RMSD <= threshold); zinc engagement breaks ties.')
report['scoring_selected']=pick_scoring(summary)
report.setdefault('limitations',[]).append(
    'Success rates differ by chemotype; the pooled rate should not be quoted alone.')
report['limitations']=list(dict.fromkeys(report['limitations']))
(args.dir/'summary.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(summary['overall'].to_string()); print()
print(summary['by_chemotype'].to_string()); print()
print(summary['per_complex'].to_string())
print('\nselected scoring:',report['scoring_selected'])
