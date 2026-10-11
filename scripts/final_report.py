"""Collect every stage's saved outputs into one report: skill -> evidence map, headline
numbers with their comparators, resource measurements, and the open limitations.
Reads only files already written by the pipeline; computes no new scientific results."""
import json
import platform
import subprocess
from pathlib import Path
import pandas as pd

RESULTS=Path('results')


def read_csv_records(path, default=None):
    """CSV rows as dicts, with numbers and booleans parsed.

    csv.DictReader yields strings for everything, and this report is the machine-readable
    roll-up, so leaving "0.00345" and "True" as text would push the casting onto every consumer.
    """
    path=Path(path)
    if not path.exists():
        return default
    import csv
    def coerce(value):
        if value in ('','None','nan','NaN'):
            return None
        if value in ('True','False'):
            return value=='True'
        try:
            number=float(value)
        except ValueError:
            return value
        return int(number) if number.is_integer() and '.' not in value and 'e' not in value.lower() else number
    with path.open() as handle:
        return [{k:coerce(v) for k,v in row.items()} for row in csv.DictReader(handle)]


def load(path, default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def table(path):
    path=Path(path)
    return pd.read_csv(path) if path.exists() else None


environment=load(RESULTS/'environment.json',{})
qc=load(RESULTS/'qc/summary.json',{})
pseudobulk=load('data/pseudobulk_A549_24h.json',{})
expression=load(RESULTS/'expression/metrics.json',{})
plate=load(RESULTS/'expression_plate/metrics.json',{})
property_=load(RESULTS/'property/metrics.json',{})
chemistry=load(RESULTS/'chemistry/summary.json',{})
targets=load(RESULTS/'targets/summary.json',{})
bioactivity=load('data/bioactivity/summary.json',{})
esm=load(RESULTS/'esm_panel/manifest.json',{})
target_aware=load(RESULTS/'target_aware/summary.json',{})
pocket=load(RESULTS/'structures/pocket_comparison.json',{})
structures=load(RESULTS/'structures/inventory.json',{})
validation=load(RESULTS/'docking/validation/summary.json',{})
candidates=load(RESULTS/'docking/candidates/summary.json',{})
# dock_candidates.py nests its metrics per receptor rather than at the top level. Reading them
# flat silently yielded nulls for a run that had in fact completed, so resolve them explicitly.
matched=(candidates or {}).get('chemotype_matched_receptor') or {}
stress=load(RESULTS/'stress_test/summary.json')
chemcpa=load(RESULTS/'chemcpa/comparison.json')
chemcpa_strata=load(RESULTS/'chemcpa/snr_strata.json')
measurability={label:load(RESULTS/f'dose_regime/stage1_{label}.json')
               for label in ('A549_24h','K562_24h','MCF7_24h','A549_72h')}
measurability={k:v for k,v in measurability.items() if v}
dose_gate=load(RESULTS/'dose_regime/stage0_A549_24h.json')
time_matched=load(RESULTS/'dose_regime/time_matched_A549.json')
ablation=read_csv_records(RESULTS/'dose_regime/summary_onehot_vs_fingerprint.csv')
mediation=read_csv_records(RESULTS/'dose_regime/summary_mediation.csv')
calibrated=read_csv_records(RESULTS/'dose_regime/summary_calibrated.csv')
chemcpa_seeds=load(RESULTS/'chemcpa/seed_comparison.json')
lincs=load(RESULTS/'lincs/summary.json')
lincs_by_cell=read_csv_records(RESULTS/'lincs/summary_by_cell.csv')
lincs_by_dose=read_csv_records(RESULTS/'lincs/summary_by_dose.csv')


def best(summary, metric='rmse'):
    """Model with the lowest mean metric, and the best non-learned comparator."""
    if not summary:
        return None
    learned={k:v for k,v in summary.items() if k in ('ridge','mlp','tanimoto_knn')}
    baseline={k:v for k,v in summary.items() if k in ('zero_response','training_mean_response','dose_mean_response')}
    pick=lambda d:min(d.items(),key=lambda kv:kv[1][metric]['mean']) if d else None
    return {'best_learned':pick(learned),'best_baseline':pick(baseline)}


expression_summary=expression.get('summary_over_seeds',{})
ranked=best(expression_summary)
skills=[]


def skill(name, done, evidence, numbers=None, caveat=None):
    skills.append({'skill':name,'demonstrated':done,'evidence':evidence,'numbers':numbers,'caveat':caveat})


skill('RDKit / SMILES', bool(chemistry),
      ['cell2compound/chemistry.py','cell2compound/compounds.py','tests/test_chemistry.py','tests/test_compounds.py',
       'results/expression/compound_mapping.csv'],
      {'sciplex_names_mapped':expression.get('compounds'),
       'distinct_parent_structures':expression.get('unique_structures'),
       'unmatched_names':len(expression.get('unmatched_names',[]))},
      'Salt stripping uses RDKit ChargeParent; two sci-Plex names share one parent structure by design.')
skill('Fingerprints / descriptors', bool(chemistry),
      ['results/chemistry/descriptors.csv','results/chemistry/fingerprints.npz'],
      {'fingerprint':chemistry.get('fingerprint')},
      'Morgan fingerprints feed every model; descriptors are the property-model features.')
skill('ChEMBL', bool(targets),
      ['scripts/chembl_targets.py','results/targets/compound_chembl.csv','results/targets/mechanisms.csv',
       'scripts/bioactivity_dataset.py','data/bioactivity/pairs.csv'],
      {'chembl_release':targets.get('chembl_release'),'structures_resolved':targets.get('match'),
       'parents_with_mechanism':targets.get('parents_with_mechanism'),
       'bioactivity_pairs':bioactivity.get('pairs'),'bioactivity_structures':bioactivity.get('structures')},
      'Identity by InChIKey; connectivity-only matches are flagged, not merged silently.')
skill('Property prediction', bool(property_),
      ['scripts/property_baseline.py','results/property/metrics.json','results/property/test_predictions.csv'],
      {'summary_over_seeds':property_.get('summary_over_seeds'),'seeds':property_.get('seeds'),
       'split_sizes':property_.get('split_sizes'),'models_selected':property_.get('models_selected'),
       'empty_scaffold_policy':property_.get('empty_scaffold_policy'),
       'acyclic_sharing_empty_scaffold':property_.get('acyclic_structures_sharing_empty_scaffold')},
      'Five scaffold-disjoint splits on ESOL; 316 of 1117 structures are acyclic and share the empty '
      'Murcko scaffold, so the policy for that group is reported as an ablation.')
skill('Single-cell / gene expression', bool(expression),
      ['scripts/pseudobulk.py','scripts/single_cell_qc.py','scripts/expression_baseline.py',
       'results/qc/summary.json','results/expression/metrics.json','results/expression/per_compound_errors.csv'],
      {'cells_used':pseudobulk.get('cells_kept'),'compound_dose_groups':expression.get('groups'),
       'genes':expression.get('genes'),'control_cells':expression.get('control_cells'),
       'best_learned':ranked and ranked['best_learned'],'best_baseline':ranked and ranked['best_baseline'],
       'replicate_noise_floor':expression.get('replicate_noise_floor_all_groups_first_seed_genes')},
      'Compound-held-out over 5 split seeds; learned gain over the dose-mean baseline is small next to the '
      'replicate floor. See the mechanism stress test for whether that gain generalises beyond '
      'mechanism classes already seen in training.')
skill('ESM', bool(esm),
      ['scripts/esm_embeddings.py','results/esm_panel/manifest.json','scripts/target_aware_model.py',
       'results/target_aware/summary.json'],
      {'model':esm.get('model'),'targets':len(esm.get('targets',[])),
       'ablation':target_aware.get('summary'),
       'esm_beats_shuffled_targets':target_aware.get('esm_beats_shuffled_targets'),
       'esm_beats_none_targets':target_aware.get('esm_beats_none_targets')},
      'Evaluated against one-hot, no-target and shuffled-embedding controls; not appended for coverage.')
skill('AlphaFold', bool(pocket),
      ['scripts/pocket_comparison.py','results/structures/pocket_comparison.json',
       'results/structures/HDAC2_alphafold_zn_crystal_frame.pdb'],
      {'pocket_plddt':pocket.get('pocket_plddt'),'pocket_ca_rmsd':pocket.get('pocket_ca_rmsd_global_frame'),
       'pocket_heavy_atom_rmsd':pocket.get('pocket_heavy_atom_rmsd_global_frame'),
       'zinc_coordination':pocket.get('zinc_coordination')},
      'Retrieved predictions analysed; no local folding. AlphaFold models apo HDAC2, so the zinc is transplanted.')
skill('PDB structures', bool(structures),
      ['scripts/structure_inventory.py','results/structures/inventory.json','scripts/docking_validation.py'],
      {'experimental_chains':len((structures.get('experimental') or {}).get('chains',[])),
       'validation_structures':[s['pdb'] for s in (validation or {}).get('structures',[])]},
      'Ligand-bound chains selected by proximity of the ligand to that chain\'s catalytic zinc.')
skill('Docking', bool(validation),
      ['cell2compound/docking.py','scripts/docking_validation.py','results/docking/validation/summary.json',
       'scripts/dock_candidates.py','results/docking/candidates/summary.json'],
      {'success_rates':(validation or {}).get('success_rates_rmsd_le_2A'),
       'scoring_used_for_candidates':(candidates or {}).get('scoring'),
       'library':(candidates or {}).get('library'),
       # Candidate metrics are nested per receptor; the chemotype-matched receptor is the primary
       # read, because validation showed one receptor cannot serve both chemotypes.
       'auroc_zbg_vs_comparator':matched.get('auroc_zbg_vs_comparator'),
       'zinc_engaged_fraction':matched.get('zinc_engaged_fraction')},
      'Cognate redocking validates the protocol before candidate runs; scores are not affinities.')

if measurability:
    skill('Benchmark measurability', True,
          ['cell2compound/measurability.py','scripts/dose_regime_gate.py','scripts/dose_regime_noise.py',
           'scripts/dose_regime_ablation.py','scripts/dose_regime_time.py','scripts/dose_regime_matched.py',
           'scripts/dose_regime_summary.py',
           'results/dose_regime/summary_snr_by_dose.csv',
           'results/dose_regime/summary_onehot_vs_fingerprint.csv',
           'results/dose_regime/summary_measurability.csv',
           'results/dose_regime/summary_mediation.csv',
           'results/dose_regime/summary_calibrated.csv','docs/dose_regime_findings.md'],
          {'datasets':sorted(measurability),
           'median_snr_by_dose':{label:{f"{r['dose_nM']:g}nM":r['snr_median'] for r in rep['per_dose']}
                                 for label,rep in measurability.items()},
           'floor_is_depth_limited':{label:rep['depth_limitation'] for label,rep in measurability.items()},
           'magnitude_dose_overlap_gate':(dose_gate or {}).get('gate'),
           'class_onehot_vs_fingerprint_sd_units':{
               f"{r['dataset']}/{r['model']}":r['sd_units'] for r in (ablation or [])
               if r.get('sd_units') not in (None,'','nan')},
           # Stage 3: the matched test, and a metric that is not ours agreeing with it.
           'dose_mediated_by_snr':{r['dataset']:{
               'dose_alone':r['dose_alone'],'dose_given_snr':r['dose_given_snr'],
               'ci95_given_snr':[r['ci_lo'],r['ci_hi']],
               'ci_includes_zero':r['ci_includes_zero'],'attenuation':r['attenuation'],
               'r2_dose_only':r['r2_dose_only'],'r2_snr_only':r['r2_snr_only']}
               for r in (mediation or []) if r.get('arm')=='full'},
           'strata_with_no_dynamic_range':[f"{r['dataset']}/{r['stratum']}"
                                           for r in (calibrated or [])
                                           if str(r.get('range_exists')).lower()=='false']},
          'snr = sqrt(rms^2 - floor^2)/floor per group. The median group is below snr 1 at every dose in '
          'all three 24 h lines, so most of this benchmark is below its own measurement error; a 24-dim '
          'mechanism-class one-hot matches a 2048-bit fingerprint across 4 datasets x 4 learned models. '
          'The matched test shows dose predicts model skill only through measurability in the 24 h lines '
          '(attenuation 96-144%, conditional CI straddling zero) but NOT at 72 h, where the assay already '
          'clears its floor. A Dynamic Range Fraction against an interpolated duplicate -- a calibrated '
          'metric that is not ours -- agrees, and more strongly: the dynamic range is negative at low dose '
          'and low snr, so even a reliability-shrunk technical duplicate cannot beat the dose-mean baseline.')

if lincs:
    skill('External validation', True,
          ['scripts/lincs_gate.py','scripts/lincs_measurability.py','results/lincs/summary.json',
           'results/lincs/summary_by_cell.csv','results/lincs/summary_by_dose.csv',
           'results/lincs/conditions.csv'],
          {'source':'LINCS L1000 Phase II, GSE70138 Level 3 (per-instance)',
           **{k:lincs[k] for k in ('conditions','compounds','cell_lines','median_replicates',
                                   'median_snr','fraction_snr_above_1',
                                   'fraction_signal_exactly_zero') if k in lincs},
           'median_snr_by_cell':{r['cell_id']:r['median_snr'] for r in (lincs_by_cell or [])},
           'fraction_above_snr1_by_dose_decade':{str(r['dose_decade']):r['frac_above_1']
                                                 for r in (lincs_by_dose or [])},
           'not_testable_here':'the 1/sqrt(cells) depth law of section 2 -- L1000 is bulk, so there '
                               'is no cell count per condition',
           'not_comparable_in_level':'L1000 floors ~0.42 vs sci-Plex ~0.09; units and preprocessing '
                                     'differ, so only the verdict and the dose ordering carry across'},
          'Independent platform, lab and scale: 1,750 compounds over 9 cell lines at a median of 3 '
          'replicates. The median condition has snr 0.000 and 74% have exactly zero noise-corrected '
          'signal, so the measurability result is not a property of sci-Plex; it holds more strongly '
          'here, which makes the sci-Plex figures the optimistic ones.')

if chemcpa:
    skill('chemCPA baseline', True,
          ['scripts/build_chemcpa_input.py','scripts/build_chemcpa_embedding.py','scripts/chemcpa_predict.py',
           'scripts/chemcpa_compare.py','scripts/chemcpa_snr_strata.py','results/chemcpa/comparison.json',
           'results/chemcpa/snr_strata.json','docs/chemcpa_integration.md'],
          {'pooled':{r['model']:r['rmse'] for r in chemcpa.get('results',[])},
           'alignment_check':chemcpa.get('alignment_check'),
           'advantage_by_stratum':{r['stratum']:r.get('chemcpa_minus_kernel_ridge')
                                   for r in (chemcpa_strata or {}).get('strata',[])},
           'unmeasurable_strata':[r['stratum'] for r in (chemcpa_strata or {}).get('strata',[])
                                  if not r.get('measurable')],
           'seed_variance':chemcpa_seeds},
          'Trained from scratch on our split, since the released checkpoints would leak held-out compounds. '
          'Over five paired split seeds it is statistically indistinguishable from the Tanimoto-kernel ridge: '
          'mean margin +0.0029 in the kernel ridge\'s favour, bootstrap 95% CI [-0.0019, +0.0077], chemCPA '
          'ahead on 1 of 5 seeds. The single-seed 0.0055 advantage, and its apparent concentration in the top '
          'snr quartile, were both properties of split 42 -- the q4 margin flips sign in 3 of 5 seeds. The '
          'same model ranges 0.1016-0.1247 across seeds from the held-out set alone, which is larger than the '
          'gap between models, and is the measurability result appearing in a model comparison.')

if stress:
    skill('Generalisation stress test', True,
          ['cell2compound/pipeline.py','cell2compound/mechanism.py','scripts/mechanism_stress_test.py',
           'results/stress_test/summary.json','results/stress_test/arms.csv','results/stress_test/strata.csv'],
          {'arms':list((stress.get('arms') or {})),
           'reference_reproduces_headline':(stress.get('self_consistency') or {}).get('within_tolerance'),
           'classes_held_out':len(stress.get('loco_by_class') or {})},
          'Asks whether the response model generalises or recognises mechanism classes; the reference arm '
          'must reproduce the headline result before any arm is interpreted.')

try:
    commit=subprocess.run(['git','rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip() or None
except OSError:
    commit=None
report={'generated_from':'saved pipeline outputs only',
        'git_commit':commit,
        'environment':{**environment,'platform':platform.platform()},
        'data':{'single_cell_source':load('data/sciplex_source.json',{}),
                'pseudobulk':pseudobulk,'qc':{k:qc.get(k) for k in
                 ['cells_in','cells_kept','thresholds','control_cells_kept','control_replicate_profile_pearson',
                  'control_split_half_rmse','highest_compound_loss']}},
        'headline_results':{
            'expression_rmse_over_seeds':{k:{'mean':round(v['rmse']['mean'],4),'sd':round(v['rmse']['sd'],4)}
                                          for k,v in expression_summary.items()},
            'expression_top50_de_rmse':{k:round(v['rmse_top50_de']['mean'],4) for k,v in expression_summary.items()},
            'expression_replicate_noise_floor':expression.get('replicate_noise_floor_all_groups_first_seed_genes'),
            'expression_control_matching_ablation':{
                'pooled_default':{k:round(v['rmse']['mean'],4) for k,v in expression_summary.items()},
                'plate_matched':{k:round(v['rmse']['mean'],4) for k,v in (plate.get('summary_over_seeds') or {}).items()}},
            'measurability_median_snr_by_dose':{
                label:{f"{r['dose_nM']:g}nM":r['snr_median'] for r in rep['per_dose']}
                for label,rep in measurability.items()},
            'measurability_fraction_groups_above_noise':{
                label:{f"{r['dose_nM']:g}nM":r['frac_groups_snr_above_1'] for r in rep['per_dose']}
                for label,rep in measurability.items()},
            'chemcpa_pooled':{r['model']:r['rmse'] for r in (chemcpa or {}).get('results',[])},
            'chemcpa_advantage_by_snr_stratum':{
                r['stratum']:r.get('chemcpa_minus_kernel_ridge')
                for r in (chemcpa_strata or {}).get('strata',[]) if r['stratum'].startswith('snr_')},
            'time_matched_snr_72h_over_24h':{
                f"{a['dose_nM']:g}nM":(round(b['snr_median']/a['snr_median'],2) if a['snr_median'] else None)
                for a,b in zip((time_matched or {}).get('matched_24h',[]),
                               (time_matched or {}).get('at_72h',[]))},
            'property':property_.get('summary_over_seeds'),
            'property_empty_scaffold_ablation':{
                policy:(load(RESULTS/f'property_{policy}/metrics.json',{}) or {}).get('summary_over_seeds')
                for policy in ['singleton','group']},
            'target_aware':target_aware.get('summary'),
            'docking_validation':(validation or {}).get('success_rates_rmsd_le_2A'),
            'mechanism_stress_test':{k:(stress or {}).get(k) for k in
                ['self_consistency','arms','strata','loco_by_class']},
            'docking_candidates':{
                **{k:matched.get(k) for k in
                   ['n','median_best_score','zinc_engaged_fraction','auroc_zbg_vs_comparator',
                    'score_vs_heavy_atoms_spearman']},
                'per_receptor':(candidates or {}).get('per_receptor'),
                **{k:(candidates or {}).get(k) for k in
                   ['score_vs_hdac2_pchembl_spearman','response_rms_median',
                    'response_mannwhitney_p_zbg_greater','compounds_with_chembl_hdac2_potency']}}},
        'skill_evidence_map':skills,
        'reproduce':['/home/ruman/venvs/general/bin/python scripts/environment_audit.py',
                     '/home/ruman/venvs/general/bin/python -m pytest tests -q',
                     '/home/ruman/venvs/general/bin/python scripts/verify_sciplex.py',
                     '/home/ruman/venvs/general/bin/python scripts/pseudobulk.py data/raw/sciplex3.h5ad',
                     '/home/ruman/venvs/general/bin/python scripts/subset_single_cell.py data/raw/sciplex3.h5ad '
                     '--cell-key cell_line --cell-line A549 --drug-key perturbation --dose-key dose_value '
                     '--per-group 36 --control control --control-cap 1500 --stratify-key replicate '
                     '--out data/sciplex_subset_v2.h5ad',
                     '/home/ruman/venvs/general/bin/python scripts/single_cell_qc.py data/sciplex_subset_v2.h5ad',
                     '/home/ruman/venvs/general/bin/python -m scripts.expression_baseline data/pseudobulk_A549_24h.npz',
                     '/home/ruman/venvs/general/bin/python -m scripts.property_baseline',
                     '/home/ruman/venvs/general/bin/python -m scripts.chembl_targets',
                     '/home/ruman/venvs/general/bin/python -m scripts.bioactivity_dataset',
                     '/home/ruman/venvs/general/bin/python scripts/esm_embeddings.py data/bioactivity/targets.fasta --out results/esm_panel',
                     '/home/ruman/venvs/general/bin/python -m scripts.target_aware_model',
                     '/home/ruman/venvs/general/bin/python -m scripts.pocket_comparison',
                     '/home/ruman/venvs/general/bin/python -m scripts.docking_validation',
                     '/home/ruman/venvs/general/bin/python -m scripts.dock_candidates'],
        'open_limitations':[
            'Four datasets are now analysed (A549 24 h and 72 h, K562 24 h, MCF7 24 h), but the models are '
            'fitted and compared within a dataset; no cross-cell-line transfer model is trained.',
            'The 72 h arm covers one cell line and a 47-compound subset enriched for HDAC and Aurora, so the '
            'time comparison is run on compounds matched to the 24 h panel.',
            'Pseudobulk group means only; single-cell distributions and dose-response shape are not modelled.',
            'Two replicates per compound/dose, so the replicate noise floor is itself imprecise; the '
            'measurability conclusions rest on medians over 183-188 groups per dose, not on single groups.',
            'The replicate floor scales as ~1/sqrt(cells), so "low dose is unmeasurable" is a statement '
            'about this assay depth, not evidence that low-dose responses are biologically absent.',
            'chemCPA vs our kernel ridge is not resolved by this benchmark: over 5 paired seeds the margin is '
            '+0.0029 with a 95% CI of [-0.0019, +0.0077]. Split-to-split variation (same model, 0.1016-0.1247) '
            'exceeds the between-model difference, so no ranking is supportable. chemCPA also ran at its '
            'shipped hyperparameters; a tuned configuration was not tried.',
            'Cytotoxic compounds lose cells at QC; their responses describe surviving cells.',
            'Docking is rigid-receptor with no explicit zinc chemistry; AutoDock4Zn needs autogrid4, not installed.',
            'ESOL property model uses one seeded split; no uncertainty estimate.',
            'No claim of disease rescue, direct binding, or efficacy is made anywhere in these outputs.']}
out=RESULTS/'final_report.json'
out.write_text(json.dumps(report,indent=2,default=str)+'\n')
incomplete=[s['skill'] for s in skills if not s['demonstrated']]
print(json.dumps({'written':str(out),'skills_with_evidence':len(skills)-len(incomplete),
                  'skills_total':len(skills),'incomplete':incomplete},indent=2))
print(json.dumps(report['headline_results'],indent=2,default=str))
