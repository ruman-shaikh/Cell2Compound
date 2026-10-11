# Cell2Compound

Laptop-scale modelling of small-molecule effects on single-cell gene expression, with an
accompanying measurement-quality audit of the benchmark it is trained on.

Everything here runs on one machine: an RTX 4050 Laptop GPU (6 GiB VRAM), 22 CPU threads, 30 GB RAM.
No cluster.

## The result worth reading first

The headline is negative, and it is about the data rather than the models.

**Most of the sci-Plex chemical perturbation benchmark sits below its own measurement noise.**
Defining a per-group replicate floor as `RMSE(rep1 - rep2) / 2` and a noise-corrected
`snr = sqrt(max(rms² - floor², 0)) / floor`, the median (compound, dose) group has `snr < 1` at
every dose in all three 24 h cell lines tested.

| dataset | 10 nM | 100 nM | 1 µM | 10 µM | groups ≥ snr 1 @10 nM | @10 µM |
| --- | --- | --- | --- | --- | --- | --- |
| A549 24 h | 0.258 | 0.321 | 0.390 | 0.698 | 8.0% | 39.9% |
| K562 24 h | 0.215 | 0.218 | 0.248 | 0.605 | 8.0% | 33.9% |
| MCF7 24 h | **0.000** | **0.000** | 0.345 | 0.619 | 8.0% | 39.9% |
| A549 72 h | 0.870 | 0.996 | 0.739 | 1.256 | 36.2% | 53.5% |

Five further findings follow from it:

1. **The floor is depth-limited, not dose-limited.** Regressing `log(floor)` on `log(cells)` gives
   slopes of −0.543 / −0.454 / −0.354 / −0.421 against −0.5 for pure counting noise; adding log dose
   raises R² by at most 0.057. Dose raises the floor because cytotoxicity removes cells.
2. **Exposure time buys more signal than dose.** On the 47 compounds measured at both timepoints,
   72 h snr is 1.7–6.0× the 24 h value where depth alone predicts 1.6×.
3. **Demonstrable model skill exists only in the top snr quartile** — attainable fraction +0.208 /
   +0.216 / +0.187 across the three 24 h lines, negative in q2–q3, and undefined in q1 where the
   best baseline already sits below the replicate floor.
4. **A 24-dimensional mechanism-class one-hot matches a 2048-bit Morgan fingerprint.** Across
   4 datasets × 4 learned models, no contrast reaches 1 sd against the one-hot, and for ridge and
   the MLP the one-hot wins all eight.
5. **Dose predicts model skill only through measurability.** Regressing per-group improvement on
   log dose attenuates 96–144% once snr is conditioned on, with the conditional CI straddling zero
   — in the 24 h data. It does *not* attenuate at 72 h (20%, CI excludes zero), the one dataset
   already above its noise floor. The claim is conditional and is stated that way.

### It is not a property of sci-Plex

Checked externally on LINCS L1000 Phase II (GSE70138) — a different platform, lab and scale:
**76,397 conditions, 1,750 compounds, 9 cell lines, median 3 replicates.** Median snr **0.000**,
**74.4% of conditions have exactly zero noise-corrected signal**, and the dose ordering reproduces
monotonically (2.9% / 4.3% / 7.2% / 13.7% above snr 1 at 10 nM / 100 nM / 1 µM / 10 µM).

Only 6.5% of L1000 conditions clear snr 1, against 8–40% by dose in sci-Plex, so **the sci-Plex
figures are the optimistic ones**.

What does *not* transfer: the depth law. L1000 is bulk, so there is no cell count per condition and
`1/sqrt(cells)` cannot be tested on it. Floor *levels* are also not comparable (~0.42 vs ~0.09)
because units and preprocessing differ — only the verdict and the ordering carry.

### Confirmed under a metric that is not ours

`attainable_fraction` is our own construction, so every model is also scored by a **Dynamic Range
Fraction** against an **interpolated duplicate** positive control — both from Miller et al.
([bioRxiv 2025](https://doi.org/10.1101/2025.10.20.683304);
[*Nature Biotechnology* 2026](https://doi.org/10.1038/s41587-026-03307-w)), the paper arguing the
"deep models lose to baselines" result is a metric artifact. Adopting the instrument built to
rescue deep models is the point.

It agrees, in a stronger form: in the bottom three snr quartiles and at the lowest doses the
**dynamic range is negative** — a reliability-shrunk technical duplicate cannot beat a dose-mean
baseline, so there is no range for any model to occupy.

Our α rule differs from theirs and the code says so: theirs is `1 - p_adj` per gene per condition;
ours is the reliability ratio `s²/(s²+n²)` per gene, which needs no DE test and is the more
conservative of the two. The blend itself is their formula.

### chemCPA, over five seeds

chemCPA ([Hetzel et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/aa933b5abc1be30baece1d230ec575a7-Abstract.html))
was trained from scratch on our compound-held-out split — its released checkpoints were fitted on a
split that almost certainly contains our held-out compounds, so using them would leak.

| seed | chemCPA | Tanimoto-kernel ridge | margin |
| --- | --- | --- | --- |
| 42 | 0.1016 | 0.1071 | **−0.0055** |
| 43 | 0.1149 | 0.1132 | +0.0017 |
| 44 | 0.1247 | 0.1166 | +0.0081 |
| 45 | 0.1194 | 0.1093 | +0.0101 |
| 46 | 0.1125 | 0.1125 | −0.0000 |

Paired mean **+0.00288** favouring the kernel ridge, bootstrap 95% CI **[−0.00186, +0.00768]**;
chemCPA wins 1 of 5 seeds. **The two are statistically indistinguishable.** An earlier single-seed
run of this repository reported chemCPA ahead, and that did not survive.

The *same* model ranges 0.1016–0.1247 across seeds purely from which compounds were held out —
larger than the gap between the models. That is the measurability result appearing in a model
comparison, which is why it is reported here rather than quietly dropped. chemCPA ran at its
shipped hyperparameters throughout; a tuned configuration was not tried.

## What is in this repository

| path | contents |
| --- | --- |
| `cell2compound/` | library: pipeline, chemistry, compounds, expression models, measurability, mechanism labels, docking, splits |
| `scripts/` | one runnable stage each: QC, pseudobulk, baselines, ChEMBL, ESM, structures, docking, dose-regime study, LINCS, chemCPA integration, report |
| `tests/` | 38 tests over the pieces where a silent error would change a conclusion |
| `results/` | every result table cited above, versioned as data |
| `data/*.json` | provenance sidecars: shapes, seeds, checksums, thresholds |

Raw inputs are not versioned (2.5 GB sci-Plex h5ad, 17 GB LINCS matrix, PDB entries, cached ChEMBL
responses); `scripts/verify_sciplex.py`, `scripts/lincs_gate.py` and `scripts/chembl_targets.py`
fetch them. Derived matrices, docking poses and ESM embeddings are regenerable and excluded too.

Narrative write-ups (`docs/`, per-session status) are kept outside this repository. Some module
docstrings refer to them by filename; the numbers they cite are all reproducible from `results/`
and the scripts here.

## Pipeline

**Target.** Per `(compound, dose)` group: mean log-normalised counts minus a pooled control
profile, averaged over two replicate plates — a control-relative pseudobulk response vector over
1,000 training-selected genes.

**Features.** Morgan fingerprint (radius 2, 2048 bits) on the `ChargeParent` salt-stripped parent
structure, concatenated with `log10(dose)`.

**Models.** Three non-learned baselines (zero, training mean, dose mean), Tanimoto 5-NN, ridge,
Tanimoto-kernel ridge with an RBF on log dose, and a small GPU MLP. Hyperparameters are chosen on
validation only, then refitted on train+validation.

**Evaluation.** Compound-held-out splits grouped on canonical parent structure, so every dose and
salt form of a compound stays in one partition. Gene selection, scalers and tuning see training
data only.

## Reproduce

```bash
P=/home/ruman/venvs/general/bin/python
export NUMBA_CACHE_DIR=/tmp/cell2compound_numba MPLCONFIGDIR=/tmp/cell2compound_mpl

$P -m pytest tests -q                                    # 38 tests
$P scripts/pseudobulk.py data/raw/sciplex3.h5ad          # streams 2.5 GB in 82 s, 2.6 GB peak
$P -m scripts.expression_baseline data/pseudobulk_A549_24h.npz
$P scripts/dose_regime_gate.py                           # magnitude/dose overlap gate
$P scripts/dose_regime_noise.py  data/pseudobulk_A549_24h.npz A549_24h
$P scripts/dose_regime_ablation.py data/pseudobulk_A549_24h.npz A549_24h
$P -m scripts.dose_regime_matched data/pseudobulk_A549_24h.npz A549_24h
$P -m scripts.dose_regime_summary
$P -m scripts.final_report                               # machine-readable skill-to-evidence map
```

LINCS external check, gated on metadata before the 13 GB download:

```bash
$P -I scripts/lincs_gate.py <inst_info.txt.gz> <pert_info.txt.gz>
$P -m scripts.lincs_measurability <level3.gctx> <inst_info.txt.gz> <gene_info.txt.gz>
```

`results/final_report.json` is the machine-readable roll-up, written by `scripts/final_report.py`
from saved outputs only — 13 areas, each with its evidence files and numbers.

## Engineering notes

Things that cost real time and are recorded in the code rather than smoothed over:

- **Streaming beats subsetting.** `pseudobulk.py` streams the 2.5 GB AnnData in chunks without
  loading the matrix, aggregating all 160k A549/24 h cells in 82 s at 2.6 GB peak — which raised
  usable cells per group from 36 to ~215 and changed the headline result.
- **A fixed 20% mitochondrial cut would discard 53% of cells.** These sci-Plex nuclei sit at ~20%
  mitochondrial reads by design; QC uses an adaptive median+3·MAD cap instead.
- **Controls were being capped like treatments**, so every response was measured against a 30-cell
  mean. Controls now have their own cap and are balanced across replicates.
- **Compound naming silently dropped 36 of 188 compounds** until salt suffixes were handled; all
  188 now map to 186 distinct parent structures.
- **Docking parallelism**: independent across compounds, receptors and seeds, so a process pool took
  CPU utilisation from 354% to 1845%. Threads per run are held fixed, because Vina trajectories
  depend on thread count and changing it would invalidate cached poses.
- **chemCPA concurrency is bounded by host RAM, not VRAM.** Each run holds ~8.2 GB; four concurrent
  need ~33 GB against 30 GB installed, and the kernel OOM killer takes them silently with no
  traceback. Two is the ceiling.
- **Our MLP is device-dependent at the fourth decimal** (0.1163 on CUDA, 0.1159 on CPU), so per-seed
  scoring must hold the device fixed. The pre-specified comparator is deterministic sklearn.

## Limitations

- Two replicates per group, so each floor is itself imprecise; conclusions rest on medians over
  183–188 groups per dose.
- Pseudobulk group means only; single-cell response distributions are not modelled.
- Mechanism-class labels are our ChEMBL/annotation-derived assignment, not external ground truth.
- The 72 h arm is one cell line and 47 compounds.
- The depth law is **not** externally validated and cannot be on bulk data — it is the largest
  unvalidated claim here.
- Docking is rigid-receptor with no explicit zinc chemistry; hydroxamates never rank a correct pose
  first even in their native receptor, and that is reported wherever those rankings are used.
- No claim of direct binding, efficacy or disease rescue is made anywhere.

## Data sources

sci-Plex3 via [scPerturb](https://zenodo.org/records/13350497) · LINCS L1000 Phase II
[GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138) ·
[ChEMBL](https://www.ebi.ac.uk/chembl/) · [AlphaFold DB](https://alphafold.ebi.ac.uk/) ·
[RCSB PDB](https://www.rcsb.org/) · [chemCPA](https://github.com/theislab/chemCPA) ·
[AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina)
