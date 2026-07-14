# Phase A — scientific core hardening (results)

Additive evidence; the frozen E0-E10 artifacts are unchanged.

## A1. Label leakage (quantified)

- Rule baseline reproduces the label exactly on 100.0% of rows (macro-F1 1.0000): it **is** the label generator, not a baseline.
- A depth-3 tree on decision scalars: macro-F1 1.0000; on representational features only: 0.6842.
- Feature partition: 34 decision columns removed, 9 representational columns kept.

## A1'. De-leaked classification (representational features only)

| Model | macro-F1 |
|---|---:|
| Generating-element (de-leaked) | 0.6659 |
| Nearest-centroid (de-leaked) | 0.5223 |
| Rule oracle (reads decision scalars) | 1.0000 |
| Leaky full-feature GE (paper E3, ref) | 0.8849 |

- best alpha = 0.70. The rule oracle stays at 1.0 because it still reads the decision scalars; the honest learning task is GE-vs-centroid on representational features.

## A2. Open-set / novelty — capability the rules lack

- Rotating class-holdout, 5 folds, de-leaked features for both detectors.
- Mean AUROC: generating-element **0.8329** vs nearest-centroid 0.6648 (GE wins 4/5 folds).
- Threshold rules emit a known class with no 'unknown' output -> novelty AUROC undefined (structural capability gap).

| Held-out class | n_novel | AUROC GE | AUROC centroid |
|---|---:|---:|---:|
| borderline_avalanche | 72 | 0.7962 | 0.4776 |
| derived_good | 152 | 0.7883 | 0.5048 |
| high_complexity_good | 289 | 0.7976 | 0.6185 |
| linear_weak | 252 | 0.7823 | 0.7232 |
| non_bijective_weak | 256 | 1.0000 | 1.0000 |

## A3. Independent ranking target (nonlinearity, non-circular)

| Scorer | vs nonlinearity (indep.) | vs -diff.uniformity | vs composite (circular) |
|---|---:|---:|---:|
| composite | 0.4725 | 0.7503 | 1.0000 |
| rule_based | 0.3425 | 0.5670 | 0.9364 |
| reconstruction | -0.2189 | -0.4245 | -0.3298 |
| hybrid | 0.4526 | 0.7244 | 0.9915 |

- The last column reproduces the paper's circular agreement; the first is the honest signal against a cryptographic property the scorers never saw.

