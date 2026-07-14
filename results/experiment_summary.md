# Experiment Summary

## Corpus

- Operation rows: 1024
- Reconstructed real CET rows: 384
- Base operation rows: 8

## Classification

| Experiment | GE macro-F1 | Rule macro-F1 | Centroid macro-F1 | alpha_opt |
|---|---:|---:|---:|---:|
| E2 sanity | 1.0000 | 1.0000 | 0.9072 | 0.00 |
| E3 multiclass | 0.8849 | 1.0000 | 0.8206 | 0.35 |

## Ranking

| Method | Spearman | Top-25 overlap | Top-50 overlap |
|---|---:|---:|---:|
| rule_based_rank | 0.9364 | 0.44 | 0.58 |
| generating_element_rank | -0.3298 | 0.00 | 0.00 |
| hybrid_rank | 0.9915 | 0.96 | 0.94 |

## Alpha Sweep

- Best non-degenerate alpha: 0.35
- Best validation macro-F1: 0.7445
- alpha = 0.50 is a degenerate control point, not a candidate.

## Monte Carlo

| n | Mean macro-F1 | Std |
|---:|---:|---:|
| 50 | 0.7758 | 0.1458 |
| 100 | 0.8599 | 0.0595 |
| 200 | 0.9548 | 0.0286 |
| 500 | 0.9733 | 0.0281 |
| 1000 | 0.9969 | 0.0041 |

## Real-Data Validation

- R1 real reconstructed rows: 384
- Base rows: 8
- R2 status: not_extracted_in_v1; local PDFs used for reconstruction
- R3 status: not_provided

## Boundary Interpretation

- Binary sanity is pipeline verification because rule-based baseline equals the GE model.
- Pure reconstruction rank is not a standalone cryptographic ranking.
- No result is a proof of cryptographic security.
