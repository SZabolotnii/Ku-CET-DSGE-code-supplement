# Ku-CET-DSGE-code-supplement

Reproduction package for the paper

> **Open-Set Screening of CET Stream-Cipher Operations in a Generating-Element
> Space** — S. Zabolotnii et al., 2026 (submitted to *Pragmatic Cybersecurity*).

## What the paper shows

CET-encryption controls the cryptographic transformation with the data
themselves, which makes the space of admissible operations grow quickly and
manual screening subjective. We treat a CET-operation as an object of
**statistical pattern recognition**: each operation becomes a feature vector,
and a class-wise reconstruction model in a **generating-element (Kunchenko)
space** (`F_c·K_c = B_c`, log-MSED reconstruction score) is used to analyse it.
The basis exponents are adapted by a **parametrically-adaptive transition
polynomial (PATP)**, with the degenerate point `α = 0.5` excluded.

The claim is intentionally narrow — this is a **diagnostic pre-screening tool,
not a new cipher and not a proof of cryptographic strength.** The results
include the negative ones in full.

### Headline results (all checked by `verify_article_numbers.py`)

- **Label leakage (§4.1):** the labels are a deterministic function of the same
  features — a rule-based generator reproduces them exactly (`exact match = 1.0`),
  so the near-perfect closed-set accuracy is an artefact, not model quality.
- **De-leaked closed set (§4.2):** on 9 representational features the
  generating-element model beats nearest-centroid, macro-F1 **0.666 vs 0.522**
  (`α_opt = 0.70`).
- **Open-set detection (§4.3, the main result):** on the provenance-clean lens
  the minimum log-MSED recognises operations of unseen classes at AUROC
  **0.795 vs 0.611** (nearest-centroid), difference **+0.184**, class-cluster
  95% CI **[0.124, 0.246]** (excludes 0); on the full corpus **0.832 vs 0.658**.
  Threshold rules cannot do this task at all (no "unknown" verdict).
- **Ranking (§4.4, negative):** against an independent target (nonlinearity),
  the reconstruction rank offers no advantage over the composite score; the
  earlier Spearman 0.9915 was circular.

## Layout

```
src/cetspace/         importable library: features, models (F·K=B / log-MSED),
                      patp (α-adaptation), labels, diagnostics, pipeline
experiments/          run_phase_a.py        — leakage audit, de-leaked, open-set A2/A3
                      run_open_set_v2.py     — corrected 30-seed open-set (the KEY result)
                      run_experiments.py     — full E0–E10 programme
tests/                unit tests (features, patp, phase_a, pipeline, experiments)
data/corpus_v1/       frozen corpus v1: 1024 operations, 384 reconstructed real CET
results/              stored result logs the paper cites (open_set_v2/, phase_a/, E0–E10)
figures/              the two open-set figures used in the paper
run_all.py            end-to-end pipeline entry point
verify_article_numbers.py   asserts every headline number against results/
```

### Code ↔ paper map

| Paper | Code |
|---|---|
| §3.3 reconstruction model `F_c·K_c = B_c`, log-MSED | `src/cetspace/models.py` |
| §3.4 PATP `p_i(α)`, degenerate `α = 0.5` | `src/cetspace/patp.py` |
| §3.2 feature representation | `src/cetspace/features.py`, `operations.py` |
| §4.1 leakage audit | `results/phase_a/a1_leakage_audit/` |
| §4.2 de-leaked classification | `results/phase_a/a1_deleaked_multiclass/` |
| §4.3 open-set (30 seeds, L1–L4) | `experiments/run_open_set_v2.py`, `results/open_set_v2/` |
| §4.4 ranking (circular vs independent) | `results/phase_a/a3_independent_ranking/`, `results/e4_ranking/` |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python verify_article_numbers.py      # check stored numbers match the paper
python experiments/run_open_set_v2.py # re-run the KEY open-set result (30 seeds)
python experiments/run_phase_a.py     # leakage audit + de-leaked + open-set A2/A3
python run_all.py                     # full pipeline (E0–E10)
```

## Scope and caveats

- The corpus is predominantly **synthetic / reconstructed**; all numbers are
  preliminary diagnostics on a reconstructed CET corpus, **not** a transfer to
  real deployed ciphers. On the strictly real sub-corpus (L4) the open-set
  advantage is modest and rests on a single class (see §4.3).
- Statistical proximity in the generating-element space is **not** cryptographic
  strength. This code screens and diagnoses; it does not perform cryptanalysis.

## License & citation

MIT (see `LICENSE`). If you use this code, please cite the paper — see
`CITATION.cff`.
