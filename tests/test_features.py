from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cetspace.features import anf_degree_for_output, avalanche_matrix, hamming_distribution
from cetspace.operations import base_cet_mappings, generate_real_cet_corpus, is_bijective


def test_real_reconstructed_corpus_sizes():
    corpus = generate_real_cet_corpus()
    assert len(corpus) == 384
    base_records = [
        op for op in corpus
        if op.base_index is not None and op.output_permutation == (0, 1, 2) and op.inversion_mask == 0
    ]
    assert len(base_records) == 8


def test_base_mappings_are_bijective():
    assert all(is_bijective(mapping) for mapping in base_cet_mappings())


def test_identity_avalanche_and_hamming_distribution():
    identity = tuple(range(8))
    av = avalanche_matrix(identity)
    assert av.shape == (3, 3)
    assert np.allclose(np.diag(av), 1.0)
    hd = hamming_distribution(identity)
    assert np.isclose(hd[1], 1.0)


def test_anf_degree_identity_outputs_are_linear():
    identity = tuple(range(8))
    assert [anf_degree_for_output(identity, bit) for bit in range(3)] == [1, 1, 1]
