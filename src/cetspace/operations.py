from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import permutations
from typing import Callable

import numpy as np


N_BITS = 3
N_STATES = 2**N_BITS


@dataclass(frozen=True)
class CETOperation:
    operation_id: str
    mapping: tuple[int, ...]
    source: str
    source_detail: str
    base_index: int | None
    output_permutation: tuple[int, int, int] | None
    inversion_mask: int | None
    effective_substitution_tables: int
    control_bits: int
    reconstructed_from_public_description: bool
    label_source: str = "deterministic_features"

    def to_record(self) -> dict:
        rec = asdict(self)
        rec["mapping"] = " ".join(str(v) for v in self.mapping)
        rec["is_reconstructed"] = self.reconstructed_from_public_description
        rec["permutation_id"] = (
            "" if self.output_permutation is None else "".join(str(v) for v in self.output_permutation)
        )
        rec["output_permutation"] = (
            "" if self.output_permutation is None else " ".join(str(v) for v in self.output_permutation)
        )
        return rec


def int_to_bits(x: int, n_bits: int = N_BITS) -> tuple[int, ...]:
    return tuple((x >> shift) & 1 for shift in reversed(range(n_bits)))


def bits_to_int(bits: tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def mapping_from_rule(rule: Callable[[tuple[int, int, int]], tuple[int, int, int]]) -> tuple[int, ...]:
    return tuple(bits_to_int(rule(int_to_bits(x))) for x in range(N_STATES))


def permute_output(mapping: tuple[int, ...], order: tuple[int, int, int]) -> tuple[int, ...]:
    out = []
    for value in mapping:
        bits = int_to_bits(value)
        out.append(bits_to_int(tuple(bits[i] for i in order)))
    return tuple(out)


def invert_output_bits(mapping: tuple[int, ...], mask: int) -> tuple[int, ...]:
    return tuple(value ^ mask for value in mapping)


def is_bijective(mapping: tuple[int, ...]) -> bool:
    return sorted(mapping) == list(range(len(mapping)))


def compose(mapping_a: tuple[int, ...], mapping_b: tuple[int, ...]) -> tuple[int, ...]:
    """Return mapping_a(mapping_b(x))."""
    return tuple(mapping_a[mapping_b[x]] for x in range(len(mapping_a)))


def is_involution(mapping: tuple[int, ...]) -> bool:
    return compose(mapping, mapping) == tuple(range(len(mapping)))


def base_cet_mappings() -> list[tuple[int, ...]]:
    """Eight reversible 3-bit operations reconstructed from the public CET pattern.

    The local PDFs specify an eight-operation base group but do not provide a
    machine-readable truth-table corpus. These operations are therefore marked as
    reconstructed and used as a reproducible proxy for the real CET corpus.
    """
    return [
        mapping_from_rule(lambda b: (b[0], b[1], b[2])),
        mapping_from_rule(lambda b: (b[0], b[1], b[0] ^ b[2])),
        mapping_from_rule(lambda b: (b[0] ^ b[1], b[1], b[2])),
        mapping_from_rule(lambda b: (b[0], b[1] ^ b[2], b[2])),
        mapping_from_rule(lambda b: (b[0], b[2], b[1]) if b[0] else (b[0], b[1], b[2])),
        mapping_from_rule(lambda b: (b[2], b[1], b[0]) if b[1] else (b[0], b[1], b[2])),
        mapping_from_rule(lambda b: (b[1], b[0], b[2]) if b[2] else (b[0], b[1], b[2])),
        mapping_from_rule(lambda b: (b[0] ^ b[1], b[1] ^ b[2], b[2])),
    ]


def generate_real_cet_corpus() -> list[CETOperation]:
    operations: list[CETOperation] = []
    for base_index, base in enumerate(base_cet_mappings(), start=1):
        for perm_index, order in enumerate(permutations(range(N_BITS)), start=1):
            permuted = permute_output(base, order)
            for mask in range(N_STATES):
                mapping = invert_output_bits(permuted, mask)
                operations.append(
                    CETOperation(
                        operation_id=f"real_b{base_index:02d}_p{perm_index:02d}_m{mask:03b}",
                        mapping=mapping,
                        source="real_cet_reconstructed",
                        source_detail="67-Jun-11025.pdf; 79-Aug-11441.pdf; Springer chapter 10.1007/978-3-032-18415-3_12",
                        base_index=base_index,
                        output_permutation=order,
                        inversion_mask=mask,
                        effective_substitution_tables=8 if mask == 0 else (192 if perm_index <= 3 else 384),
                        control_bits=3,
                        reconstructed_from_public_description=True,
                        label_source="public_description_reconstruction",
                    )
                )
    return operations


def generate_borderline_corpus(seed: int = 20260617, n_borderline: int = 128) -> list[CETOperation]:
    rng = np.random.default_rng(seed + 17)
    operations: list[CETOperation] = []
    base = base_cet_mappings()
    for i in range(n_borderline):
        left = base[int(rng.integers(0, len(base)))]
        right = tuple(int(v) for v in rng.permutation(N_STATES))
        mapping = []
        for x in range(N_STATES):
            # Blend a structured CET-like mapping with a random permutation in a
            # deterministic but near-boundary way.
            mapping.append(left[x] if (x + i) % 3 else right[x])
        if len(set(mapping)) < N_STATES:
            mapping = list(dict.fromkeys(mapping))
            missing = [v for v in range(N_STATES) if v not in mapping]
            mapping = (mapping + missing)[:N_STATES]
        operations.append(
            CETOperation(
                operation_id=f"mc_borderline_{i:04d}",
                mapping=tuple(int(v) for v in mapping),
                source="monte_carlo_borderline_operation",
                source_detail="synthetic near-boundary operation",
                base_index=None,
                output_permutation=None,
                inversion_mask=None,
                effective_substitution_tables=int(rng.choice([8, 192, 384])),
                control_bits=int(rng.choice([2, 3, 4])),
                reconstructed_from_public_description=False,
                label_source="controlled_dgp_borderline",
            )
        )
    return operations


def generate_controlled_corpus(seed: int = 20260617, n_random: int = 256, n_weak: int = 256) -> list[CETOperation]:
    rng = np.random.default_rng(seed)
    operations: list[CETOperation] = []
    for i in range(n_random):
        mapping = tuple(int(v) for v in rng.permutation(N_STATES))
        operations.append(
            CETOperation(
                operation_id=f"mc_random_perm_{i:04d}",
                mapping=mapping,
                source="monte_carlo_random_permutation",
                source_detail="synthetic reversible baseline",
                base_index=None,
                output_permutation=None,
                inversion_mask=None,
                effective_substitution_tables=int(rng.choice([8, 192, 384])),
                control_bits=int(rng.choice([2, 3, 4])),
                reconstructed_from_public_description=False,
                label_source="controlled_dgp_random",
            )
        )
    for i in range(n_weak):
        mode = i % 4
        if mode == 0:
            mapping = tuple((x & 0b011) for x in range(N_STATES))
        elif mode == 1:
            mapping = tuple((x >> 1) for x in range(N_STATES))
        elif mode == 2:
            constant = int(rng.integers(0, N_STATES))
            mapping = tuple(constant for _ in range(N_STATES))
        else:
            mapping = tuple(int(v) for v in rng.integers(0, N_STATES, size=N_STATES))
        operations.append(
            CETOperation(
                operation_id=f"mc_weak_{i:04d}",
                mapping=mapping,
                source="monte_carlo_weak_operation",
                source_detail="synthetic defective/non-bijective baseline",
                base_index=None,
                output_permutation=None,
                inversion_mask=None,
                effective_substitution_tables=int(rng.choice([1, 2, 4, 8])),
                control_bits=int(rng.choice([0, 1, 2])),
                reconstructed_from_public_description=False,
                label_source="controlled_dgp_weak",
            )
        )
    return operations


def generate_full_corpus(seed: int = 20260617) -> list[CETOperation]:
    return generate_real_cet_corpus() + generate_controlled_corpus(seed=seed) + generate_borderline_corpus(seed=seed)
