from __future__ import annotations

from collections import Counter

import numpy as np

from .operations import N_BITS, N_STATES, CETOperation, int_to_bits, is_bijective, is_involution


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def avalanche_matrix(mapping: tuple[int, ...], n_bits: int = N_BITS) -> np.ndarray:
    matrix = np.zeros((n_bits, n_bits), dtype=float)
    for input_bit in range(n_bits):
        mask = 1 << (n_bits - 1 - input_bit)
        for x in range(2**n_bits):
            diff = mapping[x] ^ mapping[x ^ mask]
            for output_bit in range(n_bits):
                out_mask = 1 << (n_bits - 1 - output_bit)
                matrix[input_bit, output_bit] += 1.0 if diff & out_mask else 0.0
    return matrix / (2**n_bits)


def hamming_distribution(mapping: tuple[int, ...], n_bits: int = N_BITS) -> np.ndarray:
    counts = np.zeros(n_bits + 1, dtype=float)
    total = 0
    for bit in range(n_bits):
        mask = 1 << (n_bits - 1 - bit)
        for x in range(2**n_bits):
            counts[hamming(mapping[x], mapping[x ^ mask])] += 1
            total += 1
    return counts / total


def output_balance(mapping: tuple[int, ...], n_bits: int = N_BITS) -> np.ndarray:
    counts = np.zeros(n_bits, dtype=float)
    for value in mapping:
        bits = int_to_bits(value, n_bits)
        counts += np.array(bits, dtype=float)
    return counts / len(mapping)


def cycle_lengths(mapping: tuple[int, ...]) -> list[int]:
    seen = set()
    lengths: list[int] = []
    for start in range(len(mapping)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = mapping[current]
            length += 1
            if current < 0 or current >= len(mapping):
                return []
        lengths.append(length)
    return lengths


def anf_degree_for_output(mapping: tuple[int, ...], output_bit: int, n_bits: int = N_BITS) -> int:
    values = np.array([(int_to_bits(mapping[x], n_bits)[output_bit]) for x in range(2**n_bits)], dtype=int)
    coeffs = values.copy()
    for bit in range(n_bits):
        step = 1 << bit
        for mask in range(2**n_bits):
            if mask & step:
                coeffs[mask] ^= coeffs[mask ^ step]
    degree = 0
    for mask, coeff in enumerate(coeffs):
        if coeff:
            degree = max(degree, int(mask.bit_count()))
    return degree


def differential_uniformity_proxy(mapping: tuple[int, ...], n_bits: int = N_BITS) -> float:
    max_count = 0
    for dx in range(1, 2**n_bits):
        counter = Counter(mapping[x] ^ mapping[x ^ dx] for x in range(2**n_bits))
        max_count = max(max_count, max(counter.values()))
    return max_count / (2**n_bits)


def feature_record(operation: CETOperation) -> dict:
    mapping = operation.mapping
    av = avalanche_matrix(mapping)
    hd = hamming_distribution(mapping)
    bal = output_balance(mapping)
    cycles = cycle_lengths(mapping) if is_bijective(mapping) else []
    anf_degrees = [anf_degree_for_output(mapping, bit) for bit in range(N_BITS)]
    avalanche_error = float(np.mean(np.abs(av - 0.5)))
    balance_error = float(np.mean(np.abs(bal - 0.5)))
    bijective = is_bijective(mapping)
    involution = is_involution(mapping) if bijective else False
    max_anf_degree = max(anf_degrees)
    complexity_score = float(max_anf_degree + operation.control_bits + np.log2(max(operation.effective_substitution_tables, 1)))
    strict_avalanche = bool(np.allclose(av, 0.5))
    acceptable = bool(bijective and avalanche_error <= 0.25 and balance_error <= 0.05)

    rec: dict[str, float | int | str | bool] = {
        "operation_id": operation.operation_id,
        "source": operation.source,
        "quality_label": "acceptable" if acceptable else "weak",
        "is_real_corpus": operation.source == "real_cet_reconstructed",
        "is_base": operation.base_index is not None and operation.output_permutation == (0, 1, 2) and operation.inversion_mask == 0,
        "is_bijective": bijective,
        "is_involution": involution,
        "strict_avalanche": strict_avalanche,
        "avalanche_error": avalanche_error,
        "balance_error": balance_error,
        "differential_uniformity": differential_uniformity_proxy(mapping),
        "max_anf_degree": max_anf_degree,
        "mean_anf_degree": float(np.mean(anf_degrees)),
        "cycle_count": len(cycles),
        "max_cycle_length": max(cycles) if cycles else 0,
        "mean_cycle_length": float(np.mean(cycles)) if cycles else 0.0,
        "complexity_score": complexity_score,
        "effective_substitution_tables": operation.effective_substitution_tables,
        "control_bits": operation.control_bits,
    }
    for i, value in enumerate(mapping):
        rec[f"truth_{i}"] = value / (N_STATES - 1)
    for i, value in enumerate(av.ravel()):
        rec[f"avalanche_{i}"] = float(value)
    for i, value in enumerate(hd):
        rec[f"hamming_dist_{i}"] = float(value)
    for i, value in enumerate(bal):
        rec[f"balance_{i}"] = float(value)
    for i, value in enumerate(anf_degrees):
        rec[f"anf_degree_{i}"] = int(value)
    return rec


def numeric_feature_columns(records: list[dict]) -> list[str]:
    excluded = {"operation_id", "source", "quality_label"}
    cols = []
    for key, value in records[0].items():
        if key not in excluded and isinstance(value, (int, float, bool, np.bool_)):
            cols.append(key)
    return cols
