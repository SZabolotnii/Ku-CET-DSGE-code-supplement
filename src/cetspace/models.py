from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .patp import alpha_grid, is_degenerate_alpha, patp_transform


class StandardScaler:
    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class GeneratingElementModel:
    alpha: float
    classes_: list[str]
    scaler: StandardScaler
    coefficients: dict[str, np.ndarray]
    condition_numbers: dict[str, float]
    regularization: float = 0.01
    order: int = 3

    def _basis(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x)
        phi = patp_transform(x_scaled, alpha=self.alpha, order=self.order)
        return np.c_[np.ones(phi.shape[0]), phi]

    def msed_matrix(self, x: np.ndarray) -> np.ndarray:
        basis = self._basis(x)
        out = []
        for cls in self.classes_:
            recon = basis @ self.coefficients[cls]
            out.append(np.mean((self.scaler.transform(x) - recon) ** 2, axis=1))
        return np.vstack(out).T

    def predict(self, x: np.ndarray) -> np.ndarray:
        errors = self.msed_matrix(x)
        return np.array([self.classes_[idx] for idx in np.argmin(errors, axis=1)])

    def log_msed_features(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(self.msed_matrix(x))


def fit_generating_element_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    regularization: float = 0.01,
    order: int = 3,
) -> GeneratingElementModel:
    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train)
    phi = patp_transform(x_scaled, alpha=alpha, order=order)
    basis = np.c_[np.ones(phi.shape[0]), phi]
    classes = sorted(np.unique(y_train).tolist())
    coefficients: dict[str, np.ndarray] = {}
    condition_numbers: dict[str, float] = {}
    for cls in classes:
        rows = y_train == cls
        f_cls = basis[rows]
        b_cls = x_scaled[rows]
        gram = f_cls.T @ f_cls
        ridge = regularization * np.eye(gram.shape[0])
        coefficients[cls] = np.linalg.solve(gram + ridge, f_cls.T @ b_cls)
        condition_numbers[cls] = float(np.linalg.cond(gram + ridge))
    return GeneratingElementModel(
        alpha=alpha,
        classes_=classes,
        scaler=scaler,
        coefficients=coefficients,
        condition_numbers=condition_numbers,
        regularization=regularization,
        order=order,
    )


def split_dataset(x: np.ndarray, y: np.ndarray, seed: int = 20260617):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    test_idx = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * 0.6))
        n_val = int(round(n * 0.2))
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train : n_train + n_val])
        test_idx.extend(idx[n_train + n_val :])
    train_idx = np.array(sorted(train_idx))
    val_idx = np.array(sorted(val_idx))
    test_idx = np.array(sorted(test_idx))
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    return x_train, x_val, x_test, y_train, y_val, y_test


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for cls in sorted(set(y_true.tolist()) | set(y_pred.tolist())):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = {}
    for cls in sorted(set(y_true.tolist()) | set(y_pred.tolist())):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = int(np.sum(y_true == cls))
        report[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1),
            "support": support,
        }
    report["macro avg"] = {"f1-score": macro_f1(y_true, y_pred), "support": int(len(y_true))}
    return report


def select_alpha(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    best_alpha = 0.0
    best_score = -1.0
    for alpha in alpha_grid():
        model = fit_generating_element_model(x_train, y_train, alpha=alpha)
        pred = model.predict(x_val)
        score = macro_f1(y_val, pred)
        degenerate = is_degenerate_alpha(float(alpha))
        rows.append({"alpha": float(alpha), "macro_f1": float(score), "is_degenerate_alpha": degenerate})
        if not degenerate and score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, rows


def rule_based_predict(x_raw_records: list[dict]) -> np.ndarray:
    labels = []
    for rec in x_raw_records:
        acceptable = bool(
            rec["is_bijective"]
            and rec["avalanche_error"] <= 0.25
            and rec["balance_error"] <= 0.05
            and rec["effective_substitution_tables"] >= 8
        )
        labels.append("acceptable" if acceptable else "weak")
    return np.array(labels)


def fit_classical_baselines(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> dict[str, np.ndarray]:
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)
    centroids = {
        cls: np.mean(x_train_s[y_train == cls], axis=0)
        for cls in sorted(np.unique(y_train).tolist())
    }
    labels = []
    for row in x_test_s:
        labels.append(min(centroids, key=lambda cls: float(np.mean((row - centroids[cls]) ** 2))))
    return {"nearest_centroid": np.array(labels)}


def bootstrap_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000, seed: int = 20260617) -> dict:
    rng = np.random.default_rng(seed)
    scores = []
    indices = np.arange(len(y_true))
    for _ in range(n_bootstrap):
        sample = rng.choice(indices, size=len(indices), replace=True)
        scores.append(macro_f1(y_true[sample], y_pred[sample]))
    arr = np.array(scores)
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "n_bootstrap": n_bootstrap,
    }
