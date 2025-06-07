"""
benchmark.py

A small utility class that validates model predictions and produces a
selection of evaluation metrics—classification report, confusion matrix,
ROC curve—storing everything in an easy-to-consume dictionary.

Author: Pradhumna Guru Prasad
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import Sequence, List, Dict, Any, Optional, Literal

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

# ---------------------------------------------------------------------- #
# Type helpers
# ---------------------------------------------------------------------- #
MetricName = Literal["classification_report", "confusion_matrix", "roc_curve"]


def _as_1d_array(x: Sequence) -> np.ndarray:
    """
    Convert *x* to a 1-D NumPy array, gently complaining if the input is
    a scalar or multi-dimensional.
    """
    if isinstance(x, np.ndarray):
        if x.ndim != 1:
            raise ValueError("Expected a 1-D array.")
        return x
    if isinstance(x, numbers.Number):
        raise TypeError("A scalar is not a valid array-like input.")
    return np.asarray(x).ravel()


# ---------------------------------------------------------------------- #
# Main dataclass
# ---------------------------------------------------------------------- #
@dataclass
class Benchmark:
    """
    Usage
    -----
    >>> bench = Benchmark(y_true, y_pred, y_prob)
    >>> bench.compute()           # returns a dict of all metrics
    >>> bench.plot_roc(save_path="roc.png")   # writes the plot

    Parameters
    ----------
    y_true : 1-D array-like
    y_pred : 1-D array-like
    y_prob : 1-D array-like, optional
        Predicted probabilities (or scores) **only** needed when you want
        the ROC/AUC.
    metrics : list[str], optional
        Any subset of {"classification_report", "confusion_matrix",
        "roc_curve"}.  If None, every metric is computed.
    """

    y_true: Sequence
    y_pred: Sequence
    y_prob: Optional[Sequence] = None
    metrics: Optional[List[MetricName]] = None

    _results: Dict[str, Any] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self.y_true = _as_1d_array(self.y_true)
        self.y_pred = _as_1d_array(self.y_pred)

        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"len(y_true)={len(self.y_true)} differs from "
                f"len(y_pred)={len(self.y_pred)}"
            )

        # What to compute?
        if self.metrics is None:
            self.metrics = [
                "classification_report",
                "confusion_matrix",
                "roc_curve",
            ]

        # Extra requirements for ROC
        if "roc_curve" in self.metrics:
            if self.y_prob is None:
                raise ValueError(
                    "ROC curve requested but `y_prob` was not supplied."
                )
            self.y_prob = _as_1d_array(self.y_prob)
            if len(self.y_prob) != len(self.y_true):
                raise ValueError(
                    f"len(y_prob)={len(self.y_prob)} differs from "
                    f"len(y_true)={len(self.y_true)}"
                )

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, Any]:
        """Compute each requested metric (once) and return the results."""
        for m in self.metrics:
            # Skip if already done (allows incremental use)
            if m not in self._results:
                getattr(self, f"_compute_{m}")()
        return self._results

    def to_dict(self) -> Dict[str, Any]:
        """A defensive copy of the stored metrics."""
        return {k: v for k, v in self._results.items()}

    def plot_roc(
        self,
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
        **plot_kw,
    ) -> plt.Axes:
        """
        Draw or save the ROC curve.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Draw into this axes; otherwise, the current axes is used.
        save_path : str, optional
            When provided, the figure is written to disk via
            `fig.savefig(save_path, dpi=300)` and *not* displayed.
        plot_kw : dict
            Extra keyword arguments forwarded to `ax.plot()`.

        Returns
        -------
        ax : matplotlib Axes
            The axes containing the ROC curve.
        """
        if "roc_curve" not in self.metrics:
            raise RuntimeError(
                "ROC curve was not among the selected metrics."
            )
        if "roc_curve" not in self._results:
            self._compute_roc_curve()

        res = self._results["roc_curve"]
        ax = ax or plt.gca()
        ax.plot(
            res["fpr"],
            res["tpr"],
            label=f"ROC (AUC = {res['auc']:.3f})",
            **plot_kw,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend()

        if save_path is not None:
            fig = ax.figure
            fig.tight_layout()
            fig.savefig(save_path, dpi=300)
            plt.close(fig)

        return ax

    # ------------------------------------------------------------------ #
    # Private metric helpers
    # ------------------------------------------------------------------ #
    def _compute_classification_report(self) -> None:
        report = classification_report(
            self.y_true,
            self.y_pred,
            output_dict=True,
            zero_division=0,
        )
        self._results["classification_report"] = report

    def _compute_confusion_matrix(self) -> None:
        cm = confusion_matrix(self.y_true, self.y_pred)
        self._results["confusion_matrix"] = cm

    def _compute_roc_curve(self) -> None:
        fpr, tpr, thresh = roc_curve(self.y_true, self.y_prob)
        roc_auc = auc(fpr, tpr)
        self._results["roc_curve"] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresh,
            "auc": roc_auc,
        }


# ---------------------------------------------------------------------- #
# Demo / sanity check
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    # Toy binary classification
    y_true_demo = rng.integers(0, 2, size=200)
    y_prob_demo = rng.random(200)
    y_pred_demo = (y_prob_demo >= 0.5).astype(int)

    bench = Benchmark(
        y_true=y_true_demo,
        y_pred=y_pred_demo,
        y_prob=y_prob_demo,  # needed for ROC
        # metrics=None → compute all three
    )

    results = bench.compute()

    from pprint import pprint
    print("\n=== Classification report (macro-averaged) ===")
    pprint(results["classification_report"]["macro avg"])
    print("\n=== Confusion matrix ===\n", results["confusion_matrix"])
    print("\nAUC =", results["roc_curve"]["auc"])

    # Save ROC to file
    bench.plot_roc(save_path="roc_demo.png")
    print("\nROC curve written to roc_demo.png")
