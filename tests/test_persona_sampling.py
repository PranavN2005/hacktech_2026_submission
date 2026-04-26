"""
Stage-1 (sampling) tests for the persona pipeline.

These tests are deterministic. No LLM is involved at any point.

The pipeline under test:
    sample_trait_vectors(n, seed) → (n, 6) array of correlated Beta samples
    using a Latin Hypercube + Gaussian copula construction.

The shape, range, reproducibility, marginal, and correlation properties
are checked on the *final* sampler output. The strict LHS stratification
property (no two samples share a stratum) is checked on the underlying
uniform LHS step (`_lhs_uniform`) because the Gaussian copula step
breaks per-axis stratification by mixing dimensions through Cholesky.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import beta as scipy_beta

from backend.persona_config import (
    AXIS_KEYS,
    CORRELATION_MATRIX,
    MARGINAL_DISTRIBUTIONS,
)
from backend.persona_pipeline import (
    _lhs_uniform,
    sample_trait_vectors,
)


# ---------------------------------------------------------------------------
# Configuration sanity (axes + Σ)
# ---------------------------------------------------------------------------


class TestPersonaConfig:
    def test_axis_order(self) -> None:
        assert AXIS_KEYS == ("lr", "auth", "conf", "open", "react", "active")

    def test_correlation_matrix_shape(self) -> None:
        assert CORRELATION_MATRIX.shape == (6, 6)

    def test_correlation_matrix_symmetric(self) -> None:
        assert np.allclose(CORRELATION_MATRIX, CORRELATION_MATRIX.T)

    def test_correlation_matrix_unit_diagonal(self) -> None:
        assert np.allclose(np.diag(CORRELATION_MATRIX), 1.0)

    def test_correlation_matrix_psd(self) -> None:
        eigs = np.linalg.eigvalsh(CORRELATION_MATRIX)
        assert eigs.min() >= -1e-9, f"Σ not PSD: min eigval {eigs.min()}"

    def test_correlation_specific_pairs(self) -> None:
        """Sanity-check the literature-motivated off-diagonals."""
        i = AXIS_KEYS.index
        assert CORRELATION_MATRIX[i("lr"), i("auth")] == pytest.approx(0.45)
        assert CORRELATION_MATRIX[i("conf"), i("open")] == pytest.approx(-0.55)
        assert CORRELATION_MATRIX[i("conf"), i("auth")] == pytest.approx(0.30)
        assert CORRELATION_MATRIX[i("react"), i("open")] == pytest.approx(-0.25)
        assert CORRELATION_MATRIX[i("react"), i("active")] == pytest.approx(0.35)
        assert CORRELATION_MATRIX[i("lr"), i("open")] == pytest.approx(-0.20)

    def test_marginal_distributions_present_for_each_axis(self) -> None:
        for k in AXIS_KEYS:
            assert k in MARGINAL_DISTRIBUTIONS


# ---------------------------------------------------------------------------
# Latin Hypercube uniform helper (the underlying stratification step)
# ---------------------------------------------------------------------------


class TestLhsHelper:
    def test_lhs_coverage(self) -> None:
        """For each axis, no two samples share the same n-quantile stratum."""
        n = 10
        u = _lhs_uniform(n=n, dim=6, seed=123)
        assert u.shape == (n, 6)
        for k in range(6):
            strata = np.floor(u[:, k] * n).astype(int)
            strata = np.clip(strata, 0, n - 1)
            assert len(set(strata.tolist())) == n, (
                f"axis {k}: LHS stratification broken, deciles {sorted(strata)}"
            )

    def test_lhs_in_unit_hypercube(self) -> None:
        u = _lhs_uniform(n=200, dim=6, seed=7)
        assert (u > 0.0).all() and (u < 1.0).all()


# ---------------------------------------------------------------------------
# sample_trait_vectors — final correlated Beta samples
# ---------------------------------------------------------------------------


class TestSampleTraitVectors:
    def test_sample_shape(self) -> None:
        out = sample_trait_vectors(n=50, seed=0)
        assert out.shape == (50, 6)

    def test_sample_range(self) -> None:
        out = sample_trait_vectors(n=200, seed=1)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_sample_reproducibility(self) -> None:
        a = sample_trait_vectors(n=64, seed=42)
        b = sample_trait_vectors(n=64, seed=42)
        assert np.array_equal(a, b)

    def test_sample_different_seeds(self) -> None:
        a = sample_trait_vectors(n=64, seed=42)
        b = sample_trait_vectors(n=64, seed=43)
        assert not np.array_equal(a, b)

    def test_marginals_approximate(self) -> None:
        """Each column's empirical mean should be within 10% of its Beta mean."""
        n = 2000
        out = sample_trait_vectors(n=n, seed=7)
        for k, key in enumerate(AXIS_KEYS):
            dist = MARGINAL_DISTRIBUTIONS[key]
            theoretical_mean = scipy_beta.mean(dist["a"], dist["b"])
            empirical_mean = float(out[:, k].mean())
            tol = 0.1 * theoretical_mean if theoretical_mean > 0 else 0.05
            assert abs(empirical_mean - theoretical_mean) <= tol, (
                f"axis {key}: empirical mean {empirical_mean:.4f} "
                f"vs theoretical {theoretical_mean:.4f} (tol {tol:.4f})"
            )

    def test_correlation_preserved(self) -> None:
        """Sample Pearson(lr, auth) should be within ±0.15 of the target 0.45."""
        out = sample_trait_vectors(n=5000, seed=11)
        i = AXIS_KEYS.index("lr")
        j = AXIS_KEYS.index("auth")
        r = float(np.corrcoef(out[:, i], out[:, j])[0, 1])
        assert abs(r - 0.45) <= 0.15, f"empirical lr-auth corr {r:.3f}"

    def test_invalid_n_rejected(self) -> None:
        with pytest.raises(ValueError):
            sample_trait_vectors(n=0, seed=0)
        with pytest.raises(ValueError):
            sample_trait_vectors(n=-3, seed=0)
