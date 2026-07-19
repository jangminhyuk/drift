#!/usr/bin/env python3
"""
Unit tests for the standalone solve_dr_mmse_tac C++ solver.
Run from repo root:
    python scripts/drekf_cpp/tests/test_dr_mmse_tac.py
"""

import sys
import os
import numpy as np

# Add build dir to path
test_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(test_dir, '..', 'build')
sys.path.insert(0, build_dir)

import dr_ekf_cpp


def make_spd(n, scale=1.0):
    """Generate a random SPD matrix."""
    A = np.random.randn(n, n) * scale
    return A @ A.T + 0.1 * np.eye(n)


def test_theta_zero_equivalence():
    """When theta_w=0 and theta_v=0, solver must return nominal Kalman covariances."""
    np.random.seed(42)

    for nx in [3, 5, 9]:
        for ny in [1, 2]:
            APA = make_spd(nx)
            Sigma_w_hat = make_spd(nx, 0.3)
            Sigma_v_hat = make_spd(ny, 0.5)
            C = np.random.randn(ny, nx)

            result = dr_ekf_cpp.solve_dr_mmse_tac(
                APA, C, Sigma_w_hat, Sigma_v_hat,
                theta_w=0.0, theta_v=0.0, solver="fw_exact"
            )

            assert result.success, "Solver should succeed with theta=0"
            assert result.iterations == 0, f"Expected 0 iterations, got {result.iterations}"

            # Check returned covariances match nominals
            assert np.allclose(result.wc_Sigma_w, Sigma_w_hat, atol=1e-12), \
                f"wc_Sigma_w != Sigma_w_hat: max diff = {np.max(np.abs(result.wc_Sigma_w - Sigma_w_hat))}"
            assert np.allclose(result.wc_Sigma_v, Sigma_v_hat, atol=1e-12), \
                f"wc_Sigma_v != Sigma_v_hat: max diff = {np.max(np.abs(result.wc_Sigma_v - Sigma_v_hat))}"

            # Check Xprior = APA + Sigma_w_hat
            Xprior_expected = APA + Sigma_w_hat
            assert np.allclose(result.wc_Xprior, Xprior_expected, atol=1e-12), \
                f"wc_Xprior mismatch: max diff = {np.max(np.abs(result.wc_Xprior - Xprior_expected))}"

            # Check Xpost = posterior_cov(Xprior, Sigma_v, C)
            Xpost_expected = dr_ekf_cpp.posterior_cov(Xprior_expected, Sigma_v_hat, C)
            assert np.allclose(result.wc_Xpost, Xpost_expected, atol=1e-10), \
                f"wc_Xpost mismatch: max diff = {np.max(np.abs(result.wc_Xpost - Xpost_expected))}"

    print("  [PASS] theta=0 equivalence")


def test_scalar_ny1():
    """Test with ny=1 (scalar measurement), matching ESKF use case."""
    np.random.seed(123)
    nx = 9
    ny = 1

    APA = make_spd(nx)
    Sigma_w_hat = make_spd(nx, 0.2)
    Sigma_v_hat = np.array([[0.05]])  # scalar variance
    C = np.random.randn(ny, nx)

    for solver in ["fw", "fw_exact"]:
        result = dr_ekf_cpp.solve_dr_mmse_tac(
            APA, C, Sigma_w_hat, Sigma_v_hat,
            theta_w=0.1, theta_v=0.1, solver=solver
        )

        assert result.success, f"Solver '{solver}' failed with ny=1"
        assert result.wc_Sigma_v.shape == (1, 1), f"Wrong Sigma_v shape: {result.wc_Sigma_v.shape}"
        assert result.wc_Sigma_w.shape == (nx, nx), f"Wrong Sigma_w shape: {result.wc_Sigma_w.shape}"
        assert result.wc_Xprior.shape == (nx, nx), f"Wrong Xprior shape: {result.wc_Xprior.shape}"
        assert result.wc_Xpost.shape == (nx, nx), f"Wrong Xpost shape: {result.wc_Xpost.shape}"

        # wc_Sigma_v should be >= nominal (adversary inflates noise)
        assert result.wc_Sigma_v[0, 0] >= Sigma_v_hat[0, 0] - 1e-10, \
            f"wc_Sigma_v < nominal: {result.wc_Sigma_v[0,0]} < {Sigma_v_hat[0,0]}"

        # Xpost should be PSD
        eigvals = np.linalg.eigvalsh(result.wc_Xpost)
        assert eigvals.min() > -1e-10, f"Xpost not PSD: min_eig = {eigvals.min()}"

    print("  [PASS] scalar ny=1")


def test_psd_outputs():
    """All output matrices should be PSD."""
    np.random.seed(456)
    nx, ny = 9, 1

    for _ in range(20):
        APA = make_spd(nx)
        Sigma_w_hat = make_spd(nx, 0.1)
        Sigma_v_hat = make_spd(ny, 0.5)
        C = np.random.randn(ny, nx)
        theta_w = np.random.uniform(0.01, 0.5)
        theta_v = np.random.uniform(0.01, 0.5)

        result = dr_ekf_cpp.solve_dr_mmse_tac(
            APA, C, Sigma_w_hat, Sigma_v_hat,
            theta_w, theta_v, solver="fw_exact"
        )

        assert result.success
        for name, mat in [("wc_Sigma_v", result.wc_Sigma_v),
                          ("wc_Sigma_w", result.wc_Sigma_w),
                          ("wc_Xprior", result.wc_Xprior),
                          ("wc_Xpost", result.wc_Xpost)]:
            eigvals = np.linalg.eigvalsh(mat)
            assert eigvals.min() > -1e-8, f"{name} not PSD: min_eig = {eigvals.min()}"

    print("  [PASS] PSD outputs")


def test_xprior_consistency():
    """wc_Xprior should equal APA + wc_Sigma_w."""
    np.random.seed(789)
    nx, ny = 9, 1

    for _ in range(20):
        APA = make_spd(nx)
        Sigma_w_hat = make_spd(nx, 0.2)
        Sigma_v_hat = np.array([[0.05]])
        C = np.random.randn(ny, nx)

        result = dr_ekf_cpp.solve_dr_mmse_tac(
            APA, C, Sigma_w_hat, Sigma_v_hat,
            theta_w=0.1, theta_v=0.1, solver="fw_exact"
        )

        assert result.success
        expected = APA + result.wc_Sigma_w
        expected = 0.5 * (expected + expected.T)
        assert np.allclose(result.wc_Xprior, expected, atol=1e-10), \
            f"Xprior != APA + wc_Sigma_w: max diff = {np.max(np.abs(result.wc_Xprior - expected))}"

    print("  [PASS] Xprior consistency")


def test_warm_start_fewer_iters():
    """Warm start should use fewer or equal iterations than cold start."""
    np.random.seed(101)
    nx, ny = 9, 1

    APA = make_spd(nx)
    Sigma_w_hat = make_spd(nx, 0.2)
    Sigma_v_hat = np.array([[0.05]])
    C = np.random.randn(ny, nx)

    # Cold start
    result_cold = dr_ekf_cpp.solve_dr_mmse_tac(
        APA, C, Sigma_w_hat, Sigma_v_hat,
        theta_w=0.1, theta_v=0.1, solver="fw_exact"
    )
    assert result_cold.success

    # Warm start with solution from cold start
    result_warm = dr_ekf_cpp.solve_dr_mmse_tac(
        APA, C, Sigma_w_hat, Sigma_v_hat,
        theta_w=0.1, theta_v=0.1, solver="fw_exact",
        warm_Sigma_w=result_cold.wc_Sigma_w,
        warm_Sigma_v=result_cold.wc_Sigma_v
    )
    assert result_warm.success

    # Warm start should converge in fewer or equal iterations
    assert result_warm.iterations <= result_cold.iterations, \
        f"Warm start ({result_warm.iterations}) used more iterations than cold ({result_cold.iterations})"

    # Solutions should be similar
    assert np.allclose(result_warm.wc_Xpost, result_cold.wc_Xpost, atol=1e-6), \
        f"Warm/cold solutions differ: max diff = {np.max(np.abs(result_warm.wc_Xpost - result_cold.wc_Xpost))}"

    print(f"  [PASS] warm start (cold={result_cold.iterations}, warm={result_warm.iterations} iters)")


def test_theta_w_only():
    """theta_w > 0, theta_v = 0: only process noise is optimized."""
    np.random.seed(202)
    nx, ny = 9, 1

    APA = make_spd(nx)
    Sigma_w_hat = make_spd(nx, 0.2)
    Sigma_v_hat = np.array([[0.05]])
    C = np.random.randn(ny, nx)

    result = dr_ekf_cpp.solve_dr_mmse_tac(
        APA, C, Sigma_w_hat, Sigma_v_hat,
        theta_w=0.1, theta_v=0.0, solver="fw_exact"
    )

    assert result.success
    # Sigma_v should remain nominal when theta_v = 0
    assert np.allclose(result.wc_Sigma_v, Sigma_v_hat, atol=1e-10), \
        f"Sigma_v should be nominal when theta_v=0"

    print("  [PASS] theta_w only")


def test_theta_v_only():
    """theta_w = 0, theta_v > 0: only measurement noise is optimized."""
    np.random.seed(303)
    nx, ny = 9, 1

    APA = make_spd(nx)
    Sigma_w_hat = make_spd(nx, 0.2)
    Sigma_v_hat = np.array([[0.05]])
    C = np.random.randn(ny, nx)

    result = dr_ekf_cpp.solve_dr_mmse_tac(
        APA, C, Sigma_w_hat, Sigma_v_hat,
        theta_w=0.0, theta_v=0.1, solver="fw_exact"
    )

    assert result.success
    # Sigma_w should remain nominal when theta_w = 0
    assert np.allclose(result.wc_Sigma_w, Sigma_w_hat, atol=1e-10), \
        f"Sigma_w should be nominal when theta_w=0"

    print("  [PASS] theta_v only")


def test_fw_vs_fw_exact_agreement():
    """fw and fw_exact should give similar (not identical) results."""
    np.random.seed(404)
    nx, ny = 9, 1

    APA = make_spd(nx)
    Sigma_w_hat = make_spd(nx, 0.2)
    Sigma_v_hat = np.array([[0.05]])
    C = np.random.randn(ny, nx)

    result_fw = dr_ekf_cpp.solve_dr_mmse_tac(
        APA, C, Sigma_w_hat, Sigma_v_hat,
        theta_w=0.1, theta_v=0.1, solver="fw"
    )

    result_exact = dr_ekf_cpp.solve_dr_mmse_tac(
        APA, C, Sigma_w_hat, Sigma_v_hat,
        theta_w=0.1, theta_v=0.1, solver="fw_exact"
    )

    assert result_fw.success and result_exact.success

    # Trace of Xpost should be similar (within 10% relative)
    tr_fw = np.trace(result_fw.wc_Xpost)
    tr_exact = np.trace(result_exact.wc_Xpost)
    rel_diff = abs(tr_fw - tr_exact) / max(tr_fw, tr_exact)
    assert rel_diff < 0.1, f"fw vs fw_exact trace Xpost differ by {rel_diff:.4f}"

    print(f"  [PASS] fw vs fw_exact agreement (rel_diff={rel_diff:.6f})")


def test_rank_deficient_sigma_w():
    """Solver should not crash with rank-deficient (but PSD) Sigma_w_hat."""
    np.random.seed(505)
    nx, ny = 9, 1

    APA = make_spd(nx)
    # Rank-deficient: only first 6 dimensions have noise
    Sigma_w_hat = np.zeros((nx, nx))
    A = np.random.randn(nx, 6) * 0.1
    Sigma_w_hat = A @ A.T + 1e-12 * np.eye(nx)
    Sigma_v_hat = np.array([[0.05]])
    C = np.random.randn(ny, nx)

    result = dr_ekf_cpp.solve_dr_mmse_tac(
        APA, C, Sigma_w_hat, Sigma_v_hat,
        theta_w=0.05, theta_v=0.05, solver="fw_exact"
    )

    assert result.success, "Solver should handle rank-deficient Sigma_w_hat"

    # Output should still be PSD
    eigvals = np.linalg.eigvalsh(result.wc_Xpost)
    assert eigvals.min() > -1e-8, f"Xpost not PSD: min_eig = {eigvals.min()}"

    print("  [PASS] rank-deficient Sigma_w")


# ========================== Factored solver tests ==========================

def make_G_list(nx, nw, num_steps, seed=0):
    """Generate a list of random G_k matrices (nx x nw)."""
    rng = np.random.RandomState(seed)
    return [rng.randn(nx, nw) * 0.3 for _ in range(num_steps)]


def test_factored_theta_zero_equivalence():
    """When theta_w=0 and theta_v=0, factored solver returns nominal Kalman covariances."""
    np.random.seed(600)
    nx, nw, ny = 9, 6, 1

    APA = make_spd(nx)
    Q_hat = make_spd(nw, 0.3)
    Sigma_v_hat = np.array([[0.05]])
    C = np.random.randn(ny, nx)
    G_list = make_G_list(nx, nw, 5, seed=600)

    result = dr_ekf_cpp.solve_dr_mmse_tac_factored(
        APA, C, G_list, Q_hat, Sigma_v_hat,
        theta_w=0.0, theta_v=0.0, solver="fw_exact"
    )

    assert result.success, "Factored solver should succeed with theta=0"
    assert result.iterations == 0, f"Expected 0 iterations, got {result.iterations}"

    # Check Q_w matches nominal
    assert np.allclose(result.wc_Q_w, Q_hat, atol=1e-12), \
        f"wc_Q_w != Q_hat: max diff = {np.max(np.abs(result.wc_Q_w - Q_hat))}"

    # Check Sigma_w = sum_k G_k Q_hat G_k^T
    Sigma_w_expected = sum(G @ Q_hat @ G.T for G in G_list)
    assert np.allclose(result.wc_Sigma_w, Sigma_w_expected, atol=1e-10), \
        f"wc_Sigma_w mismatch: max diff = {np.max(np.abs(result.wc_Sigma_w - Sigma_w_expected))}"

    # Check Xprior = APA + Sigma_w
    Xprior_expected = APA + Sigma_w_expected
    assert np.allclose(result.wc_Xprior, Xprior_expected, atol=1e-10), \
        f"wc_Xprior mismatch: max diff = {np.max(np.abs(result.wc_Xprior - Xprior_expected))}"

    # Check Xpost
    Xpost_expected = dr_ekf_cpp.posterior_cov(Xprior_expected, Sigma_v_hat, C)
    assert np.allclose(result.wc_Xpost, Xpost_expected, atol=1e-10), \
        f"wc_Xpost mismatch: max diff = {np.max(np.abs(result.wc_Xpost - Xpost_expected))}"

    print("  [PASS] factored theta=0 equivalence")


def test_factored_psd_outputs():
    """All output matrices from factored solver should be PSD."""
    np.random.seed(601)
    nx, nw, ny = 9, 6, 1

    for trial in range(20):
        APA = make_spd(nx)
        Q_hat = make_spd(nw, 0.3)
        Sigma_v_hat = make_spd(ny, 0.5)
        C = np.random.randn(ny, nx)
        G_list = make_G_list(nx, nw, 3 + trial % 5, seed=601 + trial)
        theta_w = np.random.uniform(0.01, 0.5)
        theta_v = np.random.uniform(0.01, 0.5)

        result = dr_ekf_cpp.solve_dr_mmse_tac_factored(
            APA, C, G_list, Q_hat, Sigma_v_hat,
            theta_w, theta_v, solver="fw_exact"
        )

        assert result.success
        for name, mat in [("wc_Q_w", result.wc_Q_w),
                          ("wc_Sigma_v", result.wc_Sigma_v),
                          ("wc_Sigma_w", result.wc_Sigma_w),
                          ("wc_Xprior", result.wc_Xprior),
                          ("wc_Xpost", result.wc_Xpost)]:
            eigvals = np.linalg.eigvalsh(mat)
            assert eigvals.min() > -1e-8, \
                f"Trial {trial}: {name} not PSD: min_eig = {eigvals.min()}"

    print("  [PASS] factored PSD outputs")


def test_factored_consistency_with_unfactored():
    """Both solvers should give same trace(wc_Xpost) on full-rank problems."""
    np.random.seed(602)
    nx, nw, ny = 5, 5, 1  # nw=nx so factored and unfactored are comparable

    APA = make_spd(nx)
    Q_hat = make_spd(nw, 0.2)
    Sigma_v_hat = np.array([[0.05]])
    C = np.random.randn(ny, nx)

    # Use single G = I so that sum_k G_k Q G_k^T = Q (direct comparison)
    G_list = [np.eye(nx)]

    result_factored = dr_ekf_cpp.solve_dr_mmse_tac_factored(
        APA, C, G_list, Q_hat, Sigma_v_hat,
        theta_w=0.1, theta_v=0.1, solver="fw_exact"
    )

    result_unfactored = dr_ekf_cpp.solve_dr_mmse_tac(
        APA, C, Q_hat, Sigma_v_hat,
        theta_w=0.1, theta_v=0.1, solver="fw_exact"
    )

    assert result_factored.success and result_unfactored.success

    tr_f = np.trace(result_factored.wc_Xpost)
    tr_u = np.trace(result_unfactored.wc_Xpost)
    rel_diff = abs(tr_f - tr_u) / max(tr_f, tr_u)
    assert rel_diff < 0.01, \
        f"Factored vs unfactored trace(Xpost) differ by {rel_diff:.6f} (factored={tr_f:.6f}, unfactored={tr_u:.6f})"

    print(f"  [PASS] factored consistency with unfactored (rel_diff={rel_diff:.8f})")


def test_factored_warm_start():
    """Warm start should use fewer or equal iterations than cold start."""
    np.random.seed(603)
    nx, nw, ny = 9, 6, 1

    APA = make_spd(nx)
    Q_hat = make_spd(nw, 0.2)
    Sigma_v_hat = np.array([[0.05]])
    C = np.random.randn(ny, nx)
    G_list = make_G_list(nx, nw, 5, seed=603)

    # Cold start
    result_cold = dr_ekf_cpp.solve_dr_mmse_tac_factored(
        APA, C, G_list, Q_hat, Sigma_v_hat,
        theta_w=0.1, theta_v=0.1, solver="fw_exact"
    )
    assert result_cold.success

    # Warm start with solution from cold start
    result_warm = dr_ekf_cpp.solve_dr_mmse_tac_factored(
        APA, C, G_list, Q_hat, Sigma_v_hat,
        theta_w=0.1, theta_v=0.1, solver="fw_exact",
        warm_Q_w=result_cold.wc_Q_w,
        warm_Sigma_v=result_cold.wc_Sigma_v
    )
    assert result_warm.success

    assert result_warm.iterations <= result_cold.iterations, \
        f"Warm start ({result_warm.iterations}) used more iterations than cold ({result_cold.iterations})"

    assert np.allclose(result_warm.wc_Xpost, result_cold.wc_Xpost, atol=1e-6), \
        f"Warm/cold solutions differ: max diff = {np.max(np.abs(result_warm.wc_Xpost - result_cold.wc_Xpost))}"

    print(f"  [PASS] factored warm start (cold={result_cold.iterations}, warm={result_warm.iterations} iters)")


def test_factored_fw_vs_fw_exact():
    """fw and fw_exact should give similar results for factored solver."""
    np.random.seed(604)
    nx, nw, ny = 9, 6, 1

    APA = make_spd(nx)
    Q_hat = make_spd(nw, 0.2)
    Sigma_v_hat = np.array([[0.05]])
    C = np.random.randn(ny, nx)
    G_list = make_G_list(nx, nw, 5, seed=604)

    result_fw = dr_ekf_cpp.solve_dr_mmse_tac_factored(
        APA, C, G_list, Q_hat, Sigma_v_hat,
        theta_w=0.1, theta_v=0.1, solver="fw"
    )

    result_exact = dr_ekf_cpp.solve_dr_mmse_tac_factored(
        APA, C, G_list, Q_hat, Sigma_v_hat,
        theta_w=0.1, theta_v=0.1, solver="fw_exact"
    )

    assert result_fw.success and result_exact.success

    tr_fw = np.trace(result_fw.wc_Xpost)
    tr_exact = np.trace(result_exact.wc_Xpost)
    rel_diff = abs(tr_fw - tr_exact) / max(tr_fw, tr_exact)
    assert rel_diff < 0.1, f"fw vs fw_exact trace Xpost differ by {rel_diff:.4f}"

    print(f"  [PASS] factored fw vs fw_exact agreement (rel_diff={rel_diff:.6f})")


def test_factored_empty_G_list():
    """Empty G_list should behave as theta_w=0 (no process noise DR)."""
    np.random.seed(605)
    nx, nw, ny = 9, 6, 1

    APA = make_spd(nx)
    Q_hat = make_spd(nw, 0.3)
    Sigma_v_hat = np.array([[0.05]])
    C = np.random.randn(ny, nx)

    result = dr_ekf_cpp.solve_dr_mmse_tac_factored(
        APA, C, [], Q_hat, Sigma_v_hat,
        theta_w=0.1, theta_v=0.1, solver="fw_exact"
    )

    assert result.success, "Should succeed with empty G_list"

    # wc_Q_w should remain nominal (no DR on Q_w)
    assert np.allclose(result.wc_Q_w, Q_hat, atol=1e-10), \
        "Q_w should remain nominal with empty G_list"

    print("  [PASS] factored empty G_list")


if __name__ == "__main__":
    print("=" * 60)
    print("DR-MMSE TAC Standalone Solver Tests")
    print("=" * 60)

    tests = [
        ("theta=0 equivalence", test_theta_zero_equivalence),
        ("scalar ny=1", test_scalar_ny1),
        ("PSD outputs", test_psd_outputs),
        ("Xprior consistency", test_xprior_consistency),
        ("warm start", test_warm_start_fewer_iters),
        ("theta_w only", test_theta_w_only),
        ("theta_v only", test_theta_v_only),
        ("fw vs fw_exact", test_fw_vs_fw_exact_agreement),
        ("rank-deficient Sigma_w", test_rank_deficient_sigma_w),
        # Factored solver tests
        ("factored theta=0 equivalence", test_factored_theta_zero_equivalence),
        ("factored PSD outputs", test_factored_psd_outputs),
        ("factored consistency with unfactored", test_factored_consistency_with_unfactored),
        ("factored warm start", test_factored_warm_start),
        ("factored fw vs fw_exact", test_factored_fw_vs_fw_exact),
        ("factored empty G_list", test_factored_empty_G_list),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\nRunning: {name}")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'=' * 60}")
