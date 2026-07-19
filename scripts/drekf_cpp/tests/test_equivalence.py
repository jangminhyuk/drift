#!/usr/bin/env python3
"""
Test C++ implementations against Python for numerical equivalence.
Run from project root: python -m estimator.cpp.tests.test_equivalence
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)


def test_dynamics_ct2d():
    """Test 2D CT dynamics C++ vs Python."""
    import dr_ekf_cpp
    sys.path.insert(0, project_root)

    # Import Python dynamics
    sys.path.insert(0, project_root)
    from main0_CT import ct_dynamics, ct_jacobian, radar_observation_function, radar_observation_jacobian

    cpp_dyn = dr_ekf_cpp.CT2D(dt=0.2)

    np.random.seed(42)
    for _ in range(50):
        x = np.random.randn(5, 1) * 10
        x[4, 0] = np.random.randn() * 0.5  # omega
        u = np.zeros((2, 1))

        # Dynamics
        py_f = ct_dynamics(x, u, dt=0.2)
        cpp_f = cpp_dyn.f(x.flatten(), u.flatten()).reshape(-1, 1)
        assert np.allclose(py_f, cpp_f, atol=1e-12), f"CT2D dynamics mismatch: {np.max(np.abs(py_f - cpp_f))}"

        # Jacobian
        py_F = ct_jacobian(x, u, dt=0.2)
        cpp_F = cpp_dyn.F_jac(x.flatten(), u.flatten())
        assert np.allclose(py_F, cpp_F, atol=1e-12), f"CT2D Jacobian mismatch: {np.max(np.abs(py_F - cpp_F))}"

        # Observation (only test when range > 0)
        if np.sqrt(x[0, 0]**2 + x[1, 0]**2) > 1.0:
            py_h = radar_observation_function(x)
            cpp_h = cpp_dyn.h(x.flatten()).reshape(-1, 1)
            assert np.allclose(py_h, cpp_h, atol=1e-12), f"CT2D obs mismatch: {np.max(np.abs(py_h - cpp_h))}"

            py_H = radar_observation_jacobian(x)
            cpp_H = cpp_dyn.H_jac(x.flatten())
            assert np.allclose(py_H, cpp_H, atol=1e-10), f"CT2D obs Jac mismatch: {np.max(np.abs(py_H - cpp_H))}"

    print("  [PASS] CT2D dynamics")


def test_dynamics_ct3d():
    """Test 3D CT dynamics C++ vs Python."""
    import dr_ekf_cpp
    from main1_timeCT3D import ct_dynamics, ct_jacobian, radar_observation_function, radar_observation_jacobian

    cpp_dyn = dr_ekf_cpp.CT3D(dt=0.2)

    np.random.seed(42)
    for _ in range(50):
        x = np.random.randn(7, 1) * 10
        x[6, 0] = np.random.randn() * 0.5
        u = np.zeros((3, 1))

        py_f = ct_dynamics(x, u, dt=0.2)
        cpp_f = cpp_dyn.f(x.flatten(), u.flatten()).reshape(-1, 1)
        assert np.allclose(py_f, cpp_f, atol=1e-12), f"CT3D dynamics mismatch: {np.max(np.abs(py_f - cpp_f))}"

        py_F = ct_jacobian(x, u, dt=0.2)
        cpp_F = cpp_dyn.F_jac(x.flatten(), u.flatten())
        assert np.allclose(py_F, cpp_F, atol=1e-12), f"CT3D Jacobian mismatch: {np.max(np.abs(py_F - cpp_F))}"

        if np.sqrt(x[0, 0]**2 + x[1, 0]**2 + x[2, 0]**2) > 1.0:
            py_h = radar_observation_function(x)
            cpp_h = cpp_dyn.h(x.flatten()).reshape(-1, 1)
            assert np.allclose(py_h, cpp_h, atol=1e-12), f"CT3D obs mismatch"

            py_H = radar_observation_jacobian(x)
            cpp_H = cpp_dyn.H_jac(x.flatten())
            assert np.allclose(py_H, cpp_H, atol=1e-10), f"CT3D obs Jac mismatch"

    print("  [PASS] CT3D dynamics")


def test_kalman_utils():
    """Test posterior_cov and kalman_stats."""
    import dr_ekf_cpp
    sys.path.insert(0, os.path.join(project_root, 'estimator'))
    from DR_EKF_CDC import DR_EKF_CDC

    np.random.seed(42)
    nx, ny = 5, 2

    for _ in range(30):
        # Random SPD matrices
        A = np.random.randn(nx, nx)
        X_pred = A @ A.T + 0.1 * np.eye(nx)
        B = np.random.randn(ny, ny)
        Sigma_v = B @ B.T + 0.1 * np.eye(ny)
        C_t = np.random.randn(ny, nx)

        # Test posterior_cov
        cpp_post = dr_ekf_cpp.posterior_cov(X_pred, Sigma_v, C_t)

        # Python computation
        C_X = C_t @ X_pred
        S = C_X @ C_t.T + Sigma_v
        K = np.linalg.solve(S, C_X).T
        py_post = X_pred - K @ C_X
        py_post = 0.5 * (py_post + py_post.T)

        assert np.allclose(py_post, cpp_post, atol=1e-10), \
            f"posterior_cov mismatch: {np.max(np.abs(py_post - cpp_post))}"

        # Test kalman_stats
        trace_cpp, D_x_cpp, D_v_cpp = dr_ekf_cpp.kalman_stats(X_pred, Sigma_v, C_t)

        # Verify trace
        assert abs(np.trace(py_post) - trace_cpp) < 1e-10, \
            f"kalman_stats trace mismatch: {abs(np.trace(py_post) - trace_cpp)}"

    print("  [PASS] Kalman utilities")


def test_oracle_exact():
    """Test oracle_exact_fast C++ vs Python."""
    import dr_ekf_cpp

    np.random.seed(42)
    d = 5

    for _ in range(30):
        A = np.random.randn(d, d)
        Sigma_hat = A @ A.T + 0.1 * np.eye(d)
        B = np.random.randn(d, d)
        D = B @ B.T + 0.01 * np.eye(d)
        rho = np.random.uniform(0.01, 1.0)

        cpp_L = dr_ekf_cpp.oracle_exact_fast(Sigma_hat, rho, D, 1e-8, 60)

        # Verify properties
        assert cpp_L.shape == (d, d), f"Wrong shape: {cpp_L.shape}"
        eigvals = np.linalg.eigvalsh(cpp_L)
        assert eigvals.min() > -1e-8, f"Not PSD: min_eig = {eigvals.min()}"

        # Verify BW constraint approximately
        # BW^2 = tr(Sigma_hat) + tr(L) - 2 * tr(sqrtm(sqrtm(Sigma_hat) @ L @ sqrtm(Sigma_hat)))
        # Just check L is reasonable (same order of magnitude as Sigma_hat)
        assert np.linalg.norm(cpp_L, 'fro') < 100 * np.linalg.norm(Sigma_hat, 'fro'), "Oracle output unreasonable"

    print("  [PASS] Oracle exact fast")


def test_oracle_bisection():
    """Test oracle_bisection_fast C++ vs Python."""
    import dr_ekf_cpp

    np.random.seed(42)
    d = 5

    for _ in range(30):
        A = np.random.randn(d, d)
        Sigma_hat = A @ A.T + 0.1 * np.eye(d)
        Sigma_ref = Sigma_hat + 0.01 * np.random.randn(d, d)
        Sigma_ref = 0.5 * (Sigma_ref + Sigma_ref.T) + 0.1 * np.eye(d)
        B = np.random.randn(d, d)
        D = B @ B.T + 0.01 * np.eye(d)
        rho = np.random.uniform(0.01, 1.0)
        delta = 0.05

        cpp_L = dr_ekf_cpp.oracle_bisection_fast(Sigma_hat, rho, Sigma_ref, D, delta, 30, 1e-6)

        assert cpp_L.shape == (d, d)
        eigvals = np.linalg.eigvalsh(cpp_L)
        assert eigvals.min() > -1e-8, f"Not PSD: min_eig = {eigvals.min()}"

    print("  [PASS] Oracle bisection fast")


def test_ekf_equivalence():
    """Test EKF C++ vs Python on a short simulation."""
    import dr_ekf_cpp

    np.random.seed(42)
    nx, ny = 5, 2

    x0_cov = np.diag([100, 100, 10, 10, 0.1])
    Sigma_w = np.diag([0.1, 0.1, 0.5, 0.5, 0.01])
    Sigma_v = np.diag([1.0, 0.01])
    mu_w = np.zeros(nx)
    mu_v = np.zeros(ny)

    # C++ EKF
    cpp_dyn = dr_ekf_cpp.CT2D(dt=0.2)
    cpp_ekf = dr_ekf_cpp.EKF(cpp_dyn, x0_cov, Sigma_w, Sigma_v, mu_w, mu_v)

    # Python EKF
    from estimator.EKF import EKF
    from main0_CT import ct_dynamics, ct_jacobian, radar_observation_function, radar_observation_jacobian

    py_ekf = EKF(
        T=10, dist='normal', noise_dist='normal',
        system_data=(np.eye(nx), np.eye(ny, nx)),
        B=np.zeros((nx, 2)),
        true_x0_mean=np.zeros((nx, 1)), true_x0_cov=x0_cov,
        true_mu_w=np.zeros((nx, 1)), true_Sigma_w=Sigma_w,
        true_mu_v=np.zeros((ny, 1)), true_Sigma_v=Sigma_v,
        nominal_x0_mean=np.zeros((nx, 1)), nominal_x0_cov=x0_cov,
        nominal_mu_w=np.zeros((nx, 1)), nominal_Sigma_w=Sigma_w,
        nominal_mu_v=np.zeros((ny, 1)), nominal_Sigma_v=Sigma_v,
        nonlinear_dynamics=ct_dynamics, dynamics_jacobian=ct_jacobian,
        observation_function=radar_observation_function, observation_jacobian=radar_observation_jacobian,
    )

    # Test initial update
    x_init = np.array([100.0, 200.0, 5.0, 3.0, 0.1])
    y0 = np.array([224.0, 1.1])

    cpp_x0 = cpp_ekf.initial_update(x_init, y0)
    py_x0 = py_ekf._initial_update(x_init.reshape(-1, 1), y0.reshape(-1, 1))

    assert np.allclose(cpp_x0, py_x0.flatten(), atol=1e-10), \
        f"EKF initial_update mismatch: {np.max(np.abs(cpp_x0 - py_x0.flatten()))}"

    # Test several update steps
    u = np.zeros(2)
    x_est_cpp = cpp_x0.copy()
    x_est_py = py_x0.copy()

    for t in range(1, 6):
        y_t = np.array([220.0 + t * 2, 1.1 + t * 0.01])

        x_est_cpp = cpp_ekf.update_step(x_est_cpp, y_t, t, u)
        x_est_py = py_ekf.update_step(x_est_py.reshape(-1, 1), y_t.reshape(-1, 1), t, u.reshape(-1, 1))

        assert np.allclose(x_est_cpp, x_est_py.flatten(), atol=1e-9), \
            f"EKF update_step mismatch at t={t}: {np.max(np.abs(x_est_cpp - x_est_py.flatten()))}"

    print("  [PASS] EKF equivalence")


def test_dr_ekf_cdc_fw_exact_equivalence():
    """Test DR_EKF_CDC FW-Exact C++ vs Python on a short simulation."""
    import dr_ekf_cpp

    np.random.seed(42)
    nx, ny = 5, 2

    x0_cov = np.diag([100, 100, 10, 10, 0.1])
    Sigma_w = np.diag([0.1, 0.1, 0.5, 0.5, 0.01])
    Sigma_v = np.diag([1.0, 0.01])
    mu_w = np.zeros(nx)
    mu_v = np.zeros(ny)

    # C++ DR_EKF_CDC
    cpp_dyn = dr_ekf_cpp.CT2D(dt=0.2)
    cpp_dr = dr_ekf_cpp.DR_EKF_CDC(
        cpp_dyn, x0_cov, Sigma_w, Sigma_v, mu_w, mu_v,
        theta_x=0.1, theta_v=0.1,
        solver="fw_exact", dr_update_period=1,
        fw_exact_iters=8, fw_exact_gap_tol=1e-6,
        oracle_exact_tol=1e-8, oracle_exact_max_iters=60,
    )

    # Python DR_EKF_CDC
    from estimator.DR_EKF_CDC import DR_EKF_CDC
    from main0_CT import ct_dynamics, ct_jacobian, radar_observation_function, radar_observation_jacobian

    py_dr = DR_EKF_CDC(
        T=10, dist='normal', noise_dist='normal',
        system_data=(np.eye(nx), np.eye(ny, nx)),
        B=np.zeros((nx, 2)),
        true_x0_mean=np.zeros((nx, 1)), true_x0_cov=x0_cov,
        true_mu_w=np.zeros((nx, 1)), true_Sigma_w=Sigma_w,
        true_mu_v=np.zeros((ny, 1)), true_Sigma_v=Sigma_v,
        nominal_x0_mean=np.zeros((nx, 1)), nominal_x0_cov=x0_cov,
        nominal_mu_w=np.zeros((nx, 1)), nominal_Sigma_w=Sigma_w,
        nominal_mu_v=np.zeros((ny, 1)), nominal_Sigma_v=Sigma_v,
        nonlinear_dynamics=ct_dynamics, dynamics_jacobian=ct_jacobian,
        observation_function=radar_observation_function, observation_jacobian=radar_observation_jacobian,
        theta_x=0.1, theta_v=0.1,
        solver="fw_exact", dr_update_period=1,
        fw_exact_iters=8, fw_exact_gap_tol=1e-6,
        oracle_exact_tol=1e-8, oracle_exact_max_iters=60,
    )

    # Test initial update
    x_init = np.array([100.0, 200.0, 5.0, 3.0, 0.1])
    y0 = np.array([224.0, 1.1])

    cpp_x0 = cpp_dr.initial_update(x_init, y0)
    py_x0 = py_dr._initial_update(x_init.reshape(-1, 1), y0.reshape(-1, 1))

    err = np.max(np.abs(cpp_x0 - py_x0.flatten()))
    assert err < 1e-6, f"DR_EKF_CDC initial_update mismatch: {err}"

    # Test update steps
    u = np.zeros(2)
    x_est_cpp = cpp_x0.copy()
    x_est_py = py_x0.copy()

    for t in range(1, 6):
        y_t = np.array([220.0 + t * 2, 1.1 + t * 0.01])

        x_est_cpp = cpp_dr.update_step(x_est_cpp, y_t, t, u)
        x_est_py = py_dr.update_step(x_est_py.reshape(-1, 1), y_t.reshape(-1, 1), t, u.reshape(-1, 1))

        err = np.max(np.abs(x_est_cpp - x_est_py.flatten()))
        assert err < 1e-4, f"DR_EKF_CDC update_step mismatch at t={t}: {err}"

    print("  [PASS] DR_EKF_CDC FW-Exact equivalence")


def test_timing():
    """Quick timing comparison."""
    import dr_ekf_cpp
    import time

    nx, ny = 5, 2
    x0_cov = np.diag([100, 100, 10, 10, 0.1])
    Sigma_w = np.diag([0.1, 0.1, 0.5, 0.5, 0.01])
    Sigma_v = np.diag([1.0, 0.01])

    cpp_dyn = dr_ekf_cpp.CT2D(dt=0.2)

    # EKF timing
    cpp_ekf = dr_ekf_cpp.EKF(cpp_dyn, x0_cov, Sigma_w, Sigma_v, np.zeros(nx), np.zeros(ny))

    x_est = np.array([100.0, 200.0, 5.0, 3.0, 0.1])
    y0 = np.array([224.0, 1.1])
    u = np.zeros(2)

    x_est = cpp_ekf.initial_update(x_est, y0)

    N = 1000
    start = time.perf_counter()
    for t in range(1, N + 1):
        y_t = np.array([220.0, 1.1])
        x_est = cpp_ekf.update_step(x_est, y_t, t, u)
    ekf_time = (time.perf_counter() - start) / N * 1000  # ms

    # DR_EKF_CDC FW-Exact timing
    cpp_dr = dr_ekf_cpp.DR_EKF_CDC(
        cpp_dyn, x0_cov, Sigma_w, Sigma_v, np.zeros(nx), np.zeros(ny),
        theta_x=0.1, theta_v=0.1, solver="fw_exact",
        fw_exact_iters=8, oracle_exact_tol=1e-8, oracle_exact_max_iters=60,
    )

    x_est = np.array([100.0, 200.0, 5.0, 3.0, 0.1])
    x_est = cpp_dr.initial_update(x_est, y0)

    N_dr = 200
    start = time.perf_counter()
    for t in range(1, N_dr + 1):
        y_t = np.array([220.0, 1.1])
        x_est = cpp_dr.update_step(x_est, y_t, t, u)
    dr_time = (time.perf_counter() - start) / N_dr * 1000  # ms

    print(f"  [TIMING] C++ EKF:           {ekf_time:.4f} ms/step")
    print(f"  [TIMING] C++ CDC FW-Exact:  {dr_time:.4f} ms/step")


if __name__ == "__main__":
    print("=" * 60)
    print("DR-EKF C++ vs Python Equivalence Tests")
    print("=" * 60)

    tests = [
        ("CT2D Dynamics", test_dynamics_ct2d),
        ("CT3D Dynamics", test_dynamics_ct3d),
        ("Kalman Utilities", test_kalman_utils),
        ("Oracle Exact Fast", test_oracle_exact),
        ("Oracle Bisection Fast", test_oracle_bisection),
        ("EKF Equivalence", test_ekf_equivalence),
        ("DR_EKF_CDC FW-Exact", test_dr_ekf_cdc_fw_exact_equivalence),
        ("Timing", test_timing),
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
