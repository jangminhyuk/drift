#pragma once

#include "types.h"
#include <string>

namespace dr_ekf {

struct DRMMSECDCResult {
    MatXd wc_Sigma_v;   // ny x ny
    MatXd wc_Xprior;    // nx x nx
    MatXd wc_Xpost;     // nx x nx
    bool success;
    int iterations;
};

// Standalone DR-MMSE CDC solver.
// Optimizes over (X_prior, Sigma_v) within Bures-Wasserstein balls
// around (X_prior_hat, Sigma_v_hat) to maximize trace(posterior covariance).
//
// Key difference from TAC: the BW ball acts directly on the prior covariance
// X_prior (nx x nx) rather than on the process noise Q_w or Sigma_w.
//
// When theta_x <= 0 and theta_v <= 0, returns standard Kalman update (no DR).
DRMMSECDCResult solve_dr_mmse_cdc(
    const MatXd& X_prior_hat,   // nominal prior covariance (nx x nx)
    const MatXd& C,             // measurement Jacobian     (ny x nx)
    const MatXd& Sigma_v_hat,   // nominal meas noise       (ny x ny, PD)
    double theta_x, double theta_v,
    const std::string& solver = "fw_exact",
    const MatXd& warm_Sigma_v = MatXd(),  // empty = no warm start
    // FW parameters
    double fw_beta_minus1 = 1.0, double fw_tau = 2.0,
    double fw_zeta = 2.0, double fw_delta = 0.05,
    int fw_max_iters = 50, double fw_gap_tol = 1e-4,
    int fw_bisect_max_iters = 30, double fw_bisect_tol = 1e-6,
    int fw_exact_iters = 8, double fw_exact_gap_tol = 1e-6,
    double oracle_exact_tol = 1e-8, int oracle_exact_max_iters = 60);

} // namespace dr_ekf
