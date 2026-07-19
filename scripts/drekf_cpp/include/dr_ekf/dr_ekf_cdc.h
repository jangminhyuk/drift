#pragma once

#include "types.h"
#include "dynamics.h"
#ifdef DR_EKF_HAS_MOSEK
#include "mosek_sdp_solver.h"
#endif
#include <memory>

namespace dr_ekf {

struct SDPResult {
    MatXd wc_Sigma_v;
    MatXd wc_Xprior;
    MatXd wc_Xpost;
    bool success;
};

class DR_EKF_CDC {
public:
    DR_EKF_CDC(const DynamicsInterface& dynamics,
               const MatXd& nominal_x0_cov,
               const MatXd& nominal_Sigma_w,
               const MatXd& nominal_Sigma_v,
               const VecXd& nominal_mu_w,
               const VecXd& nominal_mu_v,
               double theta_x, double theta_v,
               const std::string& solver = "fw_exact",
               int dr_update_period = 1,
               // FW params
               double fw_beta_minus1 = 1.0,
               double fw_tau = 2.0,
               double fw_zeta = 2.0,
               double fw_delta = 0.05,
               int fw_max_iters = 50,
               double fw_gap_tol = 1e-4,
               int fw_bisect_max_iters = 30,
               double fw_bisect_tol = 1e-6,
               // FW-Exact params
               int fw_exact_iters = 8,
               double fw_exact_gap_tol = 1e-6,
               double oracle_exact_tol = 1e-8,
               int oracle_exact_max_iters = 60);

    VecXd initial_update(const VecXd& x_est_init, const VecXd& y0);
    VecXd update_step(const VecXd& x_est_prev, const VecXd& y_curr, int t, const VecXd& u_prev);

    // Accessors
    const MatXd& get_P() const { return P_; }
    int get_nx() const { return nx_; }
    int get_ny() const { return ny_; }
    int get_dr_solve_count() const { return dr_solve_count_; }
    int get_dr_step_count() const { return dr_step_count_; }

private:
    SDPResult solve_sdp_online(const MatXd& X_pred_hat, const MatXd& C_t);
    SDPResult solve_sdp_online_fw(const MatXd& X_pred_hat, const MatXd& C_t);
    SDPResult solve_sdp_online_fw_exact(const MatXd& X_pred_hat, const MatXd& C_t);
#ifdef DR_EKF_HAS_MOSEK
    SDPResult solve_sdp_online_mosek(const MatXd& X_pred_hat, const MatXd& C_t);
#endif

    VecXd DR_kalman_filter(const VecXd& v_mean_hat, const VecXd& x_prior,
                           const VecXd& y, int t,
                           const VecXd& u_prev, const VecXd& x_post_prev);

    const DynamicsInterface& dynamics_;
    int nx_, ny_;

    // Nominal parameters
    MatXd x0_cov_;
    MatXd Sigma_w_hat_, Sigma_v_hat_;
    VecXd mu_w_, mu_v_;

    // DR parameters
    double theta_x_, theta_v_;
    std::string solver_;
    int dr_update_period_;

    // FW parameters
    double fw_beta_minus1_, fw_tau_, fw_zeta_, fw_delta_;
    int fw_max_iters_;
    double fw_gap_tol_;
    int fw_bisect_max_iters_;
    double fw_bisect_tol_;

    // FW-Exact parameters
    int fw_exact_iters_;
    double fw_exact_gap_tol_;
    double oracle_exact_tol_;
    int oracle_exact_max_iters_;

#ifdef DR_EKF_HAS_MOSEK
    // MOSEK solver (lazy-initialized)
    std::unique_ptr<mosek_solver::CDCSolver> mosek_solver_;
#endif

    // State
    MatXd P_;
    MatXd cached_wc_Sigma_v_;
    MatXd warm_Sigma_v_;
    bool has_cached_Sigma_v_;
    bool has_warm_Sigma_v_;
    int dr_solve_count_;
    int dr_step_count_;

    // Cached identity matrices
    MatXd I_nx_;
    MatXd I_ny_;
};

} // namespace dr_ekf
