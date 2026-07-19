#include "dr_ekf/dr_ekf_cdc.h"
#include "dr_ekf/kalman_utils.h"
#include "dr_ekf/fw_oracle.h"

namespace dr_ekf {

DR_EKF_CDC::DR_EKF_CDC(
    const DynamicsInterface& dynamics,
    const MatXd& nominal_x0_cov,
    const MatXd& nominal_Sigma_w,
    const MatXd& nominal_Sigma_v,
    const VecXd& nominal_mu_w,
    const VecXd& nominal_mu_v,
    double theta_x, double theta_v,
    const std::string& solver,
    int dr_update_period,
    double fw_beta_minus1, double fw_tau, double fw_zeta, double fw_delta,
    int fw_max_iters, double fw_gap_tol,
    int fw_bisect_max_iters, double fw_bisect_tol,
    int fw_exact_iters, double fw_exact_gap_tol,
    double oracle_exact_tol, int oracle_exact_max_iters)
    : dynamics_(dynamics),
      nx_(dynamics.nx()), ny_(dynamics.ny()),
      x0_cov_(nominal_x0_cov),
      Sigma_w_hat_(nominal_Sigma_w),
      Sigma_v_hat_(nominal_Sigma_v),
      mu_w_(nominal_mu_w),
      mu_v_(nominal_mu_v),
      theta_x_(theta_x), theta_v_(theta_v),
      solver_(solver),
      dr_update_period_(dr_update_period),
      fw_beta_minus1_(fw_beta_minus1), fw_tau_(fw_tau), fw_zeta_(fw_zeta), fw_delta_(fw_delta),
      fw_max_iters_(fw_max_iters), fw_gap_tol_(fw_gap_tol),
      fw_bisect_max_iters_(fw_bisect_max_iters), fw_bisect_tol_(fw_bisect_tol),
      fw_exact_iters_(fw_exact_iters), fw_exact_gap_tol_(fw_exact_gap_tol),
      oracle_exact_tol_(oracle_exact_tol), oracle_exact_max_iters_(oracle_exact_max_iters),
      has_cached_Sigma_v_(false), has_warm_Sigma_v_(false),
      dr_solve_count_(0), dr_step_count_(0),
      I_nx_(MatXd::Identity(dynamics.nx(), dynamics.nx())),
      I_ny_(MatXd::Identity(dynamics.ny(), dynamics.ny()))
{
    if (dr_update_period < 1) {
        throw std::invalid_argument("dr_update_period must be >= 1");
    }
    if (solver != "fw" && solver != "fw_exact"
#ifdef DR_EKF_HAS_MOSEK
        && solver != "mosek"
#endif
    ) {
        throw std::invalid_argument("C++ DR_EKF_CDC only supports solver='fw' or 'fw_exact'"
#ifdef DR_EKF_HAS_MOSEK
            " or 'mosek'"
#endif
        );
    }
}

// ============================================================
// FW Solver (Algorithm 1 in DR-MMSE paper)
// ============================================================
SDPResult DR_EKF_CDC::solve_sdp_online_fw(const MatXd& X_pred_hat, const MatXd& C_t) {
    MatXd X_pred = X_pred_hat;
    MatXd Sigma_v_nom = Sigma_v_hat_;
    MatXd Sigma_v;

    // Safe warm start
    if (!has_warm_Sigma_v_) {
        Sigma_v = Sigma_v_nom;
    } else {
        MatXd Sigma_v_warm = sym(warm_Sigma_v_);
        // Ensure PD floor
        Eigen::SelfAdjointEigenSolver<MatXd> eig_hat(Sigma_v_hat_);
        double lambda_min_val = eig_hat.eigenvalues()(0);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_warm(Sigma_v_warm);
        double eig_min = eig_warm.eigenvalues()(0);
        if (eig_min < lambda_min_val) {
            Sigma_v_warm = sym(Sigma_v_warm + (lambda_min_val - eig_min) * I_ny_);
        }
        double f_nom = fw_objective_f_min(X_pred, Sigma_v_nom, C_t);
        double f_warm = fw_objective_f_min(X_pred, Sigma_v_warm, C_t);
        Sigma_v = (f_warm < f_nom) ? Sigma_v_warm : Sigma_v_nom;
    }

    double beta_prev = fw_beta_minus1_;

    // Precompute first iteration stats
    auto stats = kalman_stats(X_pred, Sigma_v, C_t);
    double f_curr = -stats.trace_post;
    MatXd D_x = stats.D_x;
    MatXd D_v = stats.D_v;

    for (int t = 0; t < fw_max_iters_; ++t) {
        // Oracle subproblem
        MatXd L_x = oracle_bisection_fast(X_pred_hat, theta_x_, X_pred, D_x, fw_delta_,
                                           fw_bisect_max_iters_, fw_bisect_tol_);
        MatXd L_v = oracle_bisection_fast(Sigma_v_hat_, theta_v_, Sigma_v, D_v, fw_delta_,
                                           fw_bisect_max_iters_, fw_bisect_tol_);

        // Direction
        MatXd d_x = L_x - X_pred;
        MatXd d_v = L_v - Sigma_v;

        // Surrogate duality gap
        double g_t = inner(d_x, D_x) + inner(d_v, D_v);

        if (g_t <= fw_gap_tol_) break;

        // Adaptive step size
        double beta_t = beta_prev / fw_zeta_;
        double d_norm2 = fro_norm2(d_x) + fro_norm2(d_v);

        if (d_norm2 <= 0 || g_t <= 0) break;

        double eta = std::min(1.0, g_t / (beta_t * d_norm2));

        // Backtracking Armijo line search
        double f_next = 0;
        for (int bt = 0; bt < 20; ++bt) {
            auto stats_next = kalman_stats(X_pred + eta * d_x, Sigma_v + eta * d_v, C_t);
            f_next = -stats_next.trace_post;
            double armijo_rhs = f_curr - eta * g_t + 0.5 * eta * eta * beta_t * d_norm2;

            if (f_next <= armijo_rhs) {
                D_x = stats_next.D_x;
                D_v = stats_next.D_v;
                break;
            }
            beta_t = fw_tau_ * beta_t;
            eta = std::min(1.0, g_t / (beta_t * d_norm2));
        }

        // Update
        X_pred = X_pred + eta * d_x;
        Sigma_v = Sigma_v + eta * d_v;
        f_curr = f_next;
        beta_prev = beta_t;
    }

    MatXd wc_Xprior = sym(X_pred);
    MatXd wc_Sigma_v = sym(Sigma_v);
    MatXd wc_Xpost = posterior_cov(wc_Xprior, wc_Sigma_v, C_t);

    return {wc_Sigma_v, wc_Xprior, wc_Xpost, true};
}

// ============================================================
// FW-Exact Solver
// ============================================================
SDPResult DR_EKF_CDC::solve_sdp_online_fw_exact(const MatXd& X_pred_hat, const MatXd& C_t) {
    MatXd X_pred = X_pred_hat;
    MatXd Sigma_v_nom = Sigma_v_hat_;
    MatXd Sigma_v;

    // Safe warm start
    if (!has_warm_Sigma_v_) {
        Sigma_v = Sigma_v_nom;
    } else {
        MatXd Sigma_v_warm = sym(warm_Sigma_v_);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_hat(Sigma_v_hat_);
        double lambda_min_val = eig_hat.eigenvalues()(0);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_warm(Sigma_v_warm);
        double eig_min = eig_warm.eigenvalues()(0);
        if (eig_min < lambda_min_val) {
            Sigma_v_warm = sym(Sigma_v_warm + (lambda_min_val - eig_min) * I_ny_);
        }
        double f_nom = fw_objective_f_min(X_pred, Sigma_v_nom, C_t);
        double f_warm = fw_objective_f_min(X_pred, Sigma_v_warm, C_t);
        Sigma_v = (f_warm < f_nom) ? Sigma_v_warm : Sigma_v_nom;
    }

    double beta_prev = fw_beta_minus1_;

    auto stats = kalman_stats(X_pred, Sigma_v, C_t);
    double f_curr = -stats.trace_post;
    MatXd D_x = stats.D_x;
    MatXd D_v = stats.D_v;

    for (int t = 0; t < fw_exact_iters_; ++t) {
        // Exact oracle
        MatXd L_x = oracle_exact_fast(X_pred_hat, theta_x_, D_x,
                                       oracle_exact_tol_, oracle_exact_max_iters_);
        MatXd L_v = oracle_exact_fast(Sigma_v_hat_, theta_v_, D_v,
                                       oracle_exact_tol_, oracle_exact_max_iters_);

        // Direction
        MatXd d_x = L_x - X_pred;
        MatXd d_v = L_v - Sigma_v;

        // Duality gap
        double g = inner(d_x, D_x) + inner(d_v, D_v);
        if (g <= fw_exact_gap_tol_) break;

        // Step size
        double beta_t = beta_prev / fw_zeta_;
        double d_norm2 = fro_norm2(d_x) + fro_norm2(d_v);
        if (d_norm2 <= 0 || g <= 0) break;

        double eta = std::min(1.0, g / (beta_t * d_norm2));

        // Backtracking
        double f_next = 0;
        for (int bt = 0; bt < 20; ++bt) {
            auto stats_next = kalman_stats(X_pred + eta * d_x, Sigma_v + eta * d_v, C_t);
            f_next = -stats_next.trace_post;
            double armijo_rhs = f_curr - eta * g + 0.5 * eta * eta * beta_t * d_norm2;

            if (f_next <= armijo_rhs) {
                D_x = stats_next.D_x;
                D_v = stats_next.D_v;
                break;
            }
            beta_t = fw_tau_ * beta_t;
            eta = std::min(1.0, g / (beta_t * d_norm2));
        }

        X_pred = X_pred + eta * d_x;
        Sigma_v = Sigma_v + eta * d_v;
        f_curr = f_next;
        beta_prev = beta_t;
    }

    MatXd wc_Xprior = sym(X_pred);
    MatXd wc_Sigma_v = sym(Sigma_v);
    MatXd wc_Xpost = posterior_cov(wc_Xprior, wc_Sigma_v, C_t);

    return {wc_Sigma_v, wc_Xprior, wc_Xpost, true};
}

#ifdef DR_EKF_HAS_MOSEK
// ============================================================
// MOSEK Solver
// ============================================================
SDPResult DR_EKF_CDC::solve_sdp_online_mosek(const MatXd& X_pred_hat, const MatXd& C_t) {
    // Lazy-initialize MOSEK solver
    if (!mosek_solver_) {
        mosek_solver_ = std::make_unique<mosek_solver::CDCSolver>(nx_, ny_);
    }

    auto result = mosek_solver_->solve(X_pred_hat, Sigma_v_hat_, theta_x_, theta_v_, C_t);
    return {result.wc_Sigma_v, result.wc_Xprior, result.wc_Xpost, result.success};
}
#endif

// ============================================================
// Solver Router
// ============================================================
SDPResult DR_EKF_CDC::solve_sdp_online(const MatXd& X_pred_hat, const MatXd& C_t) {
    if (solver_ == "fw") {
        return solve_sdp_online_fw(X_pred_hat, C_t);
    }
#ifdef DR_EKF_HAS_MOSEK
    else if (solver_ == "mosek") {
        return solve_sdp_online_mosek(X_pred_hat, C_t);
    }
#endif
    else {
        return solve_sdp_online_fw_exact(X_pred_hat, C_t);
    }
}

// ============================================================
// DR-EKF Update
// ============================================================
VecXd DR_EKF_CDC::DR_kalman_filter(
    const VecXd& v_mean_hat, const VecXd& x_prior,
    const VecXd& y, int t,
    const VecXd& u_prev, const VecXd& x_post_prev)
{
    // Linearize observation
    MatXd C_t = dynamics_.H_jac(x_prior);

    // Compute nominal prior covariance
    MatXd X_prior_nom;
    if (t == 0) {
        X_prior_nom = x0_cov_;
    } else {
        MatXd A_t = dynamics_.F_jac(x_post_prev, u_prev);
        MatXd G_t = dynamics_.G_jac(x_post_prev, u_prev);
        X_prior_nom = A_t * P_ * A_t.transpose() + G_t * Sigma_w_hat_ * G_t.transpose();
    }

    // Decide whether to solve DR optimization
    bool solve_now = (t == 0) || ((t - 1) % dr_update_period_ == 0);
    dr_step_count_++;

    MatXd wc_Sigma_v, wc_Xprior, wc_Xpost;

    if (solve_now) {
        auto result = solve_sdp_online(X_prior_nom, C_t);
        wc_Sigma_v = result.wc_Sigma_v;
        wc_Xprior = result.wc_Xprior;
        wc_Xpost = result.wc_Xpost;

        // Cache
        cached_wc_Sigma_v_ = wc_Sigma_v;
        has_cached_Sigma_v_ = true;

        // Warm start
        warm_Sigma_v_ = wc_Sigma_v;
        has_warm_Sigma_v_ = true;

        dr_solve_count_++;
    } else {
        if (has_cached_Sigma_v_) {
            wc_Sigma_v = cached_wc_Sigma_v_;
        } else {
            // Fallback: solve once
            auto result = solve_sdp_online(X_prior_nom, C_t);
            wc_Sigma_v = result.wc_Sigma_v;
            cached_wc_Sigma_v_ = wc_Sigma_v;
            has_cached_Sigma_v_ = true;
            dr_solve_count_++;
        }
        wc_Xprior = sym(X_prior_nom);
        wc_Xpost = posterior_cov(wc_Xprior, wc_Sigma_v, C_t);
    }

    // DR-EKF update: K* = X_prior* C^T (C X_prior* C^T + Sigma_v*)^{-1}
    MatXd S = C_t * wc_Xprior * C_t.transpose() + wc_Sigma_v;
    Eigen::LDLT<MatXd> ldlt(S);
    if (ldlt.info() != Eigen::Success) {
        MatXd S_reg = S + 1e-12 * I_ny_;
        ldlt.compute(S_reg);
    }
    MatXd K_star = ldlt.solve(C_t * wc_Xprior).transpose();

    // x_post = x_prior + K* (y - h(x_prior) - v_mean)
    VecXd innovation = y - (dynamics_.h(x_prior) + v_mean_hat);
    VecXd x_post = x_prior + K_star * innovation;

    P_ = wc_Xpost;
    return x_post;
}

VecXd DR_EKF_CDC::initial_update(const VecXd& x_est_init, const VecXd& y0) {
    P_ = x0_cov_;
    return DR_kalman_filter(mu_v_, x_est_init, y0, 0, VecXd::Zero(1), VecXd::Zero(nx_));
}

VecXd DR_EKF_CDC::update_step(const VecXd& x_est_prev, const VecXd& y_curr, int t, const VecXd& u_prev) {
    // Prediction
    VecXd x_prior = dynamics_.f(x_est_prev, u_prev) + mu_w_;

    // Update
    return DR_kalman_filter(mu_v_, x_prior, y_curr, t, u_prev, x_est_prev);
}

} // namespace dr_ekf
