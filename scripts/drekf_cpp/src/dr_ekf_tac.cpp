#include "dr_ekf/dr_ekf_tac.h"
#include "dr_ekf/kalman_utils.h"
#include "dr_ekf/fw_oracle.h"

namespace dr_ekf {

DR_EKF_TAC::DR_EKF_TAC(
    const DynamicsInterface& dynamics,
    const MatXd& nominal_x0_cov,
    const MatXd& nominal_Sigma_w,
    const MatXd& nominal_Sigma_v,
    const VecXd& nominal_mu_w,
    const VecXd& nominal_mu_v,
    double theta_x, double theta_v, double theta_w,
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
      theta_x_(theta_x), theta_v_(theta_v), theta_w_(theta_w),
      solver_(solver),
      dr_update_period_(dr_update_period),
      fw_beta_minus1_(fw_beta_minus1), fw_tau_(fw_tau), fw_zeta_(fw_zeta), fw_delta_(fw_delta),
      fw_max_iters_(fw_max_iters), fw_gap_tol_(fw_gap_tol),
      fw_bisect_max_iters_(fw_bisect_max_iters), fw_bisect_tol_(fw_bisect_tol),
      fw_exact_iters_(fw_exact_iters), fw_exact_gap_tol_(fw_exact_gap_tol),
      oracle_exact_tol_(oracle_exact_tol), oracle_exact_max_iters_(oracle_exact_max_iters),
      has_cached_Sigma_v_(false), has_cached_Sigma_w_(false),
      has_warm_Sigma_v_(false), has_warm_Sigma_w_(false),
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
        throw std::invalid_argument("C++ DR_EKF_TAC only supports solver='fw' or 'fw_exact'"
#ifdef DR_EKF_HAS_MOSEK
            " or 'mosek'"
#endif
        );
    }
}

// ============================================================
// t=0 Solver Router
// ============================================================
TACSDPResult DR_EKF_TAC::solve_sdp_t0(const MatXd& X_pred_hat, const MatXd& C_t) {
    if (solver_ == "fw") {
        return solve_sdp_t0_fw(X_pred_hat, C_t);
    } else if (solver_ == "fw_exact") {
        return solve_sdp_t0_fw_exact(X_pred_hat, C_t);
    }
#ifdef DR_EKF_HAS_MOSEK
    else {
        // mosek
        if (!initial_solver_) {
            initial_solver_ = std::make_unique<mosek_solver::CDCSolver>(nx_, ny_);
        }
        auto result = initial_solver_->solve(X_pred_hat, Sigma_v_hat_, theta_x_, theta_v_, C_t);
        return {result.wc_Sigma_v, Sigma_w_hat_, result.wc_Xprior, result.wc_Xpost, result.success};
    }
#endif
    // fallback (should not reach here)
    return {Sigma_v_hat_, Sigma_w_hat_, X_pred_hat, X_pred_hat, false};
}

// ============================================================
// t=0 FW Solver (CDC-like: optimize over X_pred, Sigma_v)
// ============================================================
TACSDPResult DR_EKF_TAC::solve_sdp_t0_fw(const MatXd& X_pred_hat, const MatXd& C_t) {
    MatXd X_pred = X_pred_hat;
    MatXd Sigma_v_nom = Sigma_v_hat_;
    MatXd Sigma_v;

    // Safe warm start for Sigma_v
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

    for (int t = 0; t < fw_max_iters_; ++t) {
        MatXd L_x = oracle_bisection_fast(X_pred_hat, theta_x_, X_pred, D_x, fw_delta_,
                                           fw_bisect_max_iters_, fw_bisect_tol_);
        MatXd L_v = oracle_bisection_fast(Sigma_v_hat_, theta_v_, Sigma_v, D_v, fw_delta_,
                                           fw_bisect_max_iters_, fw_bisect_tol_);

        MatXd d_x = L_x - X_pred;
        MatXd d_v = L_v - Sigma_v;

        double g_t = inner(d_x, D_x) + inner(d_v, D_v);
        if (g_t <= fw_gap_tol_) break;

        double beta_t = beta_prev / fw_zeta_;
        double d_norm2 = fro_norm2(d_x) + fro_norm2(d_v);
        if (d_norm2 <= 0 || g_t <= 0) break;

        double eta = std::min(1.0, g_t / (beta_t * d_norm2));

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

        X_pred = X_pred + eta * d_x;
        Sigma_v = Sigma_v + eta * d_v;
        f_curr = f_next;
        beta_prev = beta_t;
    }

    MatXd wc_Xprior = sym(X_pred);
    MatXd wc_Sigma_v = sym(Sigma_v);
    MatXd wc_Xpost = posterior_cov(wc_Xprior, wc_Sigma_v, C_t);

    return {wc_Sigma_v, Sigma_w_hat_, wc_Xprior, wc_Xpost, true};
}

// ============================================================
// t=0 FW-Exact Solver (CDC-like)
// ============================================================
TACSDPResult DR_EKF_TAC::solve_sdp_t0_fw_exact(const MatXd& X_pred_hat, const MatXd& C_t) {
    MatXd X_pred = X_pred_hat;
    MatXd Sigma_v_nom = Sigma_v_hat_;
    MatXd Sigma_v;

    // Safe warm start for Sigma_v
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
        MatXd L_x = oracle_exact_fast(X_pred_hat, theta_x_, D_x,
                                       oracle_exact_tol_, oracle_exact_max_iters_);
        MatXd L_v = oracle_exact_fast(Sigma_v_hat_, theta_v_, D_v,
                                       oracle_exact_tol_, oracle_exact_max_iters_);

        MatXd d_x = L_x - X_pred;
        MatXd d_v = L_v - Sigma_v;

        double g = inner(d_x, D_x) + inner(d_v, D_v);
        if (g <= fw_exact_gap_tol_) break;

        double beta_t = beta_prev / fw_zeta_;
        double d_norm2 = fro_norm2(d_x) + fro_norm2(d_v);
        if (d_norm2 <= 0 || g <= 0) break;

        double eta = std::min(1.0, g / (beta_t * d_norm2));

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

    return {wc_Sigma_v, Sigma_w_hat_, wc_Xprior, wc_Xpost, true};
}

// ============================================================
// t>0 FW Solver (optimize over Sigma_w, Sigma_v)
// ============================================================
TACSDPResult DR_EKF_TAC::solve_sdp_tgt0_fw(const MatXd& APA, const MatXd& C_t) {
    MatXd Sigma_w_nom = Sigma_w_hat_;
    MatXd Sigma_v_nom = Sigma_v_hat_;
    MatXd Sigma_w, Sigma_v;

    // Safe warm start for Sigma_w
    if (!has_warm_Sigma_w_) {
        Sigma_w = Sigma_w_nom;
    } else {
        MatXd Sigma_w_warm = sym(warm_Sigma_w_);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_hat_w(Sigma_w_hat_);
        double lambda_min_w = eig_hat_w.eigenvalues()(0);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_warm_w(Sigma_w_warm);
        double eig_min_w = eig_warm_w.eigenvalues()(0);
        if (eig_min_w < lambda_min_w) {
            Sigma_w_warm = sym(Sigma_w_warm + (lambda_min_w - eig_min_w) * I_nx_);
        }
        double f_nom_w = fw_objective_f_min(APA + Sigma_w_nom, Sigma_v_nom, C_t);
        double f_warm_w = fw_objective_f_min(APA + Sigma_w_warm, Sigma_v_nom, C_t);
        Sigma_w = (f_warm_w < f_nom_w) ? Sigma_w_warm : Sigma_w_nom;
    }

    // Safe warm start for Sigma_v
    if (!has_warm_Sigma_v_) {
        Sigma_v = Sigma_v_nom;
    } else {
        MatXd Sigma_v_warm = sym(warm_Sigma_v_);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_hat_v(Sigma_v_hat_);
        double lambda_min_v = eig_hat_v.eigenvalues()(0);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_warm_v(Sigma_v_warm);
        double eig_min_v = eig_warm_v.eigenvalues()(0);
        if (eig_min_v < lambda_min_v) {
            Sigma_v_warm = sym(Sigma_v_warm + (lambda_min_v - eig_min_v) * I_ny_);
        }
        double f_nom_v = fw_objective_f_min(APA + Sigma_w, Sigma_v_nom, C_t);
        double f_warm_v = fw_objective_f_min(APA + Sigma_w, Sigma_v_warm, C_t);
        Sigma_v = (f_warm_v < f_nom_v) ? Sigma_v_warm : Sigma_v_nom;
    }

    double beta_prev = fw_beta_minus1_;

    // X_pred = APA + Sigma_w
    auto stats = kalman_stats(APA + Sigma_w, Sigma_v, C_t);
    double f_curr = -stats.trace_post;
    MatXd D_w = stats.D_x;  // Chain rule: dX_pred/dSigma_w = I
    MatXd D_v = stats.D_v;

    for (int t = 0; t < fw_max_iters_; ++t) {
        // Oracle for Sigma_w in BW ball around Sigma_w_hat
        MatXd L_w = oracle_bisection_fast(Sigma_w_hat_, theta_w_, Sigma_w, D_w, fw_delta_,
                                           fw_bisect_max_iters_, fw_bisect_tol_);
        MatXd L_v = oracle_bisection_fast(Sigma_v_hat_, theta_v_, Sigma_v, D_v, fw_delta_,
                                           fw_bisect_max_iters_, fw_bisect_tol_);

        MatXd d_w = L_w - Sigma_w;
        MatXd d_v = L_v - Sigma_v;

        double g_t = inner(d_w, D_w) + inner(d_v, D_v);
        if (g_t <= fw_gap_tol_) break;

        double beta_t = beta_prev / fw_zeta_;
        double d_norm2 = fro_norm2(d_w) + fro_norm2(d_v);
        if (d_norm2 <= 0 || g_t <= 0) break;

        double eta = std::min(1.0, g_t / (beta_t * d_norm2));

        double f_next = 0;
        for (int bt = 0; bt < 20; ++bt) {
            auto stats_next = kalman_stats(APA + Sigma_w + eta * d_w, Sigma_v + eta * d_v, C_t);
            f_next = -stats_next.trace_post;
            double armijo_rhs = f_curr - eta * g_t + 0.5 * eta * eta * beta_t * d_norm2;

            if (f_next <= armijo_rhs) {
                D_w = stats_next.D_x;  // Chain rule
                D_v = stats_next.D_v;
                break;
            }
            beta_t = fw_tau_ * beta_t;
            eta = std::min(1.0, g_t / (beta_t * d_norm2));
        }

        Sigma_w = Sigma_w + eta * d_w;
        Sigma_v = Sigma_v + eta * d_v;
        f_curr = f_next;
        beta_prev = beta_t;
    }

    MatXd wc_Sigma_w = sym(Sigma_w);
    MatXd wc_Sigma_v = sym(Sigma_v);
    MatXd wc_Xprior = sym(APA + wc_Sigma_w);
    MatXd wc_Xpost = posterior_cov(wc_Xprior, wc_Sigma_v, C_t);

    return {wc_Sigma_v, wc_Sigma_w, wc_Xprior, wc_Xpost, true};
}

// ============================================================
// t>0 FW-Exact Solver (optimize over Sigma_w, Sigma_v)
// ============================================================
TACSDPResult DR_EKF_TAC::solve_sdp_tgt0_fw_exact(const MatXd& APA, const MatXd& C_t) {
    MatXd Sigma_w_nom = Sigma_w_hat_;
    MatXd Sigma_v_nom = Sigma_v_hat_;
    MatXd Sigma_w, Sigma_v;

    // Safe warm start for Sigma_w
    if (!has_warm_Sigma_w_) {
        Sigma_w = Sigma_w_nom;
    } else {
        MatXd Sigma_w_warm = sym(warm_Sigma_w_);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_hat_w(Sigma_w_hat_);
        double lambda_min_w = eig_hat_w.eigenvalues()(0);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_warm_w(Sigma_w_warm);
        double eig_min_w = eig_warm_w.eigenvalues()(0);
        if (eig_min_w < lambda_min_w) {
            Sigma_w_warm = sym(Sigma_w_warm + (lambda_min_w - eig_min_w) * I_nx_);
        }
        double f_nom_w = fw_objective_f_min(APA + Sigma_w_nom, Sigma_v_nom, C_t);
        double f_warm_w = fw_objective_f_min(APA + Sigma_w_warm, Sigma_v_nom, C_t);
        Sigma_w = (f_warm_w < f_nom_w) ? Sigma_w_warm : Sigma_w_nom;
    }

    // Safe warm start for Sigma_v
    if (!has_warm_Sigma_v_) {
        Sigma_v = Sigma_v_nom;
    } else {
        MatXd Sigma_v_warm = sym(warm_Sigma_v_);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_hat_v(Sigma_v_hat_);
        double lambda_min_v = eig_hat_v.eigenvalues()(0);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_warm_v(Sigma_v_warm);
        double eig_min_v = eig_warm_v.eigenvalues()(0);
        if (eig_min_v < lambda_min_v) {
            Sigma_v_warm = sym(Sigma_v_warm + (lambda_min_v - eig_min_v) * I_ny_);
        }
        double f_nom_v = fw_objective_f_min(APA + Sigma_w, Sigma_v_nom, C_t);
        double f_warm_v = fw_objective_f_min(APA + Sigma_w, Sigma_v_warm, C_t);
        Sigma_v = (f_warm_v < f_nom_v) ? Sigma_v_warm : Sigma_v_nom;
    }

    double beta_prev = fw_beta_minus1_;

    auto stats = kalman_stats(APA + Sigma_w, Sigma_v, C_t);
    double f_curr = -stats.trace_post;
    MatXd D_w = stats.D_x;  // Chain rule
    MatXd D_v = stats.D_v;

    for (int t = 0; t < fw_exact_iters_; ++t) {
        MatXd L_w = oracle_exact_fast(Sigma_w_hat_, theta_w_, D_w,
                                       oracle_exact_tol_, oracle_exact_max_iters_);
        MatXd L_v = oracle_exact_fast(Sigma_v_hat_, theta_v_, D_v,
                                       oracle_exact_tol_, oracle_exact_max_iters_);

        MatXd d_w = L_w - Sigma_w;
        MatXd d_v = L_v - Sigma_v;

        double g = inner(d_w, D_w) + inner(d_v, D_v);
        if (g <= fw_exact_gap_tol_) break;

        double beta_t = beta_prev / fw_zeta_;
        double d_norm2 = fro_norm2(d_w) + fro_norm2(d_v);
        if (d_norm2 <= 0 || g <= 0) break;

        double eta = std::min(1.0, g / (beta_t * d_norm2));

        double f_next = 0;
        for (int bt = 0; bt < 20; ++bt) {
            auto stats_next = kalman_stats(APA + Sigma_w + eta * d_w, Sigma_v + eta * d_v, C_t);
            f_next = -stats_next.trace_post;
            double armijo_rhs = f_curr - eta * g + 0.5 * eta * eta * beta_t * d_norm2;

            if (f_next <= armijo_rhs) {
                D_w = stats_next.D_x;  // Chain rule
                D_v = stats_next.D_v;
                break;
            }
            beta_t = fw_tau_ * beta_t;
            eta = std::min(1.0, g / (beta_t * d_norm2));
        }

        Sigma_w = Sigma_w + eta * d_w;
        Sigma_v = Sigma_v + eta * d_v;
        f_curr = f_next;
        beta_prev = beta_t;
    }

    MatXd wc_Sigma_w = sym(Sigma_w);
    MatXd wc_Sigma_v = sym(Sigma_v);
    MatXd wc_Xprior = sym(APA + wc_Sigma_w);
    MatXd wc_Xpost = posterior_cov(wc_Xprior, wc_Sigma_v, C_t);

    return {wc_Sigma_v, wc_Sigma_w, wc_Xprior, wc_Xpost, true};
}

// ============================================================
// DR-EKF Update
// ============================================================
VecXd DR_EKF_TAC::DR_kalman_filter(
    const VecXd& v_mean_hat, const VecXd& x_prior,
    const VecXd& y, int t,
    const VecXd& u_prev, const VecXd& x_post_prev)
{
    // Linearize observation
    MatXd C_t = dynamics_.H_jac(x_prior);

    // Decide whether to solve DR optimization
    bool solve_now = (t == 0) || ((t - 1) % dr_update_period_ == 0);
    dr_step_count_++;

    MatXd wc_Sigma_v, wc_Xprior, wc_Xpost;

    if (t == 0) {
        // Initial update: solve CDC-like SDP (only X_pred and Sigma_v ambiguity)
        auto result = solve_sdp_t0(x0_cov_, C_t);

        if (!result.success) {
            throw std::runtime_error("TAC initial SDP failed");
        }

        wc_Sigma_v = result.wc_Sigma_v;
        wc_Xprior = result.wc_Xprior;
        wc_Xpost = result.wc_Xpost;

        // Cache Sigma_v; set cached Sigma_w to nominal
        cached_wc_Sigma_v_ = wc_Sigma_v;
        has_cached_Sigma_v_ = true;
        cached_wc_Sigma_w_ = Sigma_w_hat_;
        has_cached_Sigma_w_ = true;

        // Warm start
        warm_Sigma_v_ = wc_Sigma_v;
        has_warm_Sigma_v_ = true;

        dr_solve_count_++;

    } else if (solve_now) {
        // Regular solve: TAC SDP with process noise ambiguity
        MatXd A_t = dynamics_.F_jac(x_post_prev, u_prev);
        MatXd G_t = dynamics_.G_jac(x_post_prev, u_prev);
        MatXd APA = sym(A_t * P_ * A_t.transpose());
        // Pre-transform process noise nominal into state space
        MatXd Sigma_w_hat_eff = G_t * Sigma_w_hat_ * G_t.transpose();

#ifdef DR_EKF_HAS_MOSEK
        if (solver_ == "mosek") {
            // MOSEK needs P_ and A_t separately
            if (!regular_solver_) {
                regular_solver_ = std::make_unique<mosek_solver::TACSolver>(nx_, ny_);
            }

            auto result = regular_solver_->solve(P_, A_t, C_t, Sigma_w_hat_eff, Sigma_v_hat_,
                                                  theta_w_, theta_v_);

            if (!result.success) {
                throw std::runtime_error("TAC regular SDP failed at t=" + std::to_string(t));
            }

            wc_Sigma_v = result.wc_Sigma_v;
            wc_Xprior = result.wc_Xprior;
            wc_Xpost = result.wc_Xpost;

            cached_wc_Sigma_v_ = wc_Sigma_v;
            has_cached_Sigma_v_ = true;
            cached_wc_Sigma_w_ = result.wc_Sigma_w;
            has_cached_Sigma_w_ = true;

        } else
#endif
        {
            // FW / FW-exact: temporarily use G-transformed nominal
            MatXd Sigma_w_hat_orig = Sigma_w_hat_;
            Sigma_w_hat_ = Sigma_w_hat_eff;
            TACSDPResult result;
            if (solver_ == "fw") {
                result = solve_sdp_tgt0_fw(APA, C_t);
            } else {
                result = solve_sdp_tgt0_fw_exact(APA, C_t);
            }
            Sigma_w_hat_ = Sigma_w_hat_orig;

            if (!result.success) {
                throw std::runtime_error("TAC FW SDP failed at t=" + std::to_string(t));
            }

            wc_Sigma_v = result.wc_Sigma_v;
            wc_Xprior = result.wc_Xprior;
            wc_Xpost = result.wc_Xpost;

            cached_wc_Sigma_v_ = wc_Sigma_v;
            has_cached_Sigma_v_ = true;
            cached_wc_Sigma_w_ = result.wc_Sigma_w;
            has_cached_Sigma_w_ = true;

            // Warm start
            warm_Sigma_v_ = wc_Sigma_v;
            has_warm_Sigma_v_ = true;
            warm_Sigma_w_ = result.wc_Sigma_w;
            has_warm_Sigma_w_ = true;
        }

        dr_solve_count_++;

    } else {
        // Skip step: reuse cached covariances with EKF-like recursion
        MatXd A_t = dynamics_.F_jac(x_post_prev, u_prev);
        MatXd G_t = dynamics_.G_jac(x_post_prev, u_prev);

        MatXd Sigma_w_use = has_cached_Sigma_w_ ? cached_wc_Sigma_w_ : G_t * Sigma_w_hat_ * G_t.transpose();
        MatXd Sigma_v_use = has_cached_Sigma_v_ ? cached_wc_Sigma_v_ : Sigma_v_hat_;

        // EKF-like prediction with cached worst-case Sigma_w
        MatXd X_prior_use = A_t * P_ * A_t.transpose() + Sigma_w_use;
        wc_Xprior = sym(X_prior_use);
        wc_Sigma_v = Sigma_v_use;
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

VecXd DR_EKF_TAC::initial_update(const VecXd& x_est_init, const VecXd& y0) {
    P_ = x0_cov_;
    return DR_kalman_filter(mu_v_, x_est_init, y0, 0, VecXd::Zero(1), VecXd::Zero(nx_));
}

VecXd DR_EKF_TAC::update_step(const VecXd& x_est_prev, const VecXd& y_curr, int t, const VecXd& u_prev) {
    // Prediction
    VecXd x_prior = dynamics_.f(x_est_prev, u_prev) + mu_w_;

    // Update
    return DR_kalman_filter(mu_v_, x_prior, y_curr, t, u_prev, x_est_prev);
}

} // namespace dr_ekf
