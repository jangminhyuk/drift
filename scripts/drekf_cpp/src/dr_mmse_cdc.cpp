#include "dr_ekf/dr_mmse_cdc.h"
#include "dr_ekf/kalman_utils.h"
#include "dr_ekf/fw_oracle.h"

namespace dr_ekf {

DRMMSECDCResult solve_dr_mmse_cdc(
    const MatXd& X_prior_hat,
    const MatXd& C,
    const MatXd& Sigma_v_hat,
    double theta_x, double theta_v,
    const std::string& solver,
    const MatXd& warm_Sigma_v,
    double fw_beta_minus1, double fw_tau,
    double fw_zeta, double fw_delta,
    int fw_max_iters, double fw_gap_tol,
    int fw_bisect_max_iters, double fw_bisect_tol,
    int fw_exact_iters, double fw_exact_gap_tol,
    double oracle_exact_tol, int oracle_exact_max_iters)
{
    int nx = X_prior_hat.rows();
    int ny = C.rows();
    MatXd I_ny = MatXd::Identity(ny, ny);

    // Short-circuit: theta=0 => standard Kalman (no DR)
    if (theta_x <= 0 && theta_v <= 0) {
        MatXd Xprior = sym(X_prior_hat);
        MatXd Xpost = posterior_cov(Xprior, Sigma_v_hat, C);
        return {Sigma_v_hat, Xprior, Xpost, true, 0};
    }

    // Determine iteration count and gap tolerance based on solver type
    bool use_exact = (solver == "fw_exact");
    int max_iters = use_exact ? fw_exact_iters : fw_max_iters;
    double gap_tol = use_exact ? fw_exact_gap_tol : fw_gap_tol;

    MatXd Sigma_v_nom = Sigma_v_hat;
    // X_prior always starts from nominal (no warm-start for X_prior in CDC)
    MatXd X_prior = X_prior_hat;
    MatXd Sigma_v;

    // --- Safe warm start for Sigma_v ---
    if (theta_v > 0 && warm_Sigma_v.rows() > 0) {
        MatXd Sigma_v_warm = sym(warm_Sigma_v);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_hat_v(Sigma_v_hat);
        double lambda_min_v = eig_hat_v.eigenvalues()(0);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_warm_v(Sigma_v_warm);
        double eig_min_v = eig_warm_v.eigenvalues()(0);
        if (eig_min_v < lambda_min_v) {
            Sigma_v_warm = sym(Sigma_v_warm + (lambda_min_v - eig_min_v) * I_ny);
        }
        double f_nom_v = fw_objective_f_min(X_prior, Sigma_v_nom, C);
        double f_warm_v = fw_objective_f_min(X_prior, Sigma_v_warm, C);
        Sigma_v = (f_warm_v < f_nom_v) ? Sigma_v_warm : Sigma_v_nom;
    } else {
        Sigma_v = Sigma_v_nom;
    }

    double beta_prev = fw_beta_minus1;

    // Initial objective and gradients
    auto stats = kalman_stats(X_prior, Sigma_v, C);
    double f_curr = -stats.trace_post;
    MatXd D_x = stats.D_x;
    MatXd D_v = stats.D_v;

    int iterations = 0;
    for (int t = 0; t < max_iters; ++t) {
        iterations = t + 1;

        // Oracle calls
        MatXd L_x, L_v;
        if (use_exact) {
            L_x = (theta_x > 0) ?
                oracle_exact_fast(X_prior_hat, theta_x, D_x,
                                  oracle_exact_tol, oracle_exact_max_iters) :
                X_prior_hat;
            L_v = (theta_v > 0) ?
                oracle_exact_fast(Sigma_v_hat, theta_v, D_v,
                                  oracle_exact_tol, oracle_exact_max_iters) :
                Sigma_v_nom;
        } else {
            L_x = (theta_x > 0) ?
                oracle_bisection_fast(X_prior_hat, theta_x, X_prior, D_x, fw_delta,
                                      fw_bisect_max_iters, fw_bisect_tol) :
                X_prior_hat;
            L_v = (theta_v > 0) ?
                oracle_bisection_fast(Sigma_v_hat, theta_v, Sigma_v, D_v, fw_delta,
                                      fw_bisect_max_iters, fw_bisect_tol) :
                Sigma_v_nom;
        }

        MatXd d_x = L_x - X_prior;
        MatXd d_v = L_v - Sigma_v;

        double g = inner(d_x, D_x) + inner(d_v, D_v);
        if (g <= gap_tol) break;

        double beta_t = beta_prev / fw_zeta;
        double d_norm2 = fro_norm2(d_x) + fro_norm2(d_v);
        if (d_norm2 <= 0 || g <= 0) break;

        double eta = std::min(1.0, g / (beta_t * d_norm2));

        double f_next = 0;
        for (int bt = 0; bt < 20; ++bt) {
            auto stats_next = kalman_stats(X_prior + eta * d_x,
                                           Sigma_v + eta * d_v, C);
            f_next = -stats_next.trace_post;
            double armijo_rhs = f_curr - eta * g + 0.5 * eta * eta * beta_t * d_norm2;

            if (f_next <= armijo_rhs) {
                D_x = stats_next.D_x;
                D_v = stats_next.D_v;
                break;
            }
            beta_t = fw_tau * beta_t;
            eta = std::min(1.0, g / (beta_t * d_norm2));
        }

        X_prior = X_prior + eta * d_x;
        Sigma_v = Sigma_v + eta * d_v;
        f_curr = f_next;
        beta_prev = beta_t;
    }

    MatXd wc_Xprior = sym(X_prior);
    MatXd wc_Sigma_v = sym(Sigma_v);

    // Ensure wc_Sigma_v is PD (eigenvalue floor)
    {
        Eigen::SelfAdjointEigenSolver<MatXd> eig(wc_Sigma_v);
        double eig_min = eig.eigenvalues()(0);
        if (eig_min < 1e-14) {
            wc_Sigma_v = sym(wc_Sigma_v + (1e-14 - eig_min) * I_ny);
        }
    }

    MatXd wc_Xpost = posterior_cov(wc_Xprior, wc_Sigma_v, C);

    return {wc_Sigma_v, wc_Xprior, wc_Xpost, true, iterations};
}

} // namespace dr_ekf
