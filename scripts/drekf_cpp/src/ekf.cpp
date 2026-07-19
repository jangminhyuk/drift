#include "dr_ekf/ekf.h"

namespace dr_ekf {

EKF::EKF(const DynamicsInterface& dynamics,
         const MatXd& nominal_x0_cov,
         const MatXd& nominal_Sigma_w,
         const MatXd& nominal_Sigma_v,
         const VecXd& nominal_mu_w,
         const VecXd& nominal_mu_v)
    : dynamics_(dynamics),
      nx_(dynamics.nx()), ny_(dynamics.ny()),
      P_(MatXd::Zero(dynamics.nx(), dynamics.nx())),
      Sigma_w_(nominal_Sigma_w),
      Sigma_v_(nominal_Sigma_v),
      mu_w_(nominal_mu_w),
      mu_v_(nominal_mu_v),
      I_nx_(MatXd::Identity(dynamics.nx(), dynamics.nx()))
{
    (void)nominal_x0_cov; // Used in initial_update directly
    P_ = nominal_x0_cov;  // Store for initial_update
}

VecXd EKF::initial_update(const VecXd& x_est_init, const VecXd& y0) {
    MatXd P0 = P_;  // nominal_x0_cov stored in P_ from constructor

    // Linearize observation
    MatXd C0 = dynamics_.H_jac(x_est_init);
    VecXd h0 = dynamics_.h(x_est_init);

    // Innovation covariance
    MatXd S0 = C0 * P0 * C0.transpose() + Sigma_v_;

    // Kalman gain: K0 = P0 C0^T S0^{-1}
    // solve S0 * Z = (P0 * C0^T)^T => Z = S0^{-1} (C0 * P0)
    Eigen::LDLT<MatXd> ldlt(S0);
    if (ldlt.info() != Eigen::Success) {
        MatXd S0_reg = S0 + 1e-12 * MatXd::Identity(ny_, ny_);
        ldlt.compute(S0_reg);
    }
    MatXd K0 = ldlt.solve(C0 * P0).transpose();  // nx x ny

    // Innovation
    VecXd innovation0 = y0 - (h0 + mu_v_);

    // Posterior covariance
    P_ = (I_nx_ - K0 * C0) * P0;

    return x_est_init + K0 * innovation0;
}

VecXd EKF::ekf_update(const VecXd& x_pred, const VecXd& y, int /*t*/,
                       const VecXd& u_prev, const VecXd& x_prev) {
    // Linearize dynamics
    MatXd F_t = dynamics_.F_jac(x_prev, u_prev);

    // Prediction covariance
    MatXd G_t = dynamics_.G_jac(x_prev, u_prev);
    MatXd P_pred = F_t * P_ * F_t.transpose() + G_t * Sigma_w_ * G_t.transpose();

    // Linearize observation
    MatXd C_t = dynamics_.H_jac(x_pred);
    VecXd h_pred = dynamics_.h(x_pred);

    // Innovation covariance
    MatXd S_t = C_t * P_pred * C_t.transpose() + Sigma_v_;

    // Kalman gain
    Eigen::LDLT<MatXd> ldlt(S_t);
    if (ldlt.info() != Eigen::Success) {
        MatXd S_reg = S_t + 1e-12 * MatXd::Identity(ny_, ny_);
        ldlt.compute(S_reg);
    }
    MatXd K_t = ldlt.solve(C_t * P_pred).transpose();  // nx x ny

    // Innovation
    VecXd innovation = y - (h_pred + mu_v_);

    // State update
    VecXd x_new = x_pred + K_t * innovation;

    // Covariance update
    P_ = (I_nx_ - K_t * C_t) * P_pred;

    return x_new;
}

VecXd EKF::update_step(const VecXd& x_est_prev, const VecXd& y_curr, int t, const VecXd& u_prev) {
    // Prediction
    VecXd x_pred = dynamics_.f(x_est_prev, u_prev) + mu_w_;

    // Update
    return ekf_update(x_pred, y_curr, t, u_prev, x_est_prev);
}

} // namespace dr_ekf
