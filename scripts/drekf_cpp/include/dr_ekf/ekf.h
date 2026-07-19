#pragma once

#include "types.h"
#include "dynamics.h"

namespace dr_ekf {

class EKF {
public:
    EKF(const DynamicsInterface& dynamics,
        const MatXd& nominal_x0_cov,
        const MatXd& nominal_Sigma_w,
        const MatXd& nominal_Sigma_v,
        const VecXd& nominal_mu_w,
        const VecXd& nominal_mu_v);

    VecXd initial_update(const VecXd& x_est_init, const VecXd& y0);
    VecXd update_step(const VecXd& x_est_prev, const VecXd& y_curr, int t, const VecXd& u_prev);

    // Accessors
    const MatXd& get_P() const { return P_; }
    int get_nx() const { return nx_; }
    int get_ny() const { return ny_; }

private:
    VecXd ekf_update(const VecXd& x_pred, const VecXd& y, int t,
                     const VecXd& u_prev, const VecXd& x_prev);

    const DynamicsInterface& dynamics_;
    int nx_, ny_;
    MatXd P_;
    MatXd Sigma_w_, Sigma_v_;
    VecXd mu_w_, mu_v_;
    MatXd I_nx_;
};

} // namespace dr_ekf
