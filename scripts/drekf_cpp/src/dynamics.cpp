#include "dr_ekf/dynamics.h"

namespace dr_ekf {

// ============================================================
// CT2D: 2D Coordinated Turn [px, py, vx, vy, omega]
// ============================================================

CT2D::CT2D(double dt, double omega_eps, double sensor_x, double sensor_y, double range_eps)
    : dt(dt), omega_eps(omega_eps), sensor_x(sensor_x), sensor_y(sensor_y), range_eps(range_eps) {}

VecXd CT2D::f(const VecXd& x, const VecXd& /*u*/) const {
    double px = x(0), py = x(1), vx = x(2), vy = x(3), omega = x(4);
    double phi = omega * dt;
    VecXd x_next(5);

    if (std::abs(omega) >= omega_eps) {
        double sin_phi = std::sin(phi);
        double cos_phi = std::cos(phi);
        double A = sin_phi / omega;
        double B = (1.0 - cos_phi) / omega;
        double C = (cos_phi - 1.0) / omega;
        double D = sin_phi / omega;

        x_next(0) = px + A * vx + B * vy;
        x_next(1) = py + C * vx + D * vy;
        x_next(2) = cos_phi * vx - sin_phi * vy;
        x_next(3) = sin_phi * vx + cos_phi * vy;
        x_next(4) = omega;
    } else {
        x_next(0) = px + vx * dt;
        x_next(1) = py + vy * dt;
        x_next(2) = vx;
        x_next(3) = vy;
        x_next(4) = omega;
    }
    return x_next;
}

MatXd CT2D::F_jac(const VecXd& x, const VecXd& /*u*/) const {
    double vx = x(2), vy = x(3), omega = x(4);
    double phi = omega * dt;
    MatXd F = MatXd::Zero(5, 5);

    if (std::abs(omega) >= omega_eps) {
        double sin_phi = std::sin(phi);
        double cos_phi = std::cos(phi);
        double A = sin_phi / omega;
        double B = (1.0 - cos_phi) / omega;
        double C = (cos_phi - 1.0) / omega;
        double D = sin_phi / omega;

        double dA_dw = (dt * cos_phi) / omega - sin_phi / (omega * omega);
        double dB_dw = (dt * sin_phi) / omega - (1.0 - cos_phi) / (omega * omega);
        double dC_dw = (-dt * sin_phi) / omega - (cos_phi - 1.0) / (omega * omega);
        double dD_dw = (dt * cos_phi) / omega - sin_phi / (omega * omega);
        double dcos_dw = -dt * sin_phi;
        double dsin_dw = dt * cos_phi;

        F(0, 0) = 1; F(0, 2) = A; F(0, 3) = B; F(0, 4) = vx * dA_dw + vy * dB_dw;
        F(1, 1) = 1; F(1, 2) = C; F(1, 3) = D; F(1, 4) = vx * dC_dw + vy * dD_dw;
        F(2, 2) = cos_phi; F(2, 3) = -sin_phi; F(2, 4) = vx * dcos_dw - vy * dsin_dw;
        F(3, 2) = sin_phi; F(3, 3) = cos_phi;  F(3, 4) = vx * dsin_dw + vy * dcos_dw;
        F(4, 4) = 1;
    } else {
        F(0, 0) = 1; F(0, 2) = dt;
        F(1, 1) = 1; F(1, 3) = dt;
        F(2, 2) = 1;
        F(3, 3) = 1;
        F(4, 4) = 1;
    }
    return F;
}

VecXd CT2D::h(const VecXd& x) const {
    double dx = x(0) - sensor_x;
    double dy = x(1) - sensor_y;
    double range_val = std::sqrt(dx * dx + dy * dy);
    double bearing = std::atan2(dy, dx);
    VecXd y(2);
    y(0) = range_val;
    y(1) = bearing;
    return y;
}

MatXd CT2D::H_jac(const VecXd& x) const {
    double dx = x(0) - sensor_x;
    double dy = x(1) - sensor_y;
    double range_val = std::sqrt(dx * dx + dy * dy);
    double r = std::max(range_val, range_eps);

    MatXd H = MatXd::Zero(2, 5);
    H(0, 0) = dx / r;
    H(0, 1) = dy / r;
    H(1, 0) = -dy / (r * r);
    H(1, 1) = dx / (r * r);
    return H;
}

// ============================================================
// CT3D: 3D Coordinated Turn [px, py, pz, vx, vy, vz, omega]
// ============================================================

CT3D::CT3D(double dt, double omega_eps, double sensor_x, double sensor_y, double sensor_z,
           double range_eps, double rho_min)
    : dt(dt), omega_eps(omega_eps),
      sensor_x(sensor_x), sensor_y(sensor_y), sensor_z(sensor_z),
      range_eps(range_eps), rho_min(rho_min) {}

VecXd CT3D::f(const VecXd& x, const VecXd& /*u*/) const {
    double px = x(0), py = x(1), pz = x(2);
    double vx = x(3), vy = x(4), vz = x(5), omega = x(6);
    double phi = omega * dt;
    VecXd x_next(7);

    if (std::abs(omega) >= omega_eps) {
        double sin_phi = std::sin(phi);
        double cos_phi = std::cos(phi);
        double A = sin_phi / omega;
        double B = (1.0 - cos_phi) / omega;
        double C = (cos_phi - 1.0) / omega;
        double D = sin_phi / omega;

        x_next(0) = px + A * vx + B * vy;
        x_next(1) = py + C * vx + D * vy;
        x_next(2) = pz + vz * dt;
        x_next(3) = cos_phi * vx - sin_phi * vy;
        x_next(4) = sin_phi * vx + cos_phi * vy;
        x_next(5) = vz;
        x_next(6) = omega;
    } else {
        x_next(0) = px + vx * dt;
        x_next(1) = py + vy * dt;
        x_next(2) = pz + vz * dt;
        x_next(3) = vx;
        x_next(4) = vy;
        x_next(5) = vz;
        x_next(6) = omega;
    }
    return x_next;
}

MatXd CT3D::F_jac(const VecXd& x, const VecXd& /*u*/) const {
    double vx = x(3), vy = x(4), omega = x(6);
    double phi = omega * dt;
    MatXd F = MatXd::Zero(7, 7);

    if (std::abs(omega) >= omega_eps) {
        double sin_phi = std::sin(phi);
        double cos_phi = std::cos(phi);
        double A = sin_phi / omega;
        double B = (1.0 - cos_phi) / omega;
        double C = (cos_phi - 1.0) / omega;
        double D = sin_phi / omega;

        double dA_dw = (dt * cos_phi) / omega - sin_phi / (omega * omega);
        double dB_dw = (dt * sin_phi) / omega - (1.0 - cos_phi) / (omega * omega);
        double dC_dw = (-dt * sin_phi) / omega - (cos_phi - 1.0) / (omega * omega);
        double dD_dw = (dt * cos_phi) / omega - sin_phi / (omega * omega);
        double dcos_dw = -dt * sin_phi;
        double dsin_dw = dt * cos_phi;

        F(0, 0) = 1; F(0, 3) = A; F(0, 4) = B; F(0, 6) = vx * dA_dw + vy * dB_dw;
        F(1, 1) = 1; F(1, 3) = C; F(1, 4) = D; F(1, 6) = vx * dC_dw + vy * dD_dw;
        F(2, 2) = 1; F(2, 5) = dt;
        F(3, 3) = cos_phi; F(3, 4) = -sin_phi; F(3, 6) = vx * dcos_dw - vy * dsin_dw;
        F(4, 3) = sin_phi; F(4, 4) = cos_phi;  F(4, 6) = vx * dsin_dw + vy * dcos_dw;
        F(5, 5) = 1;
        F(6, 6) = 1;
    } else {
        F(0, 0) = 1; F(0, 3) = dt;
        F(1, 1) = 1; F(1, 4) = dt;
        F(2, 2) = 1; F(2, 5) = dt;
        F(3, 3) = 1;
        F(4, 4) = 1;
        F(5, 5) = 1;
        F(6, 6) = 1;
    }
    return F;
}

VecXd CT3D::h(const VecXd& x) const {
    double dx = x(0) - sensor_x;
    double dy = x(1) - sensor_y;
    double dz = x(2) - sensor_z;
    double rho = std::sqrt(dx * dx + dy * dy);
    double r = std::sqrt(dx * dx + dy * dy + dz * dz);

    VecXd y(3);
    y(0) = r;
    y(1) = std::atan2(dy, dx);
    y(2) = std::atan2(dz, rho);
    return y;
}

MatXd CT3D::H_jac(const VecXd& x) const {
    double dx = x(0) - sensor_x;
    double dy = x(1) - sensor_y;
    double dz = x(2) - sensor_z;
    double rho = std::sqrt(dx * dx + dy * dy);
    double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    r = std::max(r, range_eps);

    MatXd H = MatXd::Zero(3, 7);

    H(0, 0) = dx / r;
    H(0, 1) = dy / r;
    H(0, 2) = dz / r;

    if (rho < rho_min) {
        // Near singularity: zero out angle derivatives
        return H;
    }

    double rho_safe = std::max(rho, range_eps);

    // Azimuth derivatives
    H(1, 0) = -dy / (rho_safe * rho_safe);
    H(1, 1) = dx / (rho_safe * rho_safe);

    // Elevation derivatives
    H(2, 0) = -dx * dz / (rho_safe * r * r);
    H(2, 1) = -dy * dz / (rho_safe * r * r);
    H(2, 2) = rho_safe / (r * r);

    return H;
}

// =============================================================================
// TEMPLATE: MyDynamics implementation
// =============================================================================
// Uncomment and fill in after declaring the class in dynamics.h.
//
// MyDynamics::MyDynamics(double dt /*, ... */)
//     : dt(dt) /*, ... */ {}
//
// VecXd MyDynamics::f(const VecXd& x, const VecXd& /*u*/) const {
//     // Compute x_{k+1} from x_k (and optionally u_k).
//     VecXd x_next(nx());
//     // TODO: fill in your discrete-time dynamics
//     return x_next;
// }
//
// MatXd MyDynamics::F_jac(const VecXd& x, const VecXd& /*u*/) const {
//     // Jacobian df/dx at (x, u). Must be nx() x nx().
//     MatXd F = MatXd::Zero(nx(), nx());
//     // TODO: fill in the Jacobian analytically (finite-difference is too slow)
//     return F;
// }
//
// VecXd MyDynamics::h(const VecXd& x) const {
//     // Compute measurement y from state x.
//     VecXd y(ny());
//     // TODO: fill in your observation model
//     return y;
// }
//
// MatXd MyDynamics::H_jac(const VecXd& x) const {
//     // Jacobian dh/dx at x. Must be ny() x nx().
//     MatXd H = MatXd::Zero(ny(), nx());
//     // TODO: fill in the observation Jacobian analytically
//     return H;
// }

} // namespace dr_ekf
