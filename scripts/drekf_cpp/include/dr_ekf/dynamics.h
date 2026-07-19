#pragma once

#include "types.h"

namespace dr_ekf {

// Abstract dynamics interface
struct DynamicsInterface {
    virtual ~DynamicsInterface() = default;
    virtual VecXd f(const VecXd& x, const VecXd& u) const = 0;
    virtual MatXd F_jac(const VecXd& x, const VecXd& u) const = 0;
    virtual VecXd h(const VecXd& x) const = 0;
    virtual MatXd H_jac(const VecXd& x) const = 0;
    virtual int nx() const = 0;
    virtual int ny() const = 0;
    virtual int nw() const { return nx(); }
    virtual MatXd G_jac(const VecXd& /*x*/, const VecXd& /*u*/) const {
        return MatXd::Identity(nx(), nw());
    }
};

// 2D Coordinated Turn: state = [px, py, vx, vy, omega], obs = [range, bearing]
class CT2D : public DynamicsInterface {
public:
    double dt;
    double omega_eps;
    double sensor_x, sensor_y;
    double range_eps;

    CT2D(double dt = 0.2, double omega_eps = 1e-4,
         double sensor_x = 0.0, double sensor_y = 0.0,
         double range_eps = 1e-6);

    int nx() const override { return 5; }
    int ny() const override { return 2; }

    VecXd f(const VecXd& x, const VecXd& u) const override;
    MatXd F_jac(const VecXd& x, const VecXd& u) const override;
    VecXd h(const VecXd& x) const override;
    MatXd H_jac(const VecXd& x) const override;
};

// 3D Coordinated Turn: state = [px, py, pz, vx, vy, vz, omega], obs = [range, azimuth, elevation]
class CT3D : public DynamicsInterface {
public:
    double dt;
    double omega_eps;
    double sensor_x, sensor_y, sensor_z;
    double range_eps;
    double rho_min;

    CT3D(double dt = 0.2, double omega_eps = 1e-4,
         double sensor_x = 0.0, double sensor_y = 0.0, double sensor_z = 0.0,
         double range_eps = 1e-6, double rho_min = 1e-2);

    int nx() const override { return 7; }
    int ny() const override { return 3; }

    VecXd f(const VecXd& x, const VecXd& u) const override;
    MatXd F_jac(const VecXd& x, const VecXd& u) const override;
    VecXd h(const VecXd& x) const override;
    MatXd H_jac(const VecXd& x) const override;
};

// =============================================================================
// TEMPLATE: Adding a new dynamics model
// =============================================================================
// To implement your own dynamics, copy this template, rename the class, and
// fill in the four methods: f, F_jac, h, H_jac.
//
// The filter code is completely generic — it only interacts with dynamics
// through DynamicsInterface, so no filter files need modification.
//
// Steps:
//   1. dynamics.h   — Declare your class below (this file)
//   2. dynamics.cpp — Implement f, F_jac, h, H_jac
//   3. bindings.cpp — Add a pybind11 binding block (copy CT2D/CT3D pattern)
//   4. estimator/EKF.py — Add an elif branch in _create_dynamics()
//   5. Rebuild: cd estimator/cpp/build && cmake .. && make
//
// class MyDynamics : public DynamicsInterface {
// public:
//     double dt;
//     // ... your model parameters ...
//
//     MyDynamics(double dt = 0.1 /*, ... */);
//
//     int nx() const override { return /* state dimension */; }
//     int ny() const override { return /* observation dimension */; }
//
//     // x_{k+1} = f(x_k, u_k)  — discrete-time state transition
//     VecXd f(const VecXd& x, const VecXd& u) const override;
//
//     // df/dx evaluated at (x, u)  — Jacobian of f w.r.t. state
//     MatXd F_jac(const VecXd& x, const VecXd& u) const override;
//
//     // y_k = h(x_k)  — observation/measurement function
//     VecXd h(const VecXd& x) const override;
//
//     // dh/dx evaluated at x  — Jacobian of h w.r.t. state
//     MatXd H_jac(const VecXd& x) const override;
// };

} // namespace dr_ekf
