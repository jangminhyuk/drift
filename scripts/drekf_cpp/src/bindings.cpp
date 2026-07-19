#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "dr_ekf/types.h"
#include "dr_ekf/dynamics.h"
#include "dr_ekf/ekf.h"
#include "dr_ekf/dr_ekf_cdc.h"
#include "dr_ekf/dr_ekf_tac.h"
#include "dr_ekf/kalman_utils.h"
#include "dr_ekf/fw_oracle.h"
#include "dr_ekf/dr_mmse_tac.h"
#include "dr_ekf/dr_mmse_cdc.h"

namespace py = pybind11;
using namespace dr_ekf;

PYBIND11_MODULE(dr_ekf_cpp, m) {
    m.doc() = "C++ implementation of DR-EKF estimators (EKF, CDC-FW, CDC-FW-Exact, CDC-MOSEK, TAC-MOSEK, TAC-FW, TAC-FW-Exact)";

    // --- Dynamics Interface ---
    py::class_<DynamicsInterface>(m, "DynamicsInterface")
        .def("f", &DynamicsInterface::f)
        .def("F_jac", &DynamicsInterface::F_jac)
        .def("h", &DynamicsInterface::h)
        .def("H_jac", &DynamicsInterface::H_jac)
        .def("nx", &DynamicsInterface::nx)
        .def("ny", &DynamicsInterface::ny);

    py::class_<CT2D, DynamicsInterface>(m, "CT2D")
        .def(py::init<double, double, double, double, double>(),
             py::arg("dt") = 0.2,
             py::arg("omega_eps") = 1e-4,
             py::arg("sensor_x") = 0.0,
             py::arg("sensor_y") = 0.0,
             py::arg("range_eps") = 1e-6)
        .def("f", &CT2D::f)
        .def("F_jac", &CT2D::F_jac)
        .def("h", &CT2D::h)
        .def("H_jac", &CT2D::H_jac)
        .def("nx", &CT2D::nx)
        .def("ny", &CT2D::ny);

    py::class_<CT3D, DynamicsInterface>(m, "CT3D")
        .def(py::init<double, double, double, double, double, double, double>(),
             py::arg("dt") = 0.2,
             py::arg("omega_eps") = 1e-4,
             py::arg("sensor_x") = 0.0,
             py::arg("sensor_y") = 0.0,
             py::arg("sensor_z") = 0.0,
             py::arg("range_eps") = 1e-6,
             py::arg("rho_min") = 1e-2)
        .def("f", &CT3D::f)
        .def("F_jac", &CT3D::F_jac)
        .def("h", &CT3D::h)
        .def("H_jac", &CT3D::H_jac)
        .def("nx", &CT3D::nx)
        .def("ny", &CT3D::ny);

    // --- TEMPLATE: Binding for a new dynamics model ---
    // Uncomment after declaring and implementing your class.
    //
    // py::class_<MyDynamics, DynamicsInterface>(m, "MyDynamics")
    //     .def(py::init<double /*, ... */>(),
    //          py::arg("dt") = 0.1 /*, ... */)
    //     .def("f", &MyDynamics::f)
    //     .def("F_jac", &MyDynamics::F_jac)
    //     .def("h", &MyDynamics::h)
    //     .def("H_jac", &MyDynamics::H_jac)
    //     .def("nx", &MyDynamics::nx)
    //     .def("ny", &MyDynamics::ny);

    // --- EKF ---
    py::class_<EKF>(m, "EKF")
        .def(py::init<const DynamicsInterface&, const MatXd&, const MatXd&, const MatXd&, const VecXd&, const VecXd&>(),
             py::arg("dynamics"),
             py::arg("nominal_x0_cov"),
             py::arg("nominal_Sigma_w"),
             py::arg("nominal_Sigma_v"),
             py::arg("nominal_mu_w"),
             py::arg("nominal_mu_v"),
             py::keep_alive<1, 2>())  // Keep dynamics alive as long as EKF exists
        .def("initial_update", &EKF::initial_update)
        .def("update_step", &EKF::update_step)
        .def("get_P", &EKF::get_P)
        .def("get_nx", &EKF::get_nx)
        .def("get_ny", &EKF::get_ny);

    // --- DR_EKF_CDC ---
    py::class_<DR_EKF_CDC>(m, "DR_EKF_CDC")
        .def(py::init<const DynamicsInterface&, const MatXd&, const MatXd&, const MatXd&,
                       const VecXd&, const VecXd&,
                       double, double, const std::string&, int,
                       double, double, double, double, int, double, int, double,
                       int, double, double, int>(),
             py::arg("dynamics"),
             py::arg("nominal_x0_cov"),
             py::arg("nominal_Sigma_w"),
             py::arg("nominal_Sigma_v"),
             py::arg("nominal_mu_w"),
             py::arg("nominal_mu_v"),
             py::arg("theta_x"),
             py::arg("theta_v"),
             py::arg("solver") = "fw_exact",
             py::arg("dr_update_period") = 1,
             py::arg("fw_beta_minus1") = 1.0,
             py::arg("fw_tau") = 2.0,
             py::arg("fw_zeta") = 2.0,
             py::arg("fw_delta") = 0.05,
             py::arg("fw_max_iters") = 50,
             py::arg("fw_gap_tol") = 1e-4,
             py::arg("fw_bisect_max_iters") = 30,
             py::arg("fw_bisect_tol") = 1e-6,
             py::arg("fw_exact_iters") = 8,
             py::arg("fw_exact_gap_tol") = 1e-6,
             py::arg("oracle_exact_tol") = 1e-8,
             py::arg("oracle_exact_max_iters") = 60,
             py::keep_alive<1, 2>())  // Keep dynamics alive
        .def("initial_update", &DR_EKF_CDC::initial_update)
        .def("update_step", &DR_EKF_CDC::update_step)
        .def("get_P", &DR_EKF_CDC::get_P)
        .def("get_nx", &DR_EKF_CDC::get_nx)
        .def("get_ny", &DR_EKF_CDC::get_ny)
        .def("get_dr_solve_count", &DR_EKF_CDC::get_dr_solve_count)
        .def("get_dr_step_count", &DR_EKF_CDC::get_dr_step_count);

    // --- DR_EKF_TAC ---
    py::class_<DR_EKF_TAC>(m, "DR_EKF_TAC")
        .def(py::init<const DynamicsInterface&, const MatXd&, const MatXd&, const MatXd&,
                       const VecXd&, const VecXd&,
                       double, double, double, const std::string&, int,
                       double, double, double, double, int, double, int, double,
                       int, double, double, int>(),
             py::arg("dynamics"),
             py::arg("nominal_x0_cov"),
             py::arg("nominal_Sigma_w"),
             py::arg("nominal_Sigma_v"),
             py::arg("nominal_mu_w"),
             py::arg("nominal_mu_v"),
             py::arg("theta_x"),
             py::arg("theta_v"),
             py::arg("theta_w"),
             py::arg("solver") = "mosek",
             py::arg("dr_update_period") = 1,
             py::arg("fw_beta_minus1") = 1.0,
             py::arg("fw_tau") = 2.0,
             py::arg("fw_zeta") = 2.0,
             py::arg("fw_delta") = 0.05,
             py::arg("fw_max_iters") = 50,
             py::arg("fw_gap_tol") = 1e-4,
             py::arg("fw_bisect_max_iters") = 30,
             py::arg("fw_bisect_tol") = 1e-6,
             py::arg("fw_exact_iters") = 8,
             py::arg("fw_exact_gap_tol") = 1e-6,
             py::arg("oracle_exact_tol") = 1e-8,
             py::arg("oracle_exact_max_iters") = 60,
             py::keep_alive<1, 2>())  // Keep dynamics alive
        .def("initial_update", &DR_EKF_TAC::initial_update)
        .def("update_step", &DR_EKF_TAC::update_step)
        .def("get_P", &DR_EKF_TAC::get_P)
        .def("get_nx", &DR_EKF_TAC::get_nx)
        .def("get_ny", &DR_EKF_TAC::get_ny)
        .def("get_dr_solve_count", &DR_EKF_TAC::get_dr_solve_count)
        .def("get_dr_step_count", &DR_EKF_TAC::get_dr_step_count);

    // --- Standalone utility functions (for testing) ---
    m.def("posterior_cov", &posterior_cov, "Compute posterior covariance");
    m.def("kalman_stats", [](const MatXd& X_pred, const MatXd& Sigma_v, const MatXd& C_t) {
        auto result = kalman_stats(X_pred, Sigma_v, C_t);
        return py::make_tuple(result.trace_post, result.D_x, result.D_v);
    }, "Compute Kalman statistics (trace_post, D_x, D_v)");
    m.def("oracle_bisection_fast", &oracle_bisection_fast,
          py::arg("Sigma_hat"), py::arg("rho"),
          py::arg("Sigma_ref"), py::arg("D"), py::arg("delta"),
          py::arg("bisect_max_iters") = 30, py::arg("bisect_tol") = 1e-6);
    m.def("oracle_exact_fast", &oracle_exact_fast,
          py::arg("Sigma_hat"), py::arg("rho"), py::arg("D"),
          py::arg("tol") = 1e-8, py::arg("max_iters") = 60);

    // --- DRMMSEFactoredResult ---
    py::class_<DRMMSEFactoredResult>(m, "DRMMSEFactoredResult")
        .def_readonly("wc_Sigma_v", &DRMMSEFactoredResult::wc_Sigma_v)
        .def_readonly("wc_Q_w", &DRMMSEFactoredResult::wc_Q_w)
        .def_readonly("wc_Sigma_w", &DRMMSEFactoredResult::wc_Sigma_w)
        .def_readonly("wc_Xprior", &DRMMSEFactoredResult::wc_Xprior)
        .def_readonly("wc_Xpost", &DRMMSEFactoredResult::wc_Xpost)
        .def_readonly("success", &DRMMSEFactoredResult::success)
        .def_readonly("iterations", &DRMMSEFactoredResult::iterations);

    // --- DRMMSEResult ---
    py::class_<DRMMSEResult>(m, "DRMMSEResult")
        .def_readonly("wc_Sigma_v", &DRMMSEResult::wc_Sigma_v)
        .def_readonly("wc_Sigma_w", &DRMMSEResult::wc_Sigma_w)
        .def_readonly("wc_Xprior", &DRMMSEResult::wc_Xprior)
        .def_readonly("wc_Xpost", &DRMMSEResult::wc_Xpost)
        .def_readonly("success", &DRMMSEResult::success)
        .def_readonly("iterations", &DRMMSEResult::iterations);

    // --- Standalone DR-MMSE TAC solver ---
    // Use lambda to handle optional warm-start matrices (pybind11 can't default MatXd())
    m.def("solve_dr_mmse_tac",
          [](const MatXd& APA, const MatXd& C,
             const MatXd& Sigma_w_hat, const MatXd& Sigma_v_hat,
             double theta_w, double theta_v,
             const std::string& solver,
             const py::object& warm_Sigma_w_obj,
             const py::object& warm_Sigma_v_obj,
             double fw_beta_minus1, double fw_tau,
             double fw_zeta, double fw_delta,
             int fw_max_iters, double fw_gap_tol,
             int fw_bisect_max_iters, double fw_bisect_tol,
             int fw_exact_iters, double fw_exact_gap_tol,
             double oracle_exact_tol, int oracle_exact_max_iters) {
              MatXd warm_w = warm_Sigma_w_obj.is_none() ? MatXd(0, 0) :
                  warm_Sigma_w_obj.cast<MatXd>();
              MatXd warm_v = warm_Sigma_v_obj.is_none() ? MatXd(0, 0) :
                  warm_Sigma_v_obj.cast<MatXd>();
              return solve_dr_mmse_tac(
                  APA, C, Sigma_w_hat, Sigma_v_hat,
                  theta_w, theta_v, solver, warm_w, warm_v,
                  fw_beta_minus1, fw_tau, fw_zeta, fw_delta,
                  fw_max_iters, fw_gap_tol,
                  fw_bisect_max_iters, fw_bisect_tol,
                  fw_exact_iters, fw_exact_gap_tol,
                  oracle_exact_tol, oracle_exact_max_iters);
          },
          py::arg("APA"),
          py::arg("C"),
          py::arg("Sigma_w_hat"),
          py::arg("Sigma_v_hat"),
          py::arg("theta_w"),
          py::arg("theta_v"),
          py::arg("solver") = "fw_exact",
          py::arg("warm_Sigma_w") = py::none(),
          py::arg("warm_Sigma_v") = py::none(),
          py::arg("fw_beta_minus1") = 1.0,
          py::arg("fw_tau") = 2.0,
          py::arg("fw_zeta") = 2.0,
          py::arg("fw_delta") = 0.05,
          py::arg("fw_max_iters") = 50,
          py::arg("fw_gap_tol") = 1e-4,
          py::arg("fw_bisect_max_iters") = 30,
          py::arg("fw_bisect_tol") = 1e-6,
          py::arg("fw_exact_iters") = 8,
          py::arg("fw_exact_gap_tol") = 1e-6,
          py::arg("oracle_exact_tol") = 1e-8,
          py::arg("oracle_exact_max_iters") = 60);

    // --- DRMMSECDCResult ---
    py::class_<DRMMSECDCResult>(m, "DRMMSECDCResult")
        .def_readonly("wc_Sigma_v", &DRMMSECDCResult::wc_Sigma_v)
        .def_readonly("wc_Xprior", &DRMMSECDCResult::wc_Xprior)
        .def_readonly("wc_Xpost", &DRMMSECDCResult::wc_Xpost)
        .def_readonly("success", &DRMMSECDCResult::success)
        .def_readonly("iterations", &DRMMSECDCResult::iterations);

    // --- Standalone DR-MMSE CDC solver ---
    m.def("solve_dr_mmse_cdc",
          [](const MatXd& X_prior_hat, const MatXd& C,
             const MatXd& Sigma_v_hat,
             double theta_x, double theta_v,
             const std::string& solver,
             const py::object& warm_Sigma_v_obj,
             double fw_beta_minus1, double fw_tau,
             double fw_zeta, double fw_delta,
             int fw_max_iters, double fw_gap_tol,
             int fw_bisect_max_iters, double fw_bisect_tol,
             int fw_exact_iters, double fw_exact_gap_tol,
             double oracle_exact_tol, int oracle_exact_max_iters) {
              MatXd warm_v = warm_Sigma_v_obj.is_none() ? MatXd(0, 0) :
                  warm_Sigma_v_obj.cast<MatXd>();
              return solve_dr_mmse_cdc(
                  X_prior_hat, C, Sigma_v_hat,
                  theta_x, theta_v, solver, warm_v,
                  fw_beta_minus1, fw_tau, fw_zeta, fw_delta,
                  fw_max_iters, fw_gap_tol,
                  fw_bisect_max_iters, fw_bisect_tol,
                  fw_exact_iters, fw_exact_gap_tol,
                  oracle_exact_tol, oracle_exact_max_iters);
          },
          py::arg("X_prior_hat"),
          py::arg("C"),
          py::arg("Sigma_v_hat"),
          py::arg("theta_x"),
          py::arg("theta_v"),
          py::arg("solver") = "fw_exact",
          py::arg("warm_Sigma_v") = py::none(),
          py::arg("fw_beta_minus1") = 1.0,
          py::arg("fw_tau") = 2.0,
          py::arg("fw_zeta") = 2.0,
          py::arg("fw_delta") = 0.05,
          py::arg("fw_max_iters") = 50,
          py::arg("fw_gap_tol") = 1e-4,
          py::arg("fw_bisect_max_iters") = 30,
          py::arg("fw_bisect_tol") = 1e-6,
          py::arg("fw_exact_iters") = 8,
          py::arg("fw_exact_gap_tol") = 1e-6,
          py::arg("oracle_exact_tol") = 1e-8,
          py::arg("oracle_exact_max_iters") = 60);

    // --- Standalone DR-MMSE TAC factored solver ---
    m.def("solve_dr_mmse_tac_factored",
          [](const MatXd& APA, const MatXd& C,
             const std::vector<MatXd>& G_list,
             const MatXd& Q_hat, const MatXd& Sigma_v_hat,
             double theta_w, double theta_v,
             const std::string& solver,
             const py::object& warm_Q_w_obj,
             const py::object& warm_Sigma_v_obj,
             double fw_beta_minus1, double fw_tau,
             double fw_zeta, double fw_delta,
             int fw_max_iters, double fw_gap_tol,
             int fw_bisect_max_iters, double fw_bisect_tol,
             int fw_exact_iters, double fw_exact_gap_tol,
             double oracle_exact_tol, int oracle_exact_max_iters) {
              MatXd warm_q = warm_Q_w_obj.is_none() ? MatXd(0, 0) :
                  warm_Q_w_obj.cast<MatXd>();
              MatXd warm_v = warm_Sigma_v_obj.is_none() ? MatXd(0, 0) :
                  warm_Sigma_v_obj.cast<MatXd>();
              return solve_dr_mmse_tac_factored(
                  APA, C, G_list, Q_hat, Sigma_v_hat,
                  theta_w, theta_v, solver, warm_q, warm_v,
                  fw_beta_minus1, fw_tau, fw_zeta, fw_delta,
                  fw_max_iters, fw_gap_tol,
                  fw_bisect_max_iters, fw_bisect_tol,
                  fw_exact_iters, fw_exact_gap_tol,
                  oracle_exact_tol, oracle_exact_max_iters);
          },
          py::arg("APA"),
          py::arg("C"),
          py::arg("G_list"),
          py::arg("Q_hat"),
          py::arg("Sigma_v_hat"),
          py::arg("theta_w"),
          py::arg("theta_v"),
          py::arg("solver") = "fw_exact",
          py::arg("warm_Q_w") = py::none(),
          py::arg("warm_Sigma_v") = py::none(),
          py::arg("fw_beta_minus1") = 1.0,
          py::arg("fw_tau") = 2.0,
          py::arg("fw_zeta") = 2.0,
          py::arg("fw_delta") = 0.05,
          py::arg("fw_max_iters") = 50,
          py::arg("fw_gap_tol") = 1e-4,
          py::arg("fw_bisect_max_iters") = 30,
          py::arg("fw_bisect_tol") = 1e-6,
          py::arg("fw_exact_iters") = 8,
          py::arg("fw_exact_gap_tol") = 1e-6,
          py::arg("oracle_exact_tol") = 1e-8,
          py::arg("oracle_exact_max_iters") = 60);
}
