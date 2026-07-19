/**
 * pybind11 wrapper for KF-GINS with adaptive noise predictor hooks.
 *
 * Exposes the GIEngine class to Python with:
 *   - Step-by-step IMU/GNSS feeding
 *   - State/covariance readout
 *   - mu_v / R override injection (for the learned adapter)
 *   - GNSS diagnostics (innovation, NIS, std) for feature extraction
 */
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "kf-gins/gi_engine.h"
#include "kf-gins/kf_gins_types.h"
#include "common/earth.h"

namespace py = pybind11;

PYBIND11_MODULE(kfgins_py, m) {
    m.doc() = "KF-GINS Python bindings with adaptive noise predictor hooks";

    // ---- IMU data struct ----
    py::class_<IMU>(m, "IMU")
        .def(py::init<>())
        .def_readwrite("time", &IMU::time)
        .def_readwrite("dt", &IMU::dt)
        .def_readwrite("dtheta", &IMU::dtheta)
        .def_readwrite("dvel", &IMU::dvel)
        .def_readwrite("odovel", &IMU::odovel);

    // ---- GNSS data struct ----
    py::class_<GNSS>(m, "GNSS")
        .def(py::init<>())
        .def_readwrite("time", &GNSS::time)
        .def_readwrite("blh", &GNSS::blh)
        .def_readwrite("std", &GNSS::std)
        .def_readwrite("isvalid", &GNSS::isvalid);

    // ---- Attitude struct ----
    py::class_<Attitude>(m, "Attitude")
        .def(py::init<>())
        .def_readwrite("qbn", &Attitude::qbn)
        .def_readwrite("cbn", &Attitude::cbn)
        .def_readwrite("euler", &Attitude::euler);

    // ---- ImuError struct ----
    py::class_<ImuError>(m, "ImuError")
        .def(py::init<>())
        .def_readwrite("gyrbias", &ImuError::gyrbias)
        .def_readwrite("accbias", &ImuError::accbias)
        .def_readwrite("gyrscale", &ImuError::gyrscale)
        .def_readwrite("accscale", &ImuError::accscale);

    // ---- PVA (position, velocity, attitude) struct ----
    py::class_<PVA>(m, "PVA")
        .def(py::init<>())
        .def_readwrite("pos", &PVA::pos)
        .def_readwrite("vel", &PVA::vel)
        .def_readwrite("att", &PVA::att);

    // ---- NavState struct ----
    py::class_<NavState>(m, "NavState")
        .def(py::init<>())
        .def_readwrite("pos", &NavState::pos)
        .def_readwrite("vel", &NavState::vel)
        .def_readwrite("euler", &NavState::euler)
        .def_readwrite("imuerror", &NavState::imuerror);

    // ---- IMU noise parameters ----
    py::class_<ImuNoise>(m, "ImuNoise")
        .def(py::init<>())
        .def_readwrite("gyr_arw", &ImuNoise::gyr_arw)
        .def_readwrite("acc_vrw", &ImuNoise::acc_vrw)
        .def_readwrite("gyrbias_std", &ImuNoise::gyrbias_std)
        .def_readwrite("accbias_std", &ImuNoise::accbias_std)
        .def_readwrite("gyrscale_std", &ImuNoise::gyrscale_std)
        .def_readwrite("accscale_std", &ImuNoise::accscale_std)
        .def_readwrite("corr_time", &ImuNoise::corr_time);

    // ---- GINSOptions ----
    py::class_<GINSOptions>(m, "GINSOptions")
        .def(py::init<>())
        .def_readwrite("initstate", &GINSOptions::initstate)
        .def_readwrite("initstate_std", &GINSOptions::initstate_std)
        .def_readwrite("imunoise", &GINSOptions::imunoise)
        .def_readwrite("antlever", &GINSOptions::antlever);

    // ---- GNSS Diagnostics ----
    py::class_<GIEngine::GNSSDiagnostics>(m, "GNSSDiagnostics")
        .def(py::init<>())
        .def_readonly("innov", &GIEngine::GNSSDiagnostics::innov)
        .def_readonly("S", &GIEngine::GNSSDiagnostics::S)
        .def_readonly("gnss_std_in", &GIEngine::GNSSDiagnostics::gnss_std_in)
        .def_readonly("predicted_meas", &GIEngine::GNSSDiagnostics::predicted_meas)
        .def_readonly("Cov_prior", &GIEngine::GNSSDiagnostics::Cov_prior)
        .def_readonly("K", &GIEngine::GNSSDiagnostics::K)
        .def_readonly("dx", &GIEngine::GNSSDiagnostics::dx)
        .def_readonly("nis", &GIEngine::GNSSDiagnostics::nis)
        .def_readonly("valid", &GIEngine::GNSSDiagnostics::valid);

    // ---- Propagation Record (for DR solver) ----
    py::class_<GIEngine::PropagationRecord>(m, "PropagationRecord")
        .def_readonly("Phi", &GIEngine::PropagationRecord::Phi)
        .def_readonly("G", &GIEngine::PropagationRecord::G)
        .def_readonly("dt", &GIEngine::PropagationRecord::dt);

    // ---- GIEngine (the filter) ----
    py::class_<GIEngine>(m, "GIEngine")
        .def(py::init<GINSOptions&>())
        .def("addImuData", &GIEngine::addImuData,
             py::arg("imu"), py::arg("compensate") = false)
        .def("addGnssData", &GIEngine::addGnssData)
        .def("newImuProcess", &GIEngine::newImuProcess)
        .def("getNavState", &GIEngine::getNavState)
        .def("getCovariance", &GIEngine::getCovariance)
        .def("timestamp", &GIEngine::timestamp)
        // Adapter hooks
        .def("setOverrideMuV", &GIEngine::setOverrideMuV)
        .def("setOverrideR", &GIEngine::setOverrideR)
        .def("clearOverrides", &GIEngine::clearOverrides)
        .def("lastGNSSDiagnostics", &GIEngine::lastGNSSDiagnostics,
             py::return_value_policy::reference_internal)
        // DR mode
        .def("setDRMode", &GIEngine::setDRMode)
        .def("drMode", &GIEngine::drMode)
        .def("propagationRecords", &GIEngine::propagationRecords,
             py::return_value_policy::reference_internal)
        .def("lastCallHadGnssUpdate", &GIEngine::lastCallHadGnssUpdate)
        .def("lastHGnss", &GIEngine::lastHGnss,
             py::return_value_policy::reference_internal)
        .def("lastRGnss", &GIEngine::lastRGnss,
             py::return_value_policy::reference_internal)
        .def("lastInnovGnss", &GIEngine::lastInnovGnss,
             py::return_value_policy::reference_internal)
        .def("getQc", &GIEngine::getQc,
             py::return_value_policy::reference_internal)
        .def("setCov", &GIEngine::setCov)
        .def("setDx", &GIEngine::setDx)
        .def("doStateFeedback", &GIEngine::doStateFeedback)
        .def("finalizeDR", &GIEngine::finalizeDR);

    // ---- Earth model utilities (for coordinate conversion in Python) ----
    m.def("gravity", &Earth::gravity);
    m.def("blh2ecef", &Earth::blh2ecef);
    m.def("ecef2blh", &Earth::ecef2blh);
    m.def("global2local", py::overload_cast<const Eigen::Vector3d&, const Eigen::Vector3d&>(
        &Earth::global2local));
    m.def("local2global", py::overload_cast<const Eigen::Vector3d&, const Eigen::Vector3d&>(
        &Earth::local2global));
}
